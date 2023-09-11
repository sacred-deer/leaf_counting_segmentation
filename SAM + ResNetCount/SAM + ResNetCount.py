import os, argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")

import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset import PhenotypeDataset, train_augmentation, validation_augmentation, get_preprocessing
from models import ResNetCount

def visualize(total_rows, image, mask):
    plt.figure(figsize=(10,8))
    for i in range(total_rows):
        plt.subplot(total_rows,2,i*2+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(np.moveaxis(image, 0, -1))

        plt.subplot(total_rows,2,i*2+2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mask)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))




def load_sam(sam_checkpoint, model_type):
	#for now
	device = "cpu"

	sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
	sam.to(device=device)

	return SamAutomaticMaskGenerator(sam)

def main(args):

    # Loading dataset
    phenotype_dataset_train = PhenotypeDataset(root_dir=os.path.join(args.root_dir, 'train'), transform=train_augmentation(), preprocessing=get_preprocessing(preprocess_input))
    phenotype_dataset_test = PhenotypeDataset(root_dir=os.path.join(args.root_dir, 'test'), transform=validation_augmentation(), preprocessing=get_preprocessing(preprocess_input))

    batch_size = 4
    train_dataloader = DataLoader(phenotype_dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(phenotype_dataset_test, batch_size=32, shuffle=False)
    mask_generator = load_sam(model_path, model_type)

    for idx, batch in enumerate(train_dataloader):
        print(idx, batch['image'].size(), batch['count'].size())

        if idx == 3:
            plt.figure(figsize=(10,8))
            for i in range(batch['image'].shape[0]):
                plt.subplot(batch['image'].shape[0],2,i*2+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.transpose(batch['image'][i].detach().cpu().numpy(), (1,2,0)))

                plt.subplot(batch['image'].shape[0],2,i*2+2)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(batch['mask'][i].detach().cpu().numpy())
            
            plt.savefig('data.png')
            break
    
    if args.model == 'resnet':
        model = ResNetCount(models.resnet50()).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam([
            {'params': model.pretrained_model.parameters()},
            {'params': model.fc1.parameters()},
            {'params': model.fc2.parameters(), 'weight_decay': 0.01},
            {'params': model.classifier.parameters()}
        ], lr=1e-4)
    else:
        if args.model == 'unet':
            model = smp.Unet(
                encoder_name="resnet50",    
                encoder_weights="imagenet",
                in_channels=3,
                classes=30,
            ).to(device)
            

        elif args.model == 'fpn':
            model = smp.FPN(
                encoder_name="resnet50",    
                encoder_weights="imagenet",
                in_channels=3,
                classes=30,
            ).to(device)
        
        elif args.model == 'deeplabv3':
            model = smp.DeepLabV3(
                encoder_name="resnet50",    
                encoder_weights="imagenet",
                in_channels=3,
                classes=30,
            ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)


    ckpt_dir = './ckpts/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    max_epochs = 100

    for epoch in tqdm(range(max_epochs)):  # loop over the dataset multiple times
        train_loss = 0.0
        validation_metrics = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs
            inputs, masks, counts = data['image'].to(device), data['mask'].to(device), data['count'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            train_loss += loss
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for j, data in enumerate(test_dataloader, 0):
                validation_images, validation_masks, validation_counts = data['image'].to(device), data['mask'].to(device), data['count'].to(device)
                if args.model == 'resnet':
                    validation_output = torch.ceil(model(validation_images))
                    validation_metrics += torch.mean(torch.abs(validation_counts.unsqueeze(1) - validation_output))
                
                else:
                    validation_output = torch.argmax(torch.softmax(model(validation_images), -1), 1)
                    validation_target = validation_masks
                    tp, fp, fn, tn = smp.metrics.get_stats(validation_output.type(torch.int64), validation_target, mode='multiclass', num_classes=30)
                    # then compute metrics with required reduction (see metric docs)
                    validation_metrics += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        if (epoch+1) % 10 == 0:
            torch.save(model, os.path.join(ckpt_dir, '{}_epoch_{}.pth'.format(args.model, epoch+1)))
            print('Model saved!')

        # print statistics
        print(f'[{epoch + 1}] loss: {(loss / len(train_dataloader)):.3f} validation metric: {validation_metrics / len(test_dataloader):.3f}')

    print('Finished Training')

    best_model = torch.load('./ckpts/{}_epoch_100.pth'.format(args.model))

    plt.figure(figsize=(10,8))
    for i in range(5):
        n = np.random.choice(len(phenotype_dataset_test))
        data = phenotype_dataset_test[n]
        image, gt_mask, gt_count = data['image'], data['mask'], data['count']

        pr_mask = torch.argmax(torch.softmax(best_model(torch.from_numpy(image).unsqueeze(0).to(device)), -1), 1)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        plt.subplot(5,3,i*3+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(np.moveaxis(image, 0, -1))

        plt.subplot(5,3,i*3+2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gt_mask.detach().cpu().numpy())
        
        plt.subplot(5,3,i*3+3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(pr_mask)


    plt.savefig('prediction.png')

model_path = "../../sam_vit_h_4b8939.pth"
model_type = "vit_h"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='absolute directory of data')
parser.add_argument('--model', type=str, help='model name that will be trained')

args = parser.parse_args()
if not args.root_dir:
    raise('root_dir is not provided!')
main(args)

