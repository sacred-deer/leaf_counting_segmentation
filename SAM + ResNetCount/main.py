import os, argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import PhenotypeDataset, train_augmentation, validation_augmentation, get_preprocessing

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

# data_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((300, 300))])

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name="resnet34",    
        encoder_weights="imagenet",
        in_channels=3,
        classes=30,
    ).to(device)
    preprocess_input = smp.encoders.get_preprocessing_fn('resnet18', pretrained='imagenet')

    phenotype_dataset_train = PhenotypeDataset(root_dir=os.path.join(args.root_dir, 'train'), transform=train_augmentation(), preprocessing=get_preprocessing(preprocess_input))
    phenotype_dataset_test = PhenotypeDataset(root_dir=os.path.join(args.root_dir, 'test'), transform=validation_augmentation(), preprocessing=get_preprocessing(preprocess_input))

    batch_size = 4
    train_dataloader = DataLoader(phenotype_dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(phenotype_dataset_test, batch_size=32, shuffle=False)

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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    max_epochs = 100
    for epoch in tqdm(range(max_epochs)):  # loop over the dataset multiple times
        train_loss = 0.0
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
                validation_output = torch.argmax(model(validation_images), 1)
                validation_target = validation_masks
                tp, fp, fn, tn = smp.metrics.get_stats(validation_output.type(torch.int64), validation_target, mode='multiclass', num_classes=30)
                # then compute metrics with required reduction (see metric docs)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        if epoch % 10 == 0:
            torch.save(model, './ckpts/model_epoch_{}.pth'.format(epoch))
            print('Model saved!')

        # print statistics
        print(f'[{epoch + 1}] loss: {(loss / len(train_dataloader)):.3f} iou score: {iou_score:.3f}')

    print('Finished Training')

    best_model = torch.load('./ckpts/model_epoch_99.pth')

    plt.figure(figsize=(10,8))
    for i in range(5):
        n = np.random.choice(len(phenotype_dataset_test))
        data = phenotype_dataset_test[n]
        image, gt_mask, gt_count = data['image'], data['mask'], data['count']

        pr_mask = torch.argmax(best_model(torch.from_numpy(image).unsqueeze(0).to(device)), 1)
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

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='absolute directory of data')

args = parser.parse_args()
main(args)