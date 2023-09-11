import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import torchvision.transforms as transforms
from dataset import PhenotypeDataset, train_augmentation, validation_augmentation, get_preprocessing
import segmentation_models_pytorch as smp

from torch_utils.engine import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
phenotype_dataset_test = PhenotypeDataset(root_dir='/home/data/Plant_Phenotyping_Datasets/Plant/test', transform=validation_augmentation())
test_dataloader = DataLoader(phenotype_dataset_test, batch_size=8, collate_fn=phenotype_dataset_test.collate_fn, shuffle=False)
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# best_model = torch.load('./ckpts/maskrcnn_epoch_40.pth').to(device)
trans = transforms.Compose([
    transforms.ToTensor()])

def fasterrcnn_evaluate(dataset, dataloader, lower_threshold=0.0, higher_threshold=1.0):
    model_dir = os.path.join(output_dir, 'fasterrcnn')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    best_model = torch.load('./ckpts/fastmaskrcnn_epoch_100.pth').to(device)
    evaluate(best_model, dataloader, device=device)
    l1 = []
    mse = []
    for i in range(len(dataset)):
        data_dict = dataset[i]
        image_id, image_np, image, target, count = data_dict['image_id'], data_dict['image'], data_dict['image_tensor'], data_dict['target'], data_dict['count']
        target = [{k: v.to(device) for k, v in target.items()}]
        image = image.unsqueeze(0).to(device)
        output_dict = best_model(image, target)
        count = count.item()
        if len(output_dict[0]['boxes']) > 0:
            boxes = output_dict[0]['boxes'].type(torch.int32)
            scores = output_dict[0]['scores']
            # filter out boxes according to `detection_threshold`
            boxes = boxes[(scores >= lower_threshold) & (scores <= higher_threshold)]
            pred_count = len(boxes)
            l1.append(abs(count-pred_count))
            mse.append((pred_count-count)**2)
            # draw image with predicted bounding boxes
            plt.figure()

            plt.subplot(1, 2, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('Prediction (count: {})'.format(pred_count))
            img_with_bbox = utils.draw_bounding_boxes(torch.tensor(np.transpose(image_np, (2, 0, 1)), dtype=torch.uint8), boxes, colors='yellow', width=5)

            plt.imshow(np.transpose(img_with_bbox, (1,2,0)))
            plt.subplot(1, 2, 2)
            plt.xticks([])
            plt.yticks([])
            plt.title('Ground Truth (count: {})'.format(count))
            ground_truth = utils.draw_bounding_boxes(torch.tensor(np.transpose(image_np, (2, 0, 1)), dtype=torch.uint8), target[0]['boxes'], colors='yellow', width=5)
            plt.imshow(np.transpose(ground_truth, (1,2,0)))

            plt.savefig(os.path.join(model_dir, '{}_output.png'.format(image_id)), bbox_inches='tight')
            plt.close()
    print('L1: {}'.format(np.mean(np.array(l1))))
    print('MSE: {}'.format(np.mean(np.array(mse))))


def resnet_evaluate(dataset, dataloader):
    model_dir = os.path.join(output_dir, 'resnet')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    best_model = torch.load('./ckpts/resnet_epoch_60.pth').to(device)
    l1 = []
    mse = []
    for i in range(len(dataset)):
        data_dict = dataset[i]
        image, count = data_dict['image_tensor'],  data_dict['count']
        image = image.unsqueeze(0).to(device)
        if 'sam_2' in args.model:
            sam_mask = data_dict['sam_mask_2'].unsqueeze(0).to(device)
            output = best_model(sam_mask)
        elif 'sam' in args.model:
            sam_mask = data_dict['sam_mask'].unsqueeze(0).to(device)
            output = best_model(sam_mask)
        else:
            output = best_model(image)
        l1.append(torch.abs(count-output).squeeze().item())
        mse.append(((count-output)**2).squeeze().item())
    
    print('L1: {}'.format(np.mean(np.array(l1))))
    print('MSE: {}'.format(np.mean(np.array(mse))))


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model name that will be inferenced and evaluated')

args = parser.parse_args()
if not args.model:
    raise('model is not provided!')

if args.model == 'fasterrcnn':
    fasterrcnn_evaluate(phenotype_dataset_test, test_dataloader, lower_threshold=0.7)
elif 'resnet' in args.model:
    resnet_evaluate(phenotype_dataset_test, test_dataloader)