import os, argparse
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, utils
from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from torch_utils.engine import train_one_epoch, evaluate 
from dataset import PhenotypeDataset, train_augmentation, validation_augmentation, get_preprocessing
from models import ResNetCount

def plot_losses(training_losses, validation_losses, model_name, classifier_losses=[], regression_losses=[]):
    with open('{}_training_losses.txt'.format(model_name), 'w+') as file:
        file.write('\n'.join([str(l) for l in training_losses]))
    with open('{}_validation_losses.txt'.format(model_name), 'w+') as file:
        file.write('\n'.join([str(l) for l in validation_losses]))

    plt.figure()
    plt.plot(range(1, len(training_losses)+1), training_losses, label='training')
    plt.plot(range(1, len(validation_losses)+1), validation_losses, label='validation')
    plt.xlabel('Epoch')
    if model_name == 'fasterrcnn':
        plt.ylabel('Classification + Regression Loss')
    elif model_name == 'resnet':
        plt.ylabel('L1 Loss')
    elif model_name == 'unet':
        plt.ylabel('Dice Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('{}_losses.png'.format(model_name))

    if len(classifier_losses) > 0 and len(regression_losses) > 0:
        with open('{}_classifier_losses.txt'.format(model_name), 'w+') as file:
            file.write('\n'.join([str(l) for l in classifier_losses]))
        with open('{}_regression_losses.txt'.format(model_name), 'w+') as file:
            file.write('\n'.join([str(l) for l in regression_losses]))
        plt.figure()
        plt.plot(range(1, len(classifier_losses)+1), classifier_losses, color='orange', label='classifier')
        plt.plot(range(1, len(regression_losses)+1), regression_losses, color='green', label='regression')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc='upper right')
        plt.show()
        plt.savefig('{}_individual_losses.png'.format(model_name))


def get_fasterrcnn_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # convert_relu_to_mish(model.backbone)
    print(model)

    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_maskrcnn_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    # convert_relu_to_mish(model.backbone)
    print(model)

    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    phenotype_dataset_train = PhenotypeDataset(root_dir=os.path.join(args.root_dir, 'train'), transform=train_augmentation(), device=device)
    phenotype_dataset_test = PhenotypeDataset(root_dir=os.path.join(args.root_dir, 'test'), transform=validation_augmentation(), device=device)

    batch_size = 4

    train_dataloader = DataLoader(phenotype_dataset_train, batch_size=batch_size, collate_fn=phenotype_dataset_train.collate_fn, shuffle=True)
    test_dataloader = DataLoader(phenotype_dataset_test, batch_size=8, collate_fn=phenotype_dataset_test.collate_fn, shuffle=False)

    for idx, batch in enumerate(train_dataloader):
        if idx == 3:
            plt.figure(figsize=(10,8))
            for i in range(batch['image'].shape[0]):
                plt.subplot(batch['image'].shape[0],3,i*3+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.transpose(batch['image_tensor'][i].detach().cpu().numpy(), (1,2,0)))

                plt.subplot(batch['image'].shape[0],3,i*3+2)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(batch['mask'][i].detach().cpu().numpy())

                plt.subplot(batch['image'].shape[0],3,i*3+3)
                bbox = np.array(batch['target'][i]['boxes'].detach().cpu().numpy())
                print(batch['image'][i].shape)
                print(batch['image_id'][i], bbox.shape)
                img_with_bbox = utils.draw_bounding_boxes(torch.tensor(np.transpose(batch['image'][i], (2, 0, 1)), dtype=torch.uint8), torch.tensor(bbox))
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.transpose(img_with_bbox, (1,2,0)))
            
            plt.savefig('data.png')

            b_mask = batch['b_mask'][0]
            plt.figure(figsize=(20, 15))
            for i in range(len(b_mask)):
                plt.subplot(1, len(b_mask), i+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(b_mask[i])
            plt.savefig('binary_masks.png')
            break
    # ResNet Regressor defintion
    if 'resnet' in args.model:
        model = ResNetCount(models.resnet50()).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW([
            {'params': model.pretrained_model.parameters()},
            {'params': model.fc1.parameters()},
            {'params': model.fc2.parameters(), 'weight_decay': 0.01},
            {'params': model.classifier.parameters()}
        ], lr=1e-4)

    # R-CNN models definition
    elif args.model == 'fasterrcnn':
        model = get_fasterrcnn_model(num_classes=2).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=5e-4)
    elif args.model == 'maskrcnn':
        model = get_maskrcnn_model(num_classes=2).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=5e-4)

    # Semantic Segmentation models definition
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

        criterion = smp.losses.DiceLoss(mode='multiclass', smooth=.2)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

    max_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max_epochs+25,
        T_mult=1,
        verbose=True
    )


    ckpt_dir = './ckpts/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    training_losses = []
    validation_losses = []
    classifier_losses = []
    regression_losses = []

    for epoch in tqdm(range(max_epochs)):  # loop over the dataset multiple times
        train_loss = 0.0
        validation_loss = 0.0
        classifier_loss = 0.0
        regression_loss = 0.0
        model.train()

        if args.model == 'fasterrcnn' or args.model == 'maskrcnn':
            _, loss_list, _, _ = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100, scheduler=scheduler)
            train_loss = sum(loss_list)
        else:
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs
                images, masks, counts = data['image_tensor'],  data['mask'], data['count'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if 'resnet' in args.model:
                    if 'sam_2' in args.model:
                        sam_mask = data['sam_mask_2'].to(device)
                        outputs = model(sam_mask)
                    elif 'sam' in args.model:
                        sam_mask = data['sam_mask'].to(device)
                        outputs = model(sam_mask)
                    else:
                        outputs = model(images.to(device))
                    loss = criterion(outputs, counts)

                else:
                    outputs = model(images.to(device))
                    loss = criterion(outputs, masks.to(device).to(torch.int64))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        with torch.no_grad():
            if args.model == 'fasterrcnn' or args.model == 'maskrcnn':
                _, validation_loss_list, classifier_loss_list, regression_loss_list = train_one_epoch(model, optimizer, test_dataloader, device, max_epochs+1, print_freq=50, scheduler=scheduler, validation=True)
                validation_loss = sum(validation_loss_list)
                classifier_loss = sum(classifier_loss_list)
                regression_loss = sum(regression_loss_list)
            else:
                model.eval()
                for j, v_data in enumerate(test_dataloader, 0):
                    validation_images, validation_masks, validation_counts = v_data['image_tensor'], v_data['mask'], v_data['count'].to(device)

                    if 'resnet' in args.model:
                        if 'sam_2' in args.model:
                            validation_sam_mask = v_data['sam_mask_2'].to(device)
                            validation_output = torch.ceil(model(validation_sam_mask))
                        elif 'sam' in args.model:
                            validation_sam_mask = v_data['sam_mask'].to(device)
                            validation_output = torch.ceil(model(validation_sam_mask))
                        else:
                            validation_output = torch.ceil(model(validation_images.to(device)))
                        validation_loss += torch.mean(torch.abs(validation_counts.unsqueeze(1) - validation_output)).item()
                    
                    else:
                        validation_target = validation_masks.to(device)
                        
                        validation_output = model(validation_images.to(device))
                        validation_loss += criterion(validation_output, validation_target.to(torch.int64)).item()
                        # validation_output = torch.argmax(torch.softmax(model(validation_images), -1), 1)
                        # tp, fp, fn, tn = smp.metrics.get_stats(validation_output.type(torch.int64), validation_target, mode='multiclass', num_classes=30)
                        # # then compute metrics with required reduction (see metric docs)
                        # validation_metrics += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        if (epoch+1) % 10 == 0:
            torch.save(model, os.path.join(ckpt_dir, '{}_epoch_{}.pth'.format(args.model, epoch+1)))
            print('Model saved!')

        avg_training_loss = train_loss / len(train_dataloader)
        avg_validation_loss = validation_loss / len(test_dataloader)
        training_losses.append(avg_training_loss)
        validation_losses.append(avg_validation_loss)
        if 'rcnn' in args.model:
            classifier_losses.append(classifier_loss / len(test_dataloader))
            regression_losses.append(regression_loss / len(test_dataloader))
        # print statistics
        print(f'[{epoch + 1}] loss: {avg_training_loss:.3f} validation loss: {avg_validation_loss:.3f}')

    print('Finished Training')

    plot_losses(training_losses, validation_losses, args.model, classifier_losses=classifier_losses, regression_losses=regression_losses)

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='absolute directory of data')
parser.add_argument('--model', type=str, help='model name that will be trained')

args = parser.parse_args()
if not args.root_dir:
    raise('root_dir is not provided!')
main(args)