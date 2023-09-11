import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import segmentation_models_pytorch as smp

from skimage.segmentation import mark_boundaries
from lime import lime_image

import sys, os, argparse

from dataset import PhenotypeDataset, validation_augmentation, get_preprocessing
from models import ResNetCount


def setup_dataloader(path, augmentation=None, batch_size=1):
  preprocess_input = smp.encoders.get_preprocessing_fn('resnet18', pretrained='imagenet')
  if augmentation:
    phenotype_dataset = PhenotypeDataset(root_dir = path, transform = augmentation(), preprocessing=get_preprocessing(preprocess_input))
  else:
    phenotype_dataset = PhenotypeDataset(root_dir = path, transform = validation_augmentation())

  return DataLoader(phenotype_dataset, batch_size=batch_size, shuffle=True)


def get_lime_expl(trained_model, explainer, test_dataloader, device, save_path, num_explanations=5, num_samples=10000):

  def batch_predict(images):
    temp_image = np.transpose(images, (0,3,1,2))
    batch = torch.from_numpy(temp_image).to(device).float()

    with torch.no_grad():
      predictions = trained_model(batch)  

    return predictions.detach().cpu().numpy()

  print("Getting LIME explanations")

  actual_counts = np.zeros(shape=(num_explanations, 2)).astype(int)

  ## Currently only handles batch size of 1 for test_dataloader
  for i, data in enumerate(test_dataloader, 0):
    if i >= num_explanations:
      break

    image, count = data['image'], data['count']

    print('\nExplanation for test instance at index ', i)
    print('Actual count: ', count.item())
    actual_counts[i] = [i, count.item()]

    ## Prep image for LIME
    image_transform = np.array(image).squeeze()
    if image_transform.shape[0] == 3:
      image_transform = np.transpose(image_transform, (1,2,0))

    ## Train LIME
    exp = explainer.explain_instance(image_transform, 
                                     batch_predict, 
                                     top_labels=1,
                                     hide_color=0,
                                     num_samples=num_samples)
    
    temp, mask = exp.get_image_and_mask(exp.top_labels[0], 
                                        positive_only=False, 
                                        num_features=10, 
                                        hide_rest=False)
    
    img_boundary = mark_boundaries(temp, mask)
    plt.imshow(img_boundary)
    plt.savefig(os.path.join(save_path, "explanation_{}.png".format(i)))

  np.savetxt(os.path.join(save_path, "explanations.csv"), actual_counts, delimiter=",", header='id,counts')
  print("Done!")


def run_lime(data_path, model_path, save_path, do_augmentation=False, num_explanations=5, num_samples=10000):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  ## Dataloader
  if do_augmentation:
    test_dl = setup_dataloader(data_path, validation_augmentation)
  else:
    test_dl = setup_dataloader(data_path)

  ## Model
  saved_model = torch.load(model_path, map_location=device)
  saved_model.to(device)

  expl = lime_image.LimeImageExplainer()
  get_lime_expl(saved_model, 
                expl, 
                test_dl, 
                device, 
                save_path, 
                num_explanations=num_explanations, 
                num_samples=num_samples)


def main(args):
  data_full_path = os.path.join(args.root_dir, 'data/Plant_Phenotyping_Datasets/Plant/test')
  model_full_path = os.path.join(args.root_dir, 'ckpts/resnet_epoch_99.pth')
  save_full_path = os.path.join(args.root_dir, 'results')

  run_lime(data_full_path, model_full_path, save_full_path, do_augmentation=True)


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='absolute directory of project')

args = parser.parse_args()
main(args)