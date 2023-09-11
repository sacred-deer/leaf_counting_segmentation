import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class PhenotypeDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocessing=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing

        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.extract_plantnames()

    def extract_plantnames(self):
        # csv file containing target values
        target_csv = 'Leaf_counts.csv'
        self.plant_names = []
        all_dfs = []

        # append all plant names to a list
        types = os.listdir(self.root_dir)
        for type in types:
            type_dir = os.path.join(self.root_dir, type)
            plant_names = [os.path.join(type_dir, '_'.join(name.split('_')[:-1])) for name in os.listdir(type_dir) if not name[0].isupper()]
        
            # remove duplicates
            plant_names = list(dict.fromkeys(plant_names))
            self.plant_names += plant_names

            df = pd.read_csv(os.path.join(type_dir, target_csv), index_col=0, names=['count'])
            all_dfs.append(df)
        
        self.data_dict = pd.concat(all_dfs).to_dict('index')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_name = self.plant_names[idx]

        image = io.imread(img_name + '_rgb.png')
        mask = np.array(Image.open(img_name + '_label.png'))
        count = torch.tensor(self.data_dict[img_name.split('/')[-1]]['count'])

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']


        # image = self.basic_transform(image)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        
        return {'image': image, 'mask': mask, 'count': count}
    

