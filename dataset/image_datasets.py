import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.ops import masks_to_boxes
from PIL import Image

class PhenotypeDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocessing=None, device='cpu'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.device = device

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
            plant_names = [os.path.join(type_dir, '_'.join(name.split('.')[0].split('_')[:2])) for name in os.listdir(type_dir) if not name[0].isupper()]
        
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
        image_id = img_name.split('/')[-1]
        image = io.imread(img_name + '_rgb.png')
        mask = np.array(Image.open(img_name + '_label.png'))
        try:
            sam_mask = np.array(Image.open(img_name + '_segmented.png'))
        except:
            sam_mask = None

        try:
            sam_mask_2 = np.array(Image.open(img_name + '_segmented_2.png'))
        except:
            sam_mask_2 = None

        count = torch.tensor(self.data_dict[image_id]['count'])
        binary_masks = [np.where(mask == i, 1, 0) for i in range(1, np.amax(mask)+1) if np.count_nonzero(mask == i) > 0]
        binary_masks_tensor = torch.tensor(np.array(binary_masks), dtype=torch.uint8)
        # binary_masks_tensor = [m for m in binary_masks_tensor if torch.count_nonzero(m) > 0]
        bbox = masks_to_boxes(binary_masks_tensor)
        # area of the bounding boxes
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((bbox.shape[0],), dtype=torch.int64)
        classes = torch.tensor([1]*len(bbox), dtype=torch.int64).to(self.device)

        if self.transform:
            sample = self.transform(image=image, masks=[mask] + binary_masks, bboxes=bbox, class_labels=classes)
            image, bbox = sample['image'], sample['bboxes']
            masks = sample['masks']
            mask = masks[0]
            binary_masks = masks[1:]

        if self.preprocessing:
            sample = self.preprocessing(image=image, masks=[mask] + binary_masks, bboxes=bbox, class_labels=classes)
            image, bbox = sample['image'], sample['bboxes']
            masks = sample['masks']
            mask = masks[0]
            binary_masks = masks[1:]

        bbox_tensor = torch.as_tensor(bbox)
        binary_masks_tensor = torch.tensor(np.array(binary_masks), dtype=torch.uint8)
        targets = {
            'boxes': bbox_tensor, 
            'masks': binary_masks_tensor,
            'labels': classes,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': torch.tensor(idx)
        }

        image_tensor = self.basic_transform(image)
        if sam_mask is not None:
            sam_mask = self.basic_transform(sam_mask)
        if sam_mask_2 is not None:
            sam_mask_2 = self.basic_transform(sam_mask_2)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        targets['boxes'] = torch.tensor(bbox, dtype=torch.float32).to(self.device)
        
        return {'image_id': image_id, 'image': image, 'image_tensor': image_tensor, 'mask': mask, 'sam_mask': sam_mask, 'sam_mask_2': sam_mask_2, 'b_mask': binary_masks, 'count': count, 'target': targets}
    
    def collate_fn(self, batch):
        image_ids = []
        images = []
        image_tensor = []
        mask = []
        sam_mask = []
        sam_mask_2 = []
        b_mask = []
        targets = []
        count = []
        
        for b in batch:
            image_ids.append(b['image_id'])
            images.append(b['image'])
            image_tensor.append(b['image_tensor'])
            mask.append(b['mask'])
            sam_mask.append(b['sam_mask'])
            sam_mask_2.append(b['sam_mask_2'])
            b_mask.append(b['b_mask'])
            targets.append(b['target'])
            count.append(b['count'])
            
        images = np.array(images)
        try:
            image_tensor = torch.stack(image_tensor, dim=0).type(torch.float32)
        except:
            pass
        if None not in sam_mask:
            sam_mask = torch.stack(sam_mask, dim=0)
        if None not in sam_mask_2:
            sam_mask_2 = torch.stack(sam_mask_2, dim=0)
        try:
            mask = torch.stack(mask, dim=0)
        except:
            pass
        count = torch.stack(count, dim=0)[:, None]

        return {'image_id': image_ids, 'image': images, 'image_tensor': image_tensor, 'mask': mask, 'sam_mask': sam_mask, 'sam_mask_2': sam_mask_2, 'b_mask': b_mask, 'target': targets, 'count': count}
    

