import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
import torchvision.transforms as T

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.augment_data = augment_data
        
        images_dir = os.path.join(data_dir, 'scans')
        masks_dir = os.path.join(data_dir, 'labels')
        
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        
        
        split_idx = int(len(self.image_paths) * train_test_split)
        if self.train:
            self.image_paths = self.image_paths[:split_idx]
            self.mask_paths = self.mask_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.mask_paths = self.mask_paths[split_idx:]
            
        # transform to resize and normalize the images
        self.transform = T.Compose([
            # desired size
            T.Resize((size, size)),
            # to tensor
            T.ToTensor(),
            # normalize to [0, 1]
            T.Normalize(mean=[0.5], std=[0.5])  
        ])

    def __getitem__(self, idx):
        # todo

        # load image and mask from index idx of your data
        image = Image.open(self.image_paths[idx]).convert('L')  # Load as grayscale
        mask = Image.open(self.mask_paths[idx]).convert('L')  
        
        # resize and normalization
        image = self.transform(image)
        mask = self.transform(mask)
        
        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                # todo
                # flip image vertically
                image = T.functional.vflip(image)
                mask = T.functional.vflip(mask)
            elif augment_mode == 1:
                # todo
                # flip image horizontally
                image = T.functional.hflip(image)
                mask = T.functional.hflip(mask)
            elif augment_mode == 2:
                # todo
                # zoom image
                i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(1.0, 1.0))
                image = T.functional.resized_crop(image, i, j, h, w, (self.size, self.size))
                mask = T.functional.resized_crop(mask, i, j, h, w, (self.size, self.size))
            else:
                # todo
                # rotate image
                angle = np.random.uniform(-30, 30)
                image = T.functional.rotate(image, angle)
                mask = T.functional.rotate(mask, angle)

        # todo
        # return image and mask in tensors
        return image, mask

    def __len__(self):
        return len(self.image_paths)


