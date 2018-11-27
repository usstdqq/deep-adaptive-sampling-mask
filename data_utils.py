import argparse
import os
import h5py
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import Compose, CenterCrop, Scale
from tqdm import tqdm

import cv2
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])

    
    
class DatasetFromH5(Dataset):
    def __init__(self, image_dataset_dir, input_transform=None, target_transform=None):
        super(DatasetFromH5, self).__init__()
        
        image_h5_file = h5py.File(image_dataset_dir, 'r')
        image_dataset = image_h5_file['data'].value
        
        self.image_datasets = image_dataset
        self.total_count = image_dataset.shape[0]
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        image = self.image_datasets[index]
        image =  torch.from_numpy(image)

        return image

    def __len__(self):
        return self.total_count




