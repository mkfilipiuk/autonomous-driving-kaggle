import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import numpy as np
import os
import cv2

from PIL import Image


class KaggleImageMaskDataset(Dataset):
    """
    This class prepares masks for training
    For classification for N classes + background(denoted in files as -1) it
    prepares N+1 images of binary classification where (N+1)th is background

    For correspondence maps it prepares num_of_colors binary maps.

    Therefore element of dataset looks as follows:
    (image[W, H], (classification[N+1, W, H], u_channel[num_of_colors, W, H], v_channel[num_of_colors, W, H]), prediction_string)

    where W, H = 3384//4, 2710//4
    """
    
    def __init__(self, path, is_train=True, num_of_colors=256, num_of_models=79, image_size=(3384//8, 2710//8)):
        self.images_dir = os.path.join(path, "train_images" if is_train else "test_images")
        self.masks_dir = os.path.join(path, "train_targets" if is_train else "test_targets")
        data_csv = pd.read_csv(os.path.join(path, "train.csv"))
        self.images_ID = data_csv.ImageId
        self.predition_strings = data_csv.PredictionString
        self.num_of_colors = num_of_colors
        self.num_of_models = num_of_models
        self.im_transform = transforms.Compose([
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.images_ID)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.images_ID[idx]+".jpg")
        image = Image.open(img_name)

        if not self.is_train:
            return image

        mask_name = os.path.join(self.masks_dir, self.images_ID[idx]+".npy")
        masks = np.load(mask_name)
        masks = self.target_transform(masks)
        masks = masks.type(torch.LongTensor)

        prediction_string = self.predition_strings[idx]

        image = self.im_transform(image)
        
        return image, (masks[0], masks[1], masks[2]), prediction_string
