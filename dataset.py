import glob
import os

import pandas as pd
import cv2
import numpy as np
from PIL import Image
from hydra.utils import instantiate
from torch.utils.data import Dataset, Subset

class SegmentationDataset(Dataset):
    def __init__(self, csv_file):
        img_name_list = pd.read_csv(csv_file)['img_path'].tolist()

        self.images = []
        self.masks = []
        self.metadata = []

        for image in img_name_list:

            # ./dataset/train/soc_duts_te/imgs/sun_aqnqdyisdxasmdse.jpg
            identifier = image.split('/')[3]
            if identifier in ['soc_duts_te', 'soc_duts', 'soc_train', 'soc_val', 'supervisely_coarse']:
                detailed_masks = False
            else:
                detailed_masks = True

            mask_path = image.replace('imgs', 'gt').replace('jpg', 'png').replace('JPG', 'png')
            metadata = {
                'image_path': image,
                'mask_path': mask_path,
                'dataset_identifier': identifier,
                'detailed_masks': detailed_masks,
            }

            self.images.append(image)
            self.masks.append(mask_path)
            self.metadata.append(metadata)

    def __getitem__(self, index):
        img_path = self.images[index]
        target_path = self.masks[index]
        metadata = self.metadata[index]

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path).convert('L')

        # target_np = np.asarray(target, dtype=np.uint8)
        # target_np = np.where(target_np>127, 255, 0)

        background, unknown, foreground = self.generate_trimap(np.asarray(target))
        
        # Image.fromarray(np.array(target_np, dtype=np.uint8)),
        target = Image.merge('RGB', (target,
                                      Image.fromarray(unknown),
                                      Image.fromarray(foreground)))

        return img, target, metadata

    def __len__(self):
        return len(self.images) - 1

    def generate_trimap(self, mask):
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask, dilation_kernel, iterations=1)

        erosion_kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, erosion_kernel, iterations=2)

        background = np.zeros(mask.shape, dtype=np.uint8)
        background[dilated < 128] = 255

        unknown = np.zeros(mask.shape, dtype=np.uint8)
        unknown.fill(255)
        unknown[eroded > 128] = 0
        unknown[dilated < 128] = 0

        foreground = np.zeros(mask.shape, dtype=np.uint8)
        foreground[eroded > 128] = 255

        return background, unknown, foreground