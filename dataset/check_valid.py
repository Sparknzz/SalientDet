import os
import cv2
import pandas as pd
from tomorrow import threads
from tqdm import tqdm
# from skimage import io, transform, color
from PIL import Image
import numpy as np

@threads(20)
def check_(img_list):
    for img in tqdm(img_list):        
        image = Image.open(img).convert('RGB')
        label = Image.open(img.replace('imgs', 'gt').replace('jpg', 'png')).convert('L')

        i = np.array(image)
        l = np.array(label)

        if i.shape[-1]>3 or i.shape[-1]<3:
            print('img_path : {}, image : {}, label : {}'.format(img, i.shape, l.shape))

        if len(l.shape)>2:
            print('lbl_path : {}, image : {}, label : {}'.format(img, i.shape, l.shape))

if __name__=='__main__':

    df = pd.read_csv('./dataset/train.csv')
    img_list = df['img_path'].tolist()

    check_(img_list)