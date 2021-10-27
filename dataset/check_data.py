import os
import cv2
import numpy as np
from tqdm import tqdm
from tomorrow import threads
import pandas as pd

threads(10)
def make_todo(todo_list):
    for im in tqdm(todo_list):

        if os.path.exists('/home/bpfsrw_8/ningyupeng/segmentation/data/todo2/'+im+'.jpg'):
            continue

        img = cv2.imread(im)

        alpha = cv2.imread(im.replace('imgs', 'gt').replace('jpg', 'png'))/255

        if img.shape != alpha.shape: 
            print('bad image '+ im)
            continue

        bg = np.zeros(img.shape)
        bg[:,:,1] = 255

        out_img = img * alpha + (1-alpha) * bg

        himg = np.hstack([img, out_img.astype(np.uint8)])
        cv2.imwrite('/home/bpfsrw_8/ningyupeng/segmentation/data/todo2/'+im.split('/')[-1].split('.')[0]+'.jpg', himg)

if __name__ == '__main__':

    df = pd.read_csv('./dataset/train.csv')
    todo_list = df['img_path'].tolist()
    make_todo(todo_list)