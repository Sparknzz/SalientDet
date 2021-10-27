import os
import cv2
import numpy as np


# ------------------------

test_images = os.listdir('./test_data/test_images')

for img in test_images:
    if not img.endswith('jpg'):
        continue
    
    img_path = os.path.join('./test_data/test_images', img)
    ori_img = cv2.imread(img_path)
    
    img_path = img_path.replace('jpg', 'png')
    mask_img_our = cv2.imread(img_path.replace('test_images', 'our_results'))/255
    mask_img_full = cv2.imread(img_path.replace('test_images', 'new_results'))/255
    mask_img_piia = cv2.imread(img_path.replace('test_images', 'piia_results')[:-4]+'-cutout.jpg')/255
    # mask_img_piia = np.zeros(ori_img.shape)
    bg = np.zeros(ori_img.shape)
    
    bg[:, :, 1] = 255
    # mask_img_our = np.where(mask_img_our>0.9, 1, mask_img_our)
    mask_img_our = np.where(mask_img_our<0.5, 0, mask_img_our)
    # mask_img_full = np.where(mask_img_full>0.9, 1, mask_img_full)
    mask_img_full = np.where(mask_img_full<0.5, 0, mask_img_full)
        
    seg_img_our = ori_img * (mask_img_our) + bg * (1-mask_img_our)
    seg_img_full = ori_img * (mask_img_full) + bg * (1-mask_img_full)
    seg_img_piia = ori_img * (mask_img_piia) + bg * (1-mask_img_piia)
    
    h1_img = np.hstack([ori_img, seg_img_full])
    
    h2_img = np.hstack([seg_img_piia, seg_img_our])
    # pad_width = ori_img.shape[1]//2
    # h1_img = np.zeros(h2_img.shape)
    # h1_img[:,pad_width:ori_img.shape[1]+pad_width] = ori_img
    concat_img = np.vstack([h1_img, h2_img])
    
    cv2.imwrite('./test_data/compares/'+img, concat_img.astype(np.uint8))
# ------------------------------
# test_images = os.listdir('data/images')

# for img in test_images:
#     if not img.endswith('png'):
#         continue
#     img_path = os.path.join('data/images', img)
    
#     ori_img = cv2.imread(img_path)
#     mask_img_piia = cv2.imread('data/masks/'+img)
    
#     seg_img_piia = ori_img * (mask_img_piia/255)
    
#     h1_img = np.hstack([ori_img, seg_img_piia])
    
#     cv2.imwrite('data/combined/'+img, h1_img.astype(np.uint8))