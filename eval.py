import os
import PIL
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F

ori_img_dir = './dataset/eval/images'

def rescaleTo(image, output_size=448):

    w, h = image.size
    # small side set
    if h > w:
        new_h, new_w = output_size*h/w, output_size
    else:
        new_h, new_w = output_size, output_size*w/h

    new_h, new_w = int(new_h), int(new_w)

    img = image.resize((new_w, new_h), Image.BICUBIC)
    #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
    img = image.resize((448, 448), Image.ANTIALIAS)

    return img

def get_transform():
    transforms = []
    # transforms.append(Resize(440)) # TBD: keep aspect ratio
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[.5,.5,.5],
                                  std=[.5,.5,.5]))
    return Compose(transforms)


def do_eval(net):

    net.eval()

    image_paths = [os.path.join(ori_img_dir, img_path) for img_path in os.listdir(ori_img_dir)]
    transforms = get_transform()
    
    # calculate mae / max F /
    avg_mae, img_num = 0.0, len(image_paths)
    avg_prec, avg_recall = 0, 0

    running_mae = 0

    for _, img_path in enumerate(image_paths):
        image = PIL.Image.open(img_path).convert('RGB')
        image = rescaleTo(image, 448)

        gt_img = PIL.Image.open(img_path.replace('images', 'gt')).convert('RGB')
        gt_img = rescaleTo(gt_img, 448)
        
        # do predict
        with torch.no_grad():
            x = transforms(image)
            x = x.cuda().unsqueeze(dim=0)
            d1, d2, d3, d4, d5, d6, d7 = net(x)
                
        d1 = d1.detach().squeeze().cpu().numpy()    
        image = np.array(image)
        
        for h in range(448):
            for w in range(448):
                image[h][w] = image[h, w, :] * d1[h][w]
        
        gt_img = np.array(gt_img)
        
        running_mae = running_mae + np.absolute(((image/255).astype("float") - (gt_img/255).astype("float"))).mean()
        
    return running_mae/len(image_paths)