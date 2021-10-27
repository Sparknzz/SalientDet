import os
from collections import defaultdict
from glob import glob

import PIL
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F
from model import U2NETP, U2NET

def load_samples(folder_path='./test_data/test_images'):
    assert os.path.isdir(folder_path), f'Unable to open {folder_path}'
    samples = glob(os.path.join(folder_path, f'*.jpg'))
    return samples


device = 'cpu'
samples = load_samples()

def square_pad(image, fill=255):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, fill, 'constant'), padding


def get_transform():
    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    # transforms.append(Normalize(mean=[0.5, 0.5, 0.5],
    #                               std=[0.5, 0.5, 0.5]))

    return Compose(transforms)

device = 'cpu'
checkpoint = torch.load(f'./checkpoints/u2net/e_36_i_3999_val_0.0090.pth', map_location=device)
model = U2NET(3, 1).to(device=device)

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)


transforms = get_transform()

model.eval()

for i_test, img_path in enumerate(samples):
    
    im_name = img_path.split('/')[-1].split('.')[0]
    
    image = Image.open(img_path).convert('RGB')
    image_padded, padding = square_pad(image, 0)
    pw, ph = image_padded.size
    image_resized = image_padded.resize((448, 448), Image.ANTIALIAS)
        
    with torch.no_grad():
        x = transforms(image_resized)
        x = x.to(device=device).unsqueeze(dim=0)
        y_hat, d2, d3, d4, d5, d6, d7 = model(x)

        alpha_image = y_hat.mul(255)
        alpha_image = Image.fromarray(alpha_image.squeeze().cpu().detach().numpy()).convert('L') # 448, 448
        
        o_alpha = alpha_image.resize(image_padded.size)
        # o_alpha.save(im_name+'.png')
        
        (hp, vp, hp, vp) = padding
        output_alpha = np.array(o_alpha)[vp:ph-vp, hp:pw-hp]
        Image.fromarray(output_alpha).save('./test_data/new_results/'+im_name+'.png') # 448, 448