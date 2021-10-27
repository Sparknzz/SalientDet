import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
# from pytorch_forecasting.optim import Ranger

from torch.cuda.amp import autocast, GradScaler
from dataset import SegmentationDataset as SegmentDataset
# from model.u2net_ import U2NET_full
from ranger import Ranger  # this is from ranger.py
from model import U2NETP, U2NET
from transform_pil import *
from smoothing import gaussian_blur
from torch.nn import functional as F
from tqdm import tqdm

# --------------------------------------------------------
class CollateFunction:
    def __init__(self, transforms, channels_last):
        self.transforms = transforms
        self.channels_last = channels_last
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.mean = [0.5, 0.5, 0.5]
        # self.std = [0.5, 0.5, 0.5]

    def __call__(self, batch):
        tensor, target_tensor, metadata = None, None, []

        for i, sample in enumerate(batch):

            if self.transforms is not None:
                sample = self.transforms(*sample)

            if tensor is None:
                w, h = sample[0].size
                memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
                tensor = torch.zeros((len(batch), 3, h, w)).contiguous(memory_format=memory_format)
                #   - semantic maps, foreground (from trimap), unknown (from trimap)
                target_tensor = torch.zeros((len(batch), 3, h, w)).contiguous(memory_format=torch.contiguous_format)

            x, y, sample_metadata = sample

            x = np.array(x) 
            y = np.array(y)

            x = x/np.max(x)

            x[:, :, 0] = (x[:,:,0] - self.mean[0]) / self.std[0]
            x[:, :, 1] = (x[:,:,1] - self.mean[1]) / self.std[1]
            x[:, :, 2] = (x[:,:,2] - self.mean[2]) / self.std[2]

            y = y/255

            x = x.transpose(2, 0, 1)  # C x H x W
            y = y.transpose(2, 0, 1)  # C x H x W

            # print(tensor.shape)
            # print(x.shape)
            tensor[i] += torch.from_numpy(x)
            target_tensor[i] += torch.from_numpy(y)
            metadata.append(sample_metadata)

        return tensor, target_tensor, metadata


def main():
    # ------- 2. set the directory of training dataset --------
    model_name = 'u2net'
    # pass csv file contains all train images path
    epoch_num = 100000
    batch_size_train = 12
    batch_size_val = 10
    train_num = 0
    val_num = 0

    train_dataset = SegmentDataset(csv_file='./dataset/train.csv')

    train_transform = Compose1([
        RandomRotation(degrees=90), 
        # RandomResizedCrop(size=448), 
        Resize(size=512, keep_aspect_ratio=False),
        RandomCrop(size=448),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomColorJitter()
        ], metadata_key='dataset_identifier')

    train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size_train, 
                                shuffle=True, 
                                num_workers=4,
                                drop_last=True,
                                collate_fn=CollateFunction(transforms=train_transform, channels_last=False))

    valid_dataset = SegmentDataset(csv_file='./dataset/val.csv')
    val_transform = Compose1([Resize(size=448, keep_aspect_ratio=False)], metadata_key='dataset_identifier')
    valid_loader = DataLoader(valid_dataset, 
                                batch_size=batch_size_val, 
                                num_workers=1,
                                shuffle=False, 
                                collate_fn=CollateFunction(transforms=val_transform, channels_last=False))
    # ------- 3. define model --------

    # define the net
    is_mixed_precision = False
    resume = True

    net = U2NET(3,1)

    if torch.cuda.is_available():
        if is_mixed_precision:
            scaler = GradScaler()
        net = net.cuda()

    # ------- 4. define optimizer --------
    # print("---define optimizer...")
    # optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = Ranger(net.parameters(), lr=0.0001, alpha=0.5, k=6, N_sma_threshhold=5, betas=(0.9, 0.999), eps=1e-05, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")

    start_epoch = 0
    save_frq = 1000 # save the model every 2000 iterations

    if resume:
        ckpt = torch.load('/home/bpfsrw_8/ningyupeng/u2net/saved_models/u2net/e_36_i_3999_val_0.0090.pth', map_location='cuda')
        net.load_state_dict(ckpt)
        ite_num = 0
        start_epoch = 0
        best_mae = 0.0213

    mae_ = do_valid(net, valid_loader)
    # torch.save(net.state_dict(), model_dir + "epoch_%d_val_%.4f.pth" % (1, mae_))
    print(mae_)

    for epoch in range(start_epoch, epoch_num):
        best_mae = train_one_epoch(net, optimizer, train_dataloader, valid_loader, epoch)

def train_one_epoch(net, optimizer, train_dataloader, valid_loader, epoch_num, is_mixed_precision=None, save_frq=2000, best_mae=0.03):
    net.train()

    ite_num = 0
    ite_num4val = 0
    running_loss = 0.0
    running_tar_loss = 0.0

    for i, data in enumerate(train_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels, metadata = data

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # if is_mixed_precision:
        #     inputs_v = inputs_v.half()
        #     with autocast():
        #         y_hat, aux_outputs = net(inputs_v)
        #         loss, aux = criterion(aux_outputs, labels_v, metadata)

        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        # forward + backward + optimize
        # y_hat, aux_outputs = net(inputs_v)
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        aux_outputs = [d0.squeeze(dim=1), d1.squeeze(dim=1), d2.squeeze(dim=1), d3.squeeze(dim=1), d4.squeeze(dim=1), d5.squeeze(dim=1), d6.squeeze(dim=1)]
        loss, aux = criterion(aux_outputs, labels_v, metadata)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        # running_tar_loss += loss0.data.item()

        # add a eval dataloader to check mae
        # print('----------ttttt---------' + str(ite_num))
        if ite_num%2000 == 0:
            mae_ = do_valid(net, valid_loader)
            net.train()

            print('----------------- testing ------------------')
            print(best_mae)
            print(mae_)
            print('--------------------------------------------')
            if mae_ < best_mae:
                best_mae = mae_
                model_dir = os.path.join(os.getcwd(), 'saved_models', 'u2net/')
                torch.save(net.state_dict(), model_dir + "e_%d_i_%d_val_%.4f.pth" % (epoch_num, i, mae_))

        print("[epoch: %3d, batch: %5d/%5d, ite: %d] train loss: %3f " 
                % (epoch_num, (i + 1), len(train_dataloader), ite_num, running_loss / ite_num4val))

    return best_mae

# ------- 1. define loss function --------
# l1_loss = torch.nn.L1Loss()
# mse_loss = torch.nn.MSELoss()
def criterion(aux, y, metadata):
    # aux ^= [d0, d1, d2, d3, d4, d5, d6]
    # y : B, C, H, W
    def masked_l1_loss(y_hat, y, mask):
        loss = F.l1_loss(y_hat, y, reduction='none')
        loss = (loss * mask.float()).sum()
        non_zero_elements = mask.sum()+1e-8
        return loss / non_zero_elements

    def masked_bce_loss(y_hat, y, mask):
        loss = F.binary_cross_entropy(y_hat, y, reduction='none')
        loss = (loss * mask.float()).sum()
        non_zero_elements = mask.sum()+1e-8
        return loss / non_zero_elements
        
    mask = y[:, 0]
    # smooth mask to avoid bad label or background bleeding
    smoothed_mask = gaussian_blur(
        mask.unsqueeze(dim=1), (9, 9), (2.5, 2.5)).squeeze(dim=1) 
    unknown_mask = y[:, 1]

    detailed_masks = [x['detailed_masks'] for x in metadata]
    l1_mask = torch.ones(mask.shape).cuda()
    l1_details_mask = torch.zeros(mask.shape).cuda()

    # for idx, detailed_mask in enumerate(detailed_masks):
    #     if not detailed_mask:
    #         l1_mask[idx] = l1_mask[idx] - unknown_mask[idx]
    #     else:
    #         l1_details_mask[idx] = unknown_mask[idx]

    loss = 0

    for output in aux:
        # # this for segmentation 
        # loss += 2 * masked_bce_loss(output, mask, l1_mask)
        # # this loss should give some learning signals to focus on unknown areas
        # loss += 3 * masked_l1_loss(output, mask, l1_details_mask)
        # # smooth mask to avoid bad label or background bleeding
        # loss += F.mse_loss(output, smoothed_mask)

        loss += masked_l1_loss(output, mask, l1_mask)

    aux = {
        'l1_mask': l1_mask,
        'l1_detailed_mask': l1_details_mask,
        'mask': mask,
        'smoothed_mask': smoothed_mask
    }

    return loss, aux


def do_valid(net, valid_loader):
    net.eval()
    running_mae = 0
    with torch.no_grad():
        for data in tqdm(valid_loader):
            inputs, labels, metadata = data

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # forward + backward + optimize
            # y_hat, aux_outputs = net(inputs_v)
            y_hat, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            mae = np.absolute((y_hat.squeeze(dim=1).cpu().numpy().astype("float") - labels[:,0,:,:].cpu().numpy().astype("float"))).mean()
            running_mae += mae

    return running_mae/len(valid_loader)


if __name__=='__main__':
    main()
