import math
import sys
from pathlib import Path

from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import numpy as np
import os
import torchvision

import kornia
from numpy.random import randint

import torch.nn.functional as F


#FOR TESTING
#torchvision.transforms.ToPILImage()(X_aug.clamp(-1, 1).sub(-1).div(max(2, 1e-5))).convert("RGB").show()

def drop_rand_patches(X, X_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    #######################
    # X_rep: replace X with patches from X_rep. If X_rep is None, replace the patches with Noise
    # max_drop: percentage of image to be dropped
    # max_block_sz: percentage of the maximum block to be dropped
    # tolr: minimum size of the block in terms of percentage of the image size
    #######################
    
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(0, max_drop)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)
    
    tolr = (int(tolr*H), int(tolr*W))
    
    total_pix = 0
    while total_pix < n_drop_pix:
        
        # get a random block by selecting a random row, column, width, height
        rnd_r = randint(0, H-tolr[0])
        rnd_c = randint(0, W-tolr[1])
        rnd_h = min(randint(tolr[0], mx_blk_height)+rnd_r, H) #rnd_r is alread added - this is not height anymore
        rnd_w = min(randint(tolr[1], mx_blk_width)+rnd_c, W)
        
        if X_rep is None:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.empty((C, rnd_h-rnd_r, rnd_w-rnd_c), dtype=X.dtype, device='cuda').normal_()
        else:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = X_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]    
         
        total_pix = total_pix + (rnd_h-rnd_r)*(rnd_w-rnd_c)

    return X

def rgb2gray_patch(X, tolr=0.05):

    C, H, W = X.size()
    tolr = (int(tolr*H), int(tolr*W))
     
    # get a random block by selecting a random row, column, width, height
    rnd_r = randint(0, H-tolr[0])
    rnd_c = randint(0, W-tolr[1])
    rnd_h = min(randint(tolr[0], H)+rnd_r, H) #rnd_r is alread added - this is not height anymore
    rnd_w = min(randint(tolr[1], W)+rnd_c, W)
    
    X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.mean(X[:, rnd_r:rnd_h, rnd_c:rnd_w], dim=0).unsqueeze(0).repeat(C, 1, 1)

    return X
    

def smooth_patch(X, max_kernSz=15, gauss=5, tolr=0.05):

    #get a random kernel size (odd number)
    kernSz = 2*(randint(3, max_kernSz+1)//2)+1
    gausFct = np.random.rand()*gauss + 0.1 # generate a real number between 0.1 and gauss+0.1
    
    C, H, W = X.size()
    tolr = (int(tolr*H), int(tolr*W))
     
    # get a random block by selecting a random row, column, width, height
    rnd_r = randint(0, H-tolr[0])
    rnd_c = randint(0, W-tolr[1])
    rnd_h = min(randint(tolr[0], H)+rnd_r, H) #rnd_r is alread added - this is not height anymore
    rnd_w = min(randint(tolr[1], W)+rnd_c, W)
    
    
    gauss = kornia.filters.GaussianBlur2d((kernSz, kernSz), (gausFct, gausFct))
    X[:, rnd_r:rnd_h, rnd_c:rnd_w] = gauss(X[:, rnd_r:rnd_h, rnd_c:rnd_w].unsqueeze(0))
    
    return X

def distortImages(samples):
    n_imgs = samples.size()[0] #this is batch size, but in case bad inistance happened while loading
    samples_aug = samples.detach().clone()
    for i in range(n_imgs):

        samples_aug[i] = rgb2gray_patch(samples_aug[i])

        samples_aug[i] = smooth_patch(samples_aug[i])

        samples_aug[i] = drop_rand_patches(samples_aug[i])

        idx_rnd = randint(0, n_imgs)
        if idx_rnd != i:
            samples_aug[i] = drop_rand_patches(samples_aug[i], samples_aug[idx_rnd])
      
    return samples_aug


def train_SSL(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    i = 0
    for imgs1, rots1, imgs2, rots2 in metric_logger.log_every(data_loader, print_freq, header):
        
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs1_aug = distortImages(imgs1) # Apply distortion
        rots1 = rots1.to(device, non_blocking=True)
        
        imgs2 = imgs2.to(device, non_blocking=True)
        imgs2_aug = distortImages(imgs2)
        rots2 = rots2.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            
            rot1_p, contrastive1_p, imgs1_recon, r_w, cn_w, rec_w = model(imgs1_aug)
            rot2_p, contrastive2_p, imgs2_recon, _, _, _ = model(imgs2_aug)
            
            rot_p = torch.cat([rot1_p, rot2_p], dim=0) 
            rots = torch.cat([rots1, rots2], dim=0) 
            
            imgs_recon = torch.cat([imgs1_recon, imgs2_recon], dim=0) 
            imgs = torch.cat([imgs1, imgs2], dim=0) 
            
            loss, (loss1, loss2, loss3) = criterion(rot_p, rots, 
                                                        contrastive1_p, contrastive2_p, 
                                                        imgs_recon, imgs, r_w, cn_w, rec_w)
            

        loss_value = loss.item()
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(RotationLoss=loss1.data.item())
        metric_logger.update(RotationScalar=r_w.data.item())
        metric_logger.update(ContrastiveLoss=loss2.data.item())
        metric_logger.update(ContrastiveScalar=cn_w.data.item())
        metric_logger.update(ReconstructionLoss=loss3.data.item())
        metric_logger.update(ReconstructionScalar=rec_w.data.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        i = i + 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_finetune(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.cuda.amp.autocast():       
            rot_p, contrastive_p = model(images)
            loss = criterion(rot_p, targets) + criterion(contrastive_p, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_SSL(data_loader, model, device, epoch, output_dir):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    save_recon = os.path.join(output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    
    # switch to evaluation mode
    model.eval()
    print_freq = 50
    i = 0
    for imgs1, rots1, imgs2, rots2 in metric_logger.log_every(data_loader, print_freq, header):
        imgs1 = imgs1.to(device, non_blocking=True) 
        imgs1_aug = distortImages(imgs1) # Apply distortion
        rots1 = rots1.to(device, non_blocking=True)
        
        imgs2 = imgs2.to(device, non_blocking=True)
        imgs2_aug = distortImages(imgs2)
        rots2 = rots2.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            rot1_p, contrastive1_p, imgs1_recon, r_w, cn_w, rec_w = model(imgs1_aug)
            rot2_p, contrastive2_p, imgs2_recon, _, _, _ = model(imgs2_aug)
            
            rot_p = torch.cat([rot1_p, rot2_p], dim=0) 
            rots = torch.cat([rots1, rots2], dim=0) 
            
            loss = criterion(rot_p, rots)

        acc1, acc5 = accuracy(rot_p, rots, topk=(1, 4))

        batch_size = imgs1.shape[0]*2
        
        if i%print_freq==0:

            print_out = save_recon + '/Test_epoch_' + str(epoch)  + '_Iter' + str(i) + '.jpg' 
            imagesToPrint = torch.cat([imgs1[0:min(15, batch_size)].cpu(), 
                                       imgs1_aug[0:min(15, batch_size)].cpu(),
                                       imgs1_recon[0:min(15, batch_size)].cpu(),
                                       imgs2[0:min(15, batch_size)].cpu(), 
                                       imgs2_aug[0:min(15, batch_size)].cpu(),
                                       imgs2_recon[0:min(15, batch_size)].cpu()], dim=0)
            torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, batch_size), normalize=True, range=(-1, 1))
            
            
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        i = i + 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_finetune(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    print_freq = 50
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True) 
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            rot_p, contrastive_p = model(images)
            loss = criterion(rot_p, targets) + criterion(contrastive_p, targets)

        acc1, acc5 = accuracy((rot_p+contrastive_p)/2., targets, topk=(1, 5))

        batch_size = images.shape[0]
            
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



