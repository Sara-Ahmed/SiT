from PIL import Image
import numpy as np

import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

from numpy.random import randint
import io
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms as tf
from typing import Optional, Tuple

from torchvision import transforms


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def getItem(idx, X, target = None, transform=None, training_mode = 'SSL'):
    if transform is not None:
        X = transform(X)

    return X, target



class myRandCrop(tf.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(myRandCrop, self).__init__(size, scale, ratio, interpolation)
        
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tf.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)
   
class myRandomHorizontalFlip(tf.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(myRandomHorizontalFlip, self).__init__(p=p)
        
    def forward(self, img):
        if torch.rand(1) < self.p:
            return tf.functional.hflip(img), 1
        return img, 0
    
    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
    

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
        
def GMML_replace_list(samples, corrup_prev, masks_prev, drop_type='noise', max_replace=0.35, align=16):
    if not isinstance(samples, list):
        samples = [samples]
        
    rep_drop = 1 if drop_type == '' else ( 1 / ( len(drop_type.split('-')) + 1 ) )
    
    n_imgs = samples[0].size()[0] #this is batch size, but in case bad inistance happened while loading
    masks_all = []
    aug_all = []
    for si, s in enumerate(samples):
        samples_aug = s.detach().clone()
        masks = torch.zeros_like(samples_aug)
        for i in range(n_imgs):
            idx_rnd = randint(0, n_imgs)
            if random.random() < rep_drop: 
                samples_aug[i], masks[i] = GMML_drop_rand_patches(samples_aug[i], samples[si][idx_rnd], max_replace=max_replace, align=align)
            else:
                samples_aug[i], masks[i] = corrup_prev[si][i], masks_prev[si][i]
        #samples[si] = samples_aug
        masks_all.append(masks)
        aug_all.append(samples_aug)
      
    return aug_all, masks_all

def GMML_drop_rand_patches(X, X_rep=None, drop_type='noise', max_replace=0.7, align=16, max_block_sz=0.3):
    #######################
    # max_replace: percentage of image to be replaced
    # align: align corruption with the patch sizes
    # max_block_sz: percentage of the maximum block to be dropped
    #######################
   
    np.random.seed()    
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(min(0.5, max_replace), max_replace)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)
    
    align = max(1, align)
    
    mask = torch.zeros_like(X)
    drop_t = np.random.choice(drop_type.split('-'))
    
    while mask[0].sum() < n_drop_pix:
        
        ####### get a random block to replace 
        rnd_r = ( randint(0, H-align) // align ) * align
        rnd_c = ( randint(0, W-align) // align ) * align

        rnd_h = min(randint(align, mx_blk_height), H-rnd_r)
        rnd_h = round( rnd_h / align ) * align
        rnd_w = min(randint(align, mx_blk_width), W-rnd_c)
        rnd_w = round( rnd_w / align ) * align
        
        if X_rep is not None:
            X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X_rep[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w].detach().clone()
        else:
            if drop_t == 'noise':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.empty((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device).normal_()
            elif drop_t == 'zeros':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.zeros((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
            else:
                ####### get a random block to replace from
                rnd_r2 = (randint(0, H-rnd_h) // align ) * align
                rnd_c2 = (randint(0, W-rnd_w) // align ) * align
            
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X[:, rnd_r2:rnd_r2+rnd_h, rnd_c2:rnd_c2+rnd_w].detach().clone()
            
        mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
         
    return X, mask



class DataAugmentationSiT(object):
    def __init__(self, args):
        
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align

        self.color_jitter1 = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01)], p=0.3)])

        self.color_jitter2 = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # crop 1
        self.transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            self.color_jitter1,
            GaussianBlur(0.1),
            normalize])

        # crop 2
        self.transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            self.color_jitter2,
            GaussianBlur(1.0),
            Solarization(0.2),
            normalize])
        

    def rand_rotate(self, im):
        rotate = np.random.choice([0., 90., 180., 270.])
        return tf.functional.rotate(im, rotate), rotate//90
        
    def __call__(self, image):

        ########## view 1
        # augmented 
        im1 = self.transfo1(image)
        im1, rot1 = self.rand_rotate(im1)
        
        # corrupted 
        if self.drop_perc > 0:
            im_corr1, im_mask1 = GMML_drop_rand_patches(im1.detach().clone(), max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)
        else:
            im_corr1, im_mask1 = im1, torch.zeros_like(im1)

        ########## view 2
        # augmented 
        im2 = self.transfo2(image)
        im2, rot2 = self.rand_rotate(im2)
        
        # corrupted 
        if self.drop_perc > 0:
            im_corr2, im_mask2 = GMML_drop_rand_patches(im2.detach().clone(), max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)
        else:
            im_corr2, im_mask2 = im2, torch.zeros_like(im2)


        return [im1, im2], [rot1, rot2], [im_corr1, im_corr2], [im_mask1, im_mask2]

