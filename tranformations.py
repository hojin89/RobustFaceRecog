import os
import sys
import math
import matplotlib.pyplot as plt
import random

import torch
import torchvision
from torchvision.transforms import InterpolationMode

#### train

def blur(images, sigma_max, is_transformed):
    if sigma_max != 0:
        for i in range(images.size(0)):  # batch size
            sigma = random.uniform(0, sigma_max)
            kernel_size = 2 * math.ceil(2.0 * sigma) + 1
            if is_transformed[i] == True:
                images[i,:,:,:] = torchvision.transforms.functional.gaussian_blur(images[i,:,:,:], kernel_size, [sigma, sigma])

            # fig, axes = plt.subplots(1,1)
            # axes.imshow(images[i,:,:,:].permute(1, 2, 0).cpu().squeeze()) # rgb
            # plt.show()

    return images

def scale(images, scale_ratio_min, is_transformed):
    if scale_ratio_min != 1:
        for i in range(images.size(0)):  # batch size
            if is_transformed[i] == True:
                scale_size = int(random.uniform(scale_ratio_min, 1.) * images.size(2)) # assume inputs are square
                image = torch.zeros_like(images[i,:,:,:])
                resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size,], InterpolationMode.BILINEAR)
                row, col = int((images.size(2)-scale_size)/2), int((images.size(3)-scale_size)/2)
                image[:,row:row+scale_size,col:col+scale_size] = resized
                images[i,:,:,:] = image

            # fig, axes = plt.subplots(1,1)
            # axes.imshow(images[i,:,:,:].permute(1, 2, 0).cpu().squeeze()) # rgb
            # plt.show()

    return images

def quantization(images, scale_ratio_min, is_transformed):
    if scale_ratio_min != 1:
        for i in range(images.size(0)):  # batch size
            if is_transformed[i] == True:
                scale_size = int(random.uniform(scale_ratio_min, 1.) * images.size(2))
                resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size,], InterpolationMode.BILINEAR)
                images[i,:,:,:] = torchvision.transforms.functional.resize(resized, [images.size(2),], InterpolationMode.NEAREST)

            # fig, axes = plt.subplots(1,1)
            # axes.imshow(images[i,:,:,:].permute(1, 2, 0).cpu().squeeze()) # rgb
            # plt.show()

    return images

#### validation

def blur_val(images, sigma):
    if sigma != 0:
        for i in range(images.size(0)):  # batch size
            kernel_size = 2 * math.ceil(2.0 * sigma) + 1
            images[i,:,:,:] = torchvision.transforms.functional.gaussian_blur(images[i,:,:,:], kernel_size, [sigma, sigma])

    return images

def scale_val(images, scale_ratio):
    if scale_ratio != 1:
        for i in range(images.size(0)):  # batch size
            scale_size = int(scale_ratio * images.size(2)) # assume inputs are square
            image = torch.zeros_like(images[i,:,:,:])
            resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size,], InterpolationMode.BILINEAR)
            row, col = int((images.size(2)-scale_size)/2), int((images.size(3)-scale_size)/2)
            image[:,row:row+scale_size,col:col+scale_size] = resized
            images[i,:,:,:] = image

    return images

def quantization_val(images, scale_ratio):
    if scale_ratio != 1:
        for i in range(images.size(0)):  # batch size
            scale_size = int(scale_ratio * images.size(2)) # assume inputs are square
            resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size,], InterpolationMode.BILINEAR)
            images[i,:,:,:] = torchvision.transforms.functional.resize(resized, [images.size(2),], InterpolationMode.NEAREST)

    return images
