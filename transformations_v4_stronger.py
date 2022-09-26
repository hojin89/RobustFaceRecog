import os
import sys
import numpy
import math
import matplotlib.pyplot as plt
import random
from numpy.random import choice

#################
#### pytorch ####
#################
import torch
import torchvision
from torchvision.transforms import InterpolationMode

def transformations_torch(images, targets, args):
    is_transformed = [True if t in args.category_orders[:args.num_categories_transformed] else False for t in targets]

    for i in range(images.size(0)):  # batch size
        if is_transformed[i] == True:

            if args.transformation_type == 'blur':
                if args.transformation_sampling == 'discrete':
                    sigma = choice(args.transformation_levels, 1)[0]
                elif args.transformation_sampling == 'continuous':
                    sigma = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
                kernel_size = 2 * math.ceil(2.0 * sigma) + 1

                if sigma != 0:
                    images[i,:,:,:] = torchvision.transforms.functional.gaussian_blur(images[i,:,:,:], kernel_size, [sigma, sigma])

            elif args.transformation_type == 'scale':
                if args.transformation_sampling == 'discrete':
                    scale_ratio = choice(args.transformation_levels, 1)[0]
                elif args.transformation_sampling == 'continuous':
                    scale_ratio = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
                scale_size = int(scale_ratio * images.size(2))  # assume inputs are square

                if args.background_sampling == 'discrete':
                    background_color = choice(args.background_colors, 1)[0]
                elif args.background_sampling == 'continuous':
                    background_color = random.uniform(min(args.background_colors), max(args.background_colors))

                if scale_ratio != 1.:
                    image = torch.ones_like(images[i,:,:,:]) * background_color
                    resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size, ], InterpolationMode.BILINEAR)
                    row, col = int((images.size(2) - scale_size) / 2), int((images.size(3) - scale_size) / 2)
                    image[:, row:row + scale_size, col:col + scale_size] = resized
                    images[i,:,:,:] = image

            elif args.transformation_type == 'quantization':
                if args.transformation_sampling == 'discrete':
                    scale_ratio = choice(args.transformation_levels, 1)[0]
                elif args.transformation_sampling == 'continuous':
                    scale_ratio = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
                scale_size = int(scale_ratio * images.size(2))  # assume inputs are square

                if scale_ratio != 1.:
                    resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size, ], InterpolationMode.BILINEAR)
                    images[i,:,:,:] = torchvision.transforms.functional.resize(resized, [images.size(2), ], InterpolationMode.NEAREST)

            elif args.transformation_type == 'rotate':
                if args.transformation_sampling == 'discrete':
                    degree = choice(args.transformation_levels, 1)[0]
                elif args.transformation_sampling == 'continuous':
                    degree = random.uniform(min(args.transformation_levels), max(args.transformation_levels))

                if args.background_sampling == 'discrete':
                    background_color = choice(args.background_colors, 1)[0]
                elif args.background_sampling == 'continuous':
                    background_color = random.uniform(min(args.background_colors), max(args.background_colors))

                if degree != 0:
                    images[i,:,:,:] = torchvision.transforms.functional.rotate(images[i, :, :, :], angle=degree, interpolation=InterpolationMode.BILINEAR, fill=background_color)

            elif args.transformation_type == 'scale_translation':
                if args.transformation_sampling == 'discrete':
                    scale_ratio = choice(args.transformation_levels, 1)[0]
                elif args.transformation_sampling == 'continuous':
                    scale_ratio = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
                scale_size = int(scale_ratio * images.size(2))  # assume inputs are square

                if args.background_sampling == 'discrete':
                    background_color = choice(args.background_colors, 1)[0]
                elif args.background_sampling == 'continuous':
                    background_color = random.uniform(min(args.background_colors), max(args.background_colors))

                if scale_ratio != 1.:
                    image = torch.ones_like(images[i,:,:,:]) * background_color
                    resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size, ], InterpolationMode.BILINEAR)
                    # row, col = int((images.size(2) - scale_size) / 2), int((images.size(3) - scale_size) / 2)
                    row, col = int(images.size(2) * args.grid_location[0]), int(images.size(3) * args.grid_location[1])
                    image[:, row:row + scale_size, col:col + scale_size] = resized
                    images[i,:,:,:] = image

            elif args.transformation_type == 'scale_circularpad':
                if args.transformation_sampling == 'discrete':
                    scale_ratio = choice(args.transformation_levels, 1)[0]
                elif args.transformation_sampling == 'continuous':
                    scale_ratio = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
                scale_size = int(scale_ratio * images.size(2))  # assume inputs are square

                if scale_ratio != 1.:
                    resized = torchvision.transforms.functional.resize(images[i,:,:,:], [scale_size, ], InterpolationMode.BILINEAR)
                    # row, col = int((images.size(2) - scale_size) / 2), int((images.size(3) - scale_size) / 2)
                    # image[:, row:row + scale_size, col:col + scale_size] = resized
                    image = pad_circular(resized, images.size(2))
                    images[i,:,:,:] = image

        #### visualize
        # fig, axes = plt.subplots(1,1)
        # axes.imshow(images[i,:,:,:].permute(1, 2, 0).cpu().squeeze()) # rgb
        # plt.show()

    return images

def pad_circular(x, target_size): # assume inputs are square and the input dimension is [C, W, H]
    num_tiles = math.ceil(target_size / x.size(1))
    if (num_tiles % 2) == 0:
        num_tiles = num_tiles + 1
    x = torch.cat([x] * num_tiles, dim=1)
    x = torch.cat([x] * num_tiles, dim=2)
    x = torchvision.transforms.functional.center_crop(x, target_size)
    return x

####################
#### tensorflow ####
####################
import tensorflow as tf
import tensorflow_addons as tfa
tf.data.experimental.enable_debug_mode()

def transformations_tf(image, target, args):
    # import pydevd; pydevd.settrace(suspend=False)

    if tf.reduce_any(tf.equal(target, args.category_orders[:args.num_categories_transformed])):

        if args.transformation_type == 'blur':
            if args.transformation_sampling == 'discrete':
                sigma = choice(args.transformation_levels, 1)[0]
            elif args.transformation_sampling == 'continuous':
                sigma = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
            kernel_size = 2 * math.ceil(2.0 * sigma) + 1

            if sigma != 0:
                image = tfa.image.gaussian_filter2d(image, filter_shape=(kernel_size, kernel_size), sigma=(sigma, sigma))

        elif args.transformation_type == 'scale':
            if args.transformation_sampling == 'discrete':
                scale_ratio = choice(args.transformation_levels, 1)[0]
            elif args.transformation_sampling == 'continuous':
                scale_ratio = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
            original_size = tf.cast(tf.shape(image)[0], tf.float32) # assume inputs are square
            scale_size = tf.round(original_size * scale_ratio)

            if args.background_sampling == 'discrete':
                background_color = choice(args.background_colors, 1)[0]
            elif args.background_sampling == 'continuous':
                background_color = random.uniform(min(args.background_colors), max(args.background_colors))
            background_color = tf.constant(background_color, tf.float32)

            if scale_ratio != 1.:
                resized = tf.image.resize(image, size=(scale_size, scale_size), method=tf.image.ResizeMethod.BILINEAR)
                paddings = [tf.floor((original_size-scale_size)/2.), tf.math.ceil((original_size-scale_size)/2.)]
                image = tf.pad(resized, [paddings, paddings, [0,0]], 'CONSTANT', constant_values=background_color)

        elif args.transformation_type == 'quantization':
            if args.transformation_sampling == 'discrete':
                scale_ratio = choice(args.transformation_levels, 1)[0]
            elif args.transformation_sampling == 'continuous':
                scale_ratio = random.uniform(min(args.transformation_levels), max(args.transformation_levels))
            original_size = tf.cast(tf.shape(image)[0], tf.float64) # assume inputs are square
            scale_size = original_size * scale_ratio

            if scale_ratio != 1.:
                image = tf.image.resize(image, size=(scale_size, scale_size), method=tf.image.ResizeMethod.BILINEAR)
                image = tf.image.resize(image, size=(original_size, original_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        elif args.transformation_type == 'rotate':
            if args.transformation_sampling == 'discrete':
                degree = choice(args.transformation_levels, 1)[0]
            elif args.transformation_sampling == 'continuous':
                degree = random.uniform(min(args.transformation_levels), max(args.transformation_levels))

            if args.background_sampling == 'discrete':
                background_color = choice(args.background_colors, 1)[0]
            elif args.background_sampling == 'continuous':
                background_color = random.uniform(min(args.background_colors), max(args.background_colors))
            background_color = tf.constant(background_color, tf.float32)

            if degree != 0:
                image = tfa.image.rotate(image, angles=degree * math.pi / 180, interpolation='BILINEAR', fill_mode='constant', fill_value=background_color)

    return image, target
