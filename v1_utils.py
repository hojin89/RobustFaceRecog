import os
import sys
import math
import time
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

term_width = 0
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    # if current < total-1:
    #     sys.stdout.write('\r')
    # else:
    #     sys.stdout.write('\n')

    sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    # Override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path, index))
        return tuple_with_path

def default_loader(path):
    image = Image.open(path)
    if len(image.size) == 2:
        image = image.convert('RGB')
    return image

class ImageFilelist(data.Dataset):
    def __init__(self, filelist, transform=None, loader=default_loader):
        self.filelist = filelist
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.filelist[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.filelist)

import numpy
import random
from PIL import Image
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
class ResizeRandomTranslation(object):

    def __init__(self, size, interpolation='bilinear'):
        assert hasattr(size, "__len__")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        size = self.size[random.choice(range(0,len(self.size)))]
        resized_img = img.resize(size=(size, size))

        if size == self.size[0]:
            out = resized_img
        else:
            out = numpy.zeros(shape=(self.size[0], self.size[0], 3))
            x = random.choice(range(0, out.shape[0] - resized_img.size[0]))
            y = random.choice(range(0, out.shape[1] - resized_img.size[1]))
            out[x:x+size, y:y+size] = numpy.asarray(resized_img)
            out = Image.fromarray(numpy.uint8(out))

        return out

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

import numbers
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import rotate
from collections.abc import Sequence
from typing import Tuple, List, Optional

def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))

def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)
    return [float(d) for d in x]

class CustomRandomRotationV1(object):

    # def __init__(self, degrees, perc, interpolation=Image.BILINEAR, expand=False, center=None, fill=0, resample=None):
    def __init__(self, interpolation=Image.BILINEAR, expand=False, center=None, fill=0, resample=None):
        super().__init__()
        # if resample is not None:
        #     interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        # if isinstance(interpolation, int):
        #     interpolation = _interpolation_modes_from_int(interpolation)

        # self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        # self.perc = perc

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @ staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def __call__(self, img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        # angle = self.get_params(self.degrees)

        chance = random.randint(1,100)
        if chance > 1:
            angle = numpy.random.normal(0, 30., 1)[0]
            angle = 90 if angle > 90 else angle
            angle = -90 if angle < -90 else angle
        else:
            angle = numpy.random.uniform(90., 180., 1)[0]
            sign = 1. if random.random() < 0.5 else -1.
            angle = angle * sign

        return rotate(img, angle, self.resample, self.expand, self.center, fill)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

class CustomRandomRotationV2(object):

    # def __init__(self, degrees, perc, interpolation=Image.BILINEAR, expand=False, center=None, fill=0, resample=None):
    def __init__(self, interpolation=Image.BILINEAR, expand=False, center=None, fill=0, resample=None):
        super().__init__()

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @ staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def __call__(self, img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        # angle = self.get_params(self.degrees)

        # chance = random.randint(1,100)
        # if chance > 1:
        angle = numpy.random.normal(0, 30., 1)[0]
        angle = 90 if angle > 90 else angle
        angle = -90 if angle < -90 else angle
        # else:
        #     angle = numpy.random.uniform(90., 180., 1)[0]
        #     sign = 1. if random.random() < 0.5 else -1.
        #     angle = angle * sign

        return rotate(img, angle, self.resample, self.expand, self.center, fill)

class CustomRandomRotationV3(object):

    # def __init__(self, degrees, perc, interpolation=Image.BILINEAR, expand=False, center=None, fill=0, resample=None):
    def __init__(self, interpolation=Image.BILINEAR, expand=False, center=None, fill=0, resample=None):
        super().__init__()

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def __call__(self, img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        # angle = self.get_params(self.degrees)

        chance = random.randint(1,1000)
        if chance > 1:
            angle = numpy.random.normal(0, 30., 1)[0]
            angle = 90 if angle > 90 else angle
            angle = -90 if angle < -90 else angle
        else:
            angle = numpy.random.uniform(90., 180., 1)[0]
            sign = 1. if random.random() < 0.5 else -1.
            angle = angle * sign

        return rotate(img, angle, self.resample, self.expand, self.center, fill)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

class CustomRandomAffine(transforms.RandomAffine):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0, fillcolors=[0, 255]):
        super(CustomRandomAffine, self).__init__(degrees, translate, scale, shear, resample, fillcolor)
        self.fillcolors = fillcolors

    def forward(self, img):

        img_size = torchvision.transforms.functional._get_image_size(img)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        # chance = random.randint(0, len(self.fillcolors)-1)
        # fillcolor = self.fillcolors[chance]
        fillcolor = random.randint(self.fillcolors[0], self.fillcolors[1])

        return torchvision.transforms.functional.affine(img, *ret, resample=self.resample, fillcolor=fillcolor)

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import statistics

    x_axis = np.arange(-180, 180, 1)
    mean = 0.
    sd = 30.
    plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
    plt.show()

    # img = Image.open('/home/tonglab/Documents/Project/MATLAB/matconvnet/projects/noise/data/googling/animal.jpg')
    # fun = CustomRandomRotationV1()
    # rotated = fun(img)
    #
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(rotated)  # Grayscale
    # plt.show()