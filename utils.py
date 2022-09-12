import os
import sys
import glob
import time
import pandas
import collections
from PIL import Image

import torch
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

def get_features(model, inputs=None, input_size=[3,224,224], is_gpu=False, key_by='module', value_by='shape'):
    features, hooks = collections.OrderedDict(), []
    def save_maps(name):
        def hook_fn(module, input, output):
            if key_by == 'name' and value_by == 'shape':
                features[name] = output.shape
            elif key_by == 'name' and value_by == 'output':
                features[name] = output
            elif key_by == 'module' and value_by == 'shape':
                features[module] = output.shape
            elif key_by == 'module' and value_by == 'output':
                features[module] = output
        return hook_fn
    def loop_layers(model, parent_name):
        for name, layer in model._modules.items():
            if parent_name != '':
                name = parent_name + '_' + name
            if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                loop_layers(layer, name)
            else:
                hook = layer.register_forward_hook(save_maps(name))
                hooks.append(hook)

    loop_layers(model, '')  # register hooks

    if inputs != None:
        _ = model(inputs)
    else:
        dummy_inputs = torch.zeros((1,input_size[0],input_size[1],input_size[2]), dtype=torch.float32)
        if is_gpu == True:
            dummy_inputs = dummy_inputs.cuda(non_blocking=True)
        _ = model(dummy_inputs)

    for h in hooks:
        h.remove()

    return features

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

class CustomDataset(torch.utils.data.Dataset):

    def read(self):
        df = pandas.read_csv(self.csv_path)
        filenames = df['filename'].tolist()
        labels = df['label'].tolist()

        images = []
        for filename in filenames:
            images.append(os.path.join(self.data_path, filename))

        classes = list(sorted(set(labels)))
        class_to_idx = {classes[i]:i for i in range(0, len(classes))}
        targets = [int(class_to_idx[label]) for label in labels]

        return images, targets, classes, class_to_idx

    def __init__(self, data_path, csv_path, transforms=None):
        self.data_path = data_path
        self.csv_path = csv_path
        self.transforms = transforms
        self.images, self.targets, self.classes, self.class_to_idx = self.read()
        self.samples = list(zip(self.images, self.targets))

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if len(image.size) == 2:
            image = image.convert('RGB')
        target = self.targets[index]

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, target, self.images[index], index)

    def __len__(self):
        return len(self.images)

# if __name__ == '__main__':