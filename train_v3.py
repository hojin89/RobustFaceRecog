###################
#### version 3 ####
###################
# the number of transformed categories can vary from 1 to 388,
# three types of transformations are tested: blur, scale, quantization, rotate
# to check the impact of background colors

import os
import sys
import re
import argparse
import numpy
import random
import collections
import matplotlib.pyplot as plt
import itertools
from copy import copy

import torch
import torchvision
import torchvision.transforms as transforms
torch.backends.cudnn.benchmark = True

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from utils import *
from transformations import *

from models.alexnet import alexnet
from models.vgg import vgg19
from models.resnet import resnet50
from models.cornet_s import CORnet_S
from models.bl_net import bl_net
from models.convlstm import ConvLSTM3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--is_slurm', default=False, type=bool, help='use slurm')
    parser.add_argument('-m', '--model_path', default='./', type=str, help='model path')
    parser.add_argument('-d', '--data_path', default='./', type=str, help='data path')
    parser.add_argument('-c', '--csv_path', default='./', type=str, help='csv path')
    parser.add_argument('-v', '--version', default=3, type=int, help='experiment version')
    parser.add_argument('-j', '--job', default=-1, type=int, help='slurm array job id')
    parser.add_argument('-i', '--id', default=-1, type=int, help='slurm array task id')
    args = parser.parse_args()

    return args

def main_torch(args):

    #### set up parameters
    num_categories = 388 # 388
    batch_size = 32
    lr = 1e-3
    total_epochs = 300
    save_epochs = 300
    input_size = (3, 100, 100)

    random.seed(args.trial) # seed fixed
    args.category_orders = [i for i in range(num_categories)]
    random.shuffle(args.category_orders) # to determine the number of transformed categories

    ####################################################################################################################
    #### dataset
    ####################################################################################################################
    train_dataset = CustomDataset(
        args.data_path,
        os.path.join(args.csv_path, 'train.csv'),
        transforms.Compose([
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.Resize(input_size[1]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    )
    val_dataset = CustomDataset(
        args.data_path,
        os.path.join(args.csv_path, 'val.csv'),
        transforms.Compose([
            transforms.Resize(input_size[1]),
            transforms.ToTensor(),
        ]),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    ####################################################################################################################
    #### model
    ####################################################################################################################
    # model = torchvision.models.alexnet(pretrained=False)
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_categories)
    if args.model_name == 'alexnet':
        model = alexnet(num_classes=num_categories)
    elif args.model_name == 'vgg19':
        model = vgg19(num_classes=num_categories)
    elif args.model_name == 'resnet50':
        model = resnet50(num_classes=num_categories)
    elif args.model_name == 'cornets':
        model = CORnet_S(num_classes=num_categories)

    #### gpus?
    args.num_gpus = torch.cuda.device_count()
    if args.num_gpus > 0:
        device_ids = [int(g) for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    elif args.num_gpus == 1:
        device = torch.device('cuda:%d'%(device_ids[0]))
        torch.cuda.set_device(device)
        model.cuda().to(device)
    elif args.num_gpus == 0:
        device = torch.device('cpu')

    #### optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    #### resume
    load_path = os.path.join(args.model_path, 'checkpoint.pth.tar')
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1

        if args.num_gpus <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in optimizer.param_groups:
            param['lr'] = lr
    else:
        print("... No checkpoint found at '{}'".format(load_path))
        start_epoch = 0

    #### learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1, last_epoch=-1)

    ####################################################################################################################
    #### train and val
    ####################################################################################################################
    for epoch in range(start_epoch, total_epochs):

        #### adjust learning rate
        if epoch < total_epochs:
            lr_scheduler.step()

        stat_file = open(os.path.join(args.model_path, 'training_stats.txt'), 'a+')
        print("Start epoch at '{}'".format(epoch))
        sys.stdout.flush()

        train(train_loader, model, loss_function, optimizer, epoch, stat_file, args)
        val(val_loader, model, loss_function, optimizer, epoch, stat_file, args)

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.model_path, 'checkpoint.pth.tar'))
        if numpy.mod(epoch, save_epochs) == save_epochs-1:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.model_path, 'checkpoint_epoch_%d.pth.tar'%(epoch)))

def train(train_loader, model, loss_function, optimizer, epoch, stat_file, args):

    model.train()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy

    for batch_index, (inputs, targets, _, _) in enumerate(train_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        inputs = transformations_torch(inputs, targets, args)

        # for i in range(inputs.size(0)):
        #     fig, axes = plt.subplots(1,1)
        #     ax = axes.imshow(inputs[i, :, :, :].permute(1, 2, 0).cpu().squeeze()) # Grayscale
        #     fig.colorbar(ax)
        #     plt.show()

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        _, num_correct1_batch = is_correct(outputs, targets, topk=1)
        _, num_correct5_batch = is_correct(outputs, targets, topk=5)

        #### accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        # progress_bar(batch_index, len(train_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, args):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy
    correct1, correct5 = [0] * len(val_loader.dataset), [0] * len(val_loader.dataset) # Correct

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        inputs = transformations_torch(inputs, targets, args)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        correct1_batch, num_correct1_batch = is_correct(outputs, targets, topk=1)
        correct5_batch, num_correct5_batch = is_correct(outputs, targets, topk=5)

        #### accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        for i, index in enumerate(indices):
            #### correct
            correct1[index] = correct1_batch.view(-1)[i].item()
            correct5[index] = torch.any(correct5_batch, dim=0).view(-1)[i].item() # top5 glitch

        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        # progress_bar(batch_index, len(val_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True)

        return correct, num_correct

def main_tf(args):

    #### set up parameters
    num_categories = 388 # 388
    batch_size = 32
    lr = 1e-3
    total_epochs = 300
    save_epochs = 300
    input_size = (100, 100, 3)

    random.seed(args.trial) # seed fixed
    args.category_orders = [i for i in range(num_categories)]
    random.shuffle(args.category_orders) # to determine the number of transformed categories

    ####################################################################################################################
    #### dataset
    ####################################################################################################################
    def parse_df(filenames, labels):
        file_paths = [os.path.join(args.data_path, filename) for filename in filenames]
        classes = list(sorted(set(labels)))
        class_to_idx = {classes[i]:i for i in range(0, len(classes))}
        targets = [int(class_to_idx[label]) for label in labels]
        return file_paths, targets, classes, class_to_idx

    def load_dataset(file_paths, targets):
        raw = tf.io.read_file(file_paths)
        # img = tf.io.decode_jpeg(raw, channels=1) # grayscale
        img = tf.io.decode_jpeg(raw, channels=3) # rgb
        img = tf.image.convert_image_dtype(img, tf.float32) # [0, 255] int to [0, 1] float32
        img = tf.image.resize(img, [input_size[0], input_size[1]])
        return img, targets

    train_preprocessing = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.Resizing(input_size[0], input_size[1]),
        # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1),
        # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    val_preprocessing = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.Resizing(input_size[0], input_size[1]),
        # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1),
    ])

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_df = pandas.read_csv(os.path.join(args.csv_path, 'train.csv'))
    file_paths, targets, _, _ = parse_df(train_df['filename'].values, train_df['label'].values)
    train_dataset = tf.data.Dataset.from_tensor_slices((file_paths, numpy.eye(numpy.max(targets) + 1)[targets]))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_df)).map(load_dataset).map(lambda x, y: (train_preprocessing(x), y), num_parallel_calls=AUTOTUNE)
    if args.transformation_type != 'none': # image transformations
        train_dataset = train_dataset.map(lambda x, y: transformations_tf(x, y, args), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    val_df = pandas.read_csv(os.path.join(args.csv_path, 'val.csv'))
    file_paths, targets, classes, class_to_idx = parse_df(val_df['filename'].values, val_df['label'].values)
    val_dataset = tf.data.Dataset.from_tensor_slices((file_paths, numpy.eye(numpy.max(targets) + 1)[targets]))
    val_dataset = val_dataset.map(load_dataset).map(lambda x, y: (val_preprocessing(x), y), num_parallel_calls=AUTOTUNE)
    if args.transformation_type != 'none': # image transformations
        val_dataset = val_dataset.map(lambda x, y: transformations_tf(x, y, args), num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # for inp, targ in train_dataset.take(4): # visualize
    #     fig, axes = plt.subplots(2,2)
    #     for i, ax in enumerate(axes.flat):
    #         ax.imshow(inp.numpy()[i,:,:,:])
    #         ax.set_title(targ.numpy()[i])
    #     plt.show()

    ####################################################################################################################
    #### model
    ####################################################################################################################

    #### gpus?
    if args.is_slurm == False:
        args.num_gpus = 0
    elif args.is_slurm == True:
        physical_devices = tf.config.list_physical_devices('GPU')
        args.num_gpus = len(physical_devices)

    if args.num_gpus == 0:
        devices = "/cpu:0"
    elif args.num_gpus == 1:
        devices = "/gpu:0"
    elif args.num_gpus == 2:
        devices = ["/gpu:0", "/gpu:1"]

    #### define model with multi-GPUs
    strategy = tf.distribute.MirroredStrategy(devices=devices) if args.num_gpus > 1 else tf.distribute.OneDeviceStrategy(device=devices)
    with strategy.scope():

        # if args.model_name == 'vgg16':
        #     model = tf.keras.applications.VGG16(input_shape=input_size, include_top=True)
        if args.model_name == 'blnet':
            n_timesteps = 8
            input_layer = tf.keras.layers.Input(input_size)
            model = bl_net(input_layer, classes=num_categories, n_timesteps=n_timesteps, cumulative_readout=False)
        elif args.model_name == 'convlstm3':
            num_timesteps = 10
            model = ConvLSTM3(input_size=input_size, num_classes=num_categories, num_timesteps=num_timesteps)
            _ = model(tf.zeros((1,) + input_size))

        load_path = os.path.join(args.model_path, 'checkpoint.hdf5')
        if os.path.isfile(load_path):
            model.load_weights(load_path)
            with open(os.path.join(args.model_path, 'training_stats.txt'), 'r') as stat_file:
                last_line = stat_file.readlines()[-1]
                start_epoch = int(re.findall(r'Epoch \((\d)\)', last_line)[0]) + 1
        else:
            start_epoch = 0

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr, decay=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    ####################################################################################################################
    #### train and val
    ####################################################################################################################
    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.model_path, 'graph'), histogram_freq=0, write_graph=True, write_images=False),
        # tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_path, "checkpoint_epoch_{epoch}"), save_weights_only=True, save_freq='epoch', period=save_epochs),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_path, 'checkpoint.hdf5'), save_weights_only=True, save_freq='epoch'),
        CustomCallback(model, args.model_path, save_epochs)
        # tf.keras.callbacks.LearningRateScheduler(),
    ]
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=total_epochs, initial_epoch=start_epoch, callbacks=callbacks, verbose=0)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, model_path, save_epochs):
        super(CustomCallback, self).__init__()
        self.model = model
        self.model_path = model_path
        self.save_epochs = save_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        keys = list(logs.keys())
        acc_key = [key for key in keys if 'accuracy' in key and 'val' not in key][-1]
        stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f})'.format(self.epoch, lr, logs['loss'], logs[acc_key])
        open(os.path.join(self.model_path, 'training_stats.txt'), 'a+').write(stat_str + '\n')
        print(stat_str)

    def on_test_batch_end(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        keys = list(logs.keys())
        acc_key = [key for key in keys if 'accuracy' in key][-1]
        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f})'.format(self.epoch, lr, logs['loss'], logs[acc_key])
        open(os.path.join(self.model_path, 'training_stats.txt'), 'a+').write(stat_str + '\n')
        print(stat_str)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_epochs == self.save_epochs-1:
            self.model.save_weights(os.path.join(self.model_path, 'checkpoint_epoch_{}.hdf5'.format(epoch)))

if __name__ == '__main__':
    args = parse_args()

    #### model format: {model name}_{transformation}_{discrete or continuous}_{}-{}_{number of categories transformed}_{trial}_{discrete or continuous}_{}
    # model name: alexnet, vgg19, resnet50, cornets, blnet, convlstm
    # transformation type: none, blur, scale, quantization
    # transformation sampling: discrete, continuous
    # transformation levels
    # number of categories transformed: 1, 50, 100, 150, 200, 250, 300, 350, 388
    # trial: 1-n
    # background sampling: discrete, continuous
    # background colors

    #### slurm or local
    if args.is_slurm == True:
        args.model_format = [
            'alexnet_scale_continuous_1-0.1_1_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_50_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_100_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_150_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_200_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_250_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_300_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_350_1_discrete_0',
            'alexnet_scale_continuous_1-0.1_388_1_discrete_0',

            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_50_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_150_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_250_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_350_1_discrete_0',
            'alexnet_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',

            'alexnet_scale_discrete_1-0.6_1_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_50_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_100_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_150_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_200_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_250_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_300_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_350_1_discrete_0',
            'alexnet_scale_discrete_1-0.6_388_1_discrete_0',

            'alexnet_scale_continuous_1-0.1_1_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_50_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_100_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_150_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_200_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_250_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_300_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_350_1_continuous_0-1',
            'alexnet_scale_continuous_1-0.1_388_1_continuous_0-1',

            # 'alexnet_rotate_continuous_0-360_1_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_50_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_100_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_150_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_200_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_250_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_300_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_350_1_discrete_0',
            # 'alexnet_rotate_continuous_0-360_388_1_discrete_0',
            #
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_1_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_50_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_100_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_150_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_200_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_250_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_300_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_350_1_discrete_0',
            # 'alexnet_rotate_discrete_0-45-90-135-180-225-270-315_388_1_discrete_0',
            #
            # 'alexnet_rotate_discrete_0-90_1_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_50_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_100_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_150_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_200_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_250_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_300_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_350_1_discrete_0',
            # 'alexnet_rotate_discrete_0-90_388_1_discrete_0',
            #
            # 'alexnet_rotate_continuous_0-360_1_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_50_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_100_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_150_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_200_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_250_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_300_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_350_1_continuous_0-1',
            # 'alexnet_rotate_continuous_0-360_388_1_continuous_0-1',
        ][args.id-1]
        args.model_path = '/om2/user/jangh/DeepLearning/RobustFaceRecog/results/v{}/{}'.format(args.version, args.model_format)
        args.data_path = '/om2/user/jangh/Datasets/FaceScrub/data/'
        args.csv_path = '/om2/user/jangh/Datasets/FaceScrub/csv/'
    elif args.is_slurm == False:
        args.model_format = 'alexnet_rotate_continuous_0-145_388_1_continuous_0-0.5'
        args.model_path = '/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v{}/{}'.format(args.version, args.model_format)
        args.data_path = '/Users/hojinjang/Desktop/Datasets/FaceScrub/data/'
        args.csv_path = '/Users/hojinjang/Desktop/Datasets/FaceScrub/csv/'

    params = args.model_format.split('_')
    args.model_name = params[0]
    args.transformation_type = params[1]
    args.transformation_sampling = params[2]
    args.transformation_levels = list(map(float, params[3].split('-')))
    args.num_categories_transformed = int(params[4])
    args.trial = int(params[5])
    if len(params) > 6:
        args.background_sampling = params[6]
        args.background_colors = list(map(float, params[7].split('-')))

    #### create model folder & write slurm id
    if args.is_slurm == True:
        os.makedirs(args.model_path, exist_ok=True)
        open(os.path.join(args.model_path, 'slurm_id.txt'), 'a+').write(str(args.job) + '_' + str(args.id) + '\n')

    if args.model_name in ['alexnet', 'vgg19', 'resnet50', 'cornets']:
        main_torch(args)
    elif args.model_name in ['blnet', 'convlstm']:
        main_tf(args)