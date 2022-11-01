###################
#### version 4 ####
###################
# the number of transformed categories can vary from 1 to 388,
# three types of transformations are tested: blur, scale, quantization, rotate
# feedforward vs. recurrent

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

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import torch
import torchvision
import torchvision.transforms as transforms
torch.backends.cudnn.benchmark = True

from utils import *
from transformations import *

from models.alexnet import alexnet
from models.vgg import vgg19
from models.resnet import resnet50
from models.cornet_s import CORnet_S
from models.bl_net import bl_net
from models.convlstm_tf import ConvLSTM3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', default=4, type=int, help='experiment version')
    parser.add_argument('-j', '--job', default=1, type=int, help='slurm array job id')
    parser.add_argument('-i', '--id', default=1, type=int, help='slurm array task id')
    parser.add_argument('-s', '--is_slurm', default=False, type=bool, help='use slurm')
    parser.add_argument('-m', '--model_path', default='./', type=str, help='model path')
    parser.add_argument('-d', '--data_path', default='./', type=str, help='data path')
    parser.add_argument('-c', '--csv_path', default='./', type=str, help='csv path')
    args = parser.parse_args()

    return args

####################
#### tensorflow ####
####################
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
    #### gpus
    ####################################################################################################################

    if args.is_slurm == False:
        args.num_gpus = 0
    elif args.is_slurm == True:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        physical_devices = tf.config.list_physical_devices('GPU')
        args.num_gpus = len(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.num_gpus == 0:
        devices = "/cpu:0"
    elif args.num_gpus == 1:
        devices = "/gpu:0"
    elif args.num_gpus == 2:
        devices = ["/gpu:0", "/gpu:1"]

    ####################################################################################################################
    #### dataset
    ####################################################################################################################
    def parse_df(filenames, labels):
        file_paths = [os.path.join(args.data_path, filename) for filename in filenames]
        classes = list(sorted(set(labels)))
        class_to_idx = {classes[i]:i for i in range(0, len(classes))}
        targets = [int(class_to_idx[label]) for label in labels]
        return file_paths, targets, classes, class_to_idx

    @tf.function
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
    # train_dataset = tf.data.Dataset.from_tensor_slices((file_paths, numpy.eye(numpy.max(targets) + 1)[targets])) # categorical crossentropy
    train_dataset = tf.data.Dataset.from_tensor_slices((file_paths, targets)) # sparse categorical crossentropy
    train_dataset = train_dataset.shuffle(buffer_size=len(train_df)).map(load_dataset).map(lambda x, y: (train_preprocessing(x), y), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(lambda x, y: transformations_tf(x, y, True, args), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    val_df = pandas.read_csv(os.path.join(args.csv_path, 'val.csv'))
    file_paths, targets, classes, class_to_idx = parse_df(val_df['filename'].values, val_df['label'].values)
    val_dataset = tf.data.Dataset.from_tensor_slices((file_paths, targets))
    val_dataset = val_dataset.map(load_dataset).map(lambda x, y: (val_preprocessing(x), y), num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: transformations_tf(x, y, True, args), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # for inp, targ in train_dataset.take(4): # visualize
    #     fig, axes = plt.subplots(2,2)
    #     for i, ax in enumerate(axes.flat):
    #         ax.imshow(inp.numpy()[i,:,:,:])
    #         ax.set_title(targ.numpy()[i])
    #     plt.show()

    ####################################################################################################################
    #### model
    ####################################################################################################################

    #### define model with multi-GPUs
    strategy = tf.distribute.MirroredStrategy(devices=devices) if args.num_gpus > 1 else tf.distribute.OneDeviceStrategy(device=devices)
    with strategy.scope():

        #### dataset parallelism (for multi-GPUs)
        # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        # val_dataset = strategy.experimental_distribute_dataset(val_dataset)

        #### optimizer & loss
        optimizer = tf.keras.optimizers.SGD(lr=lr, decay=1e-4, momentum=0.9)
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(labels, outputs):
            per_example_loss = loss_func(labels, outputs)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

        train_loss_func = tf.keras.metrics.Mean(name='train_loss')
        val_loss_func = tf.keras.metrics.Mean(name='val_loss')
        train_accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        #### define model
        # if args.model_name == 'vgg16':
        #     model = tf.keras.applications.VGG16(input_shape=input_size, include_top=True)
        if args.model_name == 'blnet-1':
            model = bl_net(tf.keras.layers.Input(input_size), classes=num_categories, n_timesteps=1, cumulative_readout=False)
        elif args.model_name == 'blnet-3':
            model = bl_net(tf.keras.layers.Input(input_size), classes=num_categories, n_timesteps=3, cumulative_readout=False)
        elif args.model_name == 'blnet-5':
            model = bl_net(tf.keras.layers.Input(input_size), classes=num_categories, n_timesteps=5, cumulative_readout=False)
        elif args.model_name == 'convlstm3-1':
            model = ConvLSTM3(input_size=input_size, num_classes=num_categories, num_timesteps=1)
            _ = model(tf.zeros((1,) + input_size))
        elif args.model_name == 'convlstm3-3':
            model = ConvLSTM3(input_size=input_size, num_classes=num_categories, num_timesteps=3)
            _ = model(tf.zeros((1,) + input_size))
        elif args.model_name == 'convlstm3-5':
            model = ConvLSTM3(input_size=input_size, num_classes=num_categories, num_timesteps=5)
            _ = model(tf.zeros((1,) + input_size))
        model.summary()

        load_path = os.path.join(args.model_path, 'checkpoint.hdf5')
        if os.path.isfile(load_path):
            model.load_weights(load_path)
            with open(os.path.join(args.model_path, 'training_stats.txt'), 'r') as stat_file:
                last_line = stat_file.readlines()[-1]
                start_epoch = int(re.findall(r'Epoch \((\d+)\)', last_line)[0]) + 1
        else:
            print("... No checkpoint found at '{}'".format(load_path))
            start_epoch = 0

    ####################################################################################################################
    #### train and val
    ####################################################################################################################
    def train(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            if ('blnet' in args.model_name and 'blnet-1' != args.model_name) or ('convlstm' in args.model_name):
                loss = 0.0
                for output in outputs:
                    loss += compute_loss(labels, output)
                outputs = outputs[-1]
            else:
                loss = compute_loss(labels, outputs)
            predict = tf.argmax(outputs, 1, output_type=tf.dtypes.int32)
            correct = tf.equal(labels, predict)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_func.update_state(loss)
        train_accuracy_func.update_state(labels, outputs)
        return loss, correct, predict

    def val(inputs):
        images, labels = inputs
        outputs = model(images, training=False)
        if ('blnet' in args.model_name and 'blnet-1' != args.model_name) or ('convlstm' in args.model_name):
            loss = 0.0
            for output in outputs:
                loss += compute_loss(labels, output)
            outputs = outputs[-1]
        else:
            loss = compute_loss(labels, outputs)
        predict = tf.argmax(outputs, 1, output_type=tf.dtypes.int32)
        correct = tf.equal(labels, predict)

        val_loss_func.update_state(loss)
        val_accuracy_func.update_state(labels, outputs)
        return loss, correct, predict

    @tf.function
    def distributed_train(dataset_inputs):
        per_replica_losses = strategy.run(train, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    @tf.function
    def distributed_val(dataset_inputs):
        return strategy.run(val, args=(dataset_inputs,))

    ####################################################################################################################
    #### run
    ####################################################################################################################
    for epoch in range(start_epoch, total_epochs):

        #### train
        progess_bar = tf.keras.utils.Progbar(len(train_dataset))
        num_batches, batch_size_sum, loss_sum, correct_epoch, predict_epoch = 0, 0, 0.0, [], []
        for x in train_dataset:
            loss_batch, correct_batch, predict_batch = distributed_train(x)
            num_batches += 1
            batch_size_sum += len(x[1])
            loss_sum += loss_batch
            # correct_epoch.append(correct_batch)
            # predict_epoch.append(predict_batch)
            # progess_bar.update(num_batches)

        train_loss = loss_sum / num_batches
        # train_correct = tf.concat(correct_epoch, axis=0)
        # train_accuracy = tf.math.count_nonzero(train_correct) / batch_size_sum
        # train_predict = tf.concat(predict_epoch, axis=0)

        # stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f})'.format(epoch, lr, train_loss, train_accuracy)
        stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f})'.format(epoch, lr, train_loss, train_accuracy_func.result())
        open(os.path.join(args.model_path, 'training_stats.txt'), 'a+').write(stat_str + '\n')
        print(stat_str)
        sys.stdout.flush()

        #### val
        progess_bar = tf.keras.utils.Progbar(len(val_dataset))
        num_batches, batch_size_sum, loss_sum, correct_epoch, predict_epoch = 0, 0, 0.0, [], []
        for x in val_dataset:
            loss_batch, correct_batch, predict_batch = distributed_val(x)
            num_batches += 1
            batch_size_sum += len(x[1])
            loss_sum += loss_batch
            # correct_epoch.append(correct_batch)
            # predict_epoch.append(predict_batch)
            # progess_bar.update(num_batches)

        val_loss = loss_sum / num_batches
        # val_correct = tf.concat(correct_epoch, axis=0)
        # val_accuracy = tf.math.count_nonzero(val_correct) / batch_size_sum
        # val_predict = tf.concat(predict_epoch, axis=0)

        # stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f})'.format(epoch, lr, val_loss, val_accuracy)
        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f})'.format(epoch, lr, val_loss, val_accuracy_func.result())
        open(os.path.join(args.model_path, 'training_stats.txt'), 'a+').write(stat_str + '\n')
        print(stat_str)
        sys.stdout.flush()

        #### save
        model.save_weights(os.path.join(args.model_path, 'checkpoint.hdf5'))
        if epoch % save_epochs == save_epochs - 1:
            model.save_weights(os.path.join(args.model_path, 'checkpoint_epoch_{}.hdf5'.format(epoch)))

        train_loss_func.reset_states()
        val_loss_func.reset_states()
        train_accuracy_func.reset_states()
        val_accuracy_func.reset_states()

#################
#### pytorch ####
#################
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
    elif args.model_name == 'vitb16':
        if args.pretrained:
            model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        else:
            model = torchvision.models.vit_b_16()

        patch_size = 7 # if input_size is (100, 100)
        model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_categories)
        model.conv_proj = nn.Conv2d(in_channels=3, out_channels=model.conv_proj.out_channels, kernel_size=patch_size, stride=patch_size)
        from torchvision.models.vision_transformer import interpolate_embeddings
        model_state = interpolate_embeddings(input_size[1], patch_size, model.state_dict())
        model.load_state_dict(model_state)
        model.image_size, model.patch_size = input_size[1], patch_size
    elif args.model_name == 'cornets-1':
        model = CORnet_S(num_classes=num_categories, num_timesteps=1)
    elif args.model_name == 'cornets-3':
        model = CORnet_S(num_classes=num_categories, num_timesteps=3)
    elif args.model_name == 'cornets-5':
        model = CORnet_S(num_classes=num_categories, num_timesteps=5)

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
        # print("Start epoch at '{}'".format(epoch)); sys.stdout.flush()

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

        inputs = transformations_torch(inputs, targets, True, args)

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
    print(stat_str); sys.stdout.flush()
def val(val_loader, model, loss_function, optimizer, epoch, stat_file, args):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy
    correct1, correct5 = [0] * len(val_loader.dataset), [0] * len(val_loader.dataset) # Correct

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        inputs = transformations_torch(inputs, targets, True, args)

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
    print(stat_str); sys.stdout.flush()

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True)

        return correct, num_correct

if __name__ == '__main__':
    args = parse_args()

    #### model format: {model name}_{pretrained}_{transformation}_{discrete or continuous}_{}-{}_{number of categories transformed}_{trial}_{discrete or continuous}_{}
    # model name: alexnet, vgg19, resnet50, cornets, blnet, convlstm
    # pretrained: True/False
    # transformation type: none, blur, scale, quantization
    # transformation sampling: discrete, continuous
    # transformation levels
    # number of categories transformed: 1, 50, 100, 150, 200, 250, 300, 350, 388
    # trial: 1-n
    # background sampling: discrete, continuous
    # background colors

    #### slurm or local
    if args.is_slurm == True:
        #### scale
        args.model_format = [
            #### scale
            # 'alexnet_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'alexnet_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'alexnet_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'alexnet_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'alexnet_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'vgg19_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'vgg19_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'vgg19_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'vgg19_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'vgg19_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'resnet50_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'resnet50_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'resnet50_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'resnet50_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'resnet50_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'vitb16_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'vitb16_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'vitb16_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'vitb16_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'vitb16_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'vitb16_pretrained_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'vitb16_pretrained_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'vitb16_pretrained_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'vitb16_pretrained_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'vitb16_pretrained_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'cornets-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'cornets-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'cornets-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'cornets-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'cornets-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'blnet-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'blnet-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'blnet-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'blnet-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'blnet-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',

            'convlstm3-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            'convlstm3-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            'convlstm3-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            'convlstm3-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            'convlstm3-1_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',

            # 'cornets-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'cornets-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'cornets-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'cornets-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'cornets-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'blnet-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'blnet-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'blnet-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'blnet-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'blnet-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',

            'convlstm3-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            'convlstm3-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            'convlstm3-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            'convlstm3-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            'convlstm3-3_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',

            # 'cornets-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'cornets-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'cornets-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'cornets-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'cornets-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
            #
            # 'blnet-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            # 'blnet-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            # 'blnet-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            # 'blnet-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            # 'blnet-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',

            'convlstm3-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_1_1_discrete_0',
            'convlstm3-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_100_1_discrete_0',
            'convlstm3-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_200_1_discrete_0',
            'convlstm3-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_300_1_discrete_0',
            'convlstm3-5_scratch_scale_discrete_1-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.1_388_1_discrete_0',
        ][args.id-1]
        args.model_path = '/om2/user/jangh/DeepLearning/RobustFaceRecog/results/v{}/{}'.format(args.version, args.model_format)
        args.data_path = '/om2/user/jangh/Datasets/FaceScrub/data/'
        args.csv_path = '/om2/user/jangh/Datasets/FaceScrub/csv/'
    elif args.is_slurm == False:
        # args.model_format = 'convlstm3-1_scratch_rotate_discrete_0-45-90-135-180-225-270-315_388_1_discrete_0'
        args.model_format = 'blnet-1_scratch_blur_discrete_0-0.5-1-1.5-2-2.5-3-3.5-4_300_1_discrete_0'
        args.model_path = '/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v{}/{}'.format(args.version, args.model_format)
        args.data_path = '/Users/hojinjang/Desktop/Datasets/FaceScrubMini/data/'
        args.csv_path = '/Users/hojinjang/Desktop/Datasets/FaceScrubMini/csv/'

    params = args.model_format.split('_')
    args.model_name = params[0]
    args.pretrained = True if params[1] == 'pretrained' else (False if params[1] == 'scratch' else None)
    args.transformation_type = params[2]
    args.transformation_sampling = params[3]
    args.transformation_levels = list(map(float, params[4].split('-')))
    args.num_categories_transformed = int(params[5])
    args.trial = int(params[6])
    if len(params) > 7:
        args.background_sampling = params[7]
        args.background_colors = list(map(float, params[8].split('-')))

    #### create model folder & write slurm id
    os.makedirs(args.model_path, exist_ok=True)
    if args.is_slurm == True:
        open(os.path.join(args.model_path, 'slurm_id.txt'), 'a+').write(str(args.job) + '_' + str(args.id) + '\n')

    if args.model_name in ['alexnet', 'vgg19', 'resnet50', 'cornets-1', 'cornets-3', 'cornets-5', 'vitb16']:
        main_torch(args)
    elif args.model_name in ['blnet-1', 'blnet-3', 'blnet-5', 'convlstm3-1', 'convlstm3-3', 'convlstm3-5']:
        main_tf(args)
    else:
        raise ValueError('Fail to load the model...')
