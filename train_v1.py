###################
#### version 1 ####
###################
# the number of transformed categories vary from 1 to 388,
# three types of transformations are tested: blur, scale, quantization.

import os
import sys
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

from models.alexnet import alexnet
from utils import *
from transformations import *

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--is_slurm', default=False, type=bool, help='use slurm')
    parser.add_argument('-m', '--model_path', default='./', type=str, help='model path')
    parser.add_argument('-d', '--data_path', default='./', type=str, help='data path')
    parser.add_argument('-c', '--csv_path', default='./', type=str, help='csv path')
    parser.add_argument('-v', '--version', default=1, type=int, help='experiment version')
    parser.add_argument('-i', '--id', default=1, type=int, help='slurm array task id')
    parser.add_argument('-t', '--trial', default=1, type=int, help='trial number')
    args = parser.parse_args()

    return args

def main(args):

    #### set up parameters
    num_categories = 388 # 388
    batch_size = 64
    lr = 1e-3
    total_epochs = 500
    save_epochs = 500

    #### model
    # model = torchvision.models.alexnet(pretrained=False)
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_categories)
    model = alexnet(num_classes=num_categories)

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
    try:
        os.makedirs(args.model_path)
    except:
        pass

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

    #### data loader
    train_dataset = CustomDataset(
        args.data_path,
        os.path.join(args.csv_path, 'train.csv'),
        transforms.Compose([
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.Resize(100),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    )

    val_dataset = CustomDataset(
        args.data_path,
        os.path.join(args.csv_path, 'val.csv'),
        transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
        ]),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    random.seed(args.trial) # seed fixed
    args.category_orders = [i for i in range(num_categories)]
    random.shuffle(args.category_orders)

    #### train and val
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

        if args.transformation_type == 'blur':
            inputs = blur(inputs, [True if t in args.category_orders[:args.num_category_transformed] else False for t in targets], 4)
        elif args.transformation_type == 'scale':
            inputs = scale(inputs, [True if t in args.category_orders[:args.num_category_transformed] else False for t in targets], 0.1)
        elif args.transformation_type == 'quantization':
            inputs = quantization(inputs, [True if t in args.category_orders[:args.num_category_transformed] else False for t in targets], 0.1)

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

        if args.transformation_type == 'blur':
            inputs = blur(inputs, [True if t in args.category_orders[:args.num_category_transformed] else False for t in targets], 4)
        elif args.transformation_type == 'scale':
            inputs = scale(inputs, [True if t in args.category_orders[:args.num_category_transformed] else False for t in targets], 0.1)
        elif args.transformation_type == 'quantization':
            inputs = quantization(inputs, [True if t in args.category_orders[:args.num_category_transformed] else False for t in targets], 0.1)

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

if __name__ == '__main__':
    args = parse_args()

    #### define parameters
    args.trial = 1
    params1 = [1, 50, 100, 150, 200, 250, 300, 350, 388] # number of transformed categories
    params2 = ['blur', 'scale', 'quantization'] # transformation types

    if args.id == 0:
        args.num_category_transformed, args.transformation_type = 0, 'none'
    else:
        args.num_category_transformed, args.transformation_type = [p for p in itertools.product(params1, params2)][args.id-1]

    #### slurm or local
    if args.is_slurm == True:
        args.model_path = '/om2/user/jangh/DeepLearning/RobustFaceRecog/results/v{}/id{}_t{}'.format(args.version, args.id, args.trial)
        args.data_path = '/om2/user/jangh/Datasets/FaceScrub/data/'
        args.csv_path = '/om2/user/jangh/Datasets/FaceScrub/csv/'
    elif args.is_slurm == False:
        args.model_path = '/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v{}/id{}_t{}'.format(args.version, args.id, args.trial)
        args.data_path = '/Users/hojinjang/Desktop/Datasets/FaceScrub/data/'
        args.csv_path = '/Users/hojinjang/Desktop/Datasets/FaceScrub/csv/'

    main(args)
