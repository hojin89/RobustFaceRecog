import os
import builtins
import argparse
import numpy
import random
import matplotlib.pyplot as plt
import collections
from copy import copy

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_categories', default=2, type=int, help='number of categories')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per GPU')
    parser.add_argument('--total_epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--save_epochs', default=5, type=int, help='number of epochs to save')
    parser.add_argument('--is_slurm', default=False, type=bool)
    parser.add_argument('--num_gpus', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world_size', default=-1, type=int, help='number of total gpus for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    args = parser.parse_args()

    return args

def main(args, model_path):
    #### ddp setting
    args.is_slurm, args.num_gpus = 'SLURM_JOB_ID' in os.environ, torch.cuda.device_count()
    if args.is_slurm:
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.rank = int(os.environ['SLURM_PROCID'])
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank) # global

    #### model
    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, args.num_categories)

    if args.is_slurm and args.world_size > 1:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        # model_without_ddp = model.module
    elif args.num_gpus > 0:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        torch.cuda.set_device(devices)
        model.cuda(devices)

    #### optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    #### resume
    try:
        os.mkdir(model_path)
    except:
        pass

    load_path = os.path.join(model_path, 'checkpoint.pth.tar')
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch'] + 1

        if args.is_slurm == False and args.world_size <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in optimizer.param_groups:
            param['initial_lr'] = args.lr
    else:
        print("... No checkpoint found at '{}'".format(load_path))
        start_epoch = 0

    #### learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1, last_epoch=-1)

    #### data loader
    dataset = CustomDataset(
        "/Users/hojinjang/Desktop/Datasets/FaceScrubMini",
    )

    train_idx, val_idx, num_val_samples = [], [], 10
    for c in range(len(dataset.classes)):
        idx = [i for i, v in enumerate(dataset.targets) if v == c]
        val_idx.extend(idx[:num_val_samples])
        train_idx.extend(idx[num_val_samples:])
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_dataset.dataset = copy(dataset)
    train_dataset.dataset.transforms = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.Resize(100),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_dataset.dataset.transforms = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
    ])

    if args.is_slurm and args.world_size > 1:
        train_sampler, val_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True), None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    torch.backends.cudnn.benchmark = True

    #### train and val
    for epoch in range(start_epoch, args.total_epochs):
        numpy.random.seed(epoch)
        random.seed(epoch)

        #### adjust learning rate
        if epoch < args.total_epochs:
            lr_scheduler.step()

        print("Start epoch at '{}'".format(epoch))
        stat_file = open(os.path.join(model_path, 'training_stats.txt'), 'a+')

        train(train_loader, model, loss_function, optimizer, epoch, stat_file, args)
        val(val_loader, model, loss_function, optimizer, epoch, stat_file, args)

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_path, 'checkpoint.pth.tar'))
        if numpy.mod(epoch, args.save_epochs) == args.save_epochs-1:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_path, 'checkpoint_epoch_%d.pth.tar'%(epoch)))

def train(train_loader, model, loss_function, optimizer, epoch, stat_file, args):

    model.train()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy

    for batch_index, (inputs, targets) in enumerate(train_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # for i in range(inputs.size(0)):
        #     fig, axes = plt.subplots(1,1)
        #     ax = axes.imshow(inputs[i, :, :, :].permute(1, 2, 0).cpu().squeeze()) # Grayscale
        #     fig.colorbar(ax)
        #     plt.show()

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        _, num_correct1_batch = is_correct(outputs, targets, topk=1)
        # _, num_correct5_batch = is_correct(outputs, targets, topk=5)

        #### accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        # num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        progress_bar(batch_index, len(train_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, args):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy
    correct1, correct5 = [0] * len(val_loader.dataset.samples), [0] * len(val_loader.dataset.samples) # Correct

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        correct1_batch, num_correct1_batch = is_correct(outputs, targets, topk=1)
        # correct5_batch, num_correct5_batch = is_correct(outputs, targets, topk=5)

        #### accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        # num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        #### correct
        for i, index in enumerate(indices):
            correct1[index] = correct1_batch.view(-1)[i].item()
            # correct5[index] = torch.any(correct5_batch, dim=0).view(-1)[i].item() # top5 glitch

        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        progress_bar(batch_index, len(val_loader.dataset) / inputs.size(0), stat_str)

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
    main(args, '/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/1_FaceScrub_AlexNet_Scratch_SGD_Epoch500_LR1e-3')