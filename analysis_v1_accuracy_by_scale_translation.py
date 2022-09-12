import os
import sys
import argparse
import numpy
import random
import collections
import matplotlib.pyplot as plt
import itertools
import pickle
from copy import copy

import torch
import torchvision
import torchvision.transforms as transforms

from models.alexnet import alexnet
from utils import *
from tranformations import *

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
    save_epochs = 250

    #### gpus?
    args.num_gpus = torch.cuda.device_count()
    if args.num_gpus > 0:
        device_ids = [int(g) for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

    #### model
    # model = torchvision.models.alexnet(pretrained=False)
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_categories)
    model = alexnet(num_classes=num_categories)

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
    # train_dataset = CustomDataset(
    #     args.data_path,
    #     os.path.join(args.csv_path, 'train.csv'),
    #     transforms.Compose([
    #         # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    #         transforms.Resize(100),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ]),
    # )

    val_dataset = CustomDataset(
        args.data_path,
        os.path.join(args.csv_path, 'val.csv'),
        transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
        ]),
    )

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    random.seed(args.trial) # seed fixed
    args.category_orders = [i for i in range(num_categories)]
    random.shuffle(args.category_orders)
    args.is_transformed = [True if t in args.category_orders[:args.num_category_transformed] else False for t in val_dataset.targets]

    #### val
    epoch = start_epoch
    path, target = zip(*val_loader.dataset.samples)

    # if args.transformation_type == 'blur':
    #     params = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
    # elif args.transformation_type == 'scale':
    #     params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # elif args.transformation_type == 'quantization':
    #     params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    params = [0.2] # scale (pick only one)
    params2 = [(0.,0.), (0,0.2), (0,0.4), (0,0.6), (0,0.8),
               (0.2, 0.), (0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8),
               (0.4, 0.), (0.4, 0.2), (0.4, 0.4), (0.4, 0.6), (0.4, 0.8),
               (0.6, 0.), (0.6, 0.2), (0.6, 0.4), (0.6, 0.6), (0.6, 0.8),
               (0.8, 0.), (0.8, 0.2), (0.8, 0.4), (0.8, 0.6), (0.8, 0.8)] # top-left relative x, y location

    accuracy1, correct1, accuracy1_within, accuracy1_across, predict = \
        numpy.zeros((len(params), len(params2))), numpy.zeros((len(params), len(params2), len(path))), \
        numpy.zeros((len(params), len(params2))), numpy.zeros((len(params), len(params2))), numpy.zeros((len(params), len(params2), len(path)))

    print("Start epoch at '{}'".format(epoch))
    stat_file = ''

    for i, param in enumerate(params):
        for j, param2 in enumerate(params2):
            accuracy1_, correct1_, accuracy1_within_, accuracy1_across_, predict_ = \
                val(val_loader, model, loss_function, optimizer, epoch, stat_file, param, param2, args)
            accuracy1[i,j] = accuracy1_
            correct1[i,j,:] = correct1_
            accuracy1_within[i,j] = accuracy1_within_
            accuracy1_across[i,j] = accuracy1_across_
            predict[i,j,:] = predict_

    data = {'path':path, 'target':target, 'params':params, 'is_transformed':args.is_transformed, 'accuracy1':accuracy1, 'correct1':correct1, 'accuracy1_within':accuracy1_within, 'accuracy1_across':accuracy1_across, 'predict':predict}
    with open(os.path.join(args.model_path, 'analysis_v1_accuracy_by_scale_translation.pickle'), 'wb') as f:
        pickle.dump(data, f)

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, param, param2, args):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy
    correct1, correct5, predict = [0] * len(val_loader.dataset), [0] * len(val_loader.dataset), [0] * len(val_loader.dataset)
    num_correct1_transformed, num_correct1_nontransformed, batch_size_sum_transformed, batch_size_sum_nontransformed = 0., 0., 0., 0.

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # if args.transformation_type == 'blur':
        #     inputs = blur_val(inputs, param)
        # elif args.transformation_type == 'scale':
        #     inputs = scale_val(inputs, param)
        # elif args.transformation_type == 'quantization':
        #     inputs = quantization_val(inputs, param)
        inputs = scale_translation_val(inputs, param, 0., param2)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        correct1_batch, num_correct1_batch, predict_batch = is_correct(outputs, targets, topk=1)
        correct5_batch, num_correct5_batch, _ = is_correct(outputs, targets, topk=5)

        correct1_batch_transformed = correct1_batch[0, numpy.where([args.is_transformed[i] for i in indices])[0]]
        correct1_batch_nontransformed = correct1_batch[0, numpy.where(numpy.invert([args.is_transformed[i] for i in indices]))[0]]
        num_correct1_transformed += correct1_batch_transformed.float().sum().item()
        num_correct1_nontransformed += correct1_batch_nontransformed.float().sum().item()
        batch_size_sum_transformed += len(correct1_batch_transformed)
        batch_size_sum_nontransformed += len(correct1_batch_nontransformed)

        #### accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        for i, index in enumerate(indices):
            #### correct
            correct1[index] = correct1_batch.view(-1)[i].item()
            correct5[index] = torch.any(correct5_batch, dim=0).view(-1)[i].item()

            #### predict
            predict[index] = predict_batch.view(-1)[i].item()

        #### accuracy within & across
        accuracy1_within = 0. if batch_size_sum_transformed == 0. else num_correct1_transformed / batch_size_sum_transformed
        accuracy1_across = 0. if batch_size_sum_nontransformed == 0. else num_correct1_nontransformed / batch_size_sum_nontransformed

        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        # progress_bar(batch_index, len(val_loader.dataset) / inputs.size(0), stat_str)

    return (num_correct1 / batch_size_sum, correct1, accuracy1_within, accuracy1_across, predict)

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True)

        return correct, num_correct, pred

if __name__ == '__main__':
    args = parse_args()

    for trial in [1,2,3]:

        #### define parameters
        args.trial = trial
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