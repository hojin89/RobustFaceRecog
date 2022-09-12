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
    save_epochs = 500

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

    #### Add identity/softmax layer
    if args.num_gpus > 1:
        # model.module.classifier = nn.Sequential(*list(model.module.classifier) + [nn.Identity(dim=1)])
        model.module.classifier = nn.Sequential(*list(model.module.classifier) + [nn.Softmax(dim=1)])
    else:
        # model.classifier = nn.Sequential(*list(model.classifier) + [nn.Identity(dim=1)])
        model.classifier = nn.Sequential(*list(model.classifier) + [nn.Softmax(dim=1)])

    #### val
    epoch = start_epoch
    path, target = zip(*val_loader.dataset.samples)

    features, args.feature_shapes = get_features(model, input_size=[3,100,100], is_gpu=args.num_gpus>0, key_by='module', value_by='shape'), []
    for key, value in features.items():
        if isinstance(key, nn.ReLU) or isinstance(key, nn.Softmax):
            args.feature_shapes.append(value)

    if args.transformation_type == 'blur':
        params, original_idx = [0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.], 0
    elif args.transformation_type == 'scale':
        params, original_idx = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], 9
    elif args.transformation_type == 'quantization':
        params, original_idx = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], 9

    print("Start epoch at '{}'".format(epoch))
    stat_file = open(os.path.join(args.model_path, 'training_stats.txt'), 'r')

    feature_outputs = [numpy.zeros((len(path), args.feature_shapes[l][1], len(params)), dtype=numpy.float32) for l in range(len(args.feature_shapes))]
    for idx, param in enumerate(params):
        feature_outputs_ = val(val_loader, model, loss_function, optimizer, epoch, stat_file, param, args)
        for layer in range(len(args.feature_shapes)):
            feature_outputs[layer][:,:,idx] = feature_outputs_[layer]

    invariance_coef, threshold = numpy.zeros((len(args.feature_shapes), len(path), len(params)-1), dtype=numpy.float32), 0.1
    for layer in range(len(args.feature_shapes)):
        i = 0
        for idx, param in enumerate(params):
            if idx != original_idx:
                normalized_feature_outputs = feature_outputs[layer][:,:,[idx, original_idx]] / numpy.max(feature_outputs[layer][:,:,[idx, original_idx]], axis=0, keepdims=True)
                normalized_feature_outputs[normalized_feature_outputs < threshold] = float('NaN')
                invariance = 1 - abs((normalized_feature_outputs[:,:,0] - normalized_feature_outputs[:,:,1]) / (normalized_feature_outputs[:,:,0] + normalized_feature_outputs[:,:,1]))
                invariance_coef[layer,:,i] = numpy.nanmean(invariance, axis=1)
                i = i + 1
    del params[original_idx]

    data = {'path':path, 'target':target, 'params':params, 'invariance_coef':invariance_coef}
    with open(os.path.join(args.model_path, 'analysis_v1_invariance_coefficient.pickle'), 'wb') as f:
        pickle.dump(data, f)

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, param, args):

    model.eval()
    feature_outputs = [numpy.zeros((len(val_loader.dataset), args.feature_shapes[l][1]), dtype=numpy.float32) for l in range(len(args.feature_shapes))]

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        if args.transformation_type == 'blur':
            inputs = blur_val(inputs, param)
        elif args.transformation_type == 'scale':
            inputs = scale_val(inputs, param)
        elif args.transformation_type == 'quantization':
            inputs = quantization_val(inputs, param)

        features, l = get_features(model, inputs=inputs, is_gpu=args.num_gpus>0, key_by='module', value_by='output'), 0
        for key, value in features.items():
            if isinstance(key, nn.ReLU) or isinstance(key, nn.Softmax):
                if l < 5:  # conv layers
                    feature_outputs[l][indices,:] = numpy.max(value.detach().cpu().numpy(), axis=(2,3))
                else:
                    feature_outputs[l][indices,:] = value.detach().cpu().numpy()
                l = l + 1

    return feature_outputs

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
