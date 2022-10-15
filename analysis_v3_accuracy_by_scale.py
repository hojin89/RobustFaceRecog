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
from transformations_original import *

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--is_slurm', default=False, type=bool, help='use slurm')
    parser.add_argument('-m', '--model_path', default='./', type=str, help='model path')
    parser.add_argument('-d', '--data_path', default='./', type=str, help='data path')
    parser.add_argument('-c', '--csv_path', default='./', type=str, help='csv path')
    parser.add_argument('-v', '--version', default=3, type=int, help='experiment version')
    parser.add_argument('-j', '--job', default=1, type=int, help='slurm array job id')
    parser.add_argument('-i', '--id', default=1, type=int, help='slurm array task id')
    args = parser.parse_args()

    return args

def main_torch(args):

    #### set up parameters
    num_categories = 388 # 388
    batch_size = 64
    lr = 1e-3
    total_epochs = 300
    save_epochs = 300

    random.seed(args.trial) # seed fixed
    args.category_orders = [i for i in range(num_categories)]
    random.shuffle(args.category_orders)

    ####################################################################################################################
    #### dataset
    ####################################################################################################################
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

    ####################################################################################################################
    #### model
    ####################################################################################################################
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

    # load_path = os.path.join(args.model_path, 'checkpoint.pth.tar')
    load_path = os.path.join(args.model_path, 'checkpoint_epoch_299.pth.tar')
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
    #### val
    ####################################################################################################################
    epoch = start_epoch
    path, target = zip(*val_loader.dataset.samples)
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] # scale

    accuracy1, correct1, accuracy1_within, accuracy1_across, predict = \
        numpy.zeros((len(params))), numpy.zeros((len(params), len(path))), \
        numpy.zeros((len(params))), numpy.zeros((len(params))), numpy.zeros((len(params), len(path)))

    print("Start epoch at '{}'".format(epoch))
    stat_file = ''

    for i, param in enumerate(params):

        #### transformation - important!
        args.transformation_type = 'scale'
        args.transformation_sampling = 'discrete'
        args.transformation_levels = [param]
        args.is_transformed = [True if t in args.category_orders[:args.num_categories_transformed] else False for t in val_dataset.targets]
        args.background_sampling = 'discrete'
        args.background_colors = [0.]

        accuracy1_, correct1_, accuracy1_within_, accuracy1_across_, predict_ = \
            val(val_loader, model, loss_function, optimizer, epoch, stat_file, args)

        accuracy1[i] = accuracy1_
        correct1[i,:] = correct1_
        accuracy1_within[i] = accuracy1_within_
        accuracy1_across[i] = accuracy1_across_
        predict[i,:] = predict_

    data = {'path':path, 'target':target, 'params':params, 'is_transformed':args.is_transformed, 'accuracy1':accuracy1, 'correct1':correct1, 'accuracy1_within':accuracy1_within, 'accuracy1_across':accuracy1_across, 'predict':predict}
    with open(os.path.join(args.model_path, 'analysis_v3_accuracy_by_scale.pickle'), 'wb') as f:
        pickle.dump(data, f)

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, args):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy
    correct1, correct5, predict = [0] * len(val_loader.dataset), [0] * len(val_loader.dataset), [0] * len(val_loader.dataset)
    num_correct1_transformed, num_correct1_nontransformed, batch_size_sum_transformed, batch_size_sum_nontransformed = 0., 0., 0., 0.

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if args.num_gpus != 0:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        inputs = transformations_torch(inputs, targets, args, training=False)

        # fig, axes = plt.subplots(1,1)
        # axes.imshow(inputs[0,:,:,:].permute(1, 2, 0).cpu().squeeze()) # rgb
        # plt.show()

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

        # stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
        #     format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy-Within ({:.4f}) Accuracy-Across ({:.4f})'. \
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), accuracy1_within, accuracy1_across)
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
    # os.makedirs(args.model_path, exist_ok=True)
    # if args.is_slurm == True:
    #     open(os.path.join(args.model_path, 'slurm_id.txt'), 'a+').write(str(args.job) + '_' + str(args.id) + '\n')

    if args.model_name in ['alexnet', 'vgg19', 'resnet50', 'cornets']:
        main_torch(args)
    # elif args.model_name in ['blnet', 'convlstm']:
    #     main_tf(args)
