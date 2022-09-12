import os
import sys
import numpy
import collections
import matplotlib.pyplot as plt
import torch

from models.alexnet import alexnet

def main(model_path):

    #### Load model
    num_categories = 388
    lr = 1e-3
    model = alexnet(num_classes=num_categories)

    #### gpus?
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device_ids = [int(g) for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    elif num_gpus == 1:
        device = torch.device('cuda:%d' % (device_ids[0]))
        torch.cuda.set_device(device)
        model.cuda().to(device)
    elif num_gpus == 0:
        device = torch.device('cpu')

    #### optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    #### resume
    load_path = os.path.join(model_path, 'checkpoint.pth.tar')
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1

        if num_gpus <= 1: # 1. Multi-GPU to single-GPU or -CPU
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

    model.eval()

    #### Visualize filters at the first layer
    filters = model.features._modules['0'].weight.data.cpu().numpy()
    num_rows = 8
    num_cols = 8
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_rows):
        for j in range(num_cols):
            idx = num_rows * i + j
            ax1 = fig.add_subplot(num_rows, num_cols, idx + 1)
            ax1.imshow(numpy.mean(filters[idx, :, :, :], axis=0).squeeze(), cmap='gray')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    #### Save weights
    # weights = {}
    #
    # def save_weights(name):
    #     def hook_fn(module, input, ouput):
    #         weights[name] = module.weight.data.cpu().numpy()
    #
    #     return hook_fn
    #
    # def get_all_layers(model, parent_name):
    #     for name, layer in model._modules.items():
    #         if parent_name != '':
    #             name = parent_name + '_' + name
    #         if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
    #             get_all_layers(layer, name)
    #         elif isinstance(layer, nn.modules.conv.Conv2d) or isinstance(layer, nn.modules.Linear):
    #             layer.register_forward_hook(save_weights(name))
    #
    # get_all_layers(model, '')  # just register hooks
    # input = torch.tensor(()).new_zeros((1, 3, 224, 224))
    # output = model(input)  # save output into visualization
    # scipy.io.savemat(os.path.join('/'.join(load_path.split("/")[:-1]),
    #                               'python_v1_analysis2_visualize_filters_epoch_%d.mat' % (test_epoch)),
    #                  {'weights': weights})

if __name__ == '__main__':

    main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id0_t1/')
    main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id14_t1/')
