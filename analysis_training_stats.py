import os
import sys
import numpy
import matplotlib.pyplot as plt

def main(model_path):

        file = open(model_path, 'r')

        train = {'epoch': [], 'lr': [], 'loss': [], 'accuracy1': [], 'accuracy5': []}
        val = {'epoch': [], 'lr': [], 'loss': [], 'accuracy1': [], 'accuracy5': []}

        line_index = 0

        for line in file:
            line = line.replace('\n', '')
            data = line.split(' ')

            if 'Train' in line:
                train['epoch'].append(float(data[2][1:-1]))
                train['lr'].append(float(data[4][1:-1]))
                train['loss'].append(float(data[6][1:-1]))
                train['accuracy1'].append(float(data[8][1:-1]))
                train['accuracy5'].append(float(data[10][1:-1]))
            elif 'Validation' in line:
                val['epoch'].append(float(data[2][1:-1]))
                val['lr'].append(float(data[4][1:-1]))
                val['loss'].append(float(data[6][1:-1]))
                val['accuracy1'].append(float(data[8][1:-1]))
                val['accuracy5'].append(float(data[10][1:-1]))

            line_index = line_index + 1

        nrows, ncols = 1, 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

        #### LR
        axes[0].plot(train['epoch'], train['lr'])
        axes[0].plot(val['epoch'], val['lr'])
        axes[0].set_title('Learning rate')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(('Train', 'Validation'), frameon=False)

        #### Loss
        axes[1].plot(train['epoch'], train['loss'])
        axes[1].plot(val['epoch'], val['loss'])
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(('Train', 'Validation'), frameon=False)

        #### Top1 accuracy
        axes[2].plot(train['epoch'], train['accuracy1'])
        axes[2].plot(val['epoch'], val['accuracy1'])
        axes[2].set_title('Top1 accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylim([0, 1])
        axes[2].yaxis.set_ticks(numpy.arange(0, 1, step=0.2))
        axes[2].legend(('Train', 'Validation'), frameon=False)

        #### Top5 accuracy
        axes[3].plot(train['epoch'], train['accuracy5'])
        axes[3].plot(val['epoch'], val['accuracy5'])
        axes[3].set_title('Top5 accuracy')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylim([0, 1])
        axes[3].yaxis.set_ticks(numpy.arange(0, 1, step=0.2))
        axes[3].legend(('Train', 'Validation'), frameon=False)

        plt.show()

if __name__ == '__main__':

    main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id1/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id2/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id3/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id4/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id5/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id6/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id7/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id8/training_stats.txt')
    # main('/Users/hojinjang/Desktop/DeepLearning/RobustFaceRecog/results/v1/id9/training_stats.txt')
