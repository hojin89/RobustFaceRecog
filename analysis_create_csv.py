import os
import sys
import pandas
import numpy
import re

def main(data_path, csv_path):

    ####################################################################################################################
    #### FaceScrub & FaceScrubMini & FaceScrubBlur & FaceScrubTurbulence
    ####################################################################################################################

    filenames = os.walk(data_path).__next__()[2]
    filenames = list(filter(lambda f: f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')), filenames))
    filenames.sort(key=lambda x: (x.split('_')[0], x.split('_')[1]))

    labels = []
    for filename in filenames:
        labels.append(filename.split('_')[0])
    classes = list(sorted(set(labels)))

    # train_indices, val_indices, num_val_samples = [], [], 10 # FaceScrub
    train_indices, val_indices, num_val_samples = [], [], 70 # FaceScrubBlur, FaceScrubTurbulence
    for c in classes:
        class_indices = [i for i, l in enumerate(labels) if l == c]
        val_indices.extend(class_indices[:num_val_samples])
        train_indices.extend(class_indices[num_val_samples:])

    df = pandas.DataFrame({
        'filename':[filenames[i] for i in train_indices],
        'label':[labels[i] for i in train_indices],
    })
    df.to_csv(os.path.join(csv_path, 'train.csv'))

    df = pandas.DataFrame({
        'filename':[filenames[i] for i in val_indices],
        'label':[labels[i] for i in val_indices],
    })
    df.to_csv(os.path.join(csv_path, 'val.csv'))

if __name__ == '__main__':

    # main('/Users/hojinjang/Desktop/Datasets/FaceScrub/data/', '/Users/hojinjang/Desktop/Datasets/FaceScrub/csv/')
    # main('/Users/hojinjang/Desktop/Datasets/FaceScrubMini/data/', '/Users/hojinjang/Desktop/Datasets/FaceScrubMini/csv/')
    # main('/Users/hojinjang/Desktop/Datasets/FaceScrubBlur/data/', '/Users/hojinjang/Desktop/Datasets/FaceScrubBlur/csv/')
    main('/om2/user/jangh/Datasets/FaceScrubTurbulence/data/', '/om2/user/jangh/Datasets/FaceScrubTurbulence/csv/')
