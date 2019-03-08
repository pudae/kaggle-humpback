from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random

import numpy as np
import pandas as pd
import scipy.misc as misc

from torch.utils.data.dataset import Dataset


class LandmarkDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 split_prefix='split.keypoint',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.transform = transform
        self.dataset_dir = dataset_dir
        if split == 'test':
            self.images_dir = os.path.join(dataset_dir, 'test')
        else:
            self.images_dir = os.path.join(dataset_dir, 'train')
        self.split_prefix = split_prefix

        self.label_columns = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y',
                              '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y']
        self.df_examples = self.load_examples()
        self.size = len(self.df_examples)

    def load_examples(self):
        if self.split == 'test':
            image_ids_path = os.path.join(self.dataset_dir,
                                         'sample_submission.csv')
            df_examples = pd.read_csv(image_ids_path)[['Image']]
            df_examples.columns = ['filename']
        elif self.split == 'all':
            image_ids_path = os.path.join(self.dataset_dir,
                                         'train.csv')
            df_examples = pd.read_csv(image_ids_path)[['Image']]
            df_examples.columns = ['filename']
        else:
            labels_path = '{}.{}.csv'.format(self.split_prefix, self.idx_fold)
            labels_path = os.path.join(self.dataset_dir, labels_path)
            df_examples = pd.read_csv(labels_path)
            df_examples = df_examples[df_examples['split'] == self.split]
            df_examples = df_examples.reset_index()
            type_dict = {k:float for k in self.label_columns}
            df_examples = df_examples.astype(type_dict)

        df_examples = df_examples.set_index('filename')
        return df_examples

    def get_weights(self):
        return None

    def __getitem__(self, index):
        example = self.df_examples.iloc[index]

        filename = example.name
        filepath = os.path.join(self.images_dir, filename)
        if self.split != 'train':
            image = misc.imread(filepath, mode='RGB')
        else:
            if random.random() > 0.5:
                image = misc.imread(filepath, mode='RGB')
            else:
                image = misc.imread(filepath, mode='L')
                image = np.stack([image, image, image], axis=2)
        assert image.ndim == 3
        assert image.shape[-1] == 3

        if '0_x' in example:
            label = example[self.label_columns].values
            label = label.astype(np.float32)
        else:
            label = None

        if self.transform is not None:
            image, label = self.transform(image, label)

        if label is not None:
            return {'image': image,
                    'label': label,
                    'key': filename}
        else:
            return {'image': image,
                    'key': filename}

    def __len__(self):
        return self.size


def test():
    dataset = LandmarkDataset('data', 'train', None)
    print(len(dataset))
    for example in dataset:
        image = example['image']
        print(example['image'].shape,
              example['image'].dtype,
              np.max(example['image']),
              example['key'],
              example['label'])
        break

    dataset = LandmarkDataset('data', 'dev', None)
    print(len(dataset))
    # for example in dataset:
    #     print(example['key'], example['label'])

    dataset = LandmarkDataset('data', 'all', None)
    print(len(dataset))

    dataset = LandmarkDataset('data', 'test', None)
    print(len(dataset))

if __name__ == '__main__':
    test()

