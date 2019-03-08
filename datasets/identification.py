from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
import scipy.misc as misc
from torch.utils.data.dataset import Dataset


class IdentificationDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 landmark_ver='5',
                 train_csv='train.csv',
                 **_):
        self.split = split
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.landmark_ver = landmark_ver
        self.train_csv = 'train.csv'

        if split == 'test':
            self.images_dir = os.path.join(dataset_dir, 'test')
        else:
            self.images_dir = os.path.join(dataset_dir, 'train')

        self.df_examples = self.load_examples()
        self.size = len(self.df_examples)
        if self.split == 'train':
            self.size = self.size * 2

    def load_examples(self):
        type_keys = ['x','y','w','h','xl','yl','xn','yn','xr','yr','xd','yd']
        type_dict = {key:float for key in type_keys}
        if self.split == 'test':
            landmark_path = os.path.join(self.dataset_dir, 'landmark.test.{}.csv'.format(self.landmark_ver))
            print(landmark_path)
            image_ids_path = os.path.join(self.dataset_dir, 'sample_submission.csv')
            df_landmarks = pd.read_csv(landmark_path)
            df_landmarks = df_landmarks.rename(columns={'filename': 'Image'})
            df = pd.read_csv(image_ids_path)

            assert len(df_landmarks) == len(df)
            df_examples = df.join(df_landmarks.set_index('Image'), on='Image')
            df_examples = df_examples.astype(type_dict)
        elif self.split == 'known_whale':
            landmark_path = os.path.join(self.dataset_dir, 'landmark.train.{}.csv'.format(self.landmark_ver))
            print(landmark_path)
            image_ids_path = os.path.join(self.dataset_dir, self.train_csv)
            df_landmarks = pd.read_csv(landmark_path)
            df_landmarks = df_landmarks.rename(columns={'filename': 'Image'})
            df = pd.read_csv(image_ids_path)

            assert len(df_landmarks) == len(df)
            df_examples = df.join(df_landmarks.set_index('Image'), on='Image')
            df_examples = df_examples.astype(type_dict)
            df_examples = df_examples[df_examples['Id'] != 'new_whale']
        else:
            landmark_path = os.path.join(self.dataset_dir, 'landmark.train.{}.csv'.format(self.landmark_ver))
            df_landmarks = pd.read_csv(landmark_path)
            df_landmarks = df_landmarks.rename(columns={'filename': 'Image'})

            image_ids_path = os.path.join(self.dataset_dir, self.train_csv)
            df = pd.read_csv(image_ids_path)

            # split
            df_counted = df.groupby('Id').count()
            df_counted = df_counted.rename(columns={'Image': 'count'})
            df = df.join(df_counted, on='Id')
            df['split'] = 'train'

            df.loc[df['count'] == 2, 'split'] = 'none'
            df.loc[df['Id'] == 'new_whale', 'split'] = 'none'

            random.seed(508)
            ids = list(sorted(list(df.loc[df['count'] == 2]['Id'].unique())))
            random.shuffle(ids)
            for Id in ids[:-400]:
                df.loc[df['Id'] == Id, 'split'] = 'train'

            dev_ids = ids[-400:]

            for Id in dev_ids:
                image_ids = list(df[df['Id'] == Id]['Image'])
                assert len(image_ids) == 2
                if self.split == 'train':
                    df.loc[df['Image'] == image_ids[0], 'split'] = 'train'
                else:
                    df.loc[df['Image'] == image_ids[0], 'split'] = 'dev'

                df.loc[df['Image'] == image_ids[1], 'split'] = 'dev'

            # split new whale
            new_whales = list(sorted(list(df.loc[df['Id'] == 'new_whale']['Image'])))
            random.shuffle(new_whales)
            len_dev = int(len(df[df['split'] == 'dev']) * 0.276)
            for key in new_whales[:len_dev]:
                df.loc[df['Image'] == key, 'split'] = 'dev'

            # join with landmark
            assert len(df_landmarks) == len(df)
            df_examples = df.join(df_landmarks.set_index('Image'), on='Image')
            df_examples = df_examples.astype(type_dict)

            # set category id
            Ids = set(list(df_examples['Id'].unique()))
            if 'new_whale' in Ids:
                Ids.remove('new_whale')
            Ids = ['new_whale'] + list(sorted(list(Ids)))
            records = [(i, key) for i, key in enumerate(Ids)]
            df_category = pd.DataFrame.from_records(records,
                                                    columns=['category', 'Id'])
            df_category = df_category.set_index('Id')
            self.num_category = len(df_category)

            df_examples = df_examples.join(df_category, on='Id')
            df_examples = df_examples.astype({'category': int})

            df_examples = df_examples[df_examples['split'] == self.split]

        def to_filepath(v):
            return os.path.join(self.images_dir, v)

        df_examples['filepath'] = df_examples['Image'].transform(to_filepath)
        return df_examples

    def get_weights(self):
        return None

    def __getitem__(self, index):
        example = self.df_examples.iloc[index % len(self.df_examples)]

        filepath = example['filepath']
        image = misc.imread(example['filepath'], mode='RGB')
        assert image.ndim == 3
        assert image.shape[-1] == 3

        if 'category' in example:
            category = example['category']
        else:
            category = None

        if self.split == 'train' and index > len(self.df_examples):
            category = category + self.num_category
            image = np.fliplr(image)
            box = [1.0 - example['x'], example['y'], example['w'], example['h']]
            landmark = [1.0 - example['xr'], example['yr'], 1.0 - example['xn'], example['yn'],
                        1.0 - example['xl'], example['yl'], 1.0 - example['xd'], example['yd']]
        else:
            box = [example['x'], example['y'], example['w'], example['h']]
            landmark = [example['xl'], example['yl'], example['xn'], example['yn'],
                        example['xr'], example['yr'], example['xd'], example['yd']]

        if self.transform is not None:
            image = self.transform(image, box, landmark, example['Image'])
            
        if category is not None:
            return {'image': image,
                    'label': category,
                    'id': example['Id'],
                    'key': example['Image']}
        else:
            return {'image': image,
                    'id': example['Id'],
                    'key': example['Image']}

    def __len__(self):
        return self.size

def test():
    dataset = IdentificationDataset('data', 'test', None)
    print(len(dataset))

    dataset = IdentificationDataset('data', 'train', None)
    print(len(dataset))

    dataset = IdentificationDataset('data', 'dev', None)
    print(len(dataset))


if __name__ == '__main__':
    test()
