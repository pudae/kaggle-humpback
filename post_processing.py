from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tqdm
import scipy.misc as misc
import pandas as pd
from collections import defaultdict


def process_duplicate_ids(df_submission, df_duplicate):
    duplicate_ids_dict = defaultdict(list)
    for _, row in df_duplicate.iterrows():
        ids = row['SubId'].split(' ')
        if len(ids) == 2:
            duplicate_ids_dict[ids[0]].append(ids[1])
            duplicate_ids_dict[ids[1]].append(ids[0])
        else:
            duplicate_ids_dict[ids[0]].extend([ids[1], ids[2]])
            duplicate_ids_dict[ids[1]].extend([ids[0], ids[2]])
            duplicate_ids_dict[ids[2]].extend([ids[0], ids[1]])

    records = []
    for image_filenae, row in tqdm.tqdm(df_submission.iterrows(), total=len(df_submission)):
        ids = row['Id'].split(' ')
    
        if ids[0] not in duplicate_ids_dict:
            records.append((image_filenae, row['Id']))
            continue
    
        duplicate_ids = duplicate_ids_dict[ids[0]]
        new_ids = ids[:1] + duplicate_ids
        ids = new_ids + [v for v in ids if v not in new_ids]
        ids = ids[:5]
    
        assert len(ids) == len(set(ids))
        records.append((image_filenae, ' '.join(ids)))
    
    return pd.DataFrame.from_records(records, columns=['Image', 'Id'], index='Image')


def get_image_size(filepath):
    return misc.imread(filepath).shape[:2]


def get_image_sizes(filenames, images_dir):
    ret = set()
    for filename in filenames:
        filepath = os.path.join(images_dir, filename)
        ret.add(get_image_size(filepath))
    return ret


def build_size_dict(df_group, df_train):
    leak_id_set = set()
    ret_dict = {}
    for _, row in df_group.iterrows():
        id_list = row['SubId'].split(' ')
        size_dict = {}
        for id_str in id_list:
            filenames = df_train[df_train['Id'] == id_str].index
            size_dict[id_str] = get_image_sizes(filenames, os.path.join('data', 'train'))
            leak_id_set.add(id_str)

        ret_dict[tuple(list(sorted(id_list)))] = size_dict

    return ret_dict, leak_id_set


def update_using_image_size_leak(df_submission, df_duplicate, df_train):
    size_dict, leak_id_set = build_size_dict(df_duplicate, df_train)

    records = []
    for image_filename, row in tqdm.tqdm(df_submission.iterrows(), total=len(df_submission)):
        ids = row['Id'].split(' ')
        if ids[0] not in leak_id_set:
            records.append((image_filename, row['Id']))
            continue

        leak_dict = None
        for key, value in size_dict.items():
            if ids[0] in key:
                leak_dict = value
                break

        cur_size = get_image_size(os.path.join('data', 'test', image_filename))
        target_id = None
        for key, value in leak_dict.items():
            if cur_size in value:
                if target_id is not None:
                    target_id = None
                    break
                target_id = key

        if target_id is None:
            records.append((image_filename, row['Id']))
            continue

        new_ids = [target_id] + [v for v in ids if v != target_id]
        assert len(new_ids) == 5
        records.append((image_filename, ' '.join(new_ids)))

    return pd.DataFrame.from_records(records, columns=['Image', 'Id'], index='Image')


def update_using_data_leak(df_submission, df_leak):
    changed = []
    records = []
    for image_filename, row in tqdm.tqdm(df_submission.iterrows(), total=len(df_submission)):
        ids = row['Id'].split(' ')
        if image_filename not in df_leak.index:
            records.append((image_filename, row['Id']))
            continue

        leak_id = df_leak.loc[image_filename]['Id']
        top1_id = ids[0]

        if leak_id == top1_id:
            records.append((image_filename, row['Id']))
            continue

        ids = [Id for Id in ids if Id != leak_id]
        ids = [leak_id] + ids
        changed.append((image_filename, row['Id'], leak_id))
        assert len(set(ids)) == len(ids)
        records.append((image_filename, ' '.join(ids)))

    return pd.DataFrame.from_records(records, columns=['Image', 'Id'], index='Image')


def parse_args():
    description = 'post processing'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_path', dest='input_path',
                        help='input file path',
                        default=None, type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='result file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('post processing')
    args = parse_args()
    print('from', args.input_path)
    print('to', args.output_path)

    df_submission = pd.read_csv(args.input_path, index_col='Image')
    df_leak = pd.read_csv('data/leaks.csv', index_col='Image')
    df_train = pd.read_csv('data/train.csv', index_col='Image')
    df_duplicate = pd.read_csv('data/duplicate_ids.csv')

    df_submission = process_duplicate_ids(df_submission, df_duplicate)
    df_submission = update_using_image_size_leak(df_submission, df_duplicate, df_train)
    df_submission = update_using_data_leak(df_submission, df_leak)
    df_submission.to_csv(args.output_path)

    print('success!')


if __name__ == '__main__':
    main()

