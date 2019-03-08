from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from datasets import get_dataloader
from transforms import get_transform
from tasks import get_task
import utils.config
import utils.checkpoint


def get_center(vectors):
    avg = np.mean(vectors, axis=0)
    if avg.ndim == 1:
        avg = avg / np.linalg.norm(avg)
    elif avg.ndim == 2:
        assert avg.shape[1] == 512
        avg = avg / np.linalg.norm(avg, axis=1, keepdims=True)
    else:
        assert False, avg.shape
    return avg


def get_nearest_k(center, features, k, threshold):
    feature_with_dis = [(feature, np.dot(center, feature)) for feature in features]
    if len(feature_with_dis) > 10:
        distances = np.array([dis for _, dis in feature_with_dis])

    filtered = [feature for feature, dis in feature_with_dis if dis > 0.5]
    if len(filtered) < len(feature_with_dis):
        distances = np.array([feature for feature, dis in feature_with_dis if dis <= 0.5])
    if len(filtered) > k:
        return filtered
    feature_with_dis = [feature for feature, dis in sorted(feature_with_dis, key=lambda v: v[1], reverse=True)]
    return feature_with_dis[:k]


def get_image_center(features):
    if len(features) < 4:
        return get_center(features)

    for _ in range(2):
        center = get_center(features)
        features = get_nearest_k(center, features, int(len(features) * 3 / 4), 0.5)
        if len(features) < 4:
            break

    return get_center(features)


def average_features(features, id_list):
    averaged_features = []
    averaged_id_list = []

    unique_ids = set(id_list)
    unique_ids = list(sorted(list(unique_ids)))

    for unique_id in unique_ids:
        assert unique_id != 'new_whale'
        cur_features = [feature for feature, Id
                        in zip(features, id_list) if Id == unique_id]
        cur_features = np.stack(cur_features, axis=0)
        if len(cur_features) == 1:
            averaged_features.append(cur_features[0])
            averaged_id_list.append(unique_id)
        else:
            averaged_feature = get_center(cur_features)
            averaged_features.append(averaged_feature)
            averaged_id_list.append(unique_id)
    averaged_features = np.stack(averaged_features, axis=0)
    assert averaged_features.shape[0] == len(averaged_id_list)

    return averaged_features, averaged_id_list


def inference(config, task, dataloader, ret_dict):
    task.get_model().eval()

    id_dict = {}
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        column_values_dict = defaultdict(list)
        metric_list_dict = defaultdict(list)
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            key_list = data['key']
            id_list = data['id']
            outputs = task.forward(images)
            predicts = task.inference(outputs=outputs)

            for key, Id, feature in zip(key_list, id_list, predicts['features'].cpu().numpy()):
                ret_dict[key].append(feature)
                id_dict[key] = Id
    return id_dict


def inference_single_tta(config, task, preprocess_opt, split, fold, flip, align, ret_dict):
    config.transform.params.align = align
    transform = 'test' if split == 'test' else 'all'
    config.data.params.landmark_ver = fold
    dataloader = get_dataloader(config, split,
                                get_transform(config, transform, flip=flip, **preprocess_opt))
    id_dict = inference(config, task, dataloader, ret_dict)
    return id_dict


def inference_single_setting(config, task, preprocess_opt, split, flip, align, landmark_folds, ret_dict):
    for i in landmark_folds:
        id_dict = inference_single_tta(config, task, preprocess_opt, split, i,
                                       flip=flip, align=align, ret_dict=ret_dict)
    
    return id_dict


def run(config, tta_flip, tta_landmark, checkpoint_name, output_path):
    train_dir = config.train.dir

    task = get_task(config)
    checkpoint = utils.checkpoint.get_checkpoint(config, checkpoint_name)
    last_epoch, step = utils.checkpoint.load_checkpoint(task.get_model(),
                                                        None,
                                                        checkpoint)

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    preprocess_opt = task.get_preprocess_opt()
    config.data.params.train_csv = 'train.csv'

    landmark_folds = range(6) if tta_landmark else [5]

    ###########################################################################
    # train features
    # non-flip
    # default 
    ret_dict = defaultdict(list)
    id_dict = inference_single_setting(config, task, preprocess_opt,
                                       'known_whale', flip=False, align=False, landmark_folds=landmark_folds, ret_dict=ret_dict)
    
    # align
    id_dict = inference_single_setting(config, task, preprocess_opt,
                                       'known_whale', flip=False, align=True, landmark_folds=landmark_folds, ret_dict=ret_dict)

    id_features_dict = defaultdict(list)
    for key, features in ret_dict.items():
        id_features_dict[id_dict[key]].extend(features)
    features_dict_ori = {key:get_image_center(features) for key, features in id_features_dict.items()}

    id_list_ori = list(sorted(features_dict_ori.keys()))
    features_ori = np.stack([features_dict_ori[Id] for Id in id_list_ori], axis=0)

    # flip
    # default 
    if tta_flip:
        ret_dict = defaultdict(list)
        id_dict = inference_single_setting(config, task, preprocess_opt,
                                           'known_whale', flip=True, align=False, landmark_folds=landmark_folds, ret_dict=ret_dict)
        
        # align
        id_dict = inference_single_setting(config, task, preprocess_opt,
                                           'known_whale', flip=True, align=True, landmark_folds=landmark_folds, ret_dict=ret_dict)

        id_features_dict = defaultdict(list)
        for key, features in ret_dict.items():
            id_features_dict[id_dict[key]].extend(features)
        features_dict_flip = {key:get_image_center(features) for key, features in id_features_dict.items()}

        id_list_flip = list(sorted(features_dict_flip.keys()))
        features_flip = np.stack([features_dict_flip[Id] for Id in id_list_flip], axis=0)

        assert id_list_ori == id_list_flip
    id_list = id_list_ori

    ###########################################################################
    # test features
    # non-flip
    # default 
    ret_dict = defaultdict(list)
    id_dict = inference_single_setting(config, task, preprocess_opt,
                                       'test', flip=False, align=False, landmark_folds=landmark_folds, ret_dict=ret_dict)

    id_dict = inference_single_setting(config, task, preprocess_opt,
                                       'test', flip=False, align=True, landmark_folds=landmark_folds, ret_dict=ret_dict)

    features_dict = {key:get_image_center(features) for key, features in ret_dict.items()}
    test_key_list_ori = list(sorted(id_dict.keys()))
    test_features_ori = np.stack([features_dict[key] for key in test_key_list_ori], axis=0)

    # flip
    # default 
    if tta_flip:
        ret_dict = defaultdict(list)
        id_dict = inference_single_setting(config, task, preprocess_opt,
                                           'test', flip=True, align=False, landmark_folds=landmark_folds, ret_dict=ret_dict)

        id_dict = inference_single_setting(config, task, preprocess_opt,
                                           'test', flip=True, align=True, landmark_folds=landmark_folds, ret_dict=ret_dict)

        features_dict = {key:get_image_center(features) for key, features in ret_dict.items()}
        test_key_list_flip = list(sorted(id_dict.keys()))
        test_features_flip = np.stack([features_dict[key] for key in test_key_list_flip], axis=0)

        assert test_key_list_flip == test_key_list_ori
    key_list = test_key_list_ori

    # calculate distance
    m_ori = np.matmul(test_features_ori, features_ori.transpose())
    if tta_flip:
        m_flip = np.matmul(test_features_flip, features_flip.transpose())
        m = np.mean(np.stack([m_ori, m_flip],axis=0), axis=0)
    else:
        m = m_ori

    records = []
    for i, scores in enumerate(m):
        records.append(tuple([key_list[i]] + ['{:08f}'.format(v) for v in scores]))

    columns = ['Image'] + id_list
    assert len(records[0]) == len(columns)

    df_distance = pd.DataFrame.from_records(records, columns=columns)
    df_distance.to_csv(output_path, index=False)


def parse_args():
    description = 'inference similarities of whales'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument('--checkpoint_name', dest='checkpoint_name',
                        help='checkpoint name',
                        default='best.score.pth', type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output path',
                        default='output.csv', type=str)
    parser.add_argument('--tta_flip', dest='tta_flip',
                        help='tta flip',
                        default=1, type=int)
    parser.add_argument('--tta_landmark', dest='tta_landmark',
                        help='tta landmark',
                        default=1, type=int)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('inference similarities of whales')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    dir_name = os.path.dirname(args.output_path)
    os.makedirs(dir_name, exist_ok=True)
    run(config,
        tta_flip=args.tta_flip==1,
        tta_landmark=args.tta_landmark==1,
        checkpoint_name=args.checkpoint_name,
        output_path=args.output_path)
    print('success!')


if __name__ == '__main__':
    main()

