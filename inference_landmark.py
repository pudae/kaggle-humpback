from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch

from datasets import get_dataloader
from transforms import get_transform, from_norm_bgr
from tasks import get_task
import utils.config
import utils.checkpoint


def inference(config, task, dataloader):
    keys = []
    task.get_model().eval()
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        column_values_dict = defaultdict(list)
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            keys = data['key']
            outputs = task.forward(images)
            predicts = task.inference(outputs=outputs)

            boxes = predicts['boxes'].cpu().numpy()
            landmarks = predicts['landmarks'].cpu().numpy()
            probabilities = predicts['probabilities'].cpu().numpy()

            column_values_dict['boxes'].extend(boxes)
            column_values_dict['landmarks'].extend(landmarks)
            column_values_dict['keys'].extend(keys)

        keys = column_values_dict['keys']
        boxes = np.array(column_values_dict['boxes'])
        landmarks = np.array(column_values_dict['landmarks'])

    records = []
    for key, box, landmark in zip(keys, boxes, landmarks):
        records.append((key, box[0], box[1], box[2], box[3],
                        landmark[0], landmark[1], landmark[2], landmark[3],
                        landmark[4], landmark[5], landmark[6], landmark[7]))

    df = pd.DataFrame.from_records(records,
                                   columns=['filename', 'x', 'y', 'w', 'h',
                                            'xl', 'yl', 'xn', 'yn', 'xr', 'yr', 'xd', 'yd'])
    return df


def run(config, split, checkpoint_name, output_path):
    train_dir = config.train.dir

    task = get_task(config)
    checkpoint = utils.checkpoint.get_checkpoint(config, checkpoint_name)
    last_epoch, step = utils.checkpoint.load_checkpoint(task.get_model(),
                                                        None,
                                                        checkpoint)

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))

    preprocess_opt = task.get_preprocess_opt()
    dataloader = get_dataloader(config, split,
                                get_transform(config, split, **preprocess_opt))

    df = inference(config, task, dataloader)
    df.to_csv(output_path, index=False)


def parse_args():
    description = 'inference box & landmark'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument('--split', dest='split',
                        help='split name',
                        default=None, type=str)
    parser.add_argument('--checkpoint_name', dest='checkpoint_name',
                        help='checkpoint name',
                        default='best.score.pth', type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output path',
                        default='output.csv', type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('inference box & landmark')
    args = parse_args()

    config = utils.config.load(args.config_file)
    run(config, args.split, args.checkpoint_name, args.output_path)

    print('success!')


if __name__ == '__main__':
    main()

