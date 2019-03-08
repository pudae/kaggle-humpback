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


def get_top5(scores, id_list, threshold):
    used = set()
    ret_ids = []

    indices = np.argsort(scores)[::-1]

    for index in indices:
        l = id_list[index]
        s = scores[index]
        # if l in used:
        #     continue

        if 'new_whale' not in used and s < threshold:
            used.add('new_whale')
            ret_ids.append('new_whale')

        used.add(l)
        ret_ids.append(l)
        if len(ret_ids) >= 5:
            break

    return ret_ids[:5]


def fill(df_submission, df_input, threshold):
    df_submission = df_submission.copy()
    assert len(df_submission[df_submission['Id'] == '']) == len(df_submission)

    m = df_input.values

    id_list = df_input.columns
    new_whale_count = 0
    for row_index, scores in enumerate(m):
        top5 = get_top5(scores, id_list, threshold=threshold)
        if top5[0] == 'new_whale':
            new_whale_count += 1
        df_submission.iat[row_index,0] = ' '.join(top5)

    assert len(df_submission[df_submission['Id'] == '']) == 0
    return (abs(new_whale_count - 2197),
            new_whale_count,
            threshold,
            df_submission)


def ensemble(input_paths):
    df_list = [pd.read_csv(input_path, index_col='Image')
               for input_path in input_paths]
    df = sum(df_list) / len(df_list)
    return df


def run(input_path, output_path, threshold):
    df_input = ensemble(input_path.split(','))
    df_submission = pd.read_csv('data/sample_submission.csv', index_col='Image')
    df_submission['Id'] = ''

    m = df_input.values

    df_list = []
    thresholds = np.arange(0.5, 0.3, -0.01) if threshold is None else [threshold]
    for threshold in tqdm.tqdm(thresholds):
        df_list.append(fill(df_submission, df_input, threshold))

    sorted_df = list(sorted(df_list, key=lambda x: x[:2]))
    error, new_whale_count, threshold, df_submission = sorted_df[0]
    df_submission.to_csv(output_path)


def parse_args():
    description = 'Make submission'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_path', dest='input_path',
                        help='input path',
                        default='input.csv', type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output path',
                        default='output.csv', type=str)
    parser.add_argument('--threshold', dest='threshold',
                        help='threshold for new whale',
                        default=None, type=float)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    dirname = os.path.dirname(args.output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    run(args.input_path, args.output_path, args.threshold)


if __name__ == '__main__':
    main()
