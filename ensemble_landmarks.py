import os
import argparse
import tqdm
import json
import cv2
import scipy.misc as misc
import scipy.ndimage as ndimage
import numpy as np
import pandas as pd


IMAGES_DIR = 'data/train'


REF_PTS = np.array([[ 29.27,  54.37],
                    [158.69, 164.33],
                    [290.02,  59.29],
                    [157.62, 276.19]], dtype=np.float32)


def get_error(values):
    l2_list = []
    for i in range(len(values)-1):
        for j in range(i, len(values)):
            l2_list.append(np.sqrt(np.sum((values[i] - values[j])**2)))

    return sum(l2_list) / len(l2_list)


def get_box_and_landmark(row):
    x = row['x']
    y = row['y']
    w = np.clip(row['w'] * 1.3, 0, 1.0)
    h = np.clip(row['h'] * 1.3, 0, 1.0)

    x1 = np.clip(x - w / 2, 0, 1.0)
    y1 = np.clip(y - h / 2, 0, 1.0)
    x2 = np.clip(x + w / 2, 0, 1.0)
    y2 = np.clip(y + h / 2, 0, 1.0)

    xl = np.clip(row['xl'], 0.0, 1.0)
    yl = np.clip(row['yl'], 0.0, 1.0)
    xn = np.clip(row['xn'], 0.0, 1.0)
    yn = np.clip(row['yn'], 0.0, 1.0)
    xr = np.clip(row['xr'], 0.0, 1.0)
    yr = np.clip(row['yr'], 0.0, 1.0)
    xd = np.clip(row['xd'], 0.0, 1.0)
    yd = np.clip(row['yd'], 0.0, 1.0)

    return np.array([x1,y1,x2,y2]), np.array([[xl,yl],[xn,yn],[xr,yr],[xd,yd]])


def get_iou(box1, box2):
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    intersection = (x2-x1) * (y2-y1)
    iou = intersection / (area1 + area2 - intersection)
    return iou


def get_top2(boxes):
    ious = []
    for i in range(len(boxes)-1):
        for j in range(i, len(boxes)):
            iou = get_iou(boxes[i], boxes[j])
            ious.append((i,j,iou))
    return list(sorted(ious, key=lambda v:v[2]))[-1]


def ensemble_test(input_path, output_path):
    df_landmarks = [pd.read_csv(p) for p in input_path.split(',')]

    records = []
    for i in tqdm.trange(len(df_landmarks[0])):
        filename = df_landmarks[0].iloc[i]['filename']
        box_and_landmarks = [get_box_and_landmark(df_landmarks[j].iloc[i])
                             for j in range(len(df_landmarks))]
        boxes = [v[0] for v in box_and_landmarks]
        landmarks = [v[1] for v in box_and_landmarks]
        l2_boxes = get_error(boxes)
        l2_landmarks = get_error(landmarks)
        records.append((filename, boxes, landmarks, l2_boxes, l2_landmarks))

    final_records = []
    records = list(sorted(records, key=lambda v: v[4], reverse=True))
    for filename, boxes, landmarks, l2_boxes, l2_landmarks in records:
        i, j, iou = get_top2(boxes)
        box = (boxes[i] + boxes[j]) / 2
        landmark = (landmarks[i] + landmarks[j]) / 2

        xc = (box[2] - box[0]) / 2
        yc = (box[3] - box[1]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]

        xl, yl = landmark[0][0], landmark[0][1]
        xn, yn = landmark[1][0], landmark[1][1]
        xr, yr = landmark[2][0], landmark[2][1]
        xd, yd = landmark[3][0], landmark[3][1]

        final_records.append((filename, xc, yc, w, h, xl, yl, xn, yn, xr, yr, xd, yd))

    columns = ['filename', 'x', 'y', 'w', 'h', 'xl', 'yl', 'xn', 'yn', 'xr', 'yr', 'xd', 'yd']
    df = pd.DataFrame.from_records(final_records, columns=columns)
    df.to_csv(output_path, index=False)


def get_box_and_landmark(row):
    x = row['x']
    y = row['y']
    w = np.clip(row['w'] * 1.3, 0, 1.0)
    h = np.clip(row['h'] * 1.3, 0, 1.0)

    x1 = np.clip(x - w / 2, 0, 1.0)
    y1 = np.clip(y - h / 2, 0, 1.0)
    x2 = np.clip(x + w / 2, 0, 1.0)
    y2 = np.clip(y + h / 2, 0, 1.0)

    xl = np.clip(row['xl'], 0.0, 1.0)
    yl = np.clip(row['yl'], 0.0, 1.0)
    xn = np.clip(row['xn'], 0.0, 1.0)
    yn = np.clip(row['yn'], 0.0, 1.0)
    xr = np.clip(row['xr'], 0.0, 1.0)
    yr = np.clip(row['yr'], 0.0, 1.0)
    xd = np.clip(row['xd'], 0.0, 1.0)
    yd = np.clip(row['yd'], 0.0, 1.0)

    return np.array([x1,y1,x2,y2]), np.array([[xl,yl],[xn,yn],[xr,yr],[xd,yd]])


def load_csv_annotation():
    df = pd.read_csv('data/split.keypoint.0.csv', index_col='filename')
    del df['split']

    new_records = []
    for filename, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        H,W,_ = misc.imread(os.path.join('data/train', filename), mode='RGB').shape
        # 1, 3, 5, 8
        for i in range(10):
            row['{}_x'.format(i)] = row['{}_x'.format(i)] / W
            row['{}_y'.format(i)] = row['{}_y'.format(i)] / H

        x1 = np.min([row['{}_x'.format(i)] for i in range(10)])
        y1 = np.min([row['{}_y'.format(i)] for i in range(10)])
        x2 = np.max([row['{}_x'.format(i)] for i in range(10)])
        y2 = np.max([row['{}_y'.format(i)] for i in range(10)])

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, 1.0)
        y2 = min(y2, 1.0)

        x = (x2 + x1) / 2
        y = (y2 + y1) / 2
        w = x2 - x1
        h = y2 - y1

        xl, yl = row['1_x'], row['1_y']
        xn, yn = row['3_x'], row['3_y']
        xr, yr = row['5_x'], row['5_y']
        xd, yd = row['8_x'], row['8_y']

        new_records.append((filename,
                            x, y, w, h,
                            xl, yl, xn, yn, xr, yr, xd, yd))

    columns = ['filename', 'x', 'y', 'w', 'h', 'xl', 'yl', 'xn', 'yn', 'xr', 'yr', 'xd', 'yd']
    return pd.DataFrame.from_records(new_records, columns=columns, index='filename')


def load_json_annotation():
    with open('data/annotations.addition.json', 'r') as fid:
        j = json.loads(fid.read())

    records = []
    for item in tqdm.tqdm(j):
        face = [v for v in item['annotations'] if v['class'] == 'Face'][0]
        image = misc.imread(os.path.join('data', item['filename']), mode='RGB')
        H,W,_ = misc.imread(os.path.join('data', item['filename']), mode='RGB').shape
        x1 = max(0, face['x']) / W
        y1 = max(0, face['y']) / H
        w = min(1.0, face['width'] / W)
        h = min(1.0, face['height'] / H)
        xc = x1 + w / 2
        yc = y1 + h / 2

        points = [v for v in item['annotations'] if v['class'] == 'point']
        if len(points) != 4:
            print(item['filename'])
            continue
        xl = min(1.0, max(0, points[0]['x'] / W))
        yl = min(1.0, max(0, points[0]['y'] / H))
        xn = min(1.0, max(0, points[1]['x'] / W))
        yn = min(1.0, max(0, points[1]['y'] / H))
        xr = min(1.0, max(0, points[2]['x'] / W))
        yr = min(1.0, max(0, points[2]['y'] / H))
        xd = min(1.0, max(0, points[3]['x'] / W))
        yd = min(1.0, max(0, points[3]['y'] / H))

        filename = os.path.basename(item['filename'])
        records.append((filename,
                        xc, yc, w, h,
                        xl, yl, xn, yn, xr, yr, xd, yd))

    columns = ['filename', 'x', 'y', 'w', 'h', 'xl', 'yl', 'xn', 'yn', 'xr', 'yr', 'xd', 'yd']
    return pd.DataFrame.from_records(records, columns=columns, index='filename')


def ensemble_train(input_path, output_path):
    df_landmarks = [pd.read_csv(p) for p in input_path.split(',')]

    df_csv_annotation = load_csv_annotation()
    df_json_annotation = load_json_annotation()

    records = []
    for i in tqdm.trange(len(df_landmarks[0])):
        filename = df_landmarks[0].iloc[i]['filename']
        box_and_landmarks = [get_box_and_landmark(df_landmarks[j].iloc[i])
                             for j in range(len(df_landmarks))]
        boxes = [v[0] for v in box_and_landmarks]
        landmarks = [v[1] for v in box_and_landmarks]
        l2_boxes = get_error(boxes)
        l2_landmarks = get_error(landmarks)
        records.append((filename, boxes, landmarks, l2_boxes, l2_landmarks))

    final_records = []
    records = list(sorted(records, key=lambda v: v[4], reverse=True))
    for filename, boxes, landmarks, l2_boxes, l2_landmarks in tqdm.tqdm(records):
        if filename in df_json_annotation.index:
            row = df_json_annotation.loc[filename]
            xc = row['x']
            yc = row['y']
            w = row['w']
            h = row['h']
            xl = row['xl']
            yl = row['yl']
            xn = row['xn']
            yn = row['yn']
            xr = row['xr']
            yr = row['yr']
            xd = row['xd']
            yd = row['yd']
        elif filename in df_csv_annotation.index:
            row = df_csv_annotation.loc[filename]
            xc = row['x']
            yc = row['y']
            w = row['w']
            h = row['h']
            xl = row['xl']
            yl = row['yl']
            xn = row['xn']
            yn = row['yn']
            xr = row['xr']
            yr = row['yr']
            xd = row['xd']
            yd = row['yd']
        else:
            i, j, iou = get_top2(boxes)
            box = (boxes[i] + boxes[j]) / 2
            landmark = (landmarks[i] + landmarks[j]) / 2

            xc = (box[2] - box[0]) / 2
            yc = (box[3] - box[1]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]

            xl, yl = landmark[0][0], landmark[0][1]
            xn, yn = landmark[1][0], landmark[1][1]
            xr, yr = landmark[2][0], landmark[2][1]
            xd, yd = landmark[3][0], landmark[3][1]
        final_records.append((filename, xc, yc, w, h, xl, yl, xn, yn, xr, yr, xd, yd))

    columns = ['filename', 'x', 'y', 'w', 'h', 'xl', 'yl', 'xn', 'yn', 'xr', 'yr', 'xd', 'yd']
    df = pd.DataFrame.from_records(final_records, columns=columns)
    df.to_csv(output_path, index=False)


def run(split, input_path, output_path):
    if split == 'train':
        ensemble_train(input_path, output_path)
    elif split == 'test':
        ensemble_test(input_path, output_path)
    else:
        assert False


def parse_args():
    description = 'ensemble landmarks'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_path', dest='input_path',
                        help='input path',
                        default='input.csv', type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output path',
                        default='output.csv', type=str)
    parser.add_argument('--split', dest='split',
                        help='split name',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    run(args.split, args.input_path, args.output_path)


if __name__ == '__main__':
    main()
