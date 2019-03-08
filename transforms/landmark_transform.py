from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math
import cv2
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa
from .utils import to_norm_bgr


def landmark_transform(split, size, mean, std, input_space, input_range, **_):
    fliplr_aug = iaa.Fliplr(1.0)
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.AverageBlur(k=(3,3))),
        iaa.Sometimes(0.5, iaa.MotionBlur(k=(3,5))),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={'x': (0.9,1.1), 'y': (0.9,1.1)},
            translate_percent={'x': (-0.05,0.05), 'y': (-0.05,0.05)},
            rotate=(-10,10)
            )),
        # iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.8,1.0))),
        ], random_order=True)

    def transform(image, label):
        image = image.astype(np.float32)

        H, W, _ = image.shape
        if split == 'test' or split == 'all':
            label = None
            image = cv2.resize(image, (size, size))
        else:
            fliplr = random.random() > 0.5
            if split == 'train':
                for l in label:
                    if math.isnan(l):
                        print('------------------------')
                        print('Huk!!')
                        print(label.shape)
                        print(label)
                keypoints = [ia.Keypoint(x=int(label[i*2]), y=int(label[i*2+1]))
                             for i in range(10)]

                landmarks = ia.KeypointsOnImage(keypoints, shape=(H,W))

                seq_det = seq.to_deterministic()
                assert image.shape[0] > 0 and image.shape[1] > 0, '{}'.format(image.shape)
                image = seq_det.augment_images([image])[0]
                landmarks = seq_det.augment_keypoints([landmarks])[0]
                if fliplr:
                    image = fliplr_aug.augment_images([image])[0]
                    landmarks = fliplr_aug.augment_keypoints([landmarks])[0]

                H, W, _ = image.shape
                xs = np.array([landmarks.keypoints[i].x / W for i in range(10)], dtype=np.float32)
                ys = np.array([landmarks.keypoints[i].y / H for i in range(10)], dtype=np.float32)
            else:
                H, W, _ = image.shape
                xs = np.array([label[i*2] / W for i in range(10)], dtype=np.float32)
                ys = np.array([label[i*2+1] / H for i in range(10)], dtype=np.float32)

            # bounding box
            x1, x2 = np.min(xs), np.max(xs)
            y1, y2 = np.min(ys), np.max(ys)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = np.clip((x2 - x1) * 1.2, 0, 1.0)
            h = np.clip((y2 - y1) * 1.1, 0, 1.0)

            # landmarks
            if split == 'train' and fliplr:
                xl = np.clip(xs[5], 0, 1.0)
                yl = np.clip(ys[5], 0, 1.0)
                xn = np.clip(xs[3], 0, 1.0)
                yn = np.clip(ys[3], 0, 1.0)
                xr = np.clip(xs[1], 0, 1.0)
                yr = np.clip(ys[1], 0, 1.0)
                xd = np.clip(xs[8], 0, 1.0)
                yd = np.clip(ys[8], 0, 1.0)
            else:
                xl = np.clip(xs[1], 0, 1.0)
                yl = np.clip(ys[1], 0, 1.0)
                xn = np.clip(xs[3], 0, 1.0)
                yn = np.clip(ys[3], 0, 1.0)
                xr = np.clip(xs[5], 0, 1.0)
                yr = np.clip(ys[5], 0, 1.0)
                xd = np.clip(xs[8], 0, 1.0)
                yd = np.clip(ys[8], 0, 1.0)

            label = np.array([x,y,w,h,xl,yl,xn,yn,xr,yr,xd,yd], dtype=np.float32)
            image = cv2.resize(image, (size, size))

        image = image.astype(np.float32)
        image = to_norm_bgr(image, mean, std, input_space, input_range)
        image = np.transpose(image, [2,0,1])
        return image, label

    return transform

