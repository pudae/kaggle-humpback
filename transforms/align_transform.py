from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from albumentations import Compose
from albumentations import Resize, HorizontalFlip, ToGray, Rotate
from albumentations import Blur, MotionBlur, GaussNoise
from albumentations import RandomBrightnessContrast, RandomGamma
from albumentations import ShiftScaleRotate

from .utils import to_norm_bgr


REF_PTS = np.array([[ 29.27,  54.37],
                    [158.69, 164.33],
                    [290.02,  59.29],
                    [157.62, 276.19]], dtype=np.float32)

ROT_180 = ['2872f5a1a.jpg',
           '02bdec750.jpg',
           '9d23d989a.jpg',
           '91b25cfc6.jpg']

ROT_90 = ['3a16eaf31.jpg',
          '6089968f6.jpg',
          'f3f2023c6.jpg',
          'd79731004.jpg',
          '74580a97d.jpg',
          'ac833887c.jpg',
          '22ca7df21.jpg',
          '7fba36f8c.jpg',
          'dde21203c.jpg',
          'a7dd3f508.jpg',
          '5fe0c6963.jpg',
          '7cabb0c94.jpg',
          '64ad3b4cd.jpg',
          '799f86aec.jpg',
          'ed309eb49.jpg',
          '6ee4e7f28.jpg']

ROT_270 = ['45d81f7ca.jpg',
           '5db3505ac.jpg',
           '32cf34278.jpg',
           '879a8b7bc.jpg',
           '6a4ea1d99.jpg',
           '155116572.jpg',
           '84a58826e.jpg',
           'faec9335c.jpg',
           'e9577799d.jpg',
           '5cd62b399.jpg',
           'bf84bad36.jpg',
           '94bcbacdc.jpg',
           '3308f94e2.jpg',
           'efae7b997.jpg',
           '2f30ae22c.jpg',
           'cff1c391d.jpg',
           '34ff88204.jpg',
           'b5cad5e97.jpg']


def align_transform(split, size, mean, std, input_space, input_range,
                    flip=False, align=True, align_p=1.0, **_):
    print('[align_transform] align:', align)
    print('[align_transform] align_p:', align_p)
    print('[align_transform] flip:', flip)
    resize = Resize(height=size, width=size, always_apply=True)
    flip_aug = HorizontalFlip(always_apply=True)

    rotate_90 = iaa.Affine(rotate=90)
    rotate_180 = iaa.Affine(rotate=180)
    rotate_270 = iaa.Affine(rotate=270)

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.AverageBlur(k=(3,3))),
        iaa.Sometimes(0.5, iaa.MotionBlur(k=(3,5))),
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.Multiply((0.9, 1.1), per_channel=0.5),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={'x': (0.9,1.1), 'y': (0.9,1.1)},
            translate_percent={'x': (-0.05,0.05), 'y': (-0.05,0.05)},
            shear=(-10,10),
            rotate=(-10,10)
            )),
        iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.8,1.0))),
        ], random_order=True)

    def transform(image, box, landmark, filename=None):
        image = image.astype(np.float32)
        box = np.array(box).astype(np.float32)
        landmark = np.array(landmark).astype(np.float32)

        H, W, _ = image.shape
        if split == 'train':
            keypoints = [ia.Keypoint(x=int(landmark[i*2]*W), y=int(landmark[i*2+1]*H))
                         for i in range(4)]
            landmarks = ia.KeypointsOnImage(keypoints, shape=(H,W))
            boxes = [ia.BoundingBox(x1=int((box[0]-box[2]/2)*W),
                                    y1=int((box[1]-box[3]/2)*H),
                                    x2=int((box[0]+box[2]/2)*W),
                                    y2=int((box[1]+box[3]/2)*H))]
            boxes = ia.BoundingBoxesOnImage(boxes, shape=(H,W))

            seq_det = seq.to_deterministic()
            assert image.shape[0] > 0 and image.shape[1] > 0, '{}'.format(image.shape)
            image = seq_det.augment_images([image])[0]
            landmarks = seq_det.augment_keypoints([landmarks])[0]
            boxes = seq_det.augment_bounding_boxes([boxes])[0].bounding_boxes[0]

            landmark = np.array([landmarks.keypoints[0].x,
                                 landmarks.keypoints[0].y,
                                 landmarks.keypoints[1].x,
                                 landmarks.keypoints[1].y,
                                 landmarks.keypoints[2].x,
                                 landmarks.keypoints[2].y,
                                 landmarks.keypoints[3].x,
                                 landmarks.keypoints[3].y], dtype=np.float32)
            landmark = landmark / [W,H,W,H,W,H,W,H]
            boxes = np.array([(boxes.x1 + boxes.x2)/2, (boxes.y1 + boxes.y2)/2,
                              boxes.x2 - boxes.x1, boxes.y2 - boxes.y1], dtype=np.float32)
            box = boxes / [W,H,W,H]
        elif filename is not None:
            keypoints = [ia.Keypoint(x=int(landmark[i*2]*W), y=int(landmark[i*2+1]*H))
                         for i in range(4)]
            landmarks = ia.KeypointsOnImage(keypoints, shape=(H,W))
            boxes = [ia.BoundingBox(x1=int((box[0]-box[2]/2)*W),
                                    y1=int((box[1]-box[3]/2)*H),
                                    x2=int((box[0]+box[2]/2)*W),
                                    y2=int((box[1]+box[3]/2)*H))]
            boxes = ia.BoundingBoxesOnImage(boxes, shape=(H,W))
            if filename in ROT_90:
                rot_det = rotate_90.to_deterministic()
            elif filename in ROT_180:
                rot_det = rotate_180.to_deterministic()
            elif filename in ROT_270:
                rot_det = rotate_270.to_deterministic()
            else:
                rot_det = None

            if rot_det is not None:
                image = rot_det.augment_images([image])[0]
                landmarks = rot_det.augment_keypoints([landmarks])[0]
                boxes = rot_det.augment_bounding_boxes([boxes])[0].bounding_boxes[0]

                landmark = np.array([landmarks.keypoints[0].x,
                                     landmarks.keypoints[0].y,
                                     landmarks.keypoints[1].x,
                                     landmarks.keypoints[1].y,
                                     landmarks.keypoints[2].x,
                                     landmarks.keypoints[2].y,
                                     landmarks.keypoints[3].x,
                                     landmarks.keypoints[3].y], dtype=np.float32)
                landmark = landmark / [W,H,W,H,W,H,W,H]
                boxes = np.array([(boxes.x1 + boxes.x2)/2, (boxes.y1 + boxes.y2)/2,
                                  boxes.x2 - boxes.x1, boxes.y2 - boxes.y1], dtype=np.float32)
                box = boxes / [W,H,W,H]

        if split == 'train':
            xc = box[0] + (random.random() * 0.10 - 0.05)
            yc = box[1] + (random.random() * 0.10 - 0.05)
            w = box[2] * (1.0 + random.uniform(0.2,0.4))
            h = box[3] * (1.0 + random.uniform(0.2,0.4))
        else:
            xc = box[0]
            yc = box[1]
            w = box[2] * 1.3
            h = box[3] * 1.3

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        l_x1 = np.min(np.array([landmark[0], landmark[2], landmark[4], landmark[6]]))
        l_y1 = np.min(np.array([landmark[1], landmark[3], landmark[5], landmark[7]]))
        l_x2 = np.max(np.array([landmark[0], landmark[2], landmark[4], landmark[6]]))
        l_y2 = np.max(np.array([landmark[1], landmark[3], landmark[5], landmark[7]]))
        
        x1 = np.clip(np.min([l_x1, x1]), 0.0, 1.0)
        y1 = np.clip(np.min([l_y1, y1]), 0.0, 1.0)
        x2 = np.clip(np.max([l_x2, x2]), 0.0, 1.0)
        y2 = np.clip(np.max([l_y2, y2]), 0.0, 1.0)

        w = x2 - x1
        h = y2 - y1

        if split == 'train':
            xl = np.clip((landmark[0] + (random.random() * 0.10 - 0.05) - x1) / w, 0.0, 1.0)
            yl = np.clip((landmark[1] + (random.random() * 0.10 - 0.05) - y1) / h, 0.0, 1.0)
            xn = np.clip((landmark[2] + (random.random() * 0.10 - 0.05) - x1) / w, 0.0, 1.0)
            yn = np.clip((landmark[3] + (random.random() * 0.10 - 0.05) - y1) / h, 0.0, 1.0)
            xr = np.clip((landmark[4] + (random.random() * 0.10 - 0.05) - x1) / w, 0.0, 1.0)
            yr = np.clip((landmark[5] + (random.random() * 0.10 - 0.05) - y1) / h, 0.0, 1.0)
            xd = np.clip((landmark[6] + (random.random() * 0.10 - 0.05) - x1) / w, 0.0, 1.0)
            yd = np.clip((landmark[7] + (random.random() * 0.10 - 0.05) - y1) / h, 0.0, 1.0)
        else:
            xl = np.clip((landmark[0] - x1) / w, 0.0, 1.0)
            yl = np.clip((landmark[1] - y1) / h, 0.0, 1.0)
            xn = np.clip((landmark[2] - x1) / w, 0.0, 1.0)
            yn = np.clip((landmark[3] - y1) / h, 0.0, 1.0)
            xr = np.clip((landmark[4] - x1) / w, 0.0, 1.0)
            yr = np.clip((landmark[5] - y1) / h, 0.0, 1.0)
            xd = np.clip((landmark[6] - x1) / w, 0.0, 1.0)
            yd = np.clip((landmark[7] - y1) / h, 0.0, 1.0)

        image = image[int(y1*H):int(y2*H), int(x1*W):int(x2*W)]
        image = resize(image=image)['image']

        if align:
            if split != 'train' or random.random() <= align_p:
                lm = np.array([[xl,yl], [xn,yn], [xr,yr], [xd,yd]], dtype=np.float32)
                lm = lm * [size, size]
                lm = np.hstack((lm, np.ones((lm.shape[0], 1))))
                M = np.linalg.lstsq(lm, REF_PTS, rcond=1)[0]
                image = cv2.warpAffine(image, M.T[:2], dsize=(size, size))

        if flip:
            image = flip_aug(image=image)['image']

        image = to_norm_bgr(image, mean, std, input_space, input_range)
        image = np.transpose(image, [2,0,1])
        image = image.astype(np.float32)
        return image

    return transform

