from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_model
from losses import get_loss
from transforms import from_norm_bgr


class LandmarkDetector(object):
    # (X, Y, W, H)
    SCALE = [0.5, 1.0, 2.0]
    RATIO = [[0.30, 0.30],
             [0.60, 0.15],
             [0.15, 0.60]]
    NUM_OUTPUTS = 1+4+2*4

    def __init__(self, config):
        self.anchors = [np.array(LandmarkDetector.RATIO) * s for s in LandmarkDetector.SCALE]
        self.anchors = np.concatenate(self.anchors, axis=0)
        assert self.anchors.shape == (len(LandmarkDetector.SCALE) * len(LandmarkDetector.RATIO), 2)
        self.feature_size = config.model.params.feature_size
        self.num_anchors = len(LandmarkDetector.SCALE) * len(LandmarkDetector.RATIO)

        num_outputs = LandmarkDetector.NUM_OUTPUTS
        self.model = get_model(config, num_outputs=num_outputs)
        self.model.avgpool = nn.AdaptiveAvgPool2d(self.feature_size)
        in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Conv2d(in_channels=in_features,
                                           out_channels=len(self.anchors)*num_outputs,
                                           kernel_size=1)
        def logits(self, features):
            x = self.avgpool(features)
            x = self.last_linear(x)
            return x

        self.model.logits = types.MethodType(logits, self.model)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.preprocess_opt = {'mean': self.model.mean,
                               'std': self.model.std,
                               'input_range': self.model.input_range,
                               'input_space': self.model.input_space}

        self.criterion = get_loss(config)
        self.cls_criterion = F.binary_cross_entropy_with_logits

    def get_model(self):
        return self.model

    def get_preprocess_opt(self):
        return self.preprocess_opt

    def forward(self, images, labels=None, **_):
        return self.model(images)

    def inference(self, images=None, outputs=None, labels=None, **_):
        if outputs is None:
            assert images is not None
            outputs = self.model(images)

        num_outputs = LandmarkDetector.NUM_OUTPUTS
        outputs = outputs.view(-1,num_outputs,self.num_anchors,self.feature_size,self.feature_size)
        anchors = self._get_anchors()

        B,C,A,H,W = outputs.size()
        outputs = outputs.view(B,C,A*H*W)
        anchors = torch.stack([anchors]*B, dim=0)
        anchors = anchors.view(B,-1,A*H*W)

        scores, indices = torch.max(outputs[:,0], dim=1)
        outputs = outputs[torch.arange(B), :, indices]
        anchors = anchors[torch.arange(B), :, indices]
        boxes = self._targets_to_boxes(outputs[:,1:5], anchors)
        landmarks = self._targets_to_landmarks(outputs[:,5:], anchors)
        probabilities = F.sigmoid(scores)
        return {'boxes': boxes, 'landmarks': landmarks, 'probabilities': probabilities}

    def _get_anchors(self):
        anchors = []
        denom = self.feature_size*2
        for y in np.arange(1/denom, 1.0, 2/denom):
            for x in np.arange(1/denom, 1.0, 2/denom):
                for w, h in self.anchors:
                    anchors.append([x, y, w, h])
        # row x column x num_anchors x 4
        # 5 x 5 x 9 x 4
        anchors = np.array(anchors).reshape((self.feature_size,self.feature_size,self.num_anchors,4))
        # row x column x num_anchors x 4 => 4 x num_anchors x row x col
        anchors = np.transpose(anchors, (3,2,0,1))
        anchors = torch.FloatTensor(anchors).cuda()
        assert anchors.size() == (4,self.num_anchors,self.feature_size,self.feature_size)
        return anchors

    def loss(self, outputs, labels, **_):
        num_outputs = LandmarkDetector.NUM_OUTPUTS
        outputs = outputs.view(-1,num_outputs,self.num_anchors,self.feature_size,self.feature_size)
        anchors = self._get_anchors()

        output_boxes = self._targets_to_boxes(outputs[:,1:5], anchors.unsqueeze(0))
        output_landmarks = self._targets_to_landmarks(outputs[:,5:], anchors.unsqueeze(0))

        box_targets = self._boxes_to_targets(labels[:,:4], anchors)
        landmark_targets = self._landmarks_to_targets(labels[:,4:], anchors)
        cls_targets, target_on_off = self._get_cls_targets(labels, anchors.unsqueeze(0))

        assert cls_targets.size() == target_on_off.size()
        assert cls_targets.size() == outputs[:,:1].size()

        outputs = outputs * target_on_off

        loss_box = self.criterion(outputs[:,1:5], box_targets)
        loss_landmark = self.criterion(outputs[:,5:], landmark_targets)
        loss_cls = self.cls_criterion(outputs[:,:1], cls_targets)
        return (loss_box + loss_landmark) * 5 + loss_cls * 0.5


    def metrics(self, boxes, landmarks, probabilities, labels, **_):
        iou = torch.mean(self._get_iou(boxes, labels[:,:4])).item()
        l2 = torch.mean(torch.sqrt(torch.sum(torch.pow(landmarks - labels[:,4:], 2), dim=1)))

        return {'score': iou, 'iou': iou, 'l2': l2}

    def annotate_to_images(self, images, labels, predicts, **_):
        assert images.dim() == 4
        assert labels.dim() == 2

        boxes = predicts['boxes']
        landmarks = predicts['landmarks']
        probabilities = predicts['probabilities']

        ious = self._get_iou(boxes, labels[:,:4])
        iou_1, indices_1 = torch.topk(ious, 2, largest=False)
        iou_2, indices_2 = torch.topk(ious, 2, largest=True)
        indices = torch.cat([indices_1, indices_2], dim=0)

        images = images.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        landmarks = landmarks.detach().cpu().numpy()
        probabilities = probabilities.detach().cpu().numpy()
        ious = ious.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        images = images[indices]
        labels = labels[indices]
        boxes = boxes[indices]
        landmarks = landmarks[indices]
        probabilities = probabilities[indices]
        ious = ious[indices]

        annotated_images = []
        for item in zip(images, labels, boxes, landmarks, probabilities, ious):
            image, label, box, landmark, probability, iou = item
            if image.shape[0] == 3:
                image = np.transpose(image, [1,2,0])

            H, W, _ = image.shape
            label = label * [W,H,W,H, W,H,W,H,W,H,W,H]
            label = label.astype(np.int32)

            box = box * [W,H,W,H]
            box = box.astype(np.int32)

            landmark = landmark * [W,H,W,H,W,H,W,H]
            landmark = landmark.astype(np.int32)

            label_box_x1 = int(label[0] - label[2] / 2)
            label_box_y1 = int(label[1] - label[3] / 2)
            label_box_x2 = int(label[0] + label[2] / 2)
            label_box_y2 = int(label[1] + label[3] / 2)

            predict_box_x1 = int(box[0] - box[2] / 2)
            predict_box_y1 = int(box[1] - box[3] / 2)
            predict_box_x2 = int(box[0] + box[2] / 2)
            predict_box_y2 = int(box[1] + box[3] / 2)

            label_landmarks = [(int(label[4]), int(label[5])),
                               (int(label[6]), int(label[7])),
                               (int(label[8]), int(label[9])),
                               (int(label[10]), int(label[11]))]

            predict_landmarks = [(int(landmark[0]), int(landmark[1])),
                                 (int(landmark[2]), int(landmark[3])),
                                 (int(landmark[4]), int(landmark[5])),
                                 (int(landmark[6]), int(landmark[7]))]

            image = from_norm_bgr(image, **self.preprocess_opt)
            image = image.astype('uint8')
            image = image.copy()
            cv2.rectangle(image,
                          (label_box_x1, label_box_y1), (label_box_x2, label_box_y2),
                          (0,0,255), thickness=3)
            cv2.rectangle(image,
                          (predict_box_x1, predict_box_y1), (predict_box_x2, predict_box_y2),
                          (255,0,0), thickness=3)
            for i, (x, y) in enumerate(label_landmarks):
                if i == 0:
                    cv2.circle(image, (x,y), 4, (0,255,0), thickness=-1)
                elif i == 2:
                    cv2.circle(image, (x,y), 4, (0,0,255), thickness=-1)
                else:
                    cv2.circle(image, (x,y), 4, (0,255,255), thickness=-1)
            for x, y in predict_landmarks:
                cv2.circle(image, (x,y), 4, (255,0,0), thickness=-1)

            image = image.copy()
            cv2.putText(image, '{:.04f}, {:.04f}'.format(iou, probability),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=cv2.LINE_AA)
            image = np.array(image)
            image = np.transpose(image, [2,0,1])
            annotated_images.append(image)

        return annotated_images

    def to_dataframe(self, key_list, boxes, probabilities):
        print(len(key_list), len(boxes), len(probabilities))
        print(key_list[0])
        print(boxes[0])
        print(probabilities[0])

        records = []
        for key, box, probability in zip(key_list, boxes, probabilities):
            x, y, w, h = box
            records.append((key, x, y, w, h, probability))

        df = pd.DataFrame.from_records(
            records, columns=['key', 'x', 'y', 'w', 'h', 'probability'])
        df = df.set_index('key')
        return df


    def _get_cls_targets(self, labels, anchors):
        # assert labels.size() == anchors.size()[:2], '{} vs {}'.format(labels.size(), anchors.size())
        B, _ = labels.size()

        ious = torch.zeros((labels.size(0), anchors.size(2), anchors.size(3), anchors.size(4))).cuda()
        for i in range(anchors.size(2)):
            for y in range(anchors.size(3)):
                for x in range(anchors.size(4)):
                    ious[:,i,y,x] = self._get_iou(labels, anchors[:,:,i,y,x])

        # ious = (B,9,9,9)
        ious_max, _ = torch.max(ious, dim=1, keepdim=False)
        # ious_max: (B,9,9)
        ious_max = ious_max.view(B, -1)
        _, ious_max_indices = torch.max(ious_max, dim=1,  keepdim=False)

        # targets: 1 if ious > 0.75 else 0
        # ious Bx1x9x9
        targets = torch.zeros_like(ious)
        on_off = torch.zeros_like(ious)

        thres_pos = 0.75
        thres_neg = 0.40

        targets[ious > thres_pos] = 1.0
        on_off[ious > thres_pos] = 1.0
        on_off[ious < thres_neg] = 1.0

        targets = targets.float()
        on_off = on_off.float()
        return (targets.view(labels.size(0),1,anchors.size(2),anchors.size(3),anchors.size(4)),
                on_off.view(labels.size(0),1,anchors.size(2),anchors.size(3),anchors.size(4)))

    def _boxes_to_targets(self, boxes, anchors):
        if len(boxes.size()) == 2:
            assert boxes.size(1) == anchors.size(0)
            boxes = boxes.view(boxes.size(0), boxes.size(1), 1, 1, 1)
        tx = (boxes[:,0,:,:,:] - anchors[0,:,:,:]) / anchors[2,:,:,:]
        ty = (boxes[:,1,:,:,:] - anchors[1,:,:,:]) / anchors[3,:,:,:]
        tw = torch.log(boxes[:,2,:,:,:] / anchors[2,:,:,:])
        th = torch.log(boxes[:,3,:,:,:] / anchors[3,:,:,:])
        return torch.stack([tx,ty,tw,th], dim=1)

    def _targets_to_boxes(self, targets, anchors):
        x = anchors[:,2] * targets[:,0] + anchors[:,0]
        y = anchors[:,3] * targets[:,1] + anchors[:,1]
        w = anchors[:,2] * torch.exp(targets[:,2])
        h = anchors[:,3] * torch.exp(targets[:,3])
        return torch.stack([x,y,w,h], dim=1)

    def _landmarks_to_targets(self, landmarks, anchors):
        if len(landmarks.size()) == 2:
            assert landmarks.size(1) == 8
            landmarks = landmarks.view(landmarks.size(0), landmarks.size(1), 1, 1, 1)

        points = [
                (landmarks[:,0,:,:,:] - anchors[0,:,:,:]) / anchors[2,:,:,:],
                (landmarks[:,1,:,:,:] - anchors[1,:,:,:]) / anchors[3,:,:,:],
                (landmarks[:,2,:,:,:] - anchors[0,:,:,:]) / anchors[2,:,:,:],
                (landmarks[:,3,:,:,:] - anchors[1,:,:,:]) / anchors[3,:,:,:],
                (landmarks[:,4,:,:,:] - anchors[0,:,:,:]) / anchors[2,:,:,:],
                (landmarks[:,5,:,:,:] - anchors[1,:,:,:]) / anchors[3,:,:,:],
                (landmarks[:,6,:,:,:] - anchors[0,:,:,:]) / anchors[2,:,:,:],
                (landmarks[:,7,:,:,:] - anchors[1,:,:,:]) / anchors[3,:,:,:]]
        return torch.stack(points, dim=1)

    def _targets_to_landmarks(self, targets, anchors):
        points = [
            anchors[:,2] * targets[:,0] + anchors[:,0],
            anchors[:,3] * targets[:,1] + anchors[:,1],
            anchors[:,2] * targets[:,2] + anchors[:,0],
            anchors[:,3] * targets[:,3] + anchors[:,1],
            anchors[:,2] * targets[:,4] + anchors[:,0],
            anchors[:,3] * targets[:,5] + anchors[:,1],
            anchors[:,2] * targets[:,6] + anchors[:,0],
            anchors[:,3] * targets[:,7] + anchors[:,1]]
        return torch.stack(points, dim=1)

    def _get_iou(self, coords_a, coords_b):
        def clamp(v):
            return torch.clamp(v, min=0.0, max=1.0)

        area_a = coords_a[:,2] * coords_b[:,3]
        area_b = coords_b[:,2] * coords_b[:,3]

        left_tops_x_a = clamp(coords_a[:,0] - coords_a[:,2] / 2)
        left_tops_y_a = clamp(coords_a[:,1] - coords_a[:,3] / 2)
        right_bottoms_x_a = clamp(coords_a[:,0] + coords_a[:,2] / 2)
        right_bottoms_y_a = clamp(coords_a[:,1] + coords_a[:,3] / 2)

        left_tops_x_b = clamp(coords_b[:,0] - coords_b[:,2] / 2)
        left_tops_y_b = clamp(coords_b[:,1] - coords_b[:,3] / 2)
        right_bottoms_x_b = clamp(coords_b[:,0] + coords_b[:,2] / 2)
        right_bottoms_y_b = clamp(coords_b[:,1] + coords_b[:,3] / 2)

        left_tops_x = torch.max(left_tops_x_a, left_tops_x_b)
        left_tops_y = torch.max(left_tops_y_a, left_tops_y_b)

        right_bottoms_x = torch.min(right_bottoms_x_a, right_bottoms_x_b)
        right_bottoms_y = torch.min(right_bottoms_y_a, right_bottoms_y_b)

        width = clamp(right_bottoms_x - left_tops_x)
        height = clamp(right_bottoms_y - left_tops_y)

        intersection = width * height

        return intersection / (area_a + area_b - intersection)


def main():
    print('main')
    from utils.config import _get_default_config
    config = _get_default_config()
    config.model.params.num_outputs = 4
    config.loss.name = 'mse_loss'
    box_regressor = LandmarkDetector(config)


if __name__ == '__main__':
    import cProfile

