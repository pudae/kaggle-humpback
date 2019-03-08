from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_model
from losses import get_loss
from transforms import from_norm_bgr


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=65, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class ArcNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_outputs = config.model.params.num_outputs
        feature_size = config.model.params.feature_size
        if 'channel_size' in config.model.params:
            channel_size = config.model.params.channel_size
        else:
            channel_size = 512

        self.model = get_model(config)
        if isinstance(self.model.last_linear, nn.Conv2d):
            in_features = self.model.last_linear.in_channels
        else:
            in_features = self.model.last_linear.in_features
        self.bn1 = nn.BatchNorm2d(in_features)
        self.dropout = nn.Dropout2d(config.model.params.drop_rate, inplace=True)
        self.fc1 = nn.Linear(in_features * feature_size * feature_size, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)

        s = config.model.params.s if 's' in config.model.params else 65
        m = config.model.params.m if 'm' in config.model.params else 0.5
        self.arc = ArcModule(channel_size, num_outputs, s=s, m=m)

        if config.model.params.pretrained:
            self.mean = self.model.mean
            self.std = self.model.std
            self.input_range = self.model.input_range
            self.input_space = self.model.input_space
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.input_range = [0, 1]
            self.input_space = 'RGB'

    def forward(self, images, labels=None):
        features = self.model.features(images)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            assert self.training
            return self.arc(features, labels)
        return features

    def logits(self, features, labels):
        return self.arc(features, labels)


class Identifier(object):
    def __init__(self, config):
        self.model = self._build_model(config)
        self.num_classes = config.model.params.num_outputs
        assert self.num_classes % 2 == 0

        self.preprocess_opt = {'mean': self.model.mean,
                               'std': self.model.std,
                               'input_range': self.model.input_range,
                               'input_space': self.model.input_space}

        self.criterion = get_loss(config)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

    def get_model(self):
        return self.model

    def get_preprocess_opt(self):
        return self.preprocess_opt

    def forward(self, images, labels=None, **_):
        return self.model(images, labels)

    def inference(self, images=None, outputs=None, labels=None, **_):
        if outputs is None:
            assert images is not None
            outputs = self.forward(images, labels)

        return {'features': outputs}

    def loss(self, outputs, labels, **_):
        if self.model.training:
            labels_flip = labels +  self.num_classes // 2
            labels_flip = torch.remainder(labels_flip, self.num_classes)
            if labels_flip.dim() == 1:
                labels_flip = labels_flip.unsqueeze(-1)
            onehot = torch.zeros(outputs.size()).cuda()
            onehot.scatter_(1, labels_flip, 1)
            onehot_invert = (onehot == 0).float()
            assert onehot_invert.size() == outputs.size()
            outputs = outputs * onehot_invert - onehot_invert
            return self.criterion(outputs, labels)
        return torch.FloatTensor([0])

    def metrics(self, features=None, labels=None, find_best_threshold=False, **_):
        assert features is not None
        assert labels is not None

        if self.model.training:
            _, top5 = torch.topk(features, 5)
            labels = labels.cpu().numpy()
            top5 = top5.cpu().numpy()
            map5 = self._mapk(labels, top5)
            return {'score': map5, 'map5': map5}

        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            features = np.array(features)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = np.array(labels)

        m = np.matmul(features, np.transpose(features))
        for i in range(features.shape[0]):
            m[i,i] = -1000.0

        def get_top5(scores, indices, labels, threshold, k=5):
            used = set()
            ret_labels = []
            ret_scores = []

            for index in indices:
                l = labels[index]
                s = scores[index]
                if l in used:
                    continue

                if 0 not in used and s < threshold:
                    used.add(0)
                    ret_labels.append(0)
                    ret_scores.append(-2.0)
                if l in used:
                    continue

                used.add(l)
                ret_labels.append(l)
                ret_scores.append(s)
                if len(ret_labels) >= k:
                    break
            return ret_labels[:5], ret_scores[:5]

        thresholds = np.arange(0.40, 0.3, -0.02)
        if find_best_threshold:
            thresholds = np.arange(1.0, 0.0, -0.01)

        map5_list = []
        predict_sorted = np.argsort(m, axis=-1)[:,::-1]
        for threshold in thresholds:
            top5s = []
            for l, scores, indices in zip(labels, m, predict_sorted):
                top5_labels, top5_scores = get_top5(scores, indices, labels, threshold)
                top5s.append(np.array(top5_labels))
            map5_list.append((threshold, self._mapk(labels, top5s)))
        map5_list = list(sorted(map5_list, key=lambda x: x[1], reverse=True))
        best_thres = map5_list[0][0]
        best_score = map5_list[0][1]

        if find_best_threshold:
            score_dict = {'map5_{:.02f}'.format(t):v
                          for t, v in sorted(map5_list, key=lambda x: x[0], reverse=True)}
        else:
            score_dict = {'score': best_score,
                          'map5': best_score,
                          'thres': best_thres}
        return score_dict

    def annotate_to_images(self, images, labels, predicts, **_):
        return []

    def to_dataframe(self, **_):
        pass

    def _build_model(self, config):
        num_outputs = config.model.params.num_outputs
        self.model = ArcNet(config)
        return self.model

    def _apk(self, actual, predicted, k=5):
        """
        Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """
        if len(predicted) > k:
            predicted = predicted[:k]
        actual = [actual]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        ret = score / min(len(actual), k)
        return ret


    def _mapk(self, actual, predicted, k=5):
        """
        Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """
        return np.mean([self._apk(a, p, k) for a, p in zip(actual, predicted)])
