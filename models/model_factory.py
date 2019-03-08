from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


def get_pretrainedmodels(model_name='resnet18', num_outputs=None, pretrained=True, **_):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)

    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs,
                                      kernel_size=1, bias=True)
    else:
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_outputs)

    return model


def get_model(config, **kwargs):
    print('model name:', config.model.name)
    f = lambda **kwargs: get_pretrainedmodels(config.model.name, **kwargs)

    if config.model.params is None:
        return f(**kwargs)
    else:
        return f(**config.model.params, **kwargs)


if __name__ == '__main__':
    print('main')
    model = get_pretrainedmodels(num_outputs=10)
    print(model)
