from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def to_norm_bgr(image, mean, std, input_space, input_range):
    assert isinstance(image, (np.ndarray,))
    if input_space == 'rgb':
        image = image[...,::-1]

    if max(input_range) != 255:
        image = image / 255 * max(input_range)

    image -= mean
    image /= std
    return image


def from_norm_bgr(image, mean, std, input_space, input_range):
    assert isinstance(image, (np.ndarray,))
    image *= std
    image += mean

    if max(input_range) != 255:
        image = image / max(input_range) * 255

    if input_space == 'rgb':
        image = image[...,::-1]

    return image
