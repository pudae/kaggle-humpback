from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .landmark_transform import landmark_transform
from .align_transform import align_transform


def get_transform(config, split, params=None, **kwargs):
  f = globals().get(config.transform.name)

  if params is not None:
    return f(split, **config.transform.params, **params, **kwargs)
  else:
    return f(split, **config.transform.params, **kwargs)
