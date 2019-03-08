from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .landmark_detector import LandmarkDetector
from .identifier import Identifier


def get_task(config):
    f = globals().get(config.task.name)
    return f(config)
