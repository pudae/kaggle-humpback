from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict


def _get_default_config():
  c = edict()

  # dataset
  c.data = edict()
  c.data.name = 'BoxDataset'
  c.data.dir = 'data'
  c.data.params = edict()

  # task
  c.task = edict()
  c.task.name = 'BoxRegressor'
  c.task.params = edict()

  # model
  c.model = edict()
  c.model.name = 'resnet18'
  c.model.params = edict()

  # train
  c.train = edict()
  c.train.dir = './result/out'
  c.train.batch_size = 64
  c.train.num_epochs = 2000
  c.train.num_grad_acc = None
  c.train.log_step = 100
  c.train.save_checkpoint_epoch = 5

  # evaluation
  c.eval = edict()
  c.eval.batch_size = 64
  c.eval.cal_metric_once = False

  # optimizer
  c.optimizer = edict()
  c.optimizer.name = 'adam'
  c.optimizer.params = edict()

  # scheduler
  c.scheduler = edict()
  c.scheduler.name = 'none'
  c.scheduler.params = edict()

  # transforms
  c.transform = edict()
  c.transform.name = 'default_transform'
  c.transform.num_preprocessor = 4
  c.transform.params = edict()

  # losses
  c.loss = edict()
  c.loss.name = None
  c.loss.params = edict()

  return c


def _merge_config(src, dst):
  if not isinstance(src, edict):
    return

  for k, v in src.items():
    if isinstance(v, edict):
      _merge_config(src[k], dst[k])
    else:
      dst[k] = v


def load(config_path):
  with open(config_path, 'r') as fid:
    yaml_config = edict(yaml.load(fid))

  config = _get_default_config()
  _merge_config(yaml_config, config)

  return config
