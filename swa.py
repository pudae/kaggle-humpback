from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pprint

import torch

from datasets import get_dataloader
from transforms import get_transform
from tasks import get_task
import utils.config
import utils.checkpoint
import utils.swa as swa


def get_best_epoch(config):
    checkpoint = utils.checkpoint.get_checkpoint(config, 'best.score.pth')
    return torch.load(checkpoint)['epoch']


def get_checkpoints(config, num_checkpoint=10, epoch_end=None):
    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')

    if epoch_end is not None:
        checkpoints = [os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(e))
                       for e in range(epoch_end+1)]
    else:
        # checkpoints = [os.path.join(checkpoint_dir, 'best.score.{:04d}.pth'.format(e))
        checkpoints = [os.path.join(checkpoint_dir, 'best.score_mavg.{:04d}.pth'.format(e))
                       for e in range(10)]

    checkpoints = [p for p in checkpoints if os.path.exists(p)]
    checkpoints = checkpoints[-num_checkpoint:]
    return checkpoints



def run(config, num_checkpoint, epoch_end, output_filename):
    task = get_task(config)
    preprocess_opt = task.get_preprocess_opt()
    dataloader = get_dataloader(
            config, 'train',
            get_transform(config, 'dev', **preprocess_opt))

    model = task.get_model()
    checkpoints = get_checkpoints(config, num_checkpoint, epoch_end)
    print('checkpoints:')
    print('\n'.join(checkpoints))

    utils.checkpoint.load_checkpoint(model, None, checkpoints[0])
    for i, checkpoint in enumerate(checkpoints[1:]):
        model2 = get_task(config).get_model()
        last_epoch, _ = utils.checkpoint.load_checkpoint(model2, None, checkpoint)
        swa.moving_average(model, model2, 1. / (i + 2))

    with torch.no_grad():
        swa.bn_update(dataloader, model)

    output_name = '{}.{}.{:03d}'.format(output_filename, num_checkpoint, last_epoch)
    print('save {}'.format(output_name))
    utils.checkpoint.save_checkpoint(config, model, None, 0, 0,
                                     name=output_name,
                                     weights_dict={'state_dict': model.state_dict()})


def parse_args():
    parser = argparse.ArgumentParser(description='hpa')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_filename',
                        help='output filename',
                        default='swa', type=str)
    parser.add_argument('--num_checkpoint', dest='num_checkpoint',
                        help='number of checkpoints for averaging',
                        default=10, type=int)
    parser.add_argument('--epoch_end', dest='epoch_end',
                        help='epoch end',
                        default=None, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    run(config, args.num_checkpoint, args.epoch_end, args.output_filename)

    print('success!')


if __name__ == '__main__':
    main()
