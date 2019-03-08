from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from datasets import get_dataloader
from transforms import get_transform
from tasks import get_task
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils
import utils.config
import utils.checkpoint


def evaluate_single_epoch(config, task, dataloader, epoch,
                          writer, postfix_dict):
    task.get_model().eval()
    cal_metric_once = config.eval.cal_metric_once

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        loss_list = []
        metric_list_dict = defaultdict(list)
        data_list_dict = defaultdict(list)
        for i, data in tbar:
            images = data['image'].cuda()
            labels = data['label'].cuda()

            outputs = task.forward(images=images)
            loss = task.loss(outputs, labels)
            loss_list.append(loss.item())

            predicts = task.inference(outputs=outputs)
            if cal_metric_once:
                for key, value in predicts.items():
                    data_list_dict[key].extend(value.cpu().numpy())
                for key, value in data.items():
                    if key == 'image':
                        continue
                    data_list_dict[key + 's'].extend(value)
            else:
                metric_dict = task.metrics(labels=labels, **predicts, **data)
                for key, value in metric_dict.items():
                    metric_list_dict[key].append(value)

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('dev')
            desc += ', {:.2f} epoch'.format(f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

            if i % config.train.log_step == 0:
                log_step = int(f_epoch * 10000)
                if writer is not None:
                    annotated_images = task.annotate_to_images(images, labels, predicts)
                    for idx, annotated_image in enumerate(annotated_images):
                        writer.add_image('dev/image_{}'.format(idx), annotated_image, log_step)

        log_dict = {}
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        if cal_metric_once:
            metric_dict = task.metrics(**data_list_dict)
            for key, value in metric_dict.items():
                log_dict[key] = value
        else:
            for key, values in metric_list_dict.items():
                log_dict[key] = sum(values) / len(values)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('dev/{}'.format(key), value, epoch)
            if key in ['loss', 'score']:
                postfix_dict['dev/{}'.format(key)] = value

        return log_dict['score']


def train_single_epoch(config, task, dataloader, optimizer,
                       epoch, writer, postfix_dict):
    task.get_model().train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = total_size // batch_size

    log_dict = {}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        labels = data['label'].cuda()
        outputs = task.forward(images=images, labels=labels)
        loss = task.loss(outputs, labels)
        log_dict['loss'] = loss.item()

        predicts = task.inference(outputs=outputs, labels=labels)
        metric_dict = task.metrics(labels=labels, **predicts)
        log_dict.update(metric_dict)

        loss.backward()

        warmup = True
        if warmup and epoch < 3:
            if (i+1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            if config.train.num_grad_acc is None:
                optimizer.step()
                optimizer.zero_grad()
            elif (i+1) % config.train.num_grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key in ['lr', 'loss', 'score']:
            value = log_dict[key]
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:.2f} epoch'.format(f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % config.train.log_step == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)
                annotated_images = task.annotate_to_images(images, labels, predicts)
                for idx, annotated_image in enumerate(annotated_images):
                    writer.add_image('train/image_{}'.format(idx), annotated_image, log_step)


def train(config, task, dataloaders, optimizer, scheduler, writer, start_epoch):
    scores = []
    best_score = 0.0
    best_score_mavg = 0.0
    postfix_dict = {}
    for epoch in range(start_epoch, config.train.num_epochs):
        # train phase
        train_single_epoch(config, task, dataloaders['train'],
                           optimizer, epoch, writer, postfix_dict)

        # val phase
        score = evaluate_single_epoch(config, task, dataloaders['dev'],
                                      epoch, writer, postfix_dict)
        scores.append(score)

        if config.scheduler.name == 'reduce_lr_on_plateau':
          scheduler.step(score)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
          scheduler.step()

        if epoch % config.train.save_checkpoint_epoch == 0:
            utils.checkpoint.save_checkpoint(config, task.get_model(), optimizer,
                                             epoch, 0, keep=10)

        scores = scores[-20:]
        score_mavg = sum(scores) / len(scores)
        writer.add_scalar('dev/score_mavg', score_mavg, epoch)
        if score > best_score:
            best_score = score
            utils.checkpoint.save_checkpoint(config, task.get_model(), optimizer,
                                             epoch, keep=10, name='best.score')
            utils.checkpoint.copy_last_n_checkpoints(config, 10, 'best.score.{:04d}.pth')
        if score_mavg > best_score_mavg:
            best_score_mavg = score_mavg
            utils.checkpoint.save_checkpoint(config, task.get_model(), optimizer,
                                             epoch, keep=10, name='best.score_mavg')
            utils.checkpoint.copy_last_n_checkpoints(config, 10, 'best.score_mavg.{:04d}.pth')
    return {'score': best_score, 'score_mavg': best_score_mavg}


def run(config):
    train_dir = config.train.dir

    task = get_task(config)
    optimizer = get_optimizer(config, task.get_model().parameters())

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(task.get_model(),
                                                            optimizer,
                                                            checkpoint)
    else:
        last_epoch, step = -1, -1

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    scheduler = get_scheduler(config, optimizer, last_epoch)

    preprocess_opt = task.get_preprocess_opt()
    dataloaders = {split:get_dataloader(config, split,
                                        get_transform(config, split,
                                                      **preprocess_opt))
                   for split in ['train', 'dev']}

    writer = SummaryWriter(config.train.dir)
    train(config, task, dataloaders, optimizer, scheduler,
          writer, last_epoch+1)


def parse_args():
    description = 'Train humpback whale identification'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('Train humpback whale identification')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config)
    run(config)
    print('success!')


if __name__ == '__main__':
    main()
