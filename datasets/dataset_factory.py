from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torch.utils.data import DataLoader
import torch.utils.data.sampler

from .landmark import LandmarkDataset
from .identification import IdentificationDataset


def get_dataset(config, split, transform=None):
    f = globals().get(config.name)

    return f(config.dir,
             split=split,
             transform=transform,
             **config.params)


def get_dataloader(config, split, transform=None, **_):
    dataset = get_dataset(config.data, split, transform)

    is_train = 'train' == split
    batch_size = config.train.batch_size if is_train else config.eval.batch_size

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=config.transform.num_preprocessor,
                            pin_memory=False)
    return dataloader
