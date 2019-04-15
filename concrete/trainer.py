from __future__ import division

# import argparse
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch


def trainer(config,
            gpus_num,
            work_dir=None,
            resume_from=None,
            validate=True,
            seed=None,
            local_rank=0,
            launcher="pytorch"):

    cfg = Config.fromfile(config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if work_dir is not None:
        cfg.work_dir = work_dir
    if resume_from is not None:
        cfg.resume_from = resume_from
    cfg.gpus = gpus_num
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if seed is not None:
        logger.info('Set random seed to {}'.format(seed))
        set_random_seed(seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_dataset(cfg.data.train)
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        logger=logger)

