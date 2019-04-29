
import os
import sys
# import torch
# import mmcv
# import json
# import random
# import numpy as np
# import pandas as pd
from concrete import utils, datasets, evaluation, visualization


ROOT_DIR = os.path.abspath("../")
# Import concrete
sys.path.append(ROOT_DIR)


def integrated_coco_eval(name, cfg, coco_types):
    _, chp, _ = utils.autoload_model_info(name)

    data, _, _, _ = evaluation.detect_and_coco_eval(
        cfg, chp, eval_type=coco_types, name=name)


def integrated_voc_eval(name, cfg, voc_types, thresh):
    root, _, _ = utils.autoload_model_info(name)
    coco_dataset, concrete = datasets.load_dataset(cfg)
    if "bbox" in voc_types:
        gt, gt_num, det, det_num, class_id = \
            evaluation.voc_ap_prepare(coco_dataset, image_ids=None,
                                      class_names=None, limit=None,
                                      types="bbox", save=True)
        targets = utils.voc_ap_compute(
            coco_dataset, class_id, gt, det, load=False,
            class_names=None, types="bbox", threshold=thresh)
        visualization.plot_voc_curve(targets, "crack", thresh,
                                     save=True, name="bbox", root=root)
    if "segm" in voc_types:
        gt, gt_num, det, det_num, class_id = \
            evaluation.voc_ap_prepare(coco_dataset, image_ids=None,
                                      class_names=None, limit=None,
                                      types="segm", save=True)
        targets = utils.voc_ap_compute(
            coco_dataset, class_id, gt, det, load=False,
            class_names=None, types="segm", threshold=thresh)
        visualization.plot_voc_curve(targets, "crack", thresh,
                                     save=True, name="segm", root=root)


def integrated_loss_curve(name, save=True):
    root, _, log_path = utils.autoload_model_info(name)
    training = visualization.read_log(log_path)
    loss_path = os.path.join(root, "learning_curve.png")
    lr_path = os.path.join(root, "learing_rate.png")

    visualization.plot_training_curve(
        training, plot="loss", save=save, save_path=loss_path)
    visualization.plot_training_curve(
        training, plot="lr", save=save, save_path=lr_path)
