
import os
import sys
import torch
import mmcv
import json
import random
import numpy as np
import pandas as pd
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from mmdet import datasets
from mmdet.core import results2json
from mmdet.core.evaluation.coco_utils import fast_eval_recall
# from mmdet.core.evaluation.recall import eval_recalls
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
from concrete.utils import read_json_result, xywh2yxyx, pd_to_json

ROOT_DIR = os.path.abspath("../")
# Import concrete
sys.path.append(ROOT_DIR)
out_root = "../detection"


def extract_det_cls(result):
    id_list = []
    for i in range(len(result)):
        cls = result[i]["category_id"]
        if cls not in id_list:
            id_list.append(cls)
    return id_list


def modify_suffix(filename, suffix, remove=False):
    if not remove:
        if not filename.endswith(suffix):
            (fpath, temp) = os.path.split(filename)
            (fname, _) = os.path.splitext(temp)
            filename = os.path.join(fpath, fname) + suffix
    else:
        (fpath, temp) = os.path.split(filename)
        (fname, _) = os.path.splitext(temp)
        filename = os.path.join(fpath, fname)
    return filename


def detection_ouput(config, checkpoint, filename, gpus=1,
                    proc_per_gpu=2, show=False):
    path = checkpoint.split("/")[-2]
    if not os.path.exists(os.path.join(out_root, path)):
        os.makedirs(os.path.join(out_root, path))
    target = os.path.join(out_root, path, filename)
    if target is not None and not target.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, show)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            checkpoint,
            dataset,
            _data_func,
            range(gpus),
            workers_per_gpu=proc_per_gpu)

    model_name = cfg.work_dir.split("/")[-1]
    print("{} detection datasets done!".format(model_name))

    result_file = None
    if filename and outputs:
        print('\nWriting detected results to {}'.format(target))
        print("......")
        mmcv.dump(outputs, target)
        print("Writing done to pkl file: {}".format(target))
        if not isinstance(outputs[0], dict):
            result_file = modify_suffix(target, ".json")
            print('\nWriting formatted results to {}'.format(result_file))
            print("......")
            results2json(dataset, outputs, result_file)
            print("Writing done to json file: {}".format(result_file))
        else:
            print("\nWriting formatted results to multiple json files......")
            for i, name in enumerate(outputs[0]):
                print('===> {}th result of {} ......'.format(i, name))
                outputs_ = [out[name] for out in outputs]
                result_file = modify_suffix(target, None, remove=True)
                result_file = result_file + '_{}.json'.format(name)
                results2json(dataset, outputs_, result_file)
                print("Writing done to {}th files: {}".format(i, result_file))

    return dataset, result_file, outputs


def coco_evaluate(dataset, result_file, outputs=None,
                  eval_type=None, name=None, params=None):
    if outputs is None:
        assert result_file, "Results file must be provided!"
    assert os.path.exists(result_file), \
        "Results file doesn't exist in {}".format(result_file)

    data = None
    eval_types = eval_type or ["bbox", "segm"]
    print('\nStarting evaluate {} ......'.format(' and '.join(eval_types)))
    if eval_types == ['proposal_fast']:
        result_file = modify_suffix(result_file, ".pkl")
        data = coco_eval(result_file, eval_types, dataset.coco,
                         n=name, p=params)
    else:
        if outputs:
            if not isinstance(outputs[0], dict):
                data = coco_eval(result_file, eval_types, dataset.coco,
                                 n=name, p=params)
            else:
                print("\nMultiple evaluation......")
                for i, name in enumerate(outputs[0]):
                    print('{}th ==> Evaluating on {}'.format(i, name))
                    outputs_ = [out[name] for out in outputs]
                    result_file = \
                        result_file.split("/")[-1] + '_{}.json'.format(name)
                    results2json(dataset, outputs_, result_file)
                    data = coco_eval(result_file, eval_types, dataset.coco,
                                     n=name, p=params)
    return data


def detect_and_coco_eval(cfg, chp, filename, gpu_num=1, proc_per_gpu=2,
                         show=False, eval_type=None, name=None, params=None):
    # pklname = os.path.join(out_root, os.path.join(cfg.work_dirs.split("/")[-1], filename))
    outputs, target, dataset = detection_ouput(cfg, chp, filename, gpus=gpu_num,
                                               proc_per_gpu=proc_per_gpu, show=show)
    data = coco_evaluate(outputs, target, dataset, eval_type=eval_type,
                         name=name, params=params)

    return data, dataset, outputs, target


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def eval_package(frames, save=None):
    frame = pd.concat(frames)
    if save:
        frame.to_csv("./map_data.csv", index=0)
        print("=== Save done! ===")
    return frame


def coco_eval(result_file, result_types, coco, max_dets=(100, 300, 1000),
              no_sml=True, **kwargs):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
            ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_file, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    assert result_file.endswith('.json')
    coco_dets = coco.loadRes(result_file)

    heads = ['mAP', '50', '75', 's', 'm', 'l']
    data = {"name": [kwargs["n"]], "params": [kwargs["p"]]}

    img_ids = coco.getImgIds()
    for res_type in result_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        for i in range(len(heads)):
            key = '{}_{}'.format(res_type, heads[i])
            val = float('{:.3f}'.format(cocoEval.stats[i]))
            if no_sml:
                if heads[i] in ['s', 'm', 'l']:
                    continue
            if key not in data.keys():
                data[key] = []
            data[key].append(val)
        # print(data)
    data = pd.DataFrame(data)
    pd_to_json(data)
    return data


def convert_segm(segm):
    segm["counts"] = segm["counts"].encode()
    mask = maskUtils.decode(segm).astype(np.bool)
    assert mask.shape == tuple(segm['size']),\
        "Mismatching size of mask!"
    return mask


def build_voc_gts(dataset, image_ids, class_id, types):
    gt = [{} for _ in range(len(class_id))]
    gt_num = [0 for _ in range(len(class_id))]
    # progress bar set up
    print("[Building voc style GTS file......]")
    prog_bar = mmcv.ProgressBar(len(image_ids))
    # Load annotations gt information
    for j, image_id in enumerate(image_ids):
        _, image_meta, gt_class_id, gt_bbox, gt_mask = \
            dataset.load_gts(image_id, mode="square")
        for i, idx in enumerate(gt_class_id):
            if idx in class_id:
                ind = list(class_id).index(idx)
                if image_id not in gt[ind].keys():
                    gt[ind][image_id] = {'region': [], 'det': []}
                gt[ind][image_id]['region'].append(
                    gt_bbox[i, :] if types == "bbox" else gt_mask[:, :, i])
                gt[ind][image_id]['det'].append(False)
                gt_num[ind] += 1
            else:
                continue
        prog_bar.update()
    print("\nvoc groundtruth file building done!")
    return gt, gt_num


def build_voc_dets(result_file, class_id, types):
    assert os.path.exists(result_file), \
        "Results file doesn't exist in {}".format(result_file)

    det = [{"image_ids": [],
            "confidence": [],
            "region": []} for _ in range(len(class_id))]
    det_num = [0 for _ in range(len(class_id))]

    dets = read_json_result(result_file)
    cls_index = extract_det_cls(dets)
    print("\n[Building voc style DETS file......]")
    prog_bar = mmcv.ProgressBar(len(cls_index) * len(dets))
    for i, idx in enumerate(cls_index):
        if idx in class_id:
            ind = list(class_id).index(idx)
            for d in dets:
                det[ind]["image_ids"].append(d['image_id'] - 1)
                det[ind]["confidence"].append(float(d["score"]))
                det[ind]["region"].append(
                    np.array(xywh2yxyx(d["bbox"]), dtype=np.int32)
                    if types == "bbox" else convert_segm(d["segmentation"]))
                det_num[ind] += 1
                prog_bar.update()
        else:
            continue
    print("\nvoc detection file building done!")
    return det, det_num


def voc_ap_prepare(dataset, image_ids=None, class_names=None,
                   limit=None, types="bbox", fname=None, save=False,
                   verbose=False):
    # Pick COCO images from the dataset
    if image_ids:
        image_ids = image_ids
        limit = None
    else:
        image_ids = dataset.image_ids

    # Limit to a subset
    if limit and limit != "all":
        image_ids = random.sample(list(image_ids), limit)

    assert isinstance(image_ids, (list, int, np.ndarray)), \
        "Images list or selected image!"
    if isinstance(image_ids, int):
        print("Evaluate on Image ID {}".format(image_ids))
        image_ids = [image_ids]

    cls = dataset.class_names
    if class_names is not None:
        # class_id = [c["id"] for c in dataset.class_info if c["name"] in class_name]
        class_id = [cls.index(c) for c in class_names]
        assert len(class_id) == len(class_names), "No repeat class name!"
    else:
        class_id = dataset.class_ids[1:]

    gt, gt_num, det, det_num = None, None, None, None
    results_dir = os.path.join(out_root, dataset.config.work_dir.split("/")[-1])
    print("Ready to formatting on *{}* ...".format(types))
    # Formatting and save the groundtruth to file
    gt_fname = results_dir + "/{}_gts.pkl".format(types)
    if not os.path.exists(gt_fname):
        gt, gt_num = build_voc_gts(dataset, image_ids, class_id, types)
        if save:
            mmcv.dump(gt, gt_fname)
            print("Gt(gt numbers: {})file saved done in {}".format(gt_num, gt_fname))
    else:
        print("[Gt file already exists in {}] "
              "Please loading directly.....".format(gt_fname))
        gt = mmcv.load(gt_fname)
        print("[...Gt file loaded successfully!]")
    # Formatting and save the detection to file
    det_fname = results_dir + "/{}_dets.pkl".format(types)
    if not os.path.exists(det_fname):
        if fname:
            results_name = fname
        else:
            # default result file name
            results_name = "eval_result.json"
        results_file = os.path.join(results_dir, results_name)
        # print(results_dir)
        # print(results_name)
        det, det_num = build_voc_dets(results_file, class_id, types)
        if save:
            mmcv.dump(det, det_fname)
            print("Det(det numbers: {})file saved done in {}".format(det_num, det_fname))
    else:
        print("[Det file already exists in {}] "
              "Please loading directly.....".format(det_fname))
        det = mmcv.load(det_fname)
        print("[...Det file loaded successfully!]")

    # Check whether the class info is right
    if gt and det:
        assert len(det) == len(gt)
        if verbose:
            no_instance_class = [[], []]
            for i, (x, y) in enumerate(zip(det, gt)):
                if not x["region"]:
                    no_instance_class[0].append(cls[class_id[i]])
                if not y:
                    no_instance_class[1].append(cls[class_id[i]])
            if no_instance_class[0]:
                print("No instances of following class are detected:\n",
                      no_instance_class[0] if len(no_instance_class[0]) < 10 else len(no_instance_class[0]))
                if len(no_instance_class[0]) < 10:
                    print("Detected classes: \n{}".format(set(cls[1:]) - set(no_instance_class[0])))
            if no_instance_class[1]:
                print("No groundtruth of following class are found\n",
                      no_instance_class[1] if len(no_instance_class[1]) < 10 else len(no_instance_class[1]))
                if len(no_instance_class[1]) < 10:
                    print("Groundtruth classes: \n{}".format(set(cls[1:]) - set(no_instance_class[1])))
    else:
        print("NO GROUNDTRUTH AND DETECTION RESULTS!")

    return gt, gt_num, det, det_num, class_id


