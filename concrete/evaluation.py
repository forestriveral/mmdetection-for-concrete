
import os
import torch
import mmcv
import json
import numpy as np
import pandas as pd
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet import datasets
from mmdet.core import results2json, eval_map
from mmdet.core.evaluation.coco_utils import fast_eval_recall
# from mmdet.core.evaluation.recall import eval_recalls
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors


out_root = "../detection"


def concrete_classes():
    return ['bughole']


def read_result(file):
    with open(file, "r+") as f:
        r = json.load(f)
        return r


def coco_evaluate(config, checkpoint, output, gpus=1,
                  eval_type=['bbox', 'segm'],
                  proc_per_gpu=2,
                  show=False,
                  name=None,
                  params=None):

    path = checkpoint.split("/")[-2]
    if not os.path.exists(os.path.join(out_root, path)):
        os.makedirs(os.path.join(out_root, path))
    out = os.path.join(out_root, path, output)
    if out is not None and not out.endswith(('.pkl', '.pickle')):
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

    if out:
        print('\nwriting detected results to {}'.format(out))
        print("......")
        mmcv.dump(outputs, out)
        eval_types = eval_type
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = out
                data = coco_eval(result_file, eval_types, dataset.coco,
                                 n=name, p=params)
            else:
                if not isinstance(outputs[0], dict):
                    if out.endswith(('.pkl', '.pickle')):
                        out = os.path.join("/".join(out.split("/")[:-1]),
                                           output.split(".")[0])
                    result_file = out + '.json'
                    print('writing formatted results to {}'.format(result_file))
                    print("......")
                    results2json(dataset, outputs, result_file)
                    data = coco_eval(result_file, eval_types, dataset.coco,
                                     n=name, p=params)
                else:
                    print("======>>")
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = out + '_{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        data = coco_eval(result_file, eval_types, dataset.coco,
                                         n=name, p=params)
        return outputs, data


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
              **kwargs):
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
            if key not in data.keys():
                data[key] = []
            data[key].append(val)
        # print(data)
    data = pd.DataFrame(data)
    return data


def voc_eval(config, result_file, iou_thr=0.5):
    cfg = mmcv.Config.fromfile(config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)

    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(test_dataset)):
        ann = test_dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = gt_ignore
    if hasattr(test_dataset, 'year') and test_dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = test_dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)

