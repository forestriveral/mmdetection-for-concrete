
import os
import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors


out_root = "../detection"


def dataset_evaluate(config, checkpoint, output, gpus=1,
                     eval_type=['bbox'],
                     proc_per_gpu=2,
                     show=False):

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
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    if out.endswith(('.pkl', '.pickle')):
                        out = os.path.join("/".join(out.split("/")[:-1]),
                                           output.split(".")[0])
                    result_file = out + '.json'
                    print('writing formatted results to {}'.format(result_file))
                    print("......")
                    results2json(dataset, outputs, result_file)
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    print("======>>")
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = out + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)
        return outputs


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

