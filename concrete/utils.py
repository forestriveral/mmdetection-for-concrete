
import os
import sys
import random
import copy
import json
import mmcv
import colorsys
import collections
import pandas as pd
import numpy as np
from scipy import interpolate
from interval import Interval
from sklearn.metrics import auc

ROOT_DIR = os.path.abspath("../")
# Import concrete
sys.path.append(ROOT_DIR)
out_root = "../detection"


def voc_ap_compute(dataset, class_id, gts, dets, gt_num, load=False,
                   class_names=None, types="bbox", threshold=0.5,
                   debug=False):

    if load:
        # the root path to save results files
        results_dir = os.path.join(
            out_root + dataset.config.work_dir.split("/")[-1])
        # check whether results files exsit or not
        gt_fname = results_dir + "/{}_gts.json".format(types)
        if os.path.exists(gt_fname):
            gts = mmcv.load(gt_fname)
            print("\nLoading gt file {}".format(gt_fname))
        else:
            print("Can't find gt file {}. Prepare gt file first!".format(gt_fname))

        det_fname = results_dir + "/{}_dets.json".format(types)
        if os.path.exists(det_fname):
            dets = mmcv.load(det_fname)
            print("\nLoading det file {}".format(det_fname))
        else:
            print("Can't find det file {}. Prepare det file first!".format(det_fname))

    print("Ready to evaluate on {} ...".format(types))
    # Copy gt file
    gt = copy.deepcopy(gts)
    # gt number in each class

    # Loop for every class need to compute ap
    evaluation = ["precisions", "recalls", "maps", "fprs", "tprs", "aucs"]
    targets = {}
    for eva in evaluation:
        targets[eva] = {}
    # precisions, recalls, maps, fprs, tprs, aucs = {}, {}, {}, {}, {}, {}
    for i, cls in enumerate(class_id):
        # If there is no instance of this class detected
        if (not dets[i]["region"]) or (not dets[i]["confidence"]):
            # detection[i] = {}
            continue
        else:
            # While class_name = None means all classes
            if class_names is None:
                class_names = dataset.class_names[1:]
            for k in targets.keys():
                targets[k][class_names[i]] = []

            # sort by confidence
            sorted_ind = np.argsort(-1 * np.array(dets[i]["confidence"]))
            region = np.array(dets[i]["region"])[sorted_ind, :] if types == "bbox" else \
                np.array(dets[i]["region"]).transpose((1, 2, 0))[..., sorted_ind]
            image_ids = [dets[i]["image_ids"][x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros((len(threshold), nd)) if isinstance(threshold, list) and len(threshold) > 1 else \
                np.zeros(nd)
            fp = np.zeros((len(threshold), nd)) if isinstance(threshold, list) and len(threshold) > 1 else \
                np.zeros(nd)
            for d in range(nd):
                if image_ids[d] not in gt[i].keys():
                    if isinstance(threshold, (list, np.ndarray)):
                        assert len(threshold) > 1
                        fp[:, d] = 1
                    else:
                        fp[d] = 1
                    # print('No gt instances but detected out:', image_ids[d])
                    continue
                r = gt[i][image_ids[d]]
                bb = region[d, :].astype(np.float32)[None, ...] if types == "bbox" \
                    else region[:, :, d][:, :, None]
                ovmax = -np.inf
                bbgt = np.array(r['region']).astype(np.float32) if types == "bbox" \
                    else np.array(r['region']).transpose((1, 2, 0))

                if bbgt.size > 0:
                    # compute overlaps
                    # overlaps = voc_overlaps(bbgt, gt)
                    overlaps = compute_overlaps(bbgt, bb).transpose((1, 0)) if types == "bbox" else \
                        compute_overlaps_masks(bbgt, bb).transpose((1, 0))
                    assert overlaps.shape == (1, bbgt.shape[0]) if types == "bbox" \
                        else (1, bbgt.shape[-1])
                    ovmax = np.max(np.squeeze(overlaps))
                    jmax = np.argmax(np.squeeze(overlaps))

                if debug:
                    if d == debug:
                        debug_tools(i, r)
                        debug_tools(i, bb.shape)
                        debug_tools(i, bbgt.shape)

                        debug_tools(i, overlaps)
                        debug_tools(i, ovmax)
                        debug_tools(i, jmax)

                if isinstance(threshold, (list, np.ndarray)) and len(threshold) > 1:
                    if len(r['det']) == 1 or not isinstance(r['det'][0], list):
                        r['det'] = [copy.deepcopy(r['det']) for _ in range(len(threshold))]
                        assert r['det'][1] == [False] * len(r['det'][1])
                    for ind, t in enumerate(threshold):
                        if ovmax > t:
                            if not r['det'][ind][jmax]:
                                tp[ind, d] = 1.
                                r['det'][ind][jmax] = 1
                            else:
                                fp[ind, d] = 1.
                        else:
                            fp[ind, d] = 1.
                else:
                    assert isinstance(threshold, (float, list))
                    if isinstance(threshold, list) and len(threshold) == 1:
                        threshold = threshold[0]
                    if ovmax > threshold:
                        if not r['det'][jmax]:
                            tp[d] = 1.
                            r['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

                if debug:
                    if d == debug:
                        debug_tools(i, np.count_nonzero(tp))
                        debug_tools(i, np.count_nonzero(fp))
                        debug_tools(i, r['det'])

            # Check results
            # assert int(fp[-1] + tp[-1]) == region.shape[0]
            # compute precision recall
            # print("fp", fp)
            # print("tp", tp)
            fp = np.cumsum(fp) if fp.ndim == 1 else np.cumsum(fp, axis=1)
            tp = np.cumsum(tp) if tp.ndim == 1 else np.cumsum(tp, axis=1)
            # compute true positive rate and false negative
            fpr = fp / fp[-1] if fp.ndim == 1 else fp / fp[:, -1].reshape(fp.shape[0], 1)
            tpr = tp / tp[-1] if tp.ndim == 1 else tp / tp[:, -1].reshape(tp.shape[0], 1)

            # Compute area under curve
            area = np.array([auc(fpr, tpr)]) if tp.ndim == 1 else \
                np.array([auc(fpr[ix, :], tpr[ix, :]) for ix in range(tp.shape[0])])
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            recall = tp / float(gt_num[i])
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            if recall.ndim == 1:
                ap, prec, rec = voc_ap(recall, precision, use_07_metric=False)
                ap, prec, rec = np.array([ap]), prec[None, :], rec[None, :]
            else:
                ap = np.zeros([fp.shape[0]])
                prec = np.zeros([precision.shape[0], precision.shape[1] + 2])
                rec = np.zeros([recall.shape[0], recall.shape[1] + 2])
                for ix in range(recall.shape[0]):
                    a, p, r = voc_ap(recall[ix, :], precision[ix, :], use_07_metric=False)
                    ap[ix] = a
                    prec[ix, :] = p
                    rec[ix, :] = r

            print("\nFP:", fp.shape)
            print("TP:", tp.shape)
            print("FPR:", fpr.shape)
            print("TPR:", tpr.shape)
            print("AREA:", area, area.shape)
            print("AP", ap, ap.shape)
            if debug:
                # Debug
                # print("GT: ", gt[0])
                print("\nFP: ", fp)
                print("TP: ", tp)
                print("Recall: ", len(recall))
                print("Precision: ", len(precision))
                print("AP: ", ap)
                print("Area under curve: ", area)
                # print("FPR: ", fpr)
                # print("TPR: ", tpr)

            # Pad with start and end values to simplify the math
            fpr = np.pad(fpr, ((0, 0), (1, 1)), "constant", constant_values=(0, 1)) if fpr.ndim > 1 \
                else np.pad(fpr, (1, 1), "constant", constant_values=(0, 1))[None, :]
            tpr = np.pad(tpr, ((0, 0), (1, 1)), "constant", constant_values=(0, 1)) if tpr.ndim > 1 \
                else np.pad(tpr, (1, 1), "constant", constant_values=(0, 1))[None, :]

        for (x, y) in zip(evaluation, [prec, rec, ap, fpr, tpr, area]):
            targets[x][class_names[i]] = y

    # Clean the class that has no instances detected
    # detection = clean_duplicates(detection)
    # Zipped dict storing recalls, precisions, maps, fprs, tprs, aucs
    return targets


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    if not list(boxes1):
        return np.zeros((boxes1.shape[0], boxes2.shape[-1]))
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    if np.sum((masks2 > .5).astype(np.uint8)) == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))
    if np.sum((masks1 > .5).astype(np.uint8)) == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))

    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def debug_tools(i, output):
    if i == 0:
        print("\nOutput:\n{}".format(output))
        if isinstance(output, np.ndarray):
            print("Shape:\n{}".format(output.shape))
        if isinstance(output, (list, dict)):
            print("Length:\n{}".format(len(output)))
        # print("Type:\n{}".format(type(output)))


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        mpre, mrec = [], []
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mpre, mrec


