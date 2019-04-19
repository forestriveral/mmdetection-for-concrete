
import os
import cv2
import mmcv
import random
import colorsys
import numpy as np
from skimage import io
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import patches, lines, rcParams
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
import pycocotools.mask as maskUtils
from mmdet.core import tensor2imgs
from mmdet.core.evaluation import coco_classes, dataset_aliases


def opencv2skimage(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB)


def skimage2opencv(src):
    return cv2.cvtColor(src, cv2.COLOR_RGB2BGR)


def plot_image_debug(img):
    # img = io.imread('d:/dog.jpg')
    plt.figure()  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('Image')  # 图像题目
    plt.show()


def concrete_classes():
    return [
        'bughole'
        ]


def dataset_aliases_expanded():
    dataset_aliases["concrete"] = ["concrete"]
    return dataset_aliases


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases_expanded().items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def show_result(img, result, dataset='coco', score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            # if i == 0:
            #     print(img[mask].shape)
            #     plot_image_debug(mask)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=False,
        out_file=out_file)


def random_colors(N, bright=True, shuffle=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    color = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(color)
    return color


def show_mask_result(img, result, dataset='coco', score_thr=0.7, with_mask=True,
                     display=True, save=None):
    segm_result = None
    if with_mask:
        bbox_result, segm_result = result
    else:
        bbox_result = result
    if isinstance(dataset, str):  # add own data label to mmdet.core.class_name.py
        class_names = get_classes(dataset)
        # print(class_names)
    elif isinstance(dataset, list):
        class_names = dataset
    else:
        raise TypeError('dataset must be a valid dataset name or a list'
                        ' of class names, not {}'.format(type(dataset)))
    h, w, _ = img.shape
    img_show = img[:h, :w, :]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    if with_mask:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        color_masks = 255 * np.array(random_colors(4)).astype(np.uint8)
        for ix, i in enumerate(inds):
            # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_mask = color_masks[ix, :][None, :]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
    result_img = mmcv.imshow_det_bboxes(
        img_show,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        text_color='white',
        thickness=0,
        show=display,
        out_file=save)

    return result_img


def display_mask_result(img, result, dataset='coco', score_thr=0.5,
                        with_mask=True, with_bbox=True, display=True,
                        save=None):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    if isinstance(dataset, str):
        class_names = get_classes(dataset)
    elif isinstance(dataset, (list, tuple)) or dataset is None:
        class_names = dataset
    else:
        raise TypeError(
            'dataset must be a valid dataset name or a sequence'
            ' of class names, not {}'.format(type(dataset)))
    img = io.imread(img)
    h, w, _ = img.shape
    img_show = img[:h, :w, :]
    bboxes = np.vstack(bbox_result)
    # print(bboxes)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    colours = random_colors(len(inds))
    # draw segmentation masks
    if (segm_result is not None) and with_mask:
        colors = 255 * np.array(colours).astype(np.uint8)
        segms = mmcv.concat_list(segm_result)
        # print(inds)
        if inds.shape[0] == 0:
            print("*** No detected instances! ***")
        else:
            for ix, i in enumerate(inds):
                color_mask = colors[ix, :][None, :]
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)
    plot_det_mask(
        img_show,
        bboxes,
        labels,
        colours,
        with_bbox=with_bbox,
        class_names=class_names,
        score_thr=score_thr,
        show=display,
        out_file=save,)


def plot_det_mask(img,
                  bboxes,
                  labels,
                  colors,
                  with_bbox=True,
                  with_caption=True,
                  class_names=None,
                  score_thr=0.0,
                  figsize=(12, 12),
                  # bbox_color='green',
                  text_color='white',
                  font_scale=18,
                  show=True,
                  title='',
                  out_file=None):

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    N = bboxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    _, ax = plt.subplots(1, figsize=figsize)
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if not np.any(bbox):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        w = bbox_int[2] - bbox_int[0]
        h = bbox_int[3] - bbox_int[1]
        if with_bbox:
            p = patches.Rectangle(left_top, w, h, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=colors[i], facecolor='none')
            ax.add_patch(p)

        if with_caption:
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            ax.text(bbox_int[0], bbox_int[1] - 5, label_text,
                    color=text_color, size=font_scale, backgroundcolor="None")

    ax.imshow(img)
    if out_file is not None:
        plt.margins(0, 0)
        plt.savefig(out_file, dpi=100, bbox_inches='tight')
        print("== Save done! ==")
    if show:
        plt.show()


def initiate_detector(config, weights):
    cfg = mmcv.Config.fromfile(config)
    cfg.model.pretrained = None
    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, weights)

    return cfg, model


def multi_detect_plot(cfg, model, dataset="coco", path='images/*.jpg',
                      save='images/detected'):
    img_list = glob(path)
    for pic in img_list:
        f = mmcv.imread(pic)
        img_name = os.path.basename(pic)
        new_path = os.path.join(save, img_name)
        result = inference_detector(model, f, cfg, device='cuda:0')
        show_mask_result(pic, result, dataset=dataset, score_thr=0.6,
                         with_mask=True, display=False, save=new_path)


def single_detect_plot(cfg, model, image, dataset="coco", score_thr=0.5, save=None):
    img = mmcv.imread(image)
    result = inference_detector(model, img, cfg)
    display_mask_result(image, result, dataset=dataset, score_thr=score_thr,
                        with_mask=True, with_bbox=True, display=True, save=save)


def multi_detect(cfg, model, dataset="coco", path='images/*.jpg',
                 save='images/detected'):
    img_list = glob(path)
    for pic in img_list:
        f = mmcv.imread(pic)
        img_name = os.path.basename(pic)
        new_path = os.path.join(save, img_name)
        result = inference_detector(model, pic, cfg, device='cuda:0')
        show_mask_result(f, result, dataset=dataset, score_thr=0.6,
                         with_mask=True, display=False, save=new_path)


def single_detect(cfg, model, image, dataset="coco", save=None):
    img = mmcv.imread(image)
    result = inference_detector(model, img, cfg)
    show_mask_result(img, result, dataset=dataset, score_thr=0.6, with_mask=True, display=False,
                     save=save)


config_path = '../configs/cascade_mask_rcnn_r50_fpn_1x.py'
model_path = '../models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth'
image_path = './images/giraffe.jpg'


if __name__ == '__main__':
    c, m = initiate_detector(config_path, model_path)
    # single_detect(c, m, image_path, save="./images/detected/person.jpg")
    single_detect_plot(c, m, image_path, save="./images/detected/giraffe.jpg")
