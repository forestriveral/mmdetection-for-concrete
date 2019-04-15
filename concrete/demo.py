

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result


config = '../configs/cm_rcnn_10164.py'
weights = '../models/cascade_mask_rcnn_x101_64x4d_fpn_1x_20181218-85953a91.pth'

cfg = mmcv.Config.fromfile(config)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, weights)

# test a single image
pic = 'images/person.jpg'
img = mmcv.imread(pic)
result = inference_detector(model, img, cfg)
show_result(img, result)

# test a list of images
# imgs = ['images/dog.jpg', 'images/dogball.jpg', 'images/eagle.jpg',
#         'images/giraffe.jpg', 'images/horses.jpg', 'images/kite.jpg']
# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     print(i + 1, imgs[i])
#     show_result(imgs[i], result)

