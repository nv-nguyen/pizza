import cv2
import argparse
import numpy as np

import torch

import mmcv
from mmcv.parallel import collate, scatter

from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

######################################################################

class LoadImageWBBox:
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])

        ori_shape = img.shape
        crop_bbox = results['pred_bbox']
        img = img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        results['ori_filename'] = results['filename']
        results['img_shape_before_crop'] = ori_shape
        results['crop_bbox'] = crop_bbox
        results['pred_category_id'] = 1
        results['pred_score'] = 1
        results['image_id'] = 1

        return results

def inference_segmentor_w_bbox(model, img, bbox):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImageWBBox()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img, pred_bbox=bbox)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


######################################################################
def mask2box(mask):
    h,w = mask.shape
    y,x = np.where(mask)
    x1, y1 = x.min(), y.min()
    x2, y2 = x.max(), y.max()
    offset = 20
    x1 = max(0, x1-offset)
    y1 = max(0, y1-offset)
    x2 = min(w, x2+offset)
    y2 = min(h, y2+offset)
    crop_bbox = [int(x1), int(y1), int(x2), int(y2)]
    return crop_bbox

def load_seg_model(config_path, checkpoint_path):
    model = init_segmentor(
            config_path,
            checkpoint_path,
            device=device)
    return model

def inf_segmodel_w_mask(img, mask, seg_model):
    bbox = mask2box(mask)
    cropped_mask = inference_segmentor_w_bbox(seg_model, img, bbox)[0]
    h = bbox[3]-bbox[1]
    w = bbox[2]-bbox[0]
    cropped_mask = cv2.resize(cropped_mask, (w,h), interpolation=cv2.INTER_NEAREST)
    new_mask = np.zeros(mask.shape)
    new_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cropped_mask
    return new_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segmentation inference code')
    parser.add_argument('--img', help='image path')
    parser.add_argument('--mask', help='mask path')
    parser.add_argument('--config', help='config path')
    parser.add_argument('--checkpoint', help='checkpoint path')
    args = parser.parse_args()

    seg_model = load_seg_model(args.config, args.checkpoint)
    img = cv2.imread(args.img)
    mask = cv2.imread(args.mask, 0)
    mask[mask > 0] = 1
    mask = inf_segmodel_w_mask(img, mask, seg_model)

    cv2.imwrite('demo.jpg', (mask*255).astype(np.uint8))

