import numpy as np
import mmcv


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


class CreateTrackers(object):
    def __init__(self, real_masks, scores):
        self.init_trackers(real_masks, scores)

    def init_trackers(self, real_masks, scores):
        self.trackers = []
        self.dead_trackers = []
        for i in range(len(real_masks[0])):
            tracker = Tracker(real_masks[0][i], scores[0][i], 0, 1)
            self.trackers.append(tracker)

    def add_tracker(self, real_mask, score, start_idx, end_idx):
        # mask shape: H*W
        tracker = Tracker(real_mask, score, start_idx, end_idx)
        self.trackers.append(tracker)


class Tracker(object):
    def __init__(self, real_mask, score, start_idx, end_idx):
        self.alive = True
        self.dead_count = 0
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.real_masks = [real_mask]
        self.score = score

    def kill(self):
        self.alive = False

    def update(self, real_mask, score, idx):
        self.real_masks.append(real_mask)
        self.score += score


def convert_mask_to_xyxy_box(mask, offset=20):
    h, w = mask.shape
    y, x = np.where(mask)
    x1, y1 = x.min(), y.min()
    x2, y2 = x.max(), y.max()
    x1 = max(0, x1 - offset)
    y1 = max(0, y1 - offset)
    x2 = min(w, x2 + offset)
    y2 = min(h, y2 + offset)
    crop_bbox = [int(x1), int(y1), int(x2), int(y2)]
    return crop_bbox
