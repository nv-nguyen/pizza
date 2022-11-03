from __future__ import print_function
from lib2to3.pytree import convert

import os
import sys

sys.path.append("./videowalk")
sys.path.append("./videowalk/core")
import cv2
from mmcv.parallel import collate, scatter

from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
import glob
import numpy as np
from tqdm import tqdm

import torch
from videowalk import utils_videowalk
from videowalk.core.raft import RAFT
from videowalk.core.utils import flow_viz
from lib.track.utils import LoadImageWBBox, convert_mask_to_xyxy_box
from lib.track.visualizer import vis_bbox_and_mask, plot_tracking, mask_flow

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class TrackerSingleVideo(object):
    def __init__(self, args, frame_rate, rgb_dir, save_dir, config_raft, config_tracker):
        self.save_dir = save_dir
        # define config of tracker
        self.config_tracker = config_tracker
        # load segmentor model
        self.seg_model = init_segmentor(
            args.config_segmentor,
            args.checkpoint_segmentor,
            device=device)
        # load optical flow model
        model = torch.nn.DataParallel(RAFT(config_raft))
        model.load_state_dict(torch.load(config_raft.opflow_model_path))
        model = model.module
        model.cuda()
        model.eval()
        self.flow_model = model
        self.args = utils_videowalk.arguments.test_args()
        self.args.imgSize = self.args.cropSize
        print('Context Length:', self.args.videoLen, 'Image Size:', self.args.imgSize)

        # load rgb images and generate optical flow
        self.images = self.load_images(rgb_dir, frame_rate)
        self.image_size = self.images[0].shape
        self.flows = self.load_flow()

    def load_images(self, rgb_dir, frame_rate):
        images_png = glob.glob(rgb_dir + '/*.jpg')
        images_jpg = glob.glob(rgb_dir + '/*.png')
        if len(images_png) > len(images_jpg):
            image_paths = images_png
        else:
            image_paths = images_jpg
        image_paths = np.array(sorted(image_paths))
        image_paths = image_paths[np.arange(0, len(image_paths), int(1 / frame_rate))]
        imgs = []
        for path in image_paths:
            imgs.append(cv2.imread(path))
        print(f' Frame rate: {frame_rate}, {len(imgs)} images Loaded')
        return imgs

    def generate_flow_forward(self, image_prev, image_nex, num_iters=20):
        image_prev = torch.tensor(image_prev).cuda().permute(2, 0, 1).unsqueeze(0)
        image_nex = torch.tensor(image_nex).cuda().permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            _, flow_forward = self.flow_model(image_prev, image_nex, iters=num_iters, test_mode=True)
        flow_forward = flow_forward.squeeze()
        flow_forward = flow_forward.permute(1, 2, 0).cpu().numpy()
        return flow_forward

    def warp_flow_forward(self, img, flow, binarize=True):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        if binarize:
            res = np.equal(res, 1).astype(np.uint8)
        return res

    def load_flow(self):
        assert self.images is not None
        flows = []
        h, w, _ = self.images[0].shape
        rsz_h, rsz_w = h // 32 * 32, w // 32 * 32
        for i in tqdm(range(len(self.images) - 1)):
            img_prev = cv2.resize(self.images[i], (rsz_w, rsz_h))
            img_nex = cv2.resize(self.images[i + 1], (rsz_w, rsz_h))
            flow = self.generate_flow_forward(img_prev, img_nex, 20)
            flow[:, :, 0] = flow[:, :, 0] / rsz_w
            flow[:, :, 1] = flow[:, :, 1] / rsz_h
            flow = cv2.resize(flow, (w, h))
            flow[:, :, 0] *= w
            flow[:, :, 1] *= h
            flows.append(flow)
        print(len(flows), 'Flows Generated')
        return flows

    def inference_segmentor_with_bbox(self, img, lastest_bbox):
        cfg = self.seg_model.cfg
        device = next(self.seg_model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImageWBBox()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = dict(img=img, pred_bbox=lastest_bbox)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.seg_model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        # forward the model
        with torch.no_grad():
            result = self.seg_model(return_loss=False, rescale=True, **data)
        return result

    def inference_segmentor_with_mask(self, img, lastest_box):
        cropped_mask = self.inference_segmentor_with_bbox(img, lastest_box)[0]
        h = lastest_box[3] - lastest_box[1]
        w = lastest_box[2] - lastest_box[0]
        # resize mask
        cropped_mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        new_mask = np.zeros((self.image_size[0], self.image_size[1]))
        new_mask[lastest_box[1]:lastest_box[3], lastest_box[0]:lastest_box[2]] = cropped_mask
        return new_mask

    def inference_tracker(self, init_boxes, vis_output=True, save_mask=False):
        lastest_boxes = init_boxes
        lastest_masks = []
        for idx_image in tqdm(range(1, len(self.images))):
            # segment objects with lastest boxes
            wrap_boxes = []
            current_masks = []
            current_img = self.images[idx_image]
            for idx_obj in range(len(lastest_boxes)):
                current_mask = self.inference_segmentor_with_mask(current_img, lastest_boxes[idx_obj])
                # save this prediction
                if len(np.unique(current_mask)) == 1:
                    print(f"WARNING! Failing on object {idx_obj} in image {idx_image}, use lastest mask")
                    current_mask = lastest_masks[idx_obj]
                current_masks.append(current_mask)
                new_mask = self.warp_flow_forward(current_mask, self.flows[idx_image - 1])
                new_box = convert_mask_to_xyxy_box(new_mask)
                wrap_boxes.append(new_box)
            if vis_output:
                save_vis_path = os.path.join(self.save_dir, "tracking_vis", f"{idx_image:06d}.jpg")
                img_wrap_box = vis_bbox_and_mask(current_img[..., ::-1], xyxy_boxes=lastest_boxes, colors="COLORSPACE")
                img_mask = vis_bbox_and_mask(current_img[..., ::-1], masks=current_masks, colors="COLORSPACE")
                # map flow to rgb image
                flow = np.copy(self.flows[idx_image - 1])
                flow = mask_flow(flow=flow, masks=current_masks)
                img_flow = flow_viz.flow_to_image(flow)
                img_flow = img_flow[:, :, [2, 1, 0]] / 255.0
                plot_tracking([img_flow, img_wrap_box, img_mask], ["RAFT flow", "Propagated boxes", "Refined masks"],
                              save_vis_path)
            lastest_boxes = wrap_boxes
            lastest_masks = current_masks
            if save_mask:
                # save masks
                for idx_obj in range(len(lastest_boxes)):
                    cv2.imwrite(os.path.join(self.save_dir, "mask", f"{idx_image:06d}_{idx_obj:06d}.png"),
                                (current_masks[idx_obj] * 255).astype(np.uint8))
        print(f"Tracking done, results are saved at {self.save_dir} !")
