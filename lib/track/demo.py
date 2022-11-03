import os
import argparse
import numpy as np

import glob
from PIL import Image
from lib.track.visualizer import vis_bbox_and_mask, plot_tracking
from lib.track.tracker import TrackerSingleVideo
from lib.utils.config import Config
from lib.track.utils import convert_mask_to_xyxy_box
import pathlib


def get_list_video(dataset_path):
    list_video = [video_path for video_path in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, video_path)) and video_path.startswith("0")]
    # keep only videos containing images:
    list_removed_video, list_prefix = [], []
    for video_name in list_video:
        img_path = []
        for prefix in ["", "rgb", "color"]:
            png_paths = glob.glob(os.path.join(dataset_path, video_name, prefix, "*.png"))
            jpg_paths = glob.glob(os.path.join(dataset_path, video_name, prefix, "*.jpg"))
            img_path += png_paths + jpg_paths
            if len(png_paths + jpg_paths) > 0:
                prefix_used = prefix
        if len(img_path) == 0:
            list_removed_video.append(video_name)
        else:
            list_prefix.append(prefix_used)
    for video_name in np.unique(list_removed_video):
        list_video.remove(video_name)
    list_video_sorted = np.array(list_video)[np.argsort(list_video)]
    print(list_video_sorted)
    list_prefix_sorted = np.array(list_prefix)[np.argsort(list_video)]
    return [os.path.join(dataset_path, video_path) for video_path in list_video_sorted], list_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking inference code')
    parser.add_argument('--rgb_dir', help='RGB image dir')
    parser.add_argument('--config_global', help='config for input, output, weight dir',
                        default="./configs/config_path.json")
    parser.add_argument('--config_tracking', help='config for tracking', default="./configs/config_uvo_tracking.json")
    parser.add_argument('--config_segmentor', help='config for segmentation network',
                        default="./mmsegmentation/configs/tracking/biggest_model_clean_w_jitter.py")
    args = parser.parse_args()

    # read config_raft and config_run
    config_tracking = Config(args.config_tracking).get_config()
    config_global = Config(args.config_global).get_config()
    config_raft = config_tracking.raft
    args.checkpoint_segmentor = os.path.join(config_global.weight_dir, "seg_swin_l_uvo_finetuned.pth")
    config_raft.opflow_model_path = os.path.join(config_global.weight_dir, "raft-things.pth")
    # reset args
    list_video, list_prefix = get_list_video(config_global.input_dir)
    for idx_video, video_path in enumerate(list_video):
        video_name = os.path.basename(video_path)
        print(f"video: {video_name}, prefix: {list_prefix[idx_video]}", )
        save_dir = os.path.join(config_global.output_dir, video_name)
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(save_dir, "mask")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(save_dir, "tracking_vis")).mkdir(parents=True, exist_ok=True)
        # get init boxes with GT masks of first frame
        # BOP path: idx=1
        # Moped path: idx=0
        init_xyxy_boxes = []
        idx = 0
        init_mask_paths = glob.glob(os.path.join(video_path, "mask", f"{idx:06d}*"))
        init_mask_paths = sorted(init_mask_paths)
        for mask_path in init_mask_paths:
            mask = Image.open(mask_path)
            xyxy_box = convert_mask_to_xyxy_box(np.array(mask))
            init_xyxy_boxes.append(xyxy_box)
        # visualize first image
        try:
            first_img = Image.open(os.path.join(video_path, list_prefix[idx_video], f"{idx:06d}.png"))
        except:
            first_img = Image.open(os.path.join(video_path, list_prefix[idx_video], f"{idx:06d}.jpg"))
        draw_img = vis_bbox_and_mask(np.array(first_img), init_xyxy_boxes, colors="COLORSPACE")
        plot_tracking([draw_img, None, None], ["Initial boxes"],
                      save_path=os.path.join(save_dir, f"tracking_vis/{idx:06d}.jpg"))
        # start tracking and save results
        Tracker = TrackerSingleVideo(args=args,
                                     frame_rate=1,
                                     rgb_dir=os.path.join(video_path, list_prefix[idx_video]),
                                     save_dir=save_dir,
                                     config_raft=config_raft,
                                     config_tracker=config_tracking.tracking)
        Tracker.inference_tracker(init_boxes=init_xyxy_boxes, vis_output=True)
