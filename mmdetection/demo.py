# Copyright (c) OpenMMLab. All rights reserved.
from tqdm import tqdm
import numpy as np
import json
import os
import glob
import pathlib
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

from lib.utils.config import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_model', help='Config file of model')
    parser.add_argument('--config_run', help='Config file containing input dir and output dir')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def get_list_video(dataset_path):
    list_video = [video_path for video_path in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, video_path)) and os.path.isdir(os.path.join(dataset_path, video_path, "rgb"))]
    list_video = sorted(list_video)
    print("List videos", list_video)
    return [os.path.join(dataset_path, video_path) for video_path in list_video]


def main(args):
    # read config_run_file
    config_run = Config(args.config_run).get_config()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config_model, args.checkpoint, device=args.device)

    list_video = get_list_video(config_run.input_dir)
    for video_path in tqdm(list_video):
        video_name = os.path.basename(video_path)
        rgb_path = os.path.join(video_path, 'rgb')
        images_path = sorted(glob.glob(rgb_path + '/*'))
        # test images
        print("Total number of images", len(images_path))

        out = []
        idx_image = 0
        for path in tqdm(images_path[:10]):
            result = inference_detector(model, path)
            # write the results
            for box in result:
                info = dict()
                score = float(box[4])
                box = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
                info['image_id'] = idx_image
                info['category_id'] = int(1)
                info['bbox'] = box
                info['score'] = score
                info['file_name'] = path
                out.append(info)
                idx_image += 1
        pathlib.Path(os.path.join(config_run.output_dir, video_name)).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(config_run.output_dir, video_name, "uvo_proposals.json")
        print("Saving proposals at {}".format(save_path))
        with open(save_path, 'w') as w:
            json.dump(out, w)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config_model, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
