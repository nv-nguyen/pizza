# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
import json
from tqdm import tqdm

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from lib.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config_run', help='test config file path')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--dump-json', action='store_true', help='dump json results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_list_video(dataset_path):
    list_video = [video_path for video_path in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, video_path)) and os.path.isdir(os.path.join(dataset_path, video_path, "rgb"))]
    list_video = sorted(list_video)
    print("List videos", list_video)
    return [os.path.join(dataset_path, video_path) for video_path in list_video]


def main():
    args = parse_args()

    assert args.eval or args.format_only or args.show or args.dump_json \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    # rename fields in data.test with path in config_run
    # read config_run_file
    config_run = Config(args.config_run).get_config()
    cfg.data.test.test_mode = True
    list_video = get_list_video(config_run.input_dir)
    for video_path in tqdm(list_video):
        video_name = os.path.basename(video_path)
        cfg.data.test.data_root = os.path.join(config_run.output_dir, video_name)
        cfg.data.test.img_dir = os.path.join(config_run.output_dir, video_name)
        cfg.data.test.proposal_path = os.path.join(config_run.output_dir, video_name, "uvo_proposals.json")
        cfg.data.test.gt_path = os.path.join(config_run.output_dir, video_name, "uvo_proposals.json")
        # init distributed env first, since logger depends on the dist info.

        rank, _ = get_dist_info()
        # allows not to create
        if args.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            results = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                False,
                args.opacity,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            results = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
                False,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)
                
        rank, _ = get_dist_info()
        if rank == 0:
            warnings.warn(
                'The behavior of output has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            output_pickle_format_path = os.path.join(config_run.output_dir, video_name, "uvo_segmentation.pkl")
            print(f'\nwriting results to {output_pickle_format_path}')
            mmcv.dump(results, output_pickle_format_path)
            if args.eval:
                eval_kwargs.update(metric=args.eval)
                metric = dataset.evaluate(results, **eval_kwargs)
                metric_dict = dict(config=args.config, metric=metric)
                if args.work_dir is not None and rank == 0:
                    mmcv.dump(metric_dict, json_file, indent=4)
                if tmpdir is not None and eval_on_format_results:
                    # remove tmp dir when cityscapes evaluation
                    shutil.rmtree(tmpdir)
            if args.dump_json:
                output_json_format_path = os.path.join(config_run.output_dir, video_name, "box2seg.json")
                print(f'\nwriting results to {output_json_format_path}')
                with open(output_json_format_path, 'w') as w:
                    json.dump(results, w)


if __name__ == '__main__':
    main()
