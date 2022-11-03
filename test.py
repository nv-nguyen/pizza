import argparse
import os, sys
import torch
import numpy as np
import json
from lib.utils import gpu_utils
from lib.utils.config import Config
from lib.dataloader.utils import init_dataloader
from lib.dataset.modelNet.dataloader import ModelNet
from lib.dataset.modelNet import testing_utils
from bop_toolkit_lib.metrics import evaluate_6d_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--use_slurm', action='store_true')
parser.add_argument('--use_distributed', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--config_path', type=str, default="./configs/config_modelNet.json")
parser.add_argument('--exp_name', type=str, default="modelNet")
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()

config_run = Config(config_file=args.config_path).get_config()
# pylint: disable=no-member
save_path = os.path.join(config_run.output_path, config_run.log.weights, args.exp_name)
tb_logdir = os.path.join(config_run.output_path, config_run.log.tensorboard, args.exp_name)
trainer_logger, tb_logger, is_master, world_size, local_rank = gpu_utils.init_gpu(use_slurm=args.use_slurm,
                                                                                  use_distributed=args.use_distributed,
                                                                                  local_rank=args.local_rank,
                                                                                  ngpu=args.ngpu,
                                                                                  gpus=args.gpus,
                                                                                  save_path=save_path,
                                                                                  trainer_dir="./tmp",
                                                                                  tb_logdir=tb_logdir,
                                                                                  trainer_logger_name="tester")
# load checkpoint
print("Loading checkpoint...")
model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())

config_run.dataset.save_path = os.path.join(config_run.output_path, config_run.dataset.save_path)
# initialize dataloader
test_cats = ['bathtub', 'bookshelf', 'guitar', 'range_hood', 'sofa', 'tv_stand', 'wardrobe']
datasetLoader = {}
for cat in test_cats:
    test_loader = ModelNet(root_dir=os.path.join(config_run.input_path, config_run.dataset.modelNet_path),
                           split=cat, config_training=config_run.dataset,
                           logger=trainer_logger, is_master=is_master)
    datasetLoader[cat] = test_loader

train_sampler, datasetLoader = init_dataloader(dict_dataloader=datasetLoader, use_distributed=args.use_distributed,
                                               batch_size=config_run.train.batch_size,
                                               num_workers=config_run.train.num_workers)

testing_scores = {}
for cat in test_cats:
    test_loss = testing_utils.test_ModelNet(test_data=datasetLoader[cat], dataset_name='modelNet', epoch=0,
                                            model=model, logger=trainer_logger, cat=cat,
                                            data_path=os.path.join(config_run.input_path,
                                                                    config_run.dataset.modelNet_path),
                                            save_path=os.path.join(config_run.output_path,
                                                                    args.exp_name, cat + "_epoch_{}.json".format(0)),
                                            log_interval=config_run.log.log_interval,
                                            tb_logger=tb_logger, is_master=is_master, predict_translation=True)
    testing_scores[cat] = f"{test_loss:.2f}"

final_scores = np.zeros((len(test_cats), 3))
for idx_cat, cat in enumerate(test_cats):
    save_path=os.path.join(config_run.output_path, args.exp_name, cat + "_epoch_{}.json".format(0))
    with open(save_path) as json_file:
        data = json.load(json_file)
    output = evaluate_6d_metrics(data, data_path=os.path.join(config_run.input_path, config_run.dataset.modelNet_path+"/models_obj/"), metric='all')
    final_scores[idx_cat, :] = [output['5deg_5cm'], output['add'], output['proj']]
    print(f"{cat}: {final_scores[idx_cat, :]}")
print(f"Mean: {np.mean(final_scores, axis=0)}")