import argparse
import os, sys
import torch
from lib.utils import gpu_utils, weights
from lib.utils.config import Config
from lib.model.network import PIZZA
from lib.dataloader.utils import init_dataloader
from lib.utils.optimizer import adjust_learning_rate
from lib.dataset.modelNet.dataloader import ModelNet
from lib.dataset.modelNet import training_utils, testing_utils
from lib.utils.logger import print_and_log_info

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
                                                                                  trainer_logger_name=None)
# initialize network
model = PIZZA(backbone=config_run.model.backbone,
              img_feature_dim=config_run.model.img_feature_dim,
              multi_frame=config_run.model.multi_frame)
model.apply(weights.KaiMingInit)
model.cuda()

# load pretrained weight if backbone are ResNet50
if config_run.model.backbone.startswith("resnet"):
    print("Loading pretrained backbone...")
    pretrained_weight_path = os.path.join(config_run.input_path, config_run.model.pretrained_weights_resnet18)
    if not os.path.exists(pretrained_weight_path):
        print("Downloading pretrained weight!")
        from lib.model.resnet import model_urls
        command = f"wget {model_urls['resnet18']} -P {os.path.dirname(pretrained_weight_path)}"
        os.system(command)
    weights.load_pretrained_backbone(prefix="backbone.",
                                        model=model,
                                        pth_path=pretrained_weight_path)
# load checkpoint if it's available
if args.checkpoint is not None:
    print("Loading checkpoint...")
    weights.load_checkpoint(model=model, pth_path=args.checkpoint)

config_run.dataset.save_path = os.path.join(config_run.output_path, config_run.dataset.save_path)
# initialize dataloader
train_loader = ModelNet(root_dir=os.path.join(config_run.input_path, config_run.dataset.shapeNet_path),
                        split="full_train_shapeNet", config_training=config_run.dataset,
                        logger=trainer_logger, is_master=is_master)
datasetLoader = {"train": train_loader}

test_cats = ['bathtub', 'bookshelf', 'guitar', 'range_hood', 'sofa', 'tv_stand', 'wardrobe']
for cat in test_cats:
    test_loader = ModelNet(root_dir=os.path.join(config_run.input_path, config_run.dataset.modelNet_path),
                           split=cat, config_training=config_run.dataset,
                           logger=trainer_logger, is_master=is_master)
    datasetLoader[cat] = test_loader

train_sampler, datasetLoader = init_dataloader(dict_dataloader=datasetLoader, use_distributed=args.use_distributed,
                                               batch_size=config_run.train.batch_size,
                                               num_workers=config_run.train.num_workers)

# initialize optimizer
optimizer = torch.optim.Adam(list(model.parameters()), lr=config_run.train.optimizer.lr, weight_decay=0.0005)

# training only rotation
for epoch in range(0, 200):
    if args.use_slurm and args.use_distributed:
        train_sampler.set_epoch(epoch)

    # update learning rate
    if epoch in config_run.train.scheduler.milestones:
        adjust_learning_rate(optimizer, config_run.train.optimizer.lr, config_run.train.scheduler.gamma)

    train_loss = training_utils.train(train_data=datasetLoader["train"],
                                      model=model, dataset_name='modelNet', predict_translation=True,
                                      optimizer=optimizer, warm_up_config=[1000, config_run.train.optimizer.lr],
                                      epoch=epoch, logger=trainer_logger, log_interval=config_run.log.log_interval,
                                      tb_logger=tb_logger, is_master=is_master)
    testing_scores = {}
    for cat in test_cats:
        test_loss = testing_utils.test_ModelNet(test_data=datasetLoader[cat], dataset_name='modelNet', epoch=epoch,
                                                model=model, logger=trainer_logger, cat=cat,
                                                data_path=os.path.join(config_run.input_path,
                                                                       config_run.dataset.modelNet_path),
                                                save_path=os.path.join(config_run.output_path,
                                                                       args.exp_name, cat + "_epoch_{}.json".format(epoch)),
                                                log_interval=config_run.log.log_interval,
                                                tb_logger=tb_logger, is_master=is_master, predict_translation=True)
        testing_scores[cat] = f"{test_loss:.2f}"

    if is_master:
        print_and_log_info(trainer_logger, "Epoch {} Summary: ".format(epoch))
        print_and_log_info(trainer_logger, f"\tTesting loss: {testing_scores}")
        print_and_log_info(trainer_logger, f"\tTraining loss: {train_loss}")

        save_weight_path = os.path.join(config_run.output_path, config_run.log.weights, args.exp_name,
                                        "epoch_{}.pth".format(epoch))
        print_and_log_info(trainer_logger, "Save weight at {} ".format(save_weight_path))
        weights.save_checkpoint(model, save_weight_path)
