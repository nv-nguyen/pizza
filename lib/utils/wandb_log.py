import wandb
import os


def init_wandb_config(exp_name, config_run, save_dir, project_name="TemplateMatching"):
    os.environ["WANDB_API_KEY"] = "3e5a906da85ead4da7721bbd950f254c66eb8cd7"  # TODO delete when make the code public
    os.environ["WANDB_MODE"] = "dryrun"
    config = {"backbone": config_run.model.backbone,
              "descriptor_size": config_run.model.descriptor_size,
              "normalize_feature": config_run.model.normalize_feature,
              "loss": config_run.model.loss,
              "regression_loss": config_run.model.regression_loss,
              "image_size": config_run.dataset.image_size,
              "use_aug": config_run.dataset.use_augmentation,
              "cosine": config_run.model.use_cosine,
              "name": exp_name}
    wandb.init(project=project_name, config=config, dir=save_dir)
