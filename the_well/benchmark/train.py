import logging
import os.path as osp

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from the_well.benchmark.data import WellDataModule
from the_well.benchmark.trainer import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def train(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(cfg.data)
    num_fields_by_tensor_order = datamodule.train_dataset.num_fields_by_tensor_order
    n_scalar_components = num_fields_by_tensor_order[0]
    n_vector_components = num_fields_by_tensor_order[1]
    n_tensor_components = num_fields_by_tensor_order[2]
    # Treat tensor components as vector
    n_vector_components += 2 * n_tensor_components
    n_param = datamodule.train_dataset.num_constants

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_input_scalar_components=n_scalar_components,
        n_input_vector_components=n_vector_components,
        n_output_scalar_components=n_scalar_components,
        n_output_vector_components=n_vector_components,
        n_param_conditioning=n_param,
    )

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters()
    )

    logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
        cfg.lr_scheduler, optimizer=optimizer
    )

    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    train()
