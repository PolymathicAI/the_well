import logging
import os.path as osp

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from the_well.benchmark.data import WellDataModule
from the_well.benchmark.trainer import Trainer
from the_well.benchmark.trainer.utils import get_distrib_config, set_master_config

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)

CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


def train(cfg: DictConfig, world_size: int = 1, rank: int = 1, local_rank: int = 1):
    is_distributed = world_size > 1

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(cfg.data, world_size=world_size, rank=rank)
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
    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters()
    )

    if hasattr(cfg, "lr_scheduler"):
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
            cfg.lr_scheduler, optimizer=optimizer
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        is_distributed=is_distributed,
    )
    trainer.train()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    is_distributed, world_size, rank, local_rank = get_distrib_config()
    logger.info(f"Distributed training: {is_distributed}")
    if is_distributed:
        set_master_config()
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    train(cfg, world_size, rank, local_rank)


if __name__ == "__main__":
    main()
