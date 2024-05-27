import logging
import os.path as osp

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from the_well.benchmark.data import WellDataModule
from the_well.benchmark.trainer import Trainer
from the_well.benchmark.trainer.utils import get_distrib_config, set_master_config

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


def train(
    cfg: DictConfig,
    experiment_name: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 1,
    local_rank: int = 1,
):
    """Instantiate the different objects required for training and run the training loop."""

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(cfg.data, world_size=world_size, rank=rank)
    dset_metadata = datamodule.train_dataset.metadata
    n_input_fields = dset_metadata.n_fields + dset_metadata.n_constant_fields
    n_output_fields = dset_metadata.n_fields

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_spatial_dims=dset_metadata.n_spatial_dims,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
    )
    summary(model, depth=5)

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
        # Set LR scheduler configs based on experiment settings
        cfg.lr_scheduler.max_epochs = cfg.trainer.epochs
        cfg.lr_scheduler.warmup_epochs = int(cfg.trainer.epochs * .1)
        cfg.lr_scheduler.warmup_start_lr = (cfg.optimizer.lr 
            * .1)
        cfg.lr_scheduler.eta_min = cfg.optimizer.lr * .1
        # Instantiate LR scheduler
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
        experiment_name=experiment_name,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        is_distributed=is_distributed,
    )
    trainer.train()


def get_experiment_name(cfg: DictConfig) -> str:
    model_name = cfg.model._target_.split(".")[-1]
    return f"{cfg.data.well_dataset_name}-{cfg.name}-{model_name}-{cfg.optimizer.lr}"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Torch optimization settings
    torch.backends.cudnn.benchmark = (
        True  # If input size is fixed, this will usually the computation faster
    )
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    # Normal things
    experiment_name = get_experiment_name(cfg)
    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # Initiate wandb logging
    wandb.init(
        project="the_well",
        group=f"{cfg.data.well_dataset_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=experiment_name,
    )

    # Retrieve multiple processes context to setup DDP
    is_distributed, world_size, rank, local_rank = get_distrib_config()
    is_distributed = is_distributed and world_size > 1
    logger.info(f"Distributed training: {is_distributed}")
    if is_distributed:
        set_master_config()
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )

    train(cfg, experiment_name, is_distributed, world_size, rank, local_rank)
    wandb.finish()


if __name__ == "__main__":
    main()
