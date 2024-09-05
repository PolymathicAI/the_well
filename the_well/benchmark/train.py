import logging
import os
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
from the_well.benchmark.trainer.utils import set_master_config

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
    experiment_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 1,
    local_rank: int = 1,
    resume: bool = False,
    validation_mode: bool = False,
):
    """Instantiate the different objects required for training and run the training loop."""

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(
        cfg.data, world_size=world_size, rank=rank, data_workers=cfg.data_workers
    )
    dset_metadata = datamodule.train_dataset.metadata
    # TODO - currently just doing channel/time stacking for uniformity, but should
    # give the option of not stacking
    n_input_fields = (
        cfg.data.n_steps_input * dset_metadata.n_fields
        + dset_metadata.n_constant_fields
    )
    n_output_fields = dset_metadata.n_fields

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    print(cfg.model)
    model: torch.nn.Module = instantiate(
        cfg.model,
        dset_metadata=dset_metadata,
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
    if not validation_mode:
        optimizer: torch.optim.Optimizer = instantiate(
            cfg.optimizer, params=model.parameters()
        )
    else:
        optimizer = None

    if hasattr(cfg, "lr_scheduler") and not validation_mode:
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.epochs,
            warmup_start_lr=cfg.optimizer.lr * 0.1,
            eta_min=cfg.optimizer.lr * 0.1,
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None
    # Print final config, but also log it to experiment directory. 
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        experiment_folder=experiment_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        is_distributed=is_distributed,
    )
    if not validation_mode:
        # Save config to directory folder
        if not resume:
            with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
                OmegaConf.save(cfg, f)
        trainer.train()
    else:
        trainer.validate()


def get_experiment_name(cfg: DictConfig) -> str:
    model_name = cfg.model._target_.split(".")[-1]
    # slurm_job_id = os.environ.get("SLURM_JOB_ID", "0") - Not using for now since I think it'll be easier to just use name alone
    return f"{cfg.data.well_dataset_name}-{cfg.name}-{model_name}-{cfg.optimizer.lr}"

def configure_experiment(cfg: DictConfig):
    cfg.trainer.resume = False
    experiment_name = get_experiment_name(cfg)
    base_experiment_folder = osp.join(cfg.experiment_dir, experiment_name)
    # Want to allow for multiple runs to exist with same name or things will get very annoying
    if osp.exists(base_experiment_folder):
        prev_runs = sorted(os.listdir(base_experiment_folder), key=lambda x: int(x))
        # If we're auto-resuming, start from most recent run
        if cfg.auto_resume:
            experiment_folder = osp.join(base_experiment_folder, prev_runs[-1])
            # Load the previous yaml for exact correspondence
            cfg = OmegaConf.load(osp.join(experiment_folder, "extended_config.yaml"))
            cfg.trainer.resume = True
        # Otherwise, make a new folder
        else:
            experiment_folder = osp.join(base_experiment_folder, str(len(prev_runs)))
    # Otherwise, increment the internal folder by 1 and make a new base folder
    else:
        experiment_folder = osp.join(base_experiment_folder, "0")
    # Make the folder if it does not currently exist
    os.makedirs(experiment_folder, exist_ok=True)
    return cfg, experiment_name, experiment_folder

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Torch optimization settings
    torch.backends.cudnn.benchmark = (
        True  # If input size is fixed, this will usually the computation faster
    )
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    # Normal things
    cfg, experiment_name, experiment_folder = configure_experiment(cfg)
    
    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # Initiate wandb logging
    wandb.init(
        project=cfg.wandb_project_name,
        group=f"{cfg.data.well_dataset_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=experiment_name,
    )

    # Retrieve multiple processes context to setup DDP
    is_distributed, world_size, rank, local_rank = (
        False,
        1,
        0,
        0,
    )  # get_distrib_config()
    # is_distributed = is_distributed and world_size > 1

    logger.info(f"Distributed training: {is_distributed}")
    if is_distributed:
        set_master_config()
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    train(cfg, experiment_folder, is_distributed, world_size, rank, local_rank)
    wandb.finish()


if __name__ == "__main__":
    main()
