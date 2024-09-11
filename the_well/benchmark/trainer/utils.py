"""Utility functions to retrieve and set environment variables related to SLURM.
Largely taken from the `idr_torch` module on Jean-Zay.

"""

import logging
import os
import os.path as osp
from typing import List, Tuple, Union
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_experiment_name(cfg: DictConfig) -> str:
    model_name = cfg.model._target_.split(".")[-1]
    # slurm_job_id = os.environ.get("SLURM_JOB_ID", "0") - Not using for now since I think it'll be easier to just use name alone
    return f"{cfg.data.well_dataset_name}-{cfg.name}-{model_name}-{cfg.optimizer.lr}"

def configure_experiment(cfg: DictConfig):
    """ Works through resume logic to figure out where to save the current experiment
    and where to look to resume or validate previous experiments.

    If the user provides overrides for the folder/checkpoint/config, use them.

    If folder isn't provided, construct default. If autoresume or validation_mode is enabled,
    look for the most recent run under that directory and take the config and weights from it.

    If checkpoint is provided, use it to override any weights obtained until now. If 
    any checkpoint is available either in the folder or checkpoint override, this
    is considered a resume run. 
    
    If it's in validation mode but no checkpoint is found, throw an error. 

    If config override is provided, use it (with the weights and current output folder). 
    Otherwise start search over hierarchy.
      - If checkpoint is being used, look to see if it has an associated config file
      - If no checkpoint but folder, look in folder
      - If not, just use the default config (whatever is currently set)
    """
    # Sort out default names and folders
    resume = False
    experiment_name = get_experiment_name(cfg)
    base_experiment_folder = osp.join(cfg.experiment_dir, experiment_name)
    experiment_folder = cfg.folder_override # Default is ""
    checkpoint_file = cfg.checkpoint_override # Default is ""
    config_file = cfg.config_override # Default is ""
    validation_mode = cfg.validation_mode
    # If using default naming, check for auto-resume, otherwise make a new folder with default name
    if len(experiment_folder) == 0:
        if osp.exists(base_experiment_folder):
            prev_runs = sorted(os.listdir(base_experiment_folder), key=lambda x: int(x))
        else:
            prev_runs = []
        if (validation_mode or cfg.auto_resume) and len(prev_runs) > 0:
            experiment_folder = osp.join(base_experiment_folder, prev_runs[-1])
        elif validation_mode:
            raise ValueError(f"Validation mode enabled but no previous runs found in {base_experiment_folder}.")
        else:   
            experiment_folder = osp.join(base_experiment_folder, str(len(prev_runs)))
    # Now check for default checkpoint options - if override used, ignore
    if osp.exists(experiment_folder) and len(checkpoint_file) == 0:
        last_chpt = osp.join(experiment_folder, "checkpoints", "recent.pt")
        # If there's a checkpoint file, consider this a resume. Otherwise, this is new run.
        if osp.isfile(last_chpt):
            checkpoint_file = last_chpt
    if len(checkpoint_file) > 0:
        logger.info(f"Checkpoint found, using checkpoint file {checkpoint_file}")
    if not osp.isfile(checkpoint_file) and len(checkpoint_file) > 0:
        raise ValueError(f"Checkpoint path provided but checkpoint file {checkpoint_file} not found.")
    # Now pick a config file to use - either current, override, or related to a different override
    if len(checkpoint_file) > 0 and len(config_file) == 0:
        # Check two levels - the parent folder of the checkpoint and the experiment folder
        checkpoint_path = osp.join(osp.dirname(checkpoint_file), osp.pardir, "extended_config.yaml")
        folder_path = osp.join(experiment_folder, "extended_config.yaml")
        if osp.isfile(checkpoint_path):
            logger.info(f"Config file exists relative to checkpoint override provided, \
                            using config file {checkpoint_path}")
        elif osp.isfile(folder_path):
            logger.warn(f"Config file not found in checkpoint override path. \
                        Found in experiment folder, using config file {folder_path}. \
                        This could lead to weight compatibility issues if the checkpoints do not align with \
                        the specified folder.")
        else:
            logger.warn(f"Checkpoint override provided, but config file not found in checkpoint override path \
                        or experiment folder. Using default configuration which may not be compatible with checkpoint.")
        # resume = True
    elif len(config_file) > 0:
        logger.log(f"Config override provided, using config file {config_file}")
    elif validation_mode:
        raise ValueError(f"Validation mode enabled but no checkpoint provided or found in {experiment_folder}.")
    if len(config_file) > 0:
        cfg = OmegaConf.load(config_file)
    cfg.trainer.checkpoint_path = checkpoint_file
        # cfg.trainer.resume = resume
    # Create experiment folder if it doesn't already exist
    os.makedirs(experiment_folder, exist_ok=True)
    return cfg, experiment_name, experiment_folder

def nodelist() -> Union[List[str], str]:
    compact_nodelist = os.environ["SLURM_STEP_NODELIST"]
    try:
        from hostlist import expand_hostlist
    except ImportError:
        return compact_nodelist
    else:
        return expand_hostlist(compact_nodelist)


def get_first_host(hostlist: str) -> str:
    """
    Get the first host from SLURM's nodelist.
    Example: Nodelist="Node[1-5],Node7" -> First node: "Node1"
    Args:
        hostlist(str): the compact nodelist as given by SLURM
    Returns:
        (str): the first node to host the master process
    """
    from re import findall, split, sub

    regex = r"\[([^[\]]*)\]"
    all_replacement: list[str] = findall(regex, hostlist)
    new_values = [split("-|,", element)[0] for element in all_replacement]
    for i in range(len(new_values)):
        hostlist = sub(regex, new_values[i], hostlist, count=1)
    return hostlist.split(",")[0]


def get_master_address() -> str:
    nodes = nodelist()
    if isinstance(nodes, list):
        return nodes[0]
    return get_first_host(nodes)


def get_master_port() -> str:
    job_id = int(os.environ["SLURM_JOB_ID"])
    return str(1000 + job_id % 2000)


def get_distrib_config() -> Tuple[bool, int, int, int]:
    distrib_env_variables = ["SLURM_PROCID", "SLURM_LOCALID", "SLURM_STEP_NUM_TASKS"]
    if not (set(distrib_env_variables) <= set(os.environ)):
        is_distributed = False
        rank = 1
        local_rank = 1
        world_size = 1
        logger.debug("Slurm configuration not detected in the environment")
    else:
        is_distributed = True
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_STEP_NUM_TASKS"])
        logger.debug(
            f"Slurm configuration detected, rank {rank}({local_rank})/{world_size}"
        )
    return is_distributed, world_size, rank, local_rank


def set_master_config():
    master_address = get_master_address()
    master_port = get_master_port()
    logger.debug(f"Set master address to {master_address} and port to {master_port}")
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = master_port
