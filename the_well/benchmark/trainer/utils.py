import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)


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
        logger.debug(f"Slurm configuration detected, world size: {world_size}")
    return is_distributed, world_size, rank, local_rank
