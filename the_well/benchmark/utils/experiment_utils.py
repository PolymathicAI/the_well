import logging
import os
import os.path as osp
from typing import Tuple

from omegaconf import DictConfig, OmegaConf


def configure_paths(experiment_folder):
    """Configure the paths for the experiment with the given experiment folder."""
    # Make checkpoints directory as experiment_folder/checkpoints
    os.makedirs(osp.join(experiment_folder, "checkpoints"), exist_ok=True)
    checkpoint_folder = osp.join(experiment_folder, "checkpoints")
    # Store run/validation data as experiment_folder/artifacts
    os.makedirs(osp.join(experiment_folder, "artifacts"), exist_ok=True)
    artifact_folder = osp.join(experiment_folder, "artifacts")
    # Plot data directory as experiment_folder/viz
    os.makedirs(osp.join(experiment_folder, "viz"), exist_ok=True)
    viz_folder = osp.join(experiment_folder, "viz")
    return checkpoint_folder, artifact_folder, viz_folder


def get_experiment_name(cfg: DictConfig) -> str:
    model_name = cfg.model._target_.split(".")[-1]
    # slurm_job_id = os.environ.get("SLURM_JOB_ID", "0") - Not using for now since I think it'll be easier to just use name alone
    return f"{cfg.data.well_dataset_name}-{cfg.name}-{model_name}-{cfg.optimizer.lr}"


def configure_experiment(
    cfg: DictConfig, logger: logging.Logger
) -> Tuple[DictConfig, str, str]:
    """Works through resume logic to figure out where to save the current experiment
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



    Args:
        experiment_folder:
            Path to base folder used for experiment
        logger:
            Logger object to print messages to console
    """
    # Sort out default names and folders
    experiment_name = get_experiment_name(cfg)
    base_experiment_folder = osp.join(cfg.experiment_dir, experiment_name)
    experiment_folder = cfg.folder_override  # Default is ""
    checkpoint_file = cfg.checkpoint_override  # Default is ""
    config_file = cfg.config_override  # Default is ""
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
            raise ValueError(
                f"Validation mode enabled but no previous runs found in {base_experiment_folder}."
            )
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
        raise ValueError(
            f"Checkpoint path provided but checkpoint file {checkpoint_file} not found."
        )
    # Now pick a config file to use - either current, override, or related to a different override
    if len(checkpoint_file) > 0 and len(config_file) == 0:
        # Check two levels - the parent folder of the checkpoint and the experiment folder
        checkpoint_path = osp.join(
            osp.dirname(checkpoint_file), osp.pardir, "extended_config.yaml"
        )
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
            logger.warn(
                "Checkpoint override provided, but config file not found in checkpoint override path \
                        or experiment folder. Using default configuration which may not be compatible with checkpoint."
            )
        # resume = True
    elif len(config_file) > 0:
        logger.log(f"Config override provided, using config file {config_file}")
    elif validation_mode:
        raise ValueError(
            f"Validation mode enabled but no checkpoint provided or found in {experiment_folder}."
        )
    if len(config_file) > 0:
        cfg = OmegaConf.load(config_file)
    cfg.trainer.checkpoint_path = checkpoint_file
    # cfg.trainer.resume = resume
    # Create experiment folder if it doesn't already exist
    os.makedirs(experiment_folder, exist_ok=True)
    checkpoint_folder, artifact_folder, viz_folder = configure_paths(experiment_folder)
    return (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    )
