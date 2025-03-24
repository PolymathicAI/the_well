import logging
import os.path as osp

import hydra
import torch
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from omegaconf import DictConfig

from the_well.data import WellDataModule

logger = logging.getLogger("the_well")

CONFIG_DIR = osp.join(osp.dirname(__file__), "../../the_well/benchmark/configs")
CONFIG_NAME = "model_upload"


@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(cfg.data)
    dset_metadata = datamodule.train_dataset.metadata
    n_input_fields = (
        cfg.data.n_steps_input * dset_metadata.n_fields
        + dset_metadata.n_constant_fields
    )
    n_output_fields = dset_metadata.n_fields

    logger.info(f"Instantiate model {cfg.model._target_}")
    model = instantiate(
        cfg.model,
        dset_metadata=dset_metadata,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
    )
    assert isinstance(model, torch.nn.Module) and isinstance(
        model, PyTorchModelHubMixin
    )

    logger.info(f"Load checkpoints {cfg.model_ckpt}")
    checkpoint = torch.load(cfg.model_ckpt, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    logger.info("Uploading model.")
    # TODO: Actually upload the model.


if __name__ == "__main__":
    main()
