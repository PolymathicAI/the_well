import inspect
import logging
import pathlib

import hydra
import torch
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from omegaconf import DictConfig

from the_well.data import WellDataModule

logger = logging.getLogger("the_well")

CONFIG_DIR = (
    pathlib.Path(__file__) / ".." / ".." / ".." / "the_well" / "benchmark" / "configs"
).resolve()
CONFIG_NAME = "model_upload"


def retrive_model_path(model: torch.nn.Module) -> pathlib.Path:
    model_folder = inspect.getfile(model.__class__).split("/")[-2]
    model_path = (
        pathlib.Path(__file__)
        / ".."
        / ".."
        / ".."
        / "the_well"
        / "benchmark"
        / "models"
        / model_folder
    ).resolve()
    return model_path


@hydra.main(config_path=str(CONFIG_DIR), config_name=CONFIG_NAME)
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
        n_spatial_dims=dset_metadata.n_spatial_dims,
        spatial_resolution=dset_metadata.spatial_resolution,
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

    model_path = retrive_model_path(model)
    model_name = model.__class__.__name__
    dataset_name = str(cfg.data.well_dataset_name)
    repo_id = f"polymathic-ai/{model_name}-{dataset_name}"
    logger.info("Uploading model.")
    model_card_path = (model_path / "README.md").resolve()
    assert model_card_path.exists(), f"{model_card_path} does not exist."
    # Upload model with HF formalism
    model.push_to_hub(
        repo_id=repo_id,
        model_card_kwargs={"template_path": model_card_path},
    )


if __name__ == "__main__":
    main()
