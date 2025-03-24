import logging
import pathlib
import tempfile

import hydra
import torch
from huggingface_hub import HfApi, PyTorchModelHubMixin
from hydra.utils import instantiate
from omegaconf import DictConfig

from the_well.data import WellDataModule

logger = logging.getLogger("the_well")

CONFIG_DIR = (
    pathlib.Path(__file__) / ".." / ".." / ".." / "the_well" / "benchmark" / "configs"
).resolve()
CONFIG_NAME = "model_upload"


def link_model_card(model_path: pathlib.Path, target_file: pathlib.Path):
    """Link the README associated to the model to the current directory."""
    readme_file = model_path / "README.md"
    readme_file = readme_file.resolve()
    logger.info(f"Link {target_file=} to {readme_file=}")
    target_file.symlink_to(readme_file)


def retrieve_model_name(cfg_target: str) -> str:
    """Retrieve the name of the model folder from the hydra config target"""
    model_name = str(cfg_target.split(".")[-2])
    return model_name


def get_model_path(model_name: str) -> pathlib.Path:
    return (
        pathlib.Path(__file__)
        / ".."
        / ".."
        / ".."
        / "the_well"
        / "benchmark"
        / "models"
        / model_name
    ).resolve()


def upload_folder(folder: pathlib.Path, repo_id: str):
    api = HfApi()
    api.upload_large_folder(
        repo_id=repo_id, folder_path=folder, repo_type="dataset", private=False
    )


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

    model_name = retrieve_model_name(cfg.model._target_)
    model_path = get_model_path(model_name)
    dataset_name = str(cfg.data.well_dataset_name)
    repo_id = f"polymathic-ai/{model_name}-{dataset_name}"
    logger.info("Uploading model.")
    with tempfile.TemporaryDirectory() as tmp_dirname:
        tmp_dirname = pathlib.Path(tmp_dirname)
        # Copy model readme
        link_model_card(model_path, tmp_dirname / "README.md")
        # Save model locally with HF formalism
        model.save_pretrained(tmp_dirname)
        upload_folder(tmp_dirname, repo_id=repo_id)


if __name__ == "__main__":
    main()
