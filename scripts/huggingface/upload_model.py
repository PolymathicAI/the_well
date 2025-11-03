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

CONFIG_DIR = (pathlib.Path(__file__) / "../../../the_well/benchmark/configs").resolve(
    strict=True
)
CONFIG_NAME = "model_upload"

MODEL_ARXIV_ID = {
    "FNO": "2310.00120",
    "TFNO": "2310.00120",
    "UNetClassic": "1505.04597",
    "UNetConvNext": "2201.03545",
}


def build_model_card_kwargs(
    model: torch.nn.Module, dataset_name: str
) -> dict[str, str]:
    """Create a dictionary of arguments to populate the model template card."""
    template_path = pathlib.Path(__file__).parent / "MODEL_CARD_TEMPLATE.md"
    model_name = model.__class__.__name__
    model_path = retrive_model_path(model)
    model_arxiv_id = MODEL_ARXIV_ID[model_name]
    model_card_path = (model_path / "README.md").resolve(strict=True)
    model_readme_content = model_card_path.read_text()
    return {
        "template_path": template_path,
        "dataset": dataset_name,
        "arxiv_id": model_arxiv_id,
        "model_name": model_name,
        "model_readme": model_readme_content,
    }


def retrive_model_path(model: torch.nn.Module) -> pathlib.Path:
    model_folder = inspect.getfile(model.__class__).split("/")[-2]
    model_path = (CONFIG_DIR.parent / "models" / model_folder).resolve()
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
        _convert_="all",
    )
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, PyTorchModelHubMixin)

    logger.info(f"Load checkpoints {cfg.model_ckpt}")
    checkpoint = torch.load(cfg.model_ckpt, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    dataset_name = str(cfg.data.well_dataset_name)
    model_name = model.__class__.__name__
    repo_id = f"polymathic-ai/{model_name}-{dataset_name}"
    model_card_kwargs = build_model_card_kwargs(model, dataset_name)
    # Upload model with HF formalism
    model.push_to_hub(repo_id=repo_id, model_card_kwargs=model_card_kwargs)


if __name__ == "__main__":
    main()
