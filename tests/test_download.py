import glob
import pathlib

import pytest
from torch.utils.data import DataLoader

from the_well.data import DeltaWellDataset, WellDataset
from the_well.data.normalization import RMSNormalization, ZScoreNormalization


def test_download_dataset(downloaded_dataset):
    dataset_dir: pathlib.Path = downloaded_dataset
    hdf5_files = glob.glob(f"{dataset_dir}/data/train/*.hdf5")
    assert len(hdf5_files) == 1


@pytest.mark.parametrize("dataset_cls", [DeltaWellDataset, WellDataset])
@pytest.mark.parametrize("normalization_type", [RMSNormalization, ZScoreNormalization])
def test_dataset_with_normalization(
    downloaded_dataset, dataset_cls, normalization_type
):
    dataset_dir: pathlib.Path = downloaded_dataset
    base_dataset_dir = dataset_dir.parent
    dataset_name = dataset_dir.name

    dataset = dataset_cls(
        well_base_path=base_dataset_dir,
        well_dataset_name=dataset_name,
        well_split_name="train",
        use_normalization=True,
        normalization_type=normalization_type,
    )
    assert len(dataset) > 0


@pytest.mark.parametrize(
    "dataset_name", ["active_matter", "turbulent_radiative_layer_2D"]
)
def test_dataset_is_available_on_hf(dataset_name):
    dataset = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",  # access from HF hub
        well_dataset_name=dataset_name,
        well_split_name="valid",
    )
    train_loader = DataLoader(dataset)
    batch = next(iter(train_loader))
    assert batch is not None


@pytest.mark.parametrize("normalization_type", [RMSNormalization, ZScoreNormalization])
def test_dataset_is_available_with_normalization(normalization_type):
    dataset = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",  # access from HF hub
        well_dataset_name="active_matter",
        well_split_name="valid",
        use_normalization=True,
        normalization_type=normalization_type,
    )
    assert len(dataset) > 0
    train_loader = DataLoader(dataset)
    batch = next(iter(train_loader))
    assert batch is not None


@pytest.mark.parametrize("normalization_type", [ZScoreNormalization, RMSNormalization])
def test_dataset_is_available_with_delta_normalization(normalization_type):
    dataset = DeltaWellDataset(
        well_base_path="hf://datasets/polymathic-ai/",  # access from HF hub
        well_dataset_name="active_matter",
        well_split_name="valid",
        use_normalization=True,
        normalization_type=normalization_type,
    )
    assert len(dataset) > 0
    train_loader = DataLoader(dataset)
    batch = next(iter(train_loader))
    assert batch is not None
