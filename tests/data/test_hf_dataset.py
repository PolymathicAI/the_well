import pytest
from torch.utils.data import DataLoader

from the_well.data import DeltaWellDataset, WellDataset
from the_well.data.normalization import RMSNormalization, ZScoreNormalization


@pytest.mark.skip
@pytest.mark.parametrize(
    "dataset_name", ["active_matter", "turbulent_radiative_layer_2D"]
)
def test_dataset_is_available_on_hf(dataset_name):
    dataset = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",  # access from HF hub
        well_dataset_name=dataset_name,
        well_split_name="valid",
        use_normalization=False,
    )
    train_loader = DataLoader(dataset)
    batch = next(iter(train_loader))
    assert batch is not None


@pytest.mark.skip
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


@pytest.mark.skip
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
