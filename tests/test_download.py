import glob
import os
from unittest import TestCase

import pytest
from torch.utils.data import DataLoader

from the_well.data import WellDataset
from the_well.utils.download import well_download


@pytest.mark.order(1)
class TestDownload(TestCase):
    def test_active_matter(self):
        ACTIVE_MATTTER_DIR = os.path.abspath("datasets/active_matter")
        ACTIVE_MATTTER_DATA_DIR = os.path.join(ACTIVE_MATTTER_DIR, "data")

        self.assertTrue(os.path.isdir(ACTIVE_MATTTER_DIR))
        self.assertFalse(os.path.isdir(ACTIVE_MATTTER_DATA_DIR))

        well_download(
            base_path=".",
            dataset="active_matter",
            split="train",
            first_only=True,
        )

        self.assertTrue(os.path.isdir(ACTIVE_MATTTER_DATA_DIR))

        hdf5_files = glob.glob(f"{ACTIVE_MATTTER_DATA_DIR}/train/*.hdf5")

        self.assertTrue(len(hdf5_files) == 1)


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
