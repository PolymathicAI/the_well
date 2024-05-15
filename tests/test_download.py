import glob
import os
from unittest import TestCase

import pytest

from the_well.utils.download_script import download_files

JSON_DATASET_FILES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../the_well/utils/data_registry.json")
)


@pytest.mark.order(1)
class TestDownload(TestCase):
    def test_active_matter(self):
        ACTIVE_MATTTER_DIR = os.path.abspath("2D/active_matter")
        ACTIVE_MATTTER_DATA_DIR = os.path.join(ACTIVE_MATTTER_DIR, "data")

        self.assertTrue(os.path.isdir(ACTIVE_MATTTER_DIR))
        self.assertFalse(os.path.isdir(ACTIVE_MATTTER_DATA_DIR))
        download_files(
            json_file=JSON_DATASET_FILES,
            dataset_name="active_matter",
            output_path=".",
            sample_only=True,
        )
        self.assertTrue(os.path.isdir(ACTIVE_MATTTER_DATA_DIR))
        hdf5_files = glob.glob(f"{ACTIVE_MATTTER_DATA_DIR}/train/*.hdf5")
        self.assertTrue(len(hdf5_files))
