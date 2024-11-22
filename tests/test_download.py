import glob
import os
from unittest import TestCase

import pytest

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
