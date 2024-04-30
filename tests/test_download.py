import glob
import os
from unittest import TestCase

from download_script import download_files


class TestDownload(TestCase):
    def test_active_matter(self):
        ACTIVE_MATTTER_DIR = os.path.abspath("2D/active_matter")
        ACTIVE_MATTTER_DATA_DIR = os.path.join(ACTIVE_MATTTER_DIR, "data")

        self.assertTrue(os.path.isdir(ACTIVE_MATTTER_DIR))
        self.assertFalse(os.path.isdir(ACTIVE_MATTTER_DATA_DIR))
        download_files(dataset_name="active_matter", sample_only=True)
        self.assertTrue(os.path.isdir(ACTIVE_MATTTER_DATA_DIR))
        hdf5_files = glob.glob(f"{ACTIVE_MATTTER_DATA_DIR}/*.hdf5")
        self.assertTrue(len(hdf5_files))
