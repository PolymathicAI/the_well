import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from the_well.data.datasets import WellDataset
from the_well.data.miniwell import create_mini_well

WELL_BASE_PATH = os.environ.get("WELL_BASE_PATH")
CHECK_THEWELL_DATA_SCRIPT = os.path.join(
    os.path.dirname(__file__), "../scripts/check_thewell_data.py"
)


@unittest.skipIf(WELL_BASE_PATH is None, "WELL_BASE_PATH environment variable not set")
class TestMiniWell(unittest.TestCase):
    def test_create_mini_well(self):
        # Create temporary directory for mini dataset
        temp_dir = tempfile.mkdtemp()

        datasets_to_test = ["active_matter", "turbulent_radiative_layer_2D", "MHD_64"]

        try:
            # Load original datasets and create mini versions
            for dataset_name in datasets_to_test:
                original_dataset = WellDataset(
                    well_base_path=WELL_BASE_PATH,
                    well_dataset_name=dataset_name,
                    # Other options don't matter; we just use
                    # this to get the files and metadata
                )

                # Create mini dataset
                for split in ["train", "valid", "test"]:
                    mini_metadata = create_mini_well(
                        dataset=original_dataset,
                        output_base_path=temp_dir,
                        spatial_downsample_factor=4,
                        time_downsample_factor=2,
                        split=split,
                        max_trajectories=1,
                    )

                # Load mini dataset
                mini_dataset = WellDataset(
                    well_base_path=temp_dir,
                    well_dataset_name=dataset_name,
                    well_split_name="train",
                    n_steps_input=1,
                    n_steps_output=1,
                    use_normalization=False,
                    max_rollout_steps=2,
                )

                # Basic assertions
                self.assertEqual(len(mini_dataset.files_paths), 1)
                # Check for downsampling in both spatial dimensions
                expected_resolution = tuple(
                    dim // 4 for dim in original_dataset.metadata.spatial_resolution
                )
                self.assertEqual(mini_metadata.spatial_resolution, expected_resolution)
                self.assertLess(len(mini_dataset), len(original_dataset))

            # Run the data validation utility on all mini datasets at once
            result = subprocess.run(
                [
                    sys.executable,
                    CHECK_THEWELL_DATA_SCRIPT,
                    "--dir",
                    temp_dir,
                    "--nproc",
                    "1",
                    "--datasets",
                    *datasets_to_test,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
