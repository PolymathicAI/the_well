import os
import tempfile
import shutil
from unittest import TestCase, skipIf

import pytest
import h5py

from the_well.benchmark.data.datasets import GenericWellDataset, well_paths
from the_well.benchmark.data.miniwell import create_mini_well, load_mini_well

WELL_BASE_PATH = os.environ.get('WELL_BASE_PATH')

@skipIf(WELL_BASE_PATH is None, "WELL_BASE_PATH environment variable not set")
class TestMiniWell(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_name = "active_matter"  # Choose a small dataset for testing
        cls.original_dataset = GenericWellDataset(
            well_base_path=WELL_BASE_PATH,
            well_dataset_name=cls.dataset_name,
            well_split_name="train",
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=False,
            max_rollout_steps=2
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_create_and_load_mini_well(self):
        print("\nTesting create_and_load_mini_well")
        print(f"Original dataset path: {self.original_dataset.data_path}")
        print(f"Number of files in original dataset: {len(self.original_dataset.files_paths)}")
        
        mini_dataset_path = create_mini_well(
            dataset=self.original_dataset,
            output_base_path=self.temp_dir,
            spatial_downsample_factor=2,
            time_downsample_factor=2,
            max_samples=1
        )
        print(f"Mini dataset created at: {mini_dataset_path}")

        mini_dataset = load_mini_well(
            well_base_path=self.temp_dir,
            well_dataset_name=self.dataset_name,
            well_split_name="train",
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=False,
            max_rollout_steps=2
        )
        print(f"Mini dataset loaded from: {mini_dataset.data_path}")
        print(f"Number of files in mini dataset: {len(mini_dataset.files_paths)}")

        # Basic checks
        print("Checking dataset type")
        self.assertIsInstance(mini_dataset, GenericWellDataset)
        print("Checking dataset name")
        self.assertEqual(mini_dataset.metadata.dataset_name, self.dataset_name)
        
        # Check downsampling
        print("Checking spatial resolution")
        self.assertEqual(
            mini_dataset.metadata.spatial_resolution[0],
            self.original_dataset.metadata.spatial_resolution[0] // 2
        )
        
        # Check that we can access data
        print("Checking data access")
        sample = mini_dataset[0]
        self.assertIn('input_fields', sample)
        self.assertIn('output_fields', sample)

        # Check that the mini dataset has fewer samples
        print("Checking number of samples")
        self.assertLess(len(mini_dataset), len(self.original_dataset))

    def test_file_structure(self):
        print("\nTesting file_structure")
        mini_dataset_path = create_mini_well(
            dataset=self.original_dataset,
            output_base_path=self.temp_dir,
            spatial_downsample_factor=2,
            time_downsample_factor=2,
            max_samples=1
        )
        print(f"Mini dataset created at: {mini_dataset_path}")

        mini_file_path = os.path.join(mini_dataset_path, "data", "train")
        print(f"Checking files in: {mini_file_path}")
        mini_files = os.listdir(mini_file_path)
        print(f"Files found: {mini_files}")
        self.assertEqual(len(mini_files), 1)

        with h5py.File(os.path.join(mini_file_path, mini_files[0]), 'r') as f:
            original_file_path = self.original_dataset.files_paths[0]
            with h5py.File(original_file_path, 'r') as orig_f:
                for group_name in ['t0_fields', 't1_fields', 't2_fields']:
                    if group_name in f:
                        for field in f[group_name]:
                            orig_data = orig_f[group_name][field][:]
                            mini_data = f[group_name][field][:]
                            
                            # Check time dimension
                            if f[group_name][field].attrs.get('time_varying', False):
                                print(f"Checking time dimension for {field}")
                                self.assertEqual(mini_data.shape[0], orig_data.shape[0] // 2)
                            else:
                                print(f"Checking time dimension for {field}")
                                self.assertEqual(mini_data.shape[0], orig_data.shape[0])
                            
                            # Check spatial dimensions
                            n_spatial_dims = len(orig_data.shape) - 2  # Exclude time and channel dimensions
                            for i in range(n_spatial_dims):
                                print(f"Checking spatial dimension {i} for {field}")
                                self.assertEqual(mini_data.shape[i+1], orig_data.shape[i+1] // 2)
                            
                            # Check channel dimension (should be unchanged)
                            print(f"Checking channel dimension for {field}")
                            self.assertEqual(mini_data.shape[-1], orig_data.shape[-1])

                # Check if the 'time' dataset is properly downsampled
                if 'dimensions' in f and 'time' in f['dimensions']:
                    orig_time_steps = orig_f['dimensions']['time'].shape[0]
                    mini_time_steps = f['dimensions']['time'].shape[0]
                    print("Checking time dataset")
                    self.assertEqual(mini_time_steps, orig_time_steps // 2)

    def test_normalization_files(self):
        print("\nTesting normalization_files")
        mini_dataset_path = create_mini_well(
            dataset=self.original_dataset,
            output_base_path=self.temp_dir,
            max_samples=1
        )
        print(f"Mini dataset created at: {mini_dataset_path}")
        
        stats_path = os.path.join(mini_dataset_path, "stats")
        print(f"Checking for normalization files in: {stats_path}")
        for norm_file in ['means.pkl', 'stds.pkl']:
            file_path = os.path.join(stats_path, norm_file)
            exists = os.path.exists(file_path)
            print(f"{norm_file} exists: {exists}")
            self.assertTrue(exists)
