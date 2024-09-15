import os
import tempfile
import shutil
from unittest import TestCase, skipIf

import h5py

from the_well.benchmark.data.datasets import GenericWellDataset
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
        mini_dataset_path, mini_metadata = create_mini_well(
            dataset=self.original_dataset,
            output_base_path=self.temp_dir,
            spatial_downsample_factor=2,
            time_downsample_factor=2,
            max_samples=1
        )

        mini_dataset = load_mini_well(
            well_base_path=self.temp_dir,
            well_dataset_name=self.dataset_name,
            mini_metadata=mini_metadata,
            well_split_name="train",
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=False,
            max_rollout_steps=2
        )

        self.assertIsInstance(mini_dataset, GenericWellDataset)
        self.assertEqual(mini_dataset.metadata.dataset_name, self.dataset_name)
        
        self.assertEqual(
            mini_dataset.metadata.spatial_resolution[0],
            self.original_dataset.metadata.spatial_resolution[0] // 2
        )
        
        sample = mini_dataset[0]
        self.assertIn('input_fields', sample)
        self.assertIn('output_fields', sample)

        self.assertLess(len(mini_dataset), len(self.original_dataset))

    def test_file_structure(self):
        mini_dataset_path, _ = create_mini_well(
            dataset=self.original_dataset,
            output_base_path=self.temp_dir,
            spatial_downsample_factor=2,
            time_downsample_factor=2,
            max_samples=1
        )

        mini_file_path = os.path.join(mini_dataset_path, "data", "train")
        mini_files = os.listdir(mini_file_path)
        self.assertEqual(len(mini_files), 1)

        with h5py.File(os.path.join(mini_file_path, mini_files[0]), 'r') as f:
            original_file_path = self.original_dataset.files_paths[0]
            with h5py.File(original_file_path, 'r') as orig_f:
                for group_name in ['t0_fields', 't1_fields', 't2_fields']:
                    if group_name in f:
                        for field in f[group_name]:
                            orig_data = orig_f[group_name][field][:]
                            mini_data = f[group_name][field][:]
                            
                            if f[group_name][field].attrs.get('time_varying', False):
                                self.assertEqual(mini_data.shape[0], orig_data.shape[0] // 2)
                            else:
                                self.assertEqual(mini_data.shape[0], orig_data.shape[0])
                            
                            n_spatial_dims = len(orig_data.shape) - 2  # Exclude time and channel dimensions
                            for i in range(n_spatial_dims):
                                self.assertEqual(mini_data.shape[i+1], orig_data.shape[i+1] // 2)
                            
                            self.assertEqual(mini_data.shape[-1], orig_data.shape[-1])

                if 'dimensions' in f and 'time' in f['dimensions']:
                    orig_time_steps = orig_f['dimensions']['time'].shape[0]
                    mini_time_steps = f['dimensions']['time'].shape[0]
                    self.assertEqual(mini_time_steps, orig_time_steps // 2)

    def test_normalization_files(self):
        mini_dataset_path, mini_metadata = create_mini_well(
            dataset=self.original_dataset,
            output_base_path=self.temp_dir,
            max_samples=1
        )
        
        stats_path = os.path.join(mini_dataset_path, "stats")
        for norm_file in ['means.pkl', 'stds.pkl']:
            file_path = os.path.join(stats_path, norm_file)
            self.assertTrue(os.path.exists(file_path))
