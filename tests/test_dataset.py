import os.path
import random
import tempfile
from unittest import TestCase

import torch

from the_well.data.augmentation import (
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)
from the_well.data.datasets import (
    WellDataset,
    WellMetadata,
    maximum_stride_for_initial_index,
    raw_steps_to_possible_sample_t0s,
)
from the_well.utils.dummy_data import write_dummy_data


class TestMetadata(TestCase):
    def test_metadata(self):
        metadata = WellMetadata(
            dataset_name="test",
            n_spatial_dims=2,
            spatial_resolution=(256, 256),
            scalar_names=["whatever"],
            constant_scalar_names=["alpha", "beta"],
            field_names={0: ["energy"], 1: ["v_x", "v_y"]},
            constant_field_names={2: ["t_xx", "t_xy", "t_yx", "t_yy"]},
            boundary_condition_types=["periodic"],
            n_files=1,
            n_trajectories_per_file=[16],
            n_steps_per_trajectory=[100],
        )

        self.assertEqual(metadata.n_scalars, 1)
        self.assertEqual(metadata.n_constant_scalars, 2)
        self.assertEqual(metadata.n_fields, 3)
        self.assertEqual(metadata.n_constant_fields, 4)

        shapes = metadata.sample_shapes

        self.assertSequenceEqual(shapes["input_scalars"], [1])
        self.assertSequenceEqual(shapes["output_scalars"], [1])
        self.assertSequenceEqual(shapes["constant_scalars"], [2])

        self.assertSequenceEqual(shapes["input_fields"], [256, 256, 3])
        self.assertSequenceEqual(shapes["output_fields"], [256, 256, 3])
        self.assertSequenceEqual(shapes["constant_fields"], [256, 256, 4])


class TestDataset(TestCase):
    def test_local_dataset(self):
        dataset = WellDataset(
            well_base_path="datasets",
            well_dataset_name="active_matter",
            use_normalization=False,
        )
        self.assertTrue(len(dataset))

    def test_absolute_path_dataset(self):
        dataset = WellDataset(
            path="datasets/active_matter/data/train", use_normalization=False
        )
        self.assertTrue(len(dataset))

    def test_last_time_step(self):
        dataset = WellDataset(
            well_base_path="datasets",
            well_dataset_name="active_matter",
            use_normalization=False,
        )
        n_time_steps = dataset.n_steps_per_trajectory[0] - 1
        data = dataset[n_time_steps]
        self.assertIn("input_fields", data)
        self.assertIn("output_fields", data)

        data = dataset[len(dataset) - 1]
        self.assertIn("input_fields", data)
        self.assertIn("output_fields", data)

    def test_augmentation(self):
        dataset = WellDataset(
            well_base_path="datasets",
            well_dataset_name="active_matter",
            use_normalization=False,
            transform=Compose(
                RandomAxisFlip(),
                RandomAxisPermute(),
                RandomAxisRoll(),
            ),
        )

        i = random.randrange(len(dataset))
        data = dataset[i]

        self.assertIn("input_fields", data)
        self.assertIn("output_fields", data)

    def test_adjust_available_steps(self):
        # ex1: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 1
        #  Possible samples are: [0, 1], [1, 2], [2, 3], [3, 4], return 4
        self.assertEqual(raw_steps_to_possible_sample_t0s(5, 1, 1, 1), 4)
        # ex2: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 2
        #  Possible samples are: [0, 2], [1, 3], [2, 4], return 3
        self.assertEqual(raw_steps_to_possible_sample_t0s(5, 1, 1, 2), 3)
        # ex3: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 3
        #  Possible samples are: [0, 3], [1, 4], return 2
        self.assertEqual(raw_steps_to_possible_sample_t0s(5, 1, 1, 3), 2)
        # ex4: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 1, dt_stride = 2
        #  Possible samples are: [0, 2, 4], return 1
        self.assertEqual(raw_steps_to_possible_sample_t0s(5, 2, 1, 2), 1)
        # ex5: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 2, dt_stride = 2
        #   No possible samples, return 0
        self.assertEqual(raw_steps_to_possible_sample_t0s(5, 2, 2, 2), 0)
        # ex6: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 10, dt_stride = 2
        #  No possible samples, return 0
        self.assertEqual(raw_steps_to_possible_sample_t0s(5, 2, 10, 2), 0)

    def test_maximum_stride_for_initial_index(self):
        # ex1: time_idx=0, total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1
        #   Maximum stride is 4 - [0, 4]
        self.assertEqual(maximum_stride_for_initial_index(0, 5, 1, 1), 4)
        # ex2: time_idx=2, total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1
        #   Maximum stride is 2, [2, 4]
        self.assertEqual(maximum_stride_for_initial_index(2, 5, 1, 1), 2)
        # ex3: time_idx=1, total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1
        #   Maximum stride is 3, [1, 4]
        self.assertEqual(maximum_stride_for_initial_index(1, 5, 1, 1), 3)
        # ex4: time_idx=1, total_steps_in_trajectory = 5, n_steps_input = 5, n_steps_output = 1
        #   Maximum stride is 0
        self.assertEqual(maximum_stride_for_initial_index(5, 5, 1, 1), 0)
        # ex5: time_idx=5, total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 2
        #   Maximum stride is 0
        self.assertEqual(maximum_stride_for_initial_index(5, 5, 1, 1), 0)

    def test_dummy_dataset(self):
        with tempfile.TemporaryDirectory() as dir_name:
            filename = os.path.join(dir_name, "dummy_well_data.hdf5")
            write_dummy_data(filename)
            dataset = WellDataset(
                path=dir_name, use_normalization=False, return_grid=True
            )
            # Dummy dataset should contain 2 trajectories of 9 valid samples each
            self.assertEqual(len(dataset), 2 * 9)

            data = dataset[0]

            for key in (
                "input_fields",
                "output_fields",
                "constant_fields",
                "input_scalars",
                "output_scalars",
                "constant_scalars",
            ):
                self.assertIn(key, data)

            for key, shape in dataset.metadata.sample_shapes.items():
                if "input" in key or "output" in key:
                    self.assertSequenceEqual(data[key].shape[1:], shape)
                else:
                    self.assertSequenceEqual(data[key].shape, shape)

            data_next = dataset[1]

            for key in (
                "input_fields",
                "output_fields",
                "constant_fields",
                "input_scalars",
                "output_scalars",
                "constant_scalars",
            ):
                if "constant" in key:
                    self.assertTrue(torch.equal(data[key], data_next[key]))
                else:
                    self.assertFalse(torch.equal(data[key], data_next[key]))
