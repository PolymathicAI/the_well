from unittest import TestCase

from the_well.benchmark.data.datasets import (
    GenericWellDataset,
    maximum_stride_for_initial_index,
    raw_steps_to_possible_sample_t0s,
)


class TestDataset(TestCase):
    def test_local_dataset(self):
        dataset = GenericWellDataset(
            well_base_path=".", well_dataset_name="active_matter"
        )
        self.assertTrue(len(dataset))

    def test_absolute_path_dataset(self):
        dataset = GenericWellDataset(path="2D/active_matter/data")
        self.assertTrue(len(dataset))

    def test_no_normalization(self):
        dataset = GenericWellDataset(
            well_base_path=".",
            well_dataset_name="active_matter",
            use_normalization=False,
        )
        self.assertTrue(len(dataset))

    def test_last_time_step(self):
        dataset = GenericWellDataset(
            well_base_path=".", well_dataset_name="active_matter"
        )
        n_time_steps = dataset.total_file_steps[0] - 1
        data = dataset[n_time_steps]
        data_keys = list(data.keys())
        self.assertIn("output_fields", data_keys)

        data = dataset[len(dataset) - 1]
        data_keys = list(data.keys())
        self.assertIn("output_fields", data_keys)

    def test_transform(self):
        dataset = GenericWellDataset(
            well_base_path=".",
            well_dataset_name="active_matter",
            transforms=[lambda x: x],
        )
        self.assertTrue(len(dataset))

    def test_exclude_keys(self):
        pass

    def test_include_keys(self):
        pass

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
