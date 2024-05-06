from unittest import TestCase

from the_well.benchmark.data.datasets import GenericWellDataset


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
