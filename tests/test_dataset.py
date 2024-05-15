from unittest import TestCase

from the_well.benchmark.data.datasets import GenericWellDataset


class TestDataset(TestCase):
    def test_local_dataset(self):
        dataset = GenericWellDataset(
            well_base_path=".",
            well_dataset_name="active_matter",
            use_normalization=False,
        )
        self.assertTrue(len(dataset))

    def test_absolute_path_dataset(self):
        dataset = GenericWellDataset(
            path="2D/active_matter/data/train", use_normalization=False
        )
        self.assertTrue(len(dataset))
