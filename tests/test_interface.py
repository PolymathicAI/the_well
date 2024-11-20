from unittest import TestCase

import torch

from the_well.data.datasets import WellMetadata
from the_well.utils.interface import Interface


class FakeModel(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_features: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, output_features),
        )

    def forward(self, x):
        return self.layers(x)


class TestInterface(TestCase):
    def test_check(self):
        metadata = WellMetadata(
            dataset_name="test_dataset",
            n_spatial_dims=2,
            spatial_resolution=(256, 256),
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: ["a", "b", "c"]},
            constant_field_names={},
            boundary_condition_types=[],
            n_files=1,
            n_trajectories_per_file=[10],
            n_steps_per_trajectory=[100],
        )
        interface = Interface(metadata)
        model = FakeModel(3, 3, 128)
        self.assertTrue(interface.check_one_step(model, 1, 1))
        model = FakeModel(2, 2, 128)
        self.assertFalse(interface.check_one_step(model, 1, 1))
