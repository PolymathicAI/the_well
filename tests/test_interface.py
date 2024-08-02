from unittest import TestCase

import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
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
        metadata = GenericWellMetadata(
            "test_dataset", 2, (256, 256), 0, 0, [], 3, ["a", "b", "c"], [], 1, [10]
        )
        interface = Interface(metadata)
        model = FakeModel(3, 3, 128)
        self.assertTrue(interface.check(model, 1, 1))
        model = FakeModel(2, 2, 128)
        self.assertFalse(interface.check(model, 1, 1))
