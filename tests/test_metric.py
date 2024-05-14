from unittest import TestCase
import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
import the_well.benchmark.metrics as metrics

class TestSpatialMetrics(TestCase):
    def test_distance_to_itself(self):
        for metric in [metrics.mse,
                       metrics.nmse,
                       metrics.rmse,
                       metrics.nrmse]:
            x = torch.tensor([1., 2., 3.]).unsqueeze(-1)
            meta = GenericWellMetadata(spatial_ndims=1)
            error = metric(x, x, meta)
            self.assertAlmostEqual(error.item(), 0.0)
