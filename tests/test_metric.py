from unittest import TestCase
import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.metrics.spatial import mse, nmse, rmse, nrmse

class TestSpatialMetrics(TestCase):
    def test_distance_to_itself(self):
        for metric in [mse,
                       nmse,
                       rmse,
                       nrmse]:
            x = torch.tensor([1., 2., 3.]).unsqueeze(-1)
            meta = GenericWellMetadata(spatial_ndims=1)
            error = metric(x, x, meta)
            self.assertAlmostEqual(error.item(), 0.0)
