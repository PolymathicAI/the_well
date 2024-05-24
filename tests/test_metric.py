from unittest import TestCase

import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.metrics.spatial import MSE, NRMSE, RMSE, NMSE
from the_well.benchmark.metrics.spectral import binned_spectral_mse


class TestMetrics(TestCase):
    def test_distance_to_itself(self):
        meta = GenericWellMetadata(
            spatial_ndims=1,
            resolution=(128,),
            n_fields=1,
            n_constant_fields=0,
            dataset_name="test",
            field_names=["test"],
        )
        for metric in [
            MSE(meta),
            RMSE(meta),
            NRMSE(meta),
            #    binned_spectral_mse
        ]:
            x = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
            error = metric(x, x, meta)
            self.assertAlmostEqual(error.nansum().item(), 0.0)
