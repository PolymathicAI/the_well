from unittest import TestCase

import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.metrics.spatial import MSE, NMSE, NRMSE, RMSE


class TestMetrics(TestCase):
    def test_distance_to_itself(self):
        meta = GenericWellMetadata(
            dataset_name="test",
            n_spatial_dims=1,
            resolution=(128,),
            n_fields=1,
            field_names=["test"],
            n_constant_scalars=0,
            n_constant_fields=0,
            constant_names=[],
            boundary_condition_types=["periodic"],
            n_simulations=1,
            n_steps_per_simulation=[100],
        )
        for metric in [
            MSE(),
            RMSE(),
            NRMSE(),
            NMSE(),
            #    binned_spectral_mse
        ]:
            x = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
            error = metric(x, x, meta)
            self.assertAlmostEqual(error.nansum().item(), 0.0)
