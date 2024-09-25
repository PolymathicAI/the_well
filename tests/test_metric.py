from unittest import TestCase

import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.metrics.spatial import MSE, NMSE, NRMSE, RMSE


class TestMetrics(TestCase):
    def test_distance_to_itself(self):
        meta = GenericWellMetadata(
            dataset_name="test",
            n_spatial_dims=1,
            spatial_resolution=(128,),
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: ["test"]},
            constant_field_names={},
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
