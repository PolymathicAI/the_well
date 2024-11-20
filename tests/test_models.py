from unittest import TestCase

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from the_well.benchmark.models import FNO
from the_well.data.datasets import WellMetadata


class TestFNO(TestCase):
    def setUp(self):
        super().setUp()
        self.n_spatial_dims = 2
        self.dim_in = 5
        self.dim_out = 5
        self.n_param_conditioning = 3
        self.modes1 = 16
        self.modes2 = 16
        self.metadata = WellMetadata(
            dataset_name="fake_name",
            n_spatial_dims=2,
            spatial_resolution=(32, 32),
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: ["a", "b", "c"], 1: ["d_x", "d_y"]},
            constant_field_names={},
            boundary_condition_types=["periodic"],
            n_files=1,
            n_trajectories_per_file=[10],
            n_steps_per_trajectory=[100],
        )

    def test_model(self):
        model = FNO(
            self.dim_in,
            self.dim_out,
            self.metadata,
            self.modes1,
            self.modes2,
        )
        self.assertTrue(isinstance(model, FNO))
        x = torch.rand(8, 5, 32, 32)
        # t = torch.rand(8)
        # param = torch.rand(8, 3)
        # input = {"time": t, "x": x, "parameters": param}
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_load_conf(self):
        FNO_CONFIG_FILE = "the_well/benchmark/configs/model/fno.yaml"
        config = OmegaConf.load(FNO_CONFIG_FILE)
        model = instantiate(
            config,
            dset_metadata=self.metadata.__dict__,
            dim_in=self.dim_in,
            dim_out=self.dim_out,
        )
        self.assertTrue(isinstance(model, FNO))
        x = torch.rand(8, 5, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, x.shape)
