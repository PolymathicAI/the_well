from unittest import TestCase

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.models import FNO


class TestFNO(TestCase):
    def setUp(self):
        super().setUp()
        self.n_spatial_dims = 2
        self.dim_in = 5
        self.dim_out = 5
        self.n_param_conditioning = 3
        self.modes1 = 16
        self.modes2 = 16
        self.metadata = GenericWellMetadata(
            dataset_name="fake_name",
            n_spatial_dims=2,
            resolution=(32, 32),
            field_names=["field1", "field2", "field3", "field4", "field5"],
            n_fields=5,
            n_constant_scalars=0,
            n_constant_fields=0,
            constant_names=[],
            boundary_condition_types=["periodic"],
            n_simulations=1,
            n_steps_per_simulation=[100],
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
