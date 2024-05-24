from unittest import TestCase

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from the_well.benchmark.models import FNO


class TestFNO(TestCase):
    def setUp(self):
        super().setUp()
        self.n_spatial_dim = 2
        self.n_input_scalar = 1
        self.n_input_vector = 2
        self.n_param_conditioning = 3
        self.modes1 = 16
        self.modes2 = 16

    def test_model(self):
        model = FNO(
            self.n_spatial_dim,
            self.n_input_scalar,
            self.n_input_vector,
            self.n_input_scalar,
            self.n_input_vector,
            self.modes1,
            self.modes2,
            n_param_conditioning=self.n_param_conditioning,
        )
        self.assertTrue(isinstance(model, FNO))
        x = torch.rand(8, 1, 32, 32, 5)
        t = torch.rand(8)
        param = torch.rand(8, 3)
        input = {"time": t, "x": x, "parameters": param}
        out = model(input)
        self.assertEqual(out.shape, x.shape)

    def test_load_conf(self):
        FNO_CONFIG_FILE = "the_well/benchmark/configs/model/fno.yaml"
        config = OmegaConf.load(FNO_CONFIG_FILE)
        model = instantiate(
            config,
            n_spatial_dim=self.n_spatial_dim,
            n_input_scalar_components=self.n_input_scalar,
            n_input_vector_components=self.n_input_vector,
            n_output_scalar_components=self.n_input_scalar,
            n_output_vector_components=self.n_input_vector,
            n_param_conditioning=self.n_param_conditioning,
        )
        self.assertTrue(isinstance(model, FNO))
        x = torch.rand(8, 1, 32, 32, 5)
        t = torch.rand(8)
        param = torch.rand(8, 3)
        input = {"time": t, "x": x, "parameters": param}
        out = model(input)
        self.assertEqual(out.shape, x.shape)
