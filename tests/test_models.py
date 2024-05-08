from unittest import TestCase

from omegaconf import OmegaConf
import torch

from the_well.benchmark.models import FNO, PDEModelConfig, get_fno_model


class TestFNO(TestCase):
    def test_model(self):
        pde_config = PDEModelConfig(1, 2, 2, 3)
        conf = OmegaConf.create(
            {
                "name": "FNO-128-16m",
            }
        )
        model = get_fno_model(conf, pde_config)
        self.assertTrue(isinstance(model, FNO))
        x = torch.rand(8, 1, 5, 32, 32)
        t = torch.rand(8)
        param = torch.rand(8, 3)
        out = model(x, t, param)
        self.assertEqual(out.shape, x.shape)
