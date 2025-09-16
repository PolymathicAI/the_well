import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from the_well.benchmark.models import (
    AFNO,
    FNO,
    TFNO,
)


@pytest.mark.parametrize("fno_model", [FNO, TFNO])
@pytest.mark.parametrize("dim_in", [1, 3, 5])
@pytest.mark.parametrize("dim_out", [1, 3, 5])
@pytest.mark.parametrize("n_spatial_dims", [2, 3])
@pytest.mark.parametrize("spatial_resolution", [16, 32])
def test_fno_model(fno_model, dim_in, dim_out, n_spatial_dims, spatial_resolution):
    spatial_resolution = [spatial_resolution] * n_spatial_dims
    modes1 = 2
    modes2 = 2
    modes3 = 2
    hidden_channels = 8
    model = fno_model(
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution,
        modes1,
        modes2,
        modes3,
        hidden_channels,
    )
    batch_size = 4
    input = torch.rand((batch_size, dim_in, *spatial_resolution))
    output = model(input)
    assert output.shape == (batch_size, dim_out, *spatial_resolution)


@pytest.mark.parametrize("dim_in", [1, 3, 5])
@pytest.mark.parametrize("dim_out", [1, 3, 5])
@pytest.mark.parametrize("n_spatial_dims", [2, 3])
@pytest.mark.parametrize("spatial_resolution", [16, 32])
def test_afno(dim_in, dim_out, n_spatial_dims, spatial_resolution):
    spatial_resolution = [spatial_resolution] * n_spatial_dims
    model = AFNO(
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution,
        hidden_dim=8,
        n_blocks=2,
        cmlp_diagonal_blocks=1,
    )
    batch_size = 4
    input = torch.rand((batch_size, *spatial_resolution, dim_in))
    output = model(input)
    assert output.shape == (batch_size, *spatial_resolution, dim_out)


def test_load_fno_conf():
    FNO_CONFIG_FILE = "the_well/benchmark/configs/model/fno.yaml"
    config = OmegaConf.load(FNO_CONFIG_FILE)
    n_spatial_dims = 2
    spatial_resolution = [32, 32]
    dim_in = 2
    dim_out = dim_in
    model = instantiate(
        config,
        n_spatial_dims=n_spatial_dims,
        spatial_resolution=spatial_resolution,
        dim_in=dim_in,
        dim_out=dim_out,
    )
    assert isinstance(model, FNO)
    input = torch.rand(8, dim_in, *spatial_resolution)
    output = model(input)
    assert output.shape == input.shape
