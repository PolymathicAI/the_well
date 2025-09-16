import pytest
import torch

from the_well.benchmark.models import UNetClassic, UNetConvNext


@pytest.mark.parametrize("dim_in", [1, 3, 5])
@pytest.mark.parametrize("dim_out", [1, 3, 5])
@pytest.mark.parametrize("n_spatial_dims", [2, 3])
@pytest.mark.parametrize("spatial_resolution", [16, 32])
def test_unet(dim_in, dim_out, n_spatial_dims, spatial_resolution):
    batch_size = 4
    spatial_resolution = [spatial_resolution] * n_spatial_dims
    model = UNetClassic(dim_in, dim_out, n_spatial_dims, spatial_resolution)
    input = torch.rand((batch_size, dim_in, *spatial_resolution))
    output = model(input)
    assert output.shape == (
        batch_size,
        dim_out,
        *spatial_resolution,
    )


@pytest.mark.parametrize("dim_in", [1, 3, 5])
@pytest.mark.parametrize("dim_out", [1, 3, 5])
@pytest.mark.parametrize("n_spatial_dims", [2, 3])
@pytest.mark.parametrize("spatial_resolution", [16, 32])
def test_unet_convnext(dim_in, dim_out, n_spatial_dims, spatial_resolution):
    spatial_resolution = [spatial_resolution] * n_spatial_dims
    batch_size = 4
    model = UNetConvNext(
        dim_in, dim_out, n_spatial_dims, spatial_resolution, stages=2, init_features=16
    )
    input = torch.rand((batch_size, dim_in, *spatial_resolution))
    output = model(input)
    assert output.shape == (
        batch_size,
        dim_out,
        *spatial_resolution,
    )
