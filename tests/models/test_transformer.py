import pytest
import torch

from the_well.benchmark.models import AViT


@pytest.mark.parametrize("dim_in", [1, 3, 5])
@pytest.mark.parametrize("dim_out", [1, 3, 5])
@pytest.mark.parametrize("n_spatial_dims", [2, 3])
@pytest.mark.parametrize("spatial_resolution", [16, 32])
def test_avit(dim_in, dim_out, n_spatial_dims, spatial_resolution):
    batch_size = 4
    spatial_resolution = [spatial_resolution] * n_spatial_dims
    model = AViT(
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution,
        hidden_dim=48,
        num_heads=2,
        processor_blocks=2,
    )
    input = torch.rand((batch_size, *spatial_resolution, dim_in))
    output = model(input)
    assert output.shape == (
        batch_size,
        *spatial_resolution,
        dim_out,
    )
