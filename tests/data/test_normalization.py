import math

import torch

from the_well.data.normalization import RMSNormalization, ZScoreNormalization


def test_zscore_normalization():
    """Test the ZScoreNormalization actually provides the correct normalization.
    We consider fields whose mean and std are given by a linear function of the field index.

    """
    n_fields = 4
    h = 64
    w = 64
    t = 10
    batch_size = 64
    tol = 1e-2
    std = torch.arange(n_fields) + 1.0
    mean = torch.arange(n_fields)
    delta_mean = torch.zeros_like(mean)
    delta_std = math.sqrt(2) * std
    stats = {
        "mean": {f"field_{i}": mean[i] for i in range(n_fields)},
        "std": {f"field_{i}": std[i] for i in range(n_fields)},
        "mean_delta": {f"field_{i}": delta_mean[i] for i in range(n_fields)},
        "std_delta": {f"field_{i}": delta_std[i] for i in range(n_fields)},
    }
    normalization = ZScoreNormalization(
        stats=stats,
        core_field_names=[f"field_{i}" for i in range(n_fields)],
        core_constant_field_names=[],
    )

    input_tensor = std * torch.randn(batch_size, h, w, t, n_fields) + mean
    delta_input_tensor = input_tensor[..., 1:, :] - input_tensor[..., :-1, :]
    for i in range(n_fields):
        normalized_tensor = normalization.normalize(input_tensor[..., i], f"field_{i}")
        assert normalized_tensor.shape == (batch_size, h, w, t)
        assert torch.allclose(
            torch.mean(normalized_tensor), torch.tensor(0.0), atol=tol
        )
        assert torch.allclose(torch.std(normalized_tensor), torch.tensor(1.0), atol=tol)

        normalized_delta_tensor = normalization.delta_normalize(
            delta_input_tensor[..., i], f"field_{i}"
        )
        assert normalized_delta_tensor.shape == (batch_size, h, w, t - 1)
        assert torch.allclose(
            torch.mean(normalized_delta_tensor), torch.tensor(0.0), atol=tol
        )
        assert torch.allclose(
            torch.std(normalized_delta_tensor), torch.tensor(1.0), atol=tol
        )


def test_rms_normalization():
    """Test the RMSNormalization actually provides the correct normalization.
    We consider fields whose mean and std are given by a linear function of the field index.
    """
    n_fields = 4
    h = 64
    w = 64
    t = 10
    batch_size = 64
    tol = 1e-2
    std = torch.arange(n_fields) + 1.0
    mean = torch.arange(n_fields)
    delta_std = math.sqrt(2) * std
    stats = {
        "rms": {f"field_{i}": std[i] for i in range(n_fields)},
        "rms_delta": {f"field_{i}": delta_std[i] for i in range(n_fields)},
    }
    normalization = RMSNormalization(
        stats=stats,
        core_field_names=[f"field_{i}" for i in range(n_fields)],
        core_constant_field_names=[],
    )

    input_tensor = std * torch.randn(batch_size, h, w, t, n_fields) + mean
    delta_input_tensor = input_tensor[..., 1:, :] - input_tensor[..., :-1, :]
    for i in range(n_fields):
        normalized_tensor = normalization.normalize(input_tensor[..., i], f"field_{i}")
        assert normalized_tensor.shape == (batch_size, h, w, t)
        assert torch.allclose(
            torch.mean(normalized_tensor),
            mean[i].float() / std[i].float(),
            atol=tol,
        )
        assert torch.allclose(
            torch.std(normalized_tensor),
            torch.tensor(1.0),
            atol=tol,
        )

        normalized_delta_tensor = normalization.delta_normalize(
            delta_input_tensor[..., i], f"field_{i}"
        )
        assert normalized_delta_tensor.shape == (batch_size, h, w, t - 1)
        assert torch.allclose(
            torch.mean(normalized_delta_tensor), torch.tensor(0.0), atol=tol
        )
        assert torch.allclose(
            torch.std(normalized_delta_tensor), torch.tensor(1.0), atol=tol
        )
