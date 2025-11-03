"""Metrics that operate on full trajectories."""

import numpy as np
import torch
from einops import rearrange

from the_well.benchmark.metrics.common import TrajectoryMetric
from the_well.data.datasets import WellMetadata


class HistogramW1(TrajectoryMetric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
        eps: float = 1e-7,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Pearson Correlation Coefficient

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            Pearson correlation coefficient between x and y.
        """
        # Inputs B, T, [H], [W], [D], C with meta specifying number of spatial dims
        # Flatten space and time
        x_flat = torch.flatten(x, start_dim=-meta.n_spatial_dims - 2, end_dim=-2)
        y_flat = torch.flatten(y, start_dim=-meta.n_spatial_dims - 2, end_dim=-2)
        if normalize:
            # Compute norm along space-time (-2) and divide by it
            norm_denom = torch.mean(y_flat**2, dim=-2, keepdim=True) + eps
            x_flat = x_flat / norm_denom
            y_flat = y_flat / norm_denom

        x_sorted, _ = torch.sort(x_flat, dim=-2)
        y_sorted, _ = torch.sort(y_flat, dim=-2)

        # Calculate means along flattened axis
        return (
            (x_sorted - y_sorted).abs().mean(dim=-2, keepdim=True)
        )  # Should be B, 1, C


def vector_dtw(x, y, dtw_array, w=10, n=100):
    dtw_array[:, 0, 0] = 0.0
    for i in range(1, n):
        for j in range(max(1, i - w), min(n, i + w)):
            cost = torch.mean((x[:, i] - y[:, j]) ** 2, dim=-1)
            dtw_array[:, i, j] = cost + torch.minimum(
                torch.minimum(
                    dtw_array[:, i - 1, j],  # insertion
                    dtw_array[:, i, j - 1],
                ),  # deletion
                dtw_array[:, i - 1, j - 1],  # match
            )
    min_costs = torch.argmin(dtw_array, dim=1)
    # print("min costs", min_costs)
    distortion = min_costs - torch.arange(n, device=min_costs.device).unsqueeze(0)
    return (dtw_array[:, -1, -1] / (n)).sqrt(), distortion.float().abs().mean(-1)


class WindowedDTW(TrajectoryMetric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
        norm: bool = True,
        eps: float = 1e-7,
        window_size: int = 10,
    ) -> torch.Tensor:
        """
        Windowed Dynamic Time Warping computed a the distance between two time series
        after alignment.

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.
            window_size: Size of the window for DTW.

        Returns:
            Windowed DTW distance between x and y.
        """
        # Inputs B, T, [H], [W], [D], C with meta specifying number of spatial dims
        # Flatten space
        x_flat = torch.flatten(x, start_dim=-meta.n_spatial_dims - 1, end_dim=-2)
        y_flat = torch.flatten(y, start_dim=-meta.n_spatial_dims - 1, end_dim=-2)
        # Now B T S C
        if norm:
            # Compute norm along space (-2) and divide by it
            x_flat = x_flat / (torch.norm(x_flat, dim=-1, keepdim=True) + eps)
            y_flat = y_flat / (torch.norm(y_flat, dim=-1, keepdim=True) + eps)
        B, _, S, C = x_flat.shape
        x_flat = rearrange(x_flat, "b t s c -> (b c) t s")
        y_flat = rearrange(y_flat, "b t s c -> (b c) t s")
        # split into series and run DTW on each
        dtw_array = (
            torch.ones(
                (x_flat.shape[0], x_flat.shape[1], x_flat.shape[1]),
                device=x_flat.device,
            )
            * 1e12
        )
        dists, distortion = vector_dtw(
            x_flat, y_flat, dtw_array, w=window_size, n=x_flat.shape[1]
        )
        dists = rearrange(dists, "(b c) -> b 1 c", b=B)
        distortion = rearrange(distortion, "(b c) -> b 1 c", b=B)
        return {"dtw_distance": dists, "dtw_path_distortion": distortion}
