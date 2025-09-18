"""Metrics that operate on full trajectories."""
import numpy as np
import torch

from einops import rearrange
from the_well.benchmark.metrics import VMSE
from the_well.benchmark.metrics.common import Metric, TrajectoryMetric
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
        return (x_sorted - y_sorted).abs().mean(dim=-2, keepdim=True) # Should be B, 1, C


# class WindowedDTW(TrajectoryMetric):
#     def __init__(self,
#                  window_size: int = 5,
#                  ):
#         super().__init__()
#         self.window_size = window_size
#         # self.base_metric = base_metric
        
#     @staticmethod
#     def eval(
#         x: torch.Tensor | np.ndarray,
#         y: torch.Tensor | np.ndarray,
#         meta: WellMetadata,
#         norm: bool = True,
#         eps: float = 1e-7,
#     ) -> torch.Tensor:
#         """
#         Windowed Dynamic Time Warping

#         Args:
#             x: Input tensor.
#             y: Target tensor.
#             meta: Metadata for the dataset.
#             window_size: Size of the window for DTW.

#         Returns:
#             Windowed DTW distance between x and y.
#         """
#         try:
#             import tslearn
#         except ImportError:
#             raise ImportError("tslearn is required for WindowedDTW metric. Please install it via `pip install tslearn`.")
        
#         # Inputs B, T, [H], [W], [D], C with meta specifying number of spatial dims
#         # Flatten space
#         x_flat = torch.flatten(x, start_dim=-meta.n_spatial_dims - 1, end_dim=-2)
#         y_flat = torch.flatten(y, start_dim=-meta.n_spatial_dims - 1, end_dim=-2)

#         if norm:
#             # Compute norm along space (-2) and divide by it
#             x_flat = x_flat / (torch.norm(x_flat, dim=-1, keepdim=True) + eps)
#             y_flat = y_flat / (torch.norm(y_flat, dim=-1, keepdim=True) + eps)

#         B = x_flat.shape[0]
#         x_flat = rearrange(x_flat, 'b t c -> b c t')
#         y_flat = rearrange(y_flat, 'b t c -> b c t')
#         B, T, C = x_flat.shape
#         dtw_matrix = torch.zeros((B, T + 1, T + 1), device=x.device) + float("inf")
#         dtw_matrix[:, 0, 0] = 0

#         for i in range(1, T + 1):
#             for j in range(max(1, i - window_size), min(T + 1, i + window_size)):
#                 cost = torch.norm(x_flat[:, i - 1] - y_flat[:, j - 1], dim=-1)
#                 dtw_matrix[:, i, j] = cost + torch.min(
#                     torch.stack(
#                         [
#                             dtw_matrix[:, i - 1, j],    # Insertion
#                             dtw_matrix[:, i, j - 1],    # Deletion
#                             dtw_matrix[:, i - 1, j - 1] # Match
#                         ],
#                         dim=-1
#                     ),
#                     dim=-1
#                 ).values

#         return dtw_matrix[:, -1, -1] / T