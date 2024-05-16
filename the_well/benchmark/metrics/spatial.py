import numpy as np
import torch

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.metrics.common import metric


@metric
def mse(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: GenericWellMetadata,
):
    """
    Mean Squared Error

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        Input tensor.
    y : torch.Tensor | np.ndarray
        Target tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.

    Returns
    -------
    torch.Tensor
        Mean squared error between x and y.
    """
    spatial_dims = tuple(range(-meta.spatial_ndims - 1, -1))
    return torch.mean((x - y) ** 2, dim=spatial_dims)


@metric
def nmse(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: GenericWellMetadata,
    eps: float = 1e-7,
    norm_mode: str = "norm",
):
    """
    Normalized Mean Squared Error

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        Input tensor.
    y : torch.Tensor | np.ndarray
        Target tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.
    eps : float
        Small value to avoid division by zero. Default is 1e-7.
    norm_mode : str
        Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

    Returns
    -------
    torch.Tensor
        Normalized mean squared error between x and y.
    """
    spatial_dims = tuple(range(-meta.spatial_ndims - 1, -1))
    if norm_mode == "norm":
        norm = torch.mean(y**2, dim=spatial_dims)
    elif norm_mode == "std":
        norm = torch.std(y, dim=spatial_dims) ** 2
    else:
        raise ValueError(f"Invalid norm_mode: {norm_mode}")
    return mse(x, y, meta) / (norm + eps)


@metric
def rmse(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: GenericWellMetadata,
):
    """
    Root Mean Squared Error

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        Input tensor.
    y : torch.Tensor | np.ndarray
        Target tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.

    Returns
    -------
    torch.Tensor
        Root mean squared error between x and y.
    """
    return torch.sqrt(mse(x, y, meta))


@metric
def nrmse(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: GenericWellMetadata,
    eps: float = 1e-7,
    norm_mode: str = "norm",
):
    """
    Normalized Root Mean Squared Error

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        Input tensor.
    y : torch.Tensor | np.ndarray
        Target tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.
    eps : float
        Small value to avoid division by zero. Default is 1e-7.
    norm_mode : str
        Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

    Returns
    -------
    torch.Tensor
        Normalized root mean squared error between x and y.
    """
    return torch.sqrt(nmse(x, y, meta, eps=eps, norm_mode=norm_mode))
