import numpy as np
import torch
import torch.nn.functional as F

from the_well.benchmark.data.datasets import GenericWellMetadata
from the_well.benchmark.metrics.common import metric


def fftn(x: torch.Tensor, meta: GenericWellMetadata):
    """
    Compute the N-dimensional FFT of input tensor x. Wrapper around torch.fft.fftn.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.

    Returns
    -------
    torch.Tensor
        N-dimensional FFT of x.
    """
    spatial_dims = tuple(range(-meta.spatial_ndims - 1, -1))
    return torch.fft.fftn(x, dim=spatial_dims)


def ifftn(x: torch.Tensor, meta: GenericWellMetadata):
    """
    Compute the N-dimensional inverse FFT of input tensor x. Wrapper around torch.fft.ifftn.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.

    Returns
    -------
    torch.Tensor
        N-dimensional inverse FFT of x.
    """
    spatial_dims = tuple(range(-meta.spatial_ndims - 1, -1))
    return torch.fft.ifftn(x, dim=spatial_dims)


def power_spectrum(
    x: torch.Tensor,
    meta: GenericWellMetadata,
    bins: torch.Tensor = None,
    fourier_input: bool = False,
    sample_spacing: float = 1.0,
    return_counts: bool = False,
) -> tuple:
    """
    Compute the isotropic power spectrum of input tensor x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bins : torch.Tensor, optional
        Array of bin edges. If None, we use a default binning. The default is None.
    fourier_input : bool, optional
        If True, x is assumed to be the Fourier transform of the input data. The default is False.
    sample_spacing : float, optional
        Sample spacing. The default is 1.0.
    return_counts : bool, optional
        Return counts per bin. The default is False.

    Returns
    -------
    bins : torch.Tensor
        Array of bin edges.
    ps_mean : torch.Tensor
        Power spectrum (estimated as a mean over bins).
    ps_std : torch.Tensor
        Standard deviation of the power spectrum (estimated as a standard deviation over bins).
    counts : torch.Tensor, optional
        Counts per bin if return_counts=True.
    """
    spatial_dims = tuple(range(-meta.spatial_ndims - 1, -1))
    ndim = len(spatial_dims)
    device = x.device

    N = x.shape[spatial_dims[0]]
    for dim in spatial_dims:
        if x.shape[dim] != N:
            raise Exception("Input data must be of shape (N, ..., N).")

    # Compute array of isotropic wavenumbers
    wn = (
        (2 * np.pi * torch.fft.fftfreq(N, d=sample_spacing))
        .reshape((N,) + (1,) * (ndim - 1))
        .to(device)
    )
    wn_iso = torch.zeros((N,) * ndim).to(device)
    for i in range(ndim):
        wn_iso += torch.moveaxis(wn, 0, i) ** 2
    wn_iso = torch.sqrt(wn_iso).flatten()

    if bins is None:
        # bins = torch.sort(torch.unique(wn_iso))[0]  # Default binning
        bins = torch.linspace(0, np.pi, int(np.sqrt(N))).to(device)  # Default binning
    indices = torch.bucketize(wn_iso, bins, right=True) - 1
    indices_mask = F.one_hot(indices, num_classes=len(bins))
    counts = torch.sum(indices_mask, dim=0)

    if not fourier_input:
        x = fftn(x, meta)
    fx2 = torch.abs(x) ** 2
    fx2 = fx2.reshape(
        x.shape[: spatial_dims[0]] + (-1, x.shape[-1])
    )  # Flatten spatial dimensions

    # Compute power spectrum
    ps_mean = torch.sum(
        fx2.unsqueeze(-2) * indices_mask.unsqueeze(-1), dim=-3
    ) / counts.unsqueeze(-1)
    ps_std = torch.sqrt(
        torch.sum(
            (fx2.unsqueeze(-2) - ps_mean.unsqueeze(-3)) ** 2
            * indices_mask.unsqueeze(-1),
            dim=-3,
        )
        / counts.unsqueeze(-1)
    )

    # Discard the last bin (which has no upper limit)
    ps_mean = ps_mean[..., :-1, :]
    ps_std = ps_std[..., :-1, :]

    if return_counts:
        return bins, ps_mean, ps_std, counts
    else:
        return bins, ps_mean, ps_std


@metric
def spectral_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    meta: GenericWellMetadata,
    bins: torch.Tensor = None,
    fourier_input: bool = False,
) -> torch.Tensor:
    """
    Spectral Mean Squared Error.
    Corresponds to MSE computed after filtering over wavenumber bins in the Fourier domain.

    Default binning is a set of three (approximately) logspaced from 0 to pi.

    Note that, MSE(x, y) should (approximately) match the sum over frequency bins of the spectral MSE.

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        Input tensor.
    y : torch.Tensor | np.ndarray
        Target tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.
    bins : torch.Tensor, optional
        Tensor of bin edges. If None, we use a default binning that is a set of three (approximately) logspaced from 0 to pi. The default is None.
    fourier_input : bool, optional
        If True, x and y are assumed to be the Fourier transform of the input data. The default is False.

    Returns
    -------
    torch.Tensor
        Power spectrum mean squared error between x and y.
    """
    N = x.shape[-2]
    ndims = meta.spatial_ndims

    if bins is None:  # Default binning
        bins = torch.logspace(np.log10(2 * np.pi / N), np.log10(np.pi), 4).to(
            x.device
        )  # Low, medium, and high frequency bins
        bins[0] = 0.0  # We start from zero
    _, ps_res_mean, _, counts = power_spectrum(
        x - y, meta, bins=bins, fourier_input=fourier_input, return_counts=True
    )

    # Compute the mean squared error per bin (stems from Plancherel's formula)
    mse_per_bin = ps_res_mean * counts[:-1].unsqueeze(-1) / (N**ndims) ** 2

    return mse_per_bin
