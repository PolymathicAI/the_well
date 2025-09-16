from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from the_well.benchmark.metrics.common import Metric
from the_well.data.datasets import WellMetadata


def fftn(x: torch.Tensor, meta: WellMetadata) -> torch.Tensor:
    """
    Compute the N-dimensional FFT of input tensor x. Wrapper around torch.fft.fftn.

    Args:
        x: Input tensor.
        meta: Metadata for the dataset.

    Returns:
        N-dimensional FFT of x.
    """
    spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
    return torch.fft.fftn(x, dim=spatial_dims)


def ifftn(x: torch.Tensor, meta: WellMetadata) -> torch.Tensor:
    """
    Compute the N-dimensional inverse FFT of input tensor x. Wrapper around torch.fft.ifftn.

    Args:
        x: Input tensor.
        meta: Metadata for the dataset.

    Returns:
        N-dimensional inverse FFT of x.
    """
    spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
    return torch.fft.ifftn(x, dim=spatial_dims)


def power_spectrum(
    x: torch.Tensor,
    meta: WellMetadata,
    bins: torch.Tensor = None,
    fourier_input: bool = False,
    sample_spacing: float = 1.0,
    return_counts: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute the isotropic power spectrum of input tensor x.

    Args:
        x: Input tensor.
        bins: Array of bin edges. If None, we use a default binning. The default is None.
        fourier_input: If True, x is assumed to be the Fourier transform of the input data. The default is False.
        sample_spacing: Sample spacing. The default is 1.0.
        return_counts: Return counts per bin. The default is False.

    Returns:
        A four tuple (bins, ps_mean, ps_std, counts) containing the array of bin edges,
        the power spectrum (estimated as a mean over bins),
        the standard deviation of the power spectrum (estimated as a standard deviation over bins),
        and the counts per bin if return_counts=True.
    """
    spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
    spatial_shape = tuple(x.shape[dim] for dim in spatial_dims)
    ndim = len(spatial_dims)
    device = x.device

    # Compute array of isotropic wavenumbers
    wn_iso = torch.zeros(spatial_shape).to(device)
    for i in range(ndim):
        wn = (
            (2 * np.pi * torch.fft.fftfreq(spatial_shape[i], d=sample_spacing))
            .reshape((spatial_shape[i],) + (1,) * (ndim - 1))
            .to(device)
        )
        wn_iso += torch.moveaxis(wn, 0, i) ** 2
    wn_iso = torch.sqrt(wn_iso).flatten()

    if bins is None:
        bins = torch.linspace(
            0, wn_iso.max().item() + 1e-6, int(np.sqrt(min(spatial_shape)))
        ).to(device)  # Default binning
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
    ps_mean = torch.sum(fx2.unsqueeze(-2) * indices_mask.unsqueeze(-1), dim=-3) / (
        counts.unsqueeze(-1) + 1e-7
    )
    ps_std = torch.sqrt(
        torch.sum(
            (fx2.unsqueeze(-2) - ps_mean.unsqueeze(-3)) ** 2
            * indices_mask.unsqueeze(-1),
            dim=-3,
        )
        / (counts.unsqueeze(-1) + 1e-7)
    )

    # Discard the last bin (which has no upper limit)
    ps_mean = ps_mean[..., :-1, :]
    ps_std = ps_std[..., :-1, :]

    if return_counts:
        return bins, ps_mean, ps_std, counts
    else:
        return bins, ps_mean, ps_std


class binned_spectral_mse(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        meta: WellMetadata,
        bins: torch.Tensor = None,
        fourier_input: bool = False,
    ) -> torch.Tensor:
        """
        Binned Spectral Mean Squared Error.
        Corresponds to MSE computed after filtering over wavenumber bins in the Fourier domain.

        Default binning is a set of three (approximately) logspaced from 0 to pi.

        Note that, MSE(x, y) should match the sum over frequency bins of the spectral MSE.

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.
            bins:
                Tensor of bin edges. If None, we use a default binning that is a set of three (approximately) logspaced from 0 to pi. The default is None.
            fourier_input:
                If True, x and y are assumed to be the Fourier transform of the input data. The default is False.

        Returns:
            The power spectrum mean squared error between x and y.
        """
        spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
        spatial_shape = tuple(x.shape[dim] for dim in spatial_dims)
        prod_spatial_shape = np.prod(np.array(spatial_shape))
        ndims = meta.n_spatial_dims

        if bins is None:  # Default binning
            bins = torch.logspace(
                np.log10(2 * np.pi / max(spatial_shape)),
                np.log10(np.pi * np.sqrt(ndims) + 1e-6),
                4,
            ).to(x.device)  # Low, medium, and high frequency bins
            bins[0] = 0.0  # We start from zero
        _, ps_res_mean, _, counts = power_spectrum(
            x - y, meta, bins=bins, fourier_input=fourier_input, return_counts=True
        )

        # TODO - MAJOR DESIGN VIOLATION - BUT ITS FASTER TO IMPLEMENT THIS WAY TODAY...
        _, ps_true_mean, _, true_counts = power_spectrum(
            y, meta, bins=bins, fourier_input=fourier_input, return_counts=True
        )

        # Compute the mean squared error per bin (stems from Plancherel's formula)
        mse_per_bin = ps_res_mean * counts[:-1].unsqueeze(-1) / prod_spatial_shape**2
        true_energy_per_min = (
            ps_true_mean * true_counts[:-1].unsqueeze(-1) / prod_spatial_shape**2
        )
        nmse_per_bin = mse_per_bin / (true_energy_per_min + 1e-7)

        mse_dict = {
            f"spectral_error_mse_per_bin_{i}": mse_per_bin[..., i, :]
            for i in range(mse_per_bin.shape[-2])
        }
        nmse_dict = {
            f"spectral_error_nmse_per_bin_{i}": nmse_per_bin[..., i, :]
            for i in range(nmse_per_bin.shape[-2])
        }
        out_dict = mse_dict
        # Hacked to add this here for now - should be split with taking PS as input
        out_dict |= nmse_dict
        # TODO Figure out better way to handle multi-output losses
        return out_dict
