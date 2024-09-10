import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from the_well.benchmark.data.datasets import GenericWellMetadata, flatten_field_names


def field_histograms(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: GenericWellMetadata,
    bins: int = 100,
    title: str = None,
):
    """
    Compute histograms of the field values for tensors
    x and y and package them as dictionary for logging.

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        Input tensor.
    y : torch.Tensor | np.ndarray
        Target tensor.
    meta : GenericWellMetadata
        Metadata for the dataset.
    bins : int
        Number of bins for the histogram. Default is 100.
    log_scale : bool
        Whether to plot the histogram on a log scale. Default is False.
    title : str
        Title for the plot. Default is None.
    wandb_log : bool
        Whether to log the plot to Weights & Biases. Default is True.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    field_names = flatten_field_names(meta)
    out_dict = {}
    for i in range(x.shape[-1]):
        fig, ax = plt.subplots()
        title = f"{field_names[i]} Histogram"
        # Using these for debugging weird error
        np_y = np.nan_to_num(
            y[..., i].flatten().cpu().numpy(), nan=1000, posinf=10000, neginf=-10000
        )
        np_x = np.nan_to_num(
            x[..., i].flatten().cpu().numpy(), nan=1000, posinf=10000, neginf=-10000
        )
        use_bins = np.histogram_bin_edges(np_y, bins=bins)
        ax.hist(
            np_x,
            bins=use_bins,
            density=True,
            alpha=0.5,
            label="Predicted",
        )
        ax.hist(
            np_y,
            bins=use_bins,
            density=True,
            alpha=0.5,
            label="Target",
        )
        ax.set_xlabel("Field Value")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(title)
        out_dict[f"{meta.dataset_name}/{title}"] = wandb.Image(fig)
        plt.close()
    return out_dict


def build_1d_power_spectrum(x, spatial_dims):
    x_fft = torch.fft.fftn(x, dim=spatial_dims, norm="ortho").abs().square()
    # Return the shifted sqrt power spectrum - first average over spatial dims, then batch and time
    return torch.fft.fftshift(x_fft.mean(spatial_dims[1:]).mean(0).mean(0).sqrt())


def plot_power_spectrum_by_field(x, y, metadata):
    """
    Plot the power spectrum of the input tensor x and y.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    """
    field_names = flatten_field_names(metadata)
    spatial_dims = tuple(range(-metadata.n_spatial_dims - 1, -1))
    y_fft = build_1d_power_spectrum(y, spatial_dims)
    x_fft = build_1d_power_spectrum(x, spatial_dims)
    res_fft = build_1d_power_spectrum(y - x, spatial_dims)
    axis = torch.fft.fftshift(torch.fft.fftfreq(x.shape[spatial_dims[0]], d=1.0))

    out_dict = {}
    for i in range(x.shape[-1]):
        fig, ax = plt.subplots()
        np_x_fft = x_fft[..., i].sqrt().cpu().numpy()
        np_y_ftt = y_fft[..., i].sqrt().cpu().numpy()
        np_res_ftt = res_fft[..., i].sqrt().cpu().numpy()
        title = f"{field_names[i]} First Axis Mean Power Spectrum"
        ax.semilogy(
            axis,
            np_x_fft,
            label="Predicted Spectrum",
            alpha=0.5,
            linestyle="--",
        )
        ax.semilogy(
            axis,
            np_y_ftt,
            label="Target Spectrum",
            alpha=0.5,
            linestyle="-.",
        )
        ax.semilogy(
            axis,
            np_res_ftt,
            label="Residual Spectrum",
            alpha=0.5,
            linestyle=":",
        )
        ax.set_xlabel("Wave Number")
        ax.set_ylabel("Power spectrum")
        ax.legend()
        ax.set_title(title)
        out_dict[f"{metadata.dataset_name}/{title}"] = wandb.Image(fig)
        plt.close()
    return out_dict


def plot_all_time_metrics(time_logs):
    out_dict = {}

    for k, v in time_logs.items():
        data = [[t, loss] for t, loss in zip(range(len(v)), v)]
        table = wandb.Table(data=data, columns=["time_step", "loss"])
        fig = wandb.plot.line(table, "time_step", "loss", title=k)
        out_dict[k] = fig
    return out_dict
