import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import wandb
from the_well.benchmark.data.datasets import GenericWellMetadata, flatten_field_names


def field_histograms(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: GenericWellMetadata,
    output_dir: str,
    epoch_number: int = 0,
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
        y_hist, use_bins = np.histogram(np_y, bins=bins, density=True)
        x_hist, _ = np.histogram(np_x, bins=use_bins, density=True)
        ax.stairs(
            x_hist,
            use_bins,
            alpha=0.5,
            label="Predicted",
        )
        ax.stairs(
            y_hist,
            use_bins,
            alpha=0.5,
            label="Target",
        )
        ax.set_xlabel("Field Value")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(title)
        os.makedirs(f"{output_dir}/{meta.dataset_name}/{title}/", exist_ok=True)
        # Save to disk
        plt.savefig(f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_{title}.png")
        np.save(f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_xhist.npy", x_hist)
        np.save(f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_yhist.npy", y_hist)
        np.save(f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_bins.npy", use_bins)
        # out_dict[f"{meta.dataset_name}/{title}"] = wandb.Image(fig)
        plt.close()
    return out_dict


def build_1d_power_spectrum(x, spatial_dims):
    x_fft = torch.fft.fftn(x, dim=spatial_dims, norm="ortho").abs().square()
    # Return the shifted sqrt power spectrum - first average over spatial dims, then batch and time
    return torch.fft.fftshift(x_fft.mean(spatial_dims[1:]).mean(0).mean(0).sqrt())


def plot_power_spectrum_by_field(x: torch.Tensor | np.ndarray,
                                y: torch.Tensor | np.ndarray,
                                metadata: GenericWellMetadata,
                                output_dir: str,
                                epoch_number: int = 0,
):
    """
    Plot the power spectrum of the input tensor x and y.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
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
        title = f"{field_names[i]} Radial Power Spectrum"
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
        # out_dict[f"{metadata.dataset_name}/{title}"] = wandb.Image(fig)
        os.makedirs(f"{output_dir}/{metadata.dataset_name}/{title}", exist_ok=True)
        # Save to disk
        plt.savefig(f"{output_dir}/{metadata.dataset_name}/{title}/{title}_epoch{epoch_number}.png")
        np.save(f"{output_dir}/{metadata.dataset_name}/{title}/epoch{epoch_number}_x.npy", np_x_fft)
        np.save(f"{output_dir}/{metadata.dataset_name}/{title}/epoch{epoch_number}_y.npy", np_y_ftt)
        np.save(f"{output_dir}/{metadata.dataset_name}/{title}/epoch{epoch_number}_res.npy", np_res_ftt)
        plt.close()
    return out_dict


def plot_all_time_metrics(time_logs: dict,
                          metadata: GenericWellMetadata,
                          output_dir: str,
                          epoch_number: int = 0,
                          ):
    """ Plot loss over time for all time metrics. 
    
    """
    out_dict = {}
    os.makedirs(f"{output_dir}/{metadata.dataset_name}/rollout_losses/epoch_{epoch_number}", exist_ok=True)
    for k, v in time_logs.items():
        v = np.array(v)
        # os.makedirs(f"{output_dir}/{metadata.dataset_name}/epoch_{epoch_number}", exist_ok=True)
        title = k.split("/")[-1]
        np.save(f"{output_dir}/{metadata.dataset_name}/rollout_losses/epoch_{epoch_number}/{title}.npy", v)
        # data = [[t, loss] for t, loss in zip(range(len(v)), v)]
        # table = wandb.Table(data=data, columns=["time_step", "loss"])
        # fig = wandb.plot.line(table, "time_step", "loss", title=k)
        # out_dict[k] = fig
    return out_dict
