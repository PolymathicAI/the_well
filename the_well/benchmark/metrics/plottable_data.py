import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from the_well.data.datasets import WellMetadata, flatten_field_names


def field_histograms(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    meta: WellMetadata,
    output_dir: str,
    epoch_number: int = 0,
    bins: int = 100,
    title: str = None,
):
    """
    Compute histograms of the field values for tensors
    x and y and package them as dictionary for logging.

    Args:
        x: Predicted tensor
        y: Target tensor
        metadata: Metadata object associated with dset
        output_dir: Directory to save the plots
        epoch_number: Current epoch number
        bins: Number of bins for the histogram. Default is 100.
        log_scale: Whether to plot the histogram on a log scale. Default is False.
        title: Title for the plot. Default is None.

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
        plt.savefig(
            f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_{title}.png"
        )
        np.save(
            f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_xhist.npy",
            x_hist,
        )
        np.save(
            f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_yhist.npy",
            y_hist,
        )
        np.save(
            f"{output_dir}/{meta.dataset_name}/{title}/epoch{epoch_number}_bins.npy",
            use_bins,
        )
        plt.close()
    return out_dict


def build_1d_power_spectrum(x, spatial_dims):
    x_fft = torch.fft.fftn(x, dim=spatial_dims, norm="ortho").abs().square()
    # Return the shifted sqrt power spectrum - first average over spatial dims, then batch and time
    return torch.fft.fftshift(x_fft.mean(spatial_dims[1:]).mean(0).mean(0).sqrt())


def plot_power_spectrum_by_field(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    metadata: WellMetadata,
    output_dir: str,
    epoch_number: int = 0,
):
    """
    Plot the power spectrum of the input tensor x and y.

    Args:
        x: Predicted tensor
        y: Target tensor
        metadata: Metadata object associated with dset
        output_dir: Directory to save the plots
        epoch_number: Current epoch number
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
        subdir = f"{output_dir}/{metadata.dataset_name}/{title}"
        os.makedirs(subdir, exist_ok=True)
        # Save to disk
        plt.savefig(f"{subdir}/epoch{epoch_number}.png")
        np.save(
            f"{subdir}/epoch{epoch_number}_x.npy",
            np_x_fft,
        )
        np.save(
            f"{subdir}/epoch{epoch_number}_y.npy",
            np_y_ftt,
        )
        np.save(
            f"{subdir}/epoch{epoch_number}_res.npy",
            np_res_ftt,
        )
        plt.close()
    return dict()  # Keeping to avoid breaking downstream code


def plot_all_time_metrics(
    time_logs: dict,
    metadata: WellMetadata,
    output_dir: str,
    epoch_number: int = 0,
):
    """Plot loss over time for all time metrics.

    Args:
        time_logs: Dict of time metrics
        metadata: Metadata object associated with dset
        output_dir: Directory to save the plots
        epoch_number: Current epoch number
    """
    os.makedirs(
        f"{output_dir}/{metadata.dataset_name}/rollout_losses/epoch_{epoch_number}",
        exist_ok=True,
    )
    for k, v in time_logs.items():
        v = np.array(v)
        title = k.split("/")[-1]
        np.save(
            f"{output_dir}/{metadata.dataset_name}/rollout_losses/epoch_{epoch_number}/{title}.npy",
            v,
        )
    return dict()  # Keeping to avoid breaking downstream code


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.

    Taken from user Matthias at:
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    from mpl_toolkits import axes_grid1

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def make_video(
    predicted_images: torch.Tensor,
    true_images: torch.Tensor,
    metadata: WellMetadata,
    output_dir: str,
    epoch_number: int = 0,
):
    """Make a video of the rollout comparison.

    Predicted/true are 2/3D channels last tensors.
    """
    field_names = flatten_field_names(metadata, include_constants=False)
    dset_name = metadata.dataset_name
    ndims = metadata.n_spatial_dims
    if ndims == 3:
        # Slice the data along the middle of the last axis
        true_images = true_images[..., true_images.shape[-1] // 2, :]
        predicted_images = predicted_images[..., predicted_images.shape[-1] // 2, :]

    # TODO - Eventually just add the grid info to the metadata. This is currently very fragile
    # and probably wrong for external data.
    grid_type = metadata.grid_type
    if grid_type == "cartesian":
        coords = ["x", "y", "z"][:ndims]
    elif "spher" in grid_type and ndims == 2:
        coords = ["theta", "phi"]
    elif "spher" in grid_type and ndims == 3:
        coords = ["r" "theta", "phi"]
    else:
        # Just default to x, y, z since throwing an error here is going to be annoying
        coords = ["x", "y", "z"][:ndims]
    # Calculate the error
    error_images = (true_images - predicted_images).abs()

    if isinstance(predicted_images, torch.Tensor):
        predicted_images = predicted_images.cpu().numpy()
    if isinstance(true_images, torch.Tensor):
        true_images = true_images.cpu().numpy()
    if isinstance(error_images, torch.Tensor):
        error_images = error_images.cpu().numpy()

    # Calculate percentiles for normalization
    vmaxes, vmins = [], []
    emaxes, emins = [], []
    for i in range(len(field_names)):
        vmaxes.append(np.nanpercentile(true_images[..., i].flatten(), 99))
        vmins.append(np.nanpercentile(true_images[..., i].flatten(), 1))
        emaxes.append(np.nanpercentile(error_images[..., i].flatten(), 99.99))
        emins.append(np.nanpercentile(error_images[..., i].flatten(), 0.01))
    h, w = metadata.spatial_resolution[:2]
    fig, axes = plt.subplots(
        3,
        len(field_names),
        layout="constrained",
        # Scale chosen empirically based on what looked good in our 2D data
        figsize=(3 + 4.5 * len(field_names) * min(1, w / h), 2 + 8 * min(1, h / w)),
        sharex=True,
        sharey=True,
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0.01, wspace=0.15)
    if len(field_names) == 1:
        axes = axes[:, np.newaxis]
    suptitle = plt.suptitle(f"{dset_name} - Rollout Comparison")

    # Initialize the plot with the first frame
    ims = []
    for j, field_name in enumerate(field_names):
        axes[0, j].set_title(field_name)
        im = axes[0, j].imshow(
            true_images[0, ..., j],
            cmap="viridis",
            vmax=vmaxes[j],
            vmin=vmins[j],
            origin="lower",
        )
        add_colorbar(im)
        ims.append(im)
        im = axes[1, j].imshow(
            predicted_images[0, ..., j],
            cmap="viridis",
            vmax=vmaxes[j],
            vmin=vmins[j],
            origin="lower",
        )
        add_colorbar(im)
        ims.append(im)
        im = axes[2, j].imshow(
            error_images[0, ..., j],
            cmap="viridis",
            vmax=emaxes[j],
            vmin=emins[j],
            origin="lower",
        )
        add_colorbar(im)
        # axes[0, j].set_xlabel(coords[1])
        # axes[1, j].set_xlabel(coords[1])
        axes[2, j].set_xlabel(coords[1])

        ims.append(im)

        for i in range(3):
            plt.setp(axes[i, j].get_xticklabels(), visible=False)
            plt.setp(axes[i, j].get_yticklabels(), visible=False)
            axes[i, j].tick_params(axis="both", which="both", length=0)

    axes[0, 0].set_ylabel(f"True State\n{coords[0]}")
    axes[1, 0].set_ylabel(f"Predicted State\n{coords[0]}")
    axes[2, 0].set_ylabel(f"Error\n{coords[0]}")

    # # Update function for the animation
    def update(frame):
        for i, array in enumerate([true_images, predicted_images, error_images]):
            for j in range(len(field_names)):
                ims[i + j * 3].set_array(array[frame, ..., j])
        suptitle.set_text(f"{dset_name} - Frame {frame}")
        return ims

    # # Create the animation
    anim = FuncAnimation(
        fig, update, frames=range(true_images.shape[0]), interval=200, blit=True
    )

    # Save the animation
    write_path = f"{output_dir}/{metadata.dataset_name}/rollout_video"
    os.makedirs(write_path, exist_ok=True)
    anim.save(
        f"{write_path}/epoch{epoch_number}_{dset_name}.mp4",
        writer="ffmpeg",
        fps=max(1, int(predicted_images.shape[0] / 10)),
    )
    plt.close()
    return dict()  # Keeping to avoid breaking downstream code
