import argparse
import glob
import os.path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from the_well.data.utils import WELL_DATASETS


def plot_velocity(dataset_dir: str, output_dir: str):
    for dataset_name in tqdm(WELL_DATASETS):
        dataset_path = os.path.join(
            dataset_dir, dataset_name, "data", "valid", "*.hdf5"
        )
        hdf5_files = glob.glob(dataset_path)
        hdf5_file = hdf5_files[0]
        print(dataset_name, hdf5_file)
        with h5py.File(hdf5_file, "r") as f:
            if "velocity" in f["t1_fields"].keys():
                velocity = f["t1_fields"]["velocity"][:]
            else:
                continue

        traj = 0  # Select the trajectory
        if len(velocity.shape) != 5 or velocity.shape[-1] < 2:
            continue
        print(velocity.shape)
        for dim, label in zip([0, 1], ["x", "y"]):
            traj_to_plot = velocity[traj, :, :, :, dim]
            # Field is now of shape (n_timesteps, x, y). Let's do a subplot to plot it at t= 0, t= T/3, t= 2T/3 and t= T:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            T = traj_to_plot.shape[0]
            # Fix colorbar for all subplots:
            vmin = np.min(traj_to_plot)
            vmax = np.max(traj_to_plot)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            for i, t in enumerate([0, T // 3, (2 * T) // 3, T - 1]):
                axs[i].imshow(traj_to_plot[t], cmap="viridis", norm=norm)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            figure_filename = os.path.join(output_dir, f"{dataset_name}_{label}.png")
            fig.savefig(figure_filename, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset metadata generator")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    plot_velocity(dataset_dir, output_dir)
