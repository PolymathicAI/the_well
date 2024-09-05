import argparse
import os.path
import shutil

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np


def create_gif(
    time_series: np.ndarray,
    saving_directory: str,
    name_file: str,
    delete_imgs: bool = False,
    normalize: bool = True,
):
    time_series_min = time_series.min()
    time_series_max = time_series.max()
    if time_series.ndim > 3:
        raise ValueError("Error: The time series should be (time, height, width)")
    if not os.path.exists(saving_directory + "/img_for_gif"):
        os.makedirs(saving_directory + "/img_for_gif")
    images = []
    cmap = "magma"  #'RdBu_r' #'viridis'
    for i in range(time_series.shape[0]):
        if normalize:
            plt.imshow(
                time_series[i],
                origin="lower",
                cmap=cmap,
                vmin=time_series_min,
                vmax=time_series_max,
            )
        else:
            plt.imshow(time_series[i], cmap=cmap, origin="lower")
        plt.axis("off")
        plt.savefig(
            saving_directory + f"/img_for_gif/time_series_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        images.append(
            imageio.imread(saving_directory + f"/img_for_gif/time_series_{i}.png")
        )

    imageio.mimsave(saving_directory + "/" + name_file + ".gif", images, duration=0.1)
    if delete_imgs:
        shutil.rmtree(saving_directory + "/img_for_gif")


def get_trajectory(
    file_name: str, field_name: str, trajectory: int, dim=None
) -> np.ndarray:
    with h5py.File(file_name, "r") as file:
        for field_type in ["t0_fields", "t1_fields", "t2_fields"]:
            if field_name in file[field_type].keys():
                field = file[field_type][field_name]
                # Field is expected to be N, T, H, W, (D1, D2)
                if dim:
                    assert (
                        field.shape >= 4 + len(dim)
                    ), f"Dimension should specify the tensor dimension to retrieve in shape {field.shape}"
                    traj = field[trajectory, :, :, :, dim]
                else:
                    traj = field[trajectory, :, :, :]
                return traj
    raise IndexError(f"{field_name} not found in the fields of {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trajectory GIF creator")
    parser.add_argument("input_filename", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("trajectory", type=int)
    parser.add_argument("output_directory", type=str)
    parser.add_argument("output_filename", type=str)
    parser.add_argument("--dimension", nargs="+", type=int, default=None)
    args = parser.parse_args()
    file_name = args.input_filename
    field_name = args.field_name
    trajectory = args.trajectory
    dims = args.dimension
    print(
        f"Retrieve trajectory {trajectory} for field {field_name} in {file_name}. Optional dimension {dims}"
    )
    traj = get_trajectory(file_name, field_name, trajectory, dims)
    print(f"Trajectory of shape {traj.shape} retrieved.")
    output_dir = args.output_directory
    output_filename = args.output_filename
    print(f"Generat GIF to {output_dir}")
    create_gif(traj, output_dir, output_filename)
    print("Done.")
