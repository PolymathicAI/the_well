"""Permanently remove a problematic trajectory from a HDF5 file containing data formatted for the Well"""

import argparse

import h5py
import numpy as np


def overwrite_field(file: h5py.File, field, data: np.array):
    attrs = dict(field.attrs)
    field_name = field.name
    del file[field_name]
    dset = file.create_dataset(field_name, data=data)
    # Copy dataset attrs
    for key, val in attrs.items():
        dset.attrs[key] = val


def remove_trajectory(filename: str, trajectory_indices: list[int]):
    with h5py.File(filename, "r+") as file:
        n_traj = file.attrs["n_trajectories"]
        if n_traj <= max(trajectory_indices):
            raise IndexError(
                f"File {filename} has only {n_traj} but request to remove {trajectory_indices}"
            )
        for field_type in ["t0_fields", "t1_fields", "t2_fields"]:
            for field_name in file[field_type].keys():
                field = file[field_type][field_name]
                field_data = np.array(field)
                # Remove trajectories from data
                clean_field_data = np.delete(field_data, trajectory_indices, axis=0)
                overwrite_field(file[field_type], field, clean_field_data)
        # Update number of trajectories
        file.attrs["n_trajectories"] = n_traj - len(trajectory_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trajectory eraser")
    parser.add_argument("-f", "--filename", type=str, help="HDF5 filename")
    parser.add_argument(
        "-t",
        "--trajectories",
        type=int,
        nargs="+",
        help="Indices of trajectories to erase",
    )
    args = parser.parse_args()
    filename = args.filename
    indices = args.trajectories
    remove_trajectory(filename, indices)
