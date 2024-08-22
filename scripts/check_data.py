import argparse
import multiprocessing as mp
import os
from typing import List

import h5py
import numpy as np

data_register = [
    "/mnt/home/polymathic/ceph/the_well/datasets/acoustic_scattering_discontinuous_2d/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/acoustic_scattering_inclusions_2d/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/acoustic_scattering_maze_2d/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/active_matter/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/convective_envelope_rsg/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/euler_multi_quadrants_openBC/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/euler_multi_quadrants_periodicBC/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/helmholtz_staircase/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/MHD_64/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/MHD_256/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/pattern_formation/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/planetswe/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/post_neutron_star_merger/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/rayleigh_benard/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/rayleigh_taylor_instability/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/shear_flow/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/supernova_explosion_64/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/supernova_explosion_128/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/turbulence_gravity_cooling/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/turbulent_radiative_layer_2D/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/turbulent_radiative_layer_3D/data",
    "/mnt/home/polymathic/ceph/the_well/datasets/viscoelastic_instability/data",
]


def check_bc(bc: str) -> str:
    return bc.upper()


def verify_constant_array(f: np.ndarray) -> bool:
    return np.all(f == f[0])


def check_nan(f: np.ndarray) -> bool:
    return np.isnan(f).any()


def detect_outlier_pixels(image_array: np.ndarray, threshold: int = 10):
    mean = np.mean(image_array)
    std = np.std(image_array)

    # Calculate the absolute difference from the mean
    diff_from_mean = np.abs(image_array - mean)

    # Detect outliers
    outliers = diff_from_mean > threshold * std

    # Get the coordinates of the outliers
    outlier_positions = np.argwhere(outliers)

    # Check if there are any outliers
    has_outliers = outlier_positions.size > 0

    return outliers, outlier_positions, has_outliers


class WellFileChecker:
    def __init__(self, filename: str, modifiy: bool = False):
        self.filename = filename
        self.modify = modifiy

    def check_boundary_coditions(self, boundary_conditions):
        sub_keys = list(boundary_conditions.keys())
        for sub_key in sub_keys:
            bc_old = boundary_conditions[sub_key].attrs["bc_type"]
            bc = check_bc(bc_old)
            if self.modify:
                if bc_old != bc:
                    boundary_conditions[sub_key].attrs["bc_type"] = bc
                    temp = boundary_conditions[sub_key].attrs["bc_type"]
                    print(f"Modified {self.filename} bc_type from {bc_old} to {temp}")
            else:
                if bc_old != bc:
                    print(
                        f"need to modify {self.filename} bc_type from {bc_old} to {bc}"
                    )

    def check_dimensions(self, dimensions, n_spatial_dims: int):
        if len(dimensions.attrs["spatial_dims"]) != n_spatial_dims:
            print(f"need to modify {self.filename} spatial_dims")

    def check_scalars(self, scalars):
        pass

    def check_fields(self, fields):
        sub_keys = list(fields.keys())
        for sub_key in sub_keys:
            n_traj = fields[sub_key].shape[0]
            n_time = fields[sub_key].shape[1]
            spatial_dimensions = fields[sub_key].shape[2:4]
            if fields[sub_key].attrs["time_varying"] == True:
                for traj in range(n_traj):
                    for time in range(n_time):
                        arrays = fields[sub_key][traj, time, ...]
                        arrays = arrays.reshape(*spatial_dimensions, -1)
                        arrays = np.moveaxis(arrays, -1, 0)
                        for array in arrays:
                            if verify_constant_array(array) and time > 0:
                                print(
                                    f"Modify for CONSTANT ARRAY {array} folder, {self.filename}, {sub_key} trajectory {traj} time {time}"
                                )
                            if check_nan(array):
                                print(
                                    f"NaNs presents in  {self.filename} {sub_key} trajectory {traj} time {time}"
                                )

    def check(self):
        with h5py.File(self.filename, "r+") as file:
            keys_list = list(file.keys())
            for key in keys_list:
                if key == "boundary_conditions":
                    self.check_boundary_coditions(file[key])
                elif key == "dimensions":
                    n_spatial_dimensions = file.attrs["n_spatial_dims"]
                    self.check_dimensions(file[key], n_spatial_dimensions)
                elif key == "scalars":
                    self.check_scalars(file[key])
                elif "fields" in key:
                    self.check_fields(file[key])


def list_files(data_register: List[str]):
    folder = ["train", "test", "valid"]
    for data in data_register:
        for f in folder:
            file_path = f"{data}/{f}/"
            for file_name in os.listdir(file_path):
                full_path = os.path.join(file_path, file_name)
                yield full_path


def check_file(filename: str):
    file_checker = WellFileChecker(filename, modify)
    file_checker.check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Sanity Checker")
    parser.add_argument("-n", "--nproc", type=int, default=1)
    parser.add_argument("--modify", action="store_true")
    args = parser.parse_args()
    nproc = args.n
    modify = args.modify
    files = list_files(data_register)
    with mp.Pool(nproc) as pool:
        pool.map(check_file, files)
