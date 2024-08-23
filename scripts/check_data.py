import argparse
import multiprocessing as mp
import os
from typing import Any, Dict, List

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


class ProblemReport:
    def __init__(self, filename):
        self.filename = filename
        self.boundary_issues = ()
        self.spatial_issue = False
        self.constant_frames = {}
        self.nan_frames = {}
        self.field_averages = {}
        self.statistics = {}

    def set_boundary_issue(self, old_value, new_value):
        self.boundary_issues = (old_value, new_value)

    def set_spatial_issue(self):
        self.spatial_issue = True

    def set_constant_frame_issue(self, field: str, trajectory: int, time_step: int):
        if field in self.constant_frames:
            self.constant_frames[field].append((trajectory, time_step))
        else:
            self.constant_frames[field] = [(trajectory, time_step)]

    def set_nan_frame_issue(self, field: str, trajectory: int, time_step: int):
        if field in self.nan_frames:
            self.nan_frames[field].append((trajectory, time_step))
        else:
            self.nan_frames[field] = [(trajectory, time_step)]

    def has_issue(self) -> bool:
        return (
            len(self.boundary_issues)
            or self.spatial_issue
            or len(self.constant_frames)
            or len(self.nan_frames)
        )

    def update_field_average(self, field: str, dim: int, values: np.ndarray):
        mean_value = np.nanmean(values)
        if field in self.field_averages:
            if dim in self.field_averages[field]:
                self.field_averages[field][dim].append(mean_value)
            else:
                self.field_averages[field][dim] = [mean_value]
        else:
            self.field_averages[field] = {dim: [mean_value]}

    def compute_statistics(self):
        statistics = {}
        for field, field_dims in self.field_averages.items():
            statistics[field] = {}
            for dim, values in field_dims.items():
                mean = np.mean(values)
                std = np.std(values)
                statistics[field].update({dim: (mean, std)})
        self.statistics = statistics

    def __str__(self) -> str:
        if self.has_issue():
            report = f"{self.filename} has the following issues:\n"
            if self.boundary_issues:
                report += f"Boundary condition must replaced from {self.boundary_issues[0]} to {self.boundary_issues[1]}.\n"
            if self.spatial_issue:
                report += "Spatial dimensions must be modified.\n"
            if self.constant_frames:
                report += "Constant frames detected for (trajectory time_step):"
                for field, problems in self.constant_frames.items():
                    report += f"{field}: {problems} "
                report += "\n"
            if self.nan_frames:
                report += "Frames with NAN values detected for (trajectory time_step):"
                for field, problems in self.nan_frames.items():
                    report += f"{field}: {problems} "
                report += "\n"
        else:
            report = f"{self.filename} has no detected issue\n"
        report += f"Field statistics (mean, std): {self.statistics}\n"
        return report


class WellFileChecker:
    def __init__(self, filename: str, modifiy: bool = False):
        self.filename = filename
        self.report = ProblemReport(self.filename)
        self.modify = modifiy
        self.field_average = {}

    def check_boundary_coditions(self, boundary_conditions):
        sub_keys = list(boundary_conditions.keys())
        for sub_key in sub_keys:
            bc_old = boundary_conditions[sub_key].attrs["bc_type"]
            bc = check_bc(bc_old)
            if self.modify:
                if bc_old != bc:
                    boundary_conditions[sub_key].attrs["bc_type"] = bc
                    temp = boundary_conditions[sub_key].attrs["bc_type"]
                    self.report.set_boundary_issue(bc, temp)

            else:
                if bc_old != bc:
                    self.report.set_boundary_issue(bc_old, bc)

    def check_dimensions(self, dimensions, n_spatial_dims: int):
        if len(dimensions.attrs["spatial_dims"]) != n_spatial_dims:
            self.report.set_spatial_issue()

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
                        for dim, array in enumerate(arrays):
                            if check_nan(array):
                                self.report.set_nan_frame_issue(sub_key, traj, time)
                            elif verify_constant_array(array) and time > 0:
                                self.report.set_constant_frame_issue(
                                    sub_key, traj, time
                                )
                            else:
                                self.report.update_field_average(sub_key, dim, array)

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
        self.report.compute_statistics()
        return str(self.report)


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
        for report in pool.imap_unordered(check_file, files):
            print(report)
