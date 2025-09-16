import argparse
import multiprocessing as mp
import os
from typing import List

import h5py
import numpy as np

from the_well.data.utils import WELL_DATASETS


def check_bc(bc: str) -> str:
    return bc.upper()


def check_constant(f: np.ndarray) -> bool:
    return np.allclose(f, f.item(0))


def check_nan(f: np.ndarray) -> bool:
    return np.isnan(f).any()


def check_close_previous_frame(f: np.ndarray, prev_f: np.ndarray) -> bool:
    return np.allclose(f, prev_f)


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
        self.close_frames = {}
        self.nan_frames = {}
        self.field_averages = {}
        self.means = {}
        self.stds = {}
        self.outliers = {}

    def set_boundary_issue(self, old_value, new_value):
        self.boundary_issues = (old_value, new_value)

    def set_spatial_issue(self):
        self.spatial_issue = True

    def set_constant_frame_issue(self, field: str, trajectory: int, time_step: int):
        if trajectory in self.constant_frames:
            if field in self.constant_frames[trajectory]:
                self.constant_frames[trajectory][field].append(time_step)
            else:
                self.constant_frames[trajectory].update({field: [time_step]})
        else:
            self.constant_frames.update({trajectory: {field: [time_step]}})

    def set_close_to_previous(self, field: str, trajectory: int, time_step: int):
        if trajectory in self.close_frames:
            if field in self.close_frames[trajectory]:
                self.close_frames[trajectory][field].append(time_step)
            else:
                self.close_frames[trajectory].update({field: [time_step]})
        else:
            self.close_frames.update({trajectory: {field: [time_step]}})

    def set_nan_frame_issue(self, field: str, trajectory: int, time_step: int):
        if trajectory in self.nan_frames:
            if field in self.nan_frames[trajectory]:
                self.nan_frames[trajectory][field].append(time_step)
            else:
                self.nan_frames[trajectory].update({field: [time_step]})
        else:
            self.nan_frames.update({trajectory: {field: [time_step]}})

    def has_issue(self) -> bool:
        return (
            len(self.boundary_issues)
            or self.spatial_issue
            or len(self.constant_frames)
            or len(self.nan_frames)
        )

    def update_field_average(
        self, trajectory: int, field: str, dim: int, values: np.ndarray
    ):
        mean_value = np.nanmean(values)
        if trajectory in self.field_averages:
            if field in self.field_averages[trajectory]:
                if dim in self.field_averages[trajectory][field]:
                    self.field_averages[trajectory][field][dim].append(mean_value)
                else:
                    self.field_averages[trajectory][field][dim] = [mean_value]
            else:
                self.field_averages[trajectory][field] = {dim: [mean_value]}
        else:
            self.field_averages[trajectory] = {field: {dim: [mean_value]}}

    def compute_statistics(self):
        means = {}
        stds = {}
        for trajectory, trajectory_means in self.field_averages.items():
            means[trajectory] = {field: {} for field in trajectory_means.keys()}
            stds[trajectory] = {field: {} for field in trajectory_means.keys()}
            for field, field_means in trajectory_means.items():
                for dim, dim_means in field_means.items():
                    mean = np.mean(dim_means)
                    std = np.std(dim_means)
                    means[trajectory][field].update({dim: mean})
                    stds[trajectory][field].update({dim: std})

        self.means = means
        self.stds = stds

    def find_outliers(self, sigma_factor: int = 5):
        outliers = {}

        for trajectory, trajectory_means in self.field_averages.items():
            trajectory_ouliers = {}
            for field, field_means in trajectory_means.items():
                field_outliers = {}
                for dim, dim_means in field_means.items():
                    mean = self.means[trajectory][field][dim]
                    std = self.stds[trajectory][field][dim]
                    dim_outliers = []
                    for step, value in enumerate(dim_means):
                        if np.sqrt((value - mean) ** 2) >= sigma_factor * std:
                            dim_outliers.append(step)
                    field_outliers.update({dim: dim_outliers})
                if field_outliers:
                    trajectory_ouliers.update({field: field_outliers})
            if trajectory_ouliers:
                outliers.update({trajectory: trajectory_ouliers})

        self.outliers = outliers

    def __str__(self) -> str:
        if self.has_issue():
            report = f"{self.filename} has the following issues:\n"
            if self.boundary_issues:
                report += f"Boundary condition must replaced from {self.boundary_issues[0]} to {self.boundary_issues[1]}.\n"
            if self.spatial_issue:
                report += "Spatial dimensions must be modified.\n"
            if self.constant_frames:
                report += "Constant frames detected:\n"
                for trajectory, trajectory_issues in self.constant_frames.items():
                    report += f"Trajectory {trajectory} has constant frames: "
                    for field, constant_frames in trajectory_issues.items():
                        report += f"{field}:{len(constant_frames)}:{constant_frames} "
                    report += "\n"
            if self.close_frames:
                report += "Time frames close to previous ones detected:"
                for trajectory, trajectory_issues in self.close_frames.items():
                    report += f"Trajectory {trajectory} has close subsequent frames: "
                    for field, close_frames in trajectory_issues.items():
                        report += f"{field}:{len(close_frames)}:{close_frames} "
                    report += "\n"
            if self.nan_frames:
                report += "Frames with NAN values detected:"
                for trajectory, trajectory_issues in self.nan_frames.items():
                    report += f"Trajectory {trajectory} has NAN value frames: "
                    for field, nan_frames in trajectory_issues.items():
                        report += f"{field}:{len(nan_frames)}:{nan_frames} "
                    report += "\n"
            if self.outliers:
                for trajectory, trajectory_outliers in self.outliers.items():
                    report += f"Trajectory {trajectory} has outliers: "
                    for field, field_outliers in trajectory_outliers.items():
                        report += f"{field}: "
                        for dim, dim_outliers in field_outliers.items():
                            if dim_outliers:
                                report += f"{dim}: steps {dim_outliers} "
                    report += "\n"
        else:
            report = f"{self.filename} has no detected issue\n"
        report += f"Field statistics means: {self.means} +/- {self.stds}\n"
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

    def check_fields(self, fields, n_spatial_dims: int):
        sub_keys = list(fields.keys())
        for sub_key in sub_keys:
            if fields[sub_key].attrs["time_varying"]:
                n_traj = fields[sub_key].shape[0]
                n_time = fields[sub_key].shape[1]
                spatial_dimensions = fields[sub_key].shape[2 : 2 + n_spatial_dims]
                for traj in range(n_traj):
                    prev_arrays = None
                    for time in range(n_time):
                        arrays = fields[sub_key][traj, time, ...]
                        arrays = arrays.reshape(*spatial_dimensions, -1)
                        arrays = np.moveaxis(arrays, -1, 0)
                        for dim, array in enumerate(arrays):
                            if check_nan(array):
                                self.report.set_nan_frame_issue(sub_key, traj, time)
                            elif check_constant(array) and time > 0:
                                self.report.set_constant_frame_issue(
                                    sub_key, traj, time
                                )
                            else:
                                self.report.update_field_average(
                                    traj, sub_key, dim, array
                                )
                            if (prev_arrays is not None) and check_close_previous_frame(
                                array, prev_arrays[dim]
                            ):
                                self.report.set_close_to_previous(sub_key, traj, time)
                        prev_arrays = arrays

    def check(self):
        with h5py.File(self.filename, "r") as file:
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
                    self.check_fields(file[key], n_spatial_dimensions)
            self.report.compute_statistics()
            self.report.find_outliers()
            return str(self.report)


def list_files(data_register: List[str]):
    folder = ["train", "test", "valid"]
    for data in data_register:
        for f in folder:
            file_path = f"{data}/data/{f}/"
            for file_name in os.listdir(file_path):
                full_path = os.path.join(file_path, file_name)
                yield full_path


def check_file(filename: str):
    file_checker = WellFileChecker(filename)
    report = file_checker.check()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Sanity Checker")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Path to the Well directory where are stored the data",
    )
    parser.add_argument("-n", "--nproc", type=int, default=1)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=WELL_DATASETS,
        choices=WELL_DATASETS,
        help="Name of the dataset to check",
    )
    args = parser.parse_args()
    data_dir = args.dir
    nproc = args.nproc
    datasets_to_check = args.datasets
    data_register = [os.path.join(data_dir, dataset) for dataset in datasets_to_check]
    files = list(list_files(data_register))
    print(f"{len(files)} to check.")
    with mp.Pool(nproc) as pool:
        for report in pool.imap_unordered(check_file, files, chunksize=16):
            print(report, flush=True)
