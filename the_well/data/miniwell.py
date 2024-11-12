import copy
import os
import re
import shutil

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .datasets import WellDataset


def create_mini_well(
    dataset: WellDataset,
    output_base_path: str,
    spatial_downsample_factor: int = 4,
    time_downsample_factor: int = 2,
    max_trajectories: int = 100,
    split: str = "train",
    time_fraction: float = 1.0,
):
    dataset_name = dataset.well_dataset_name

    output_path = os.path.join(output_base_path, "datasets", dataset_name)
    os.makedirs(output_path, exist_ok=True)

    split_path = os.path.join(output_path, "data", split)
    os.makedirs(split_path, exist_ok=True)

    shutil.copy2(dataset.normalization_path, output_path)

    mini_metadata = copy.deepcopy(dataset.metadata)
    mini_metadata.spatial_resolution = tuple(
        dim // spatial_downsample_factor for dim in mini_metadata.spatial_resolution
    )

    # Update n_steps_per_simulation to reflect time downsampling
    mini_metadata.n_steps_per_simulation = [
        steps // time_downsample_factor
        for steps in mini_metadata.n_steps_per_simulation
    ]

    # Update sample_shapes to reflect new spatial resolution and number of fields
    mini_metadata.sample_shapes["input_fields"] = [
        *mini_metadata.spatial_resolution,
        mini_metadata.n_fields,
    ]
    mini_metadata.sample_shapes["output_fields"] = [
        *mini_metadata.spatial_resolution,
        mini_metadata.n_fields,
    ]

    # Update space_grid in sample_shapes to reflect new spatial resolution and spatial dimensions
    mini_metadata.sample_shapes["space_grid"] = [
        *mini_metadata.spatial_resolution,
        mini_metadata.n_spatial_dims,
    ]

    total_trajectories = 0
    split_files = [f for f in dataset.files_paths if split in f]
    for file_path in tqdm(split_files, desc=f"Processing {split} files"):
        if total_trajectories >= max_trajectories:
            break

        with h5py.File(file_path, "r") as src_file:
            num_trajectories_in_file = src_file.attrs["n_trajectories"]
            remaining_trajectories = max_trajectories - total_trajectories
            trajectories_to_process = min(
                num_trajectories_in_file, remaining_trajectories
            )

            relative_path = os.path.relpath(
                file_path, os.path.dirname(os.path.dirname(dataset.data_path))
            )
            output_file_path = os.path.join(output_path, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            with h5py.File(output_file_path, "w") as dst_file:
                process_file(
                    src_file,
                    dst_file,
                    spatial_downsample_factor,
                    time_downsample_factor,
                    time_fraction,
                    trajectories_to_process,
                )

            total_trajectories += trajectories_to_process

    return mini_metadata


def process_file(
    src_file: h5py.File,
    dst_file: h5py.File,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
    time_fraction: float,
    trajectories_to_process: int,
):
    for key, value in src_file.attrs.items():
        dst_file.attrs[key] = value

    if "spatial_resolution" in dst_file.attrs:
        old_resolution = dst_file.attrs["spatial_resolution"]
        dst_file.attrs["spatial_resolution"] = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )

    # Update spatial grid size if it exists
    if "spatial_grid_size" in dst_file.attrs:
        old_grid_size = dst_file.attrs["spatial_grid_size"]
        dst_file.attrs["spatial_grid_size"] = tuple(
            dim // spatial_downsample_factor for dim in old_grid_size
        )

    for group_name in src_file.keys():
        process_group(
            src_file[group_name],
            dst_file.create_group(group_name),
            spatial_downsample_factor,
            time_downsample_factor,
            time_fraction,
            trajectories_to_process,
            full_name=group_name,
        )

    # Update n_trajectories for this specific file
    dst_file.attrs["n_trajectories"] = trajectories_to_process


def process_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
    time_fraction: float,
    trajectories_to_process: int,
    full_name: str,
):
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            process_group(
                item,
                dst_group.create_group(name),
                spatial_downsample_factor,
                time_downsample_factor,
                time_fraction,
                trajectories_to_process,
                full_name=full_name + "/" + name,
            )
        elif isinstance(item, h5py.Dataset):
            process_dataset(
                item,
                dst_group,
                name=name,
                full_name=full_name + "/" + name,
                spatial_downsample_factor=spatial_downsample_factor,
                time_downsample_factor=time_downsample_factor,
                time_fraction=time_fraction,
                trajectories_to_process=trajectories_to_process,
            )


def process_dataset(
    src_dataset: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    full_name: str,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
    time_fraction: float,
    trajectories_to_process: int,
):
    attrs = dict(src_dataset.attrs)

    if src_dataset.shape == ():
        data = src_dataset[()]
    else:
        data = src_dataset[:]

        downsample_kws = dict(
            spatial_downsample_factor=spatial_downsample_factor,
            time_downsample_factor=time_downsample_factor,
            time_fraction=time_fraction,
        )

        if (
            re.match(r"t[012]_fields.*", full_name)
            or full_name == "additional_information/g_contravariant"
        ):
            if attrs["sample_varying"]:
                data = data[:trajectories_to_process, ...]

            if full_name.startswith("t0_fields"):
                n_tensor_dims = 0
            elif full_name.startswith("t1_fields"):
                n_tensor_dims = 1
            elif full_name.startswith("t2_fields"):
                n_tensor_dims = 2
            elif full_name == "additional_information/g_contravariant":
                n_tensor_dims = 2
            else:
                raise ValueError(f"Unknown dataset {full_name}")

            data = downsample_field(
                data,
                time_varying=attrs["time_varying"],
                spatial_filtering=True,
                n_batch_dims=int(attrs["sample_varying"]),
                n_tensor_dims=n_tensor_dims,
                **downsample_kws,
            )
        elif re.match(r"dimensions/time", full_name):
            data = downsample_field(
                data,
                time_varying=True,
                spatial_filtering=False,
                n_batch_dims=int(attrs["sample_varying"]),
                n_tensor_dims=0,
                **downsample_kws,
            )
        elif (
            re.match(r"dimensions/([xyz]|phi|theta|log_r)", full_name)
            and len(data.shape) == 1
        ):
            data = downsample_field(
                data,
                time_varying=False,
                spatial_filtering=False,
                n_batch_dims=0,
                n_tensor_dims=0,
                **downsample_kws,
            )
        elif (
            re.match(
                r"boundary_conditions/([xyz]|phi|theta|log_r)_(periodic|open|wall|wall_noslip|wall_dirichlet|open_neumann)/mask",
                full_name,
            )
            and len(data.shape) == 1
        ):
            num_elements = data.shape[0] // spatial_downsample_factor
            # We assume that the first and last elements are the only "data" in the mask
            data = np.array(
                [data[0]] + [False] * (num_elements - 2) + [data[-1]], dtype=bool
            )
        elif (
            re.match(r"boundary_conditions/xy_wall/mask", full_name)
            and len(data.shape) == 2
        ):
            data = downsample_field(
                data,
                time_varying=False,
                spatial_filtering=False,
                n_batch_dims=0,
                n_tensor_dims=0,
                **downsample_kws,
            )
        else:
            # Print info about the dataset before raising an error
            first_10_elements = data.ravel()[:10]
            raise NotImplementedError(
                f"Dataset {full_name} not implemented, with shape {data.shape}, type {data.dtype}, "
                f"attrs {attrs}, and first 10 elements {first_10_elements}"
            )

    dst_group.create_dataset(name, data=data)

    for key, value in attrs.items():
        dst_group[name].attrs[key] = value

    if "spatial_resolution" in attrs:
        old_resolution = attrs["spatial_resolution"]
        new_resolution = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )
        dst_group[name].attrs["spatial_resolution"] = new_resolution


def downsample_field(
    data,
    *,
    time_varying: bool,
    spatial_filtering: bool,
    n_batch_dims: int,
    n_tensor_dims: int,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
    time_fraction: float,
):
    n_time_dims = 1 if time_varying else 0
    n_spatial_dims = len(data.shape) - n_batch_dims - n_tensor_dims - n_time_dims

    # Compute the new time length before downsampling
    new_time_length = (
        int(data.shape[n_batch_dims] * time_fraction) if time_varying else None
    )

    # First, do time downsampling, so we can save some compute
    time_slices = (
        [slice(None)] * n_batch_dims
        + [slice(None, new_time_length, time_downsample_factor)] * n_time_dims
        + [slice(None)] * n_spatial_dims
        + [slice(None)] * n_tensor_dims
    )
    data = data[tuple(time_slices)]

    if spatial_filtering:
        spatial_sigma = (spatial_downsample_factor - 1) / 2
        sigma = (
            [0] * n_batch_dims
            + [0] * n_time_dims
            + [spatial_sigma] * n_spatial_dims
            + [0] * n_tensor_dims
        )

        # TODO: Use a better filtering method. scipy does not support
        # different filtering modes per axis, meaning we cannot support
        # the different `bc_type` options. So, for simplicity, we just
        # use the nearest neighbor mode here.
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        data = gaussian_filter(data, sigma=sigma, mode="nearest")

    # Finally, do spatial downsampling
    spatial_slices = (
        [slice(None)] * n_batch_dims
        + [slice(None)] * n_time_dims
        + [slice(None, None, spatial_downsample_factor)] * n_spatial_dims
        + [slice(None)] * n_tensor_dims
    )

    return data[tuple(spatial_slices)]
