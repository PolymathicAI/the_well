import copy
import os
import shutil

import h5py
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .datasets import GenericWellDataset


def create_mini_well(
    dataset: GenericWellDataset,
    output_base_path: str,
    spatial_downsample_factor: int = 4,
    time_downsample_factor: int = 2,
    max_samples: int = 10,
):
    dataset_name = dataset.metadata.dataset_name
    output_path = os.path.join(output_base_path, "datasets", dataset_name)
    os.makedirs(output_path, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_path = os.path.join(output_path, "data", split)
        os.makedirs(split_path, exist_ok=True)

    stats_path = os.path.join(output_path, "stats")
    os.makedirs(stats_path, exist_ok=True)

    for norm_file in ["means.pkl", "stds.pkl"]:
        src = os.path.join(dataset.normalization_path, norm_file)
        dst = os.path.join(stats_path, norm_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Make a copy of the metadata to avoid modifying the original dataset's metadata
    mini_metadata = copy.deepcopy(dataset.metadata)
    mini_metadata.spatial_resolution = tuple(
        dim // spatial_downsample_factor for dim in mini_metadata.spatial_resolution
    )

    for file_path in tqdm(dataset.files_paths[:max_samples], desc="Processing files"):
        with h5py.File(file_path, "r") as src_file:
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
                )

    return mini_metadata


def process_file(
    src_file: h5py.File,
    dst_file: h5py.File,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
):
    for key, value in src_file.attrs.items():
        dst_file.attrs[key] = value

    if "spatial_resolution" in dst_file.attrs:
        old_resolution = dst_file.attrs["spatial_resolution"]
        dst_file.attrs["spatial_resolution"] = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )

    for group_name in src_file.keys():
        process_group(
            src_file[group_name],
            dst_file.create_group(group_name),
            spatial_downsample_factor,
            time_downsample_factor,
        )


def process_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
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
            )
        elif isinstance(item, h5py.Dataset):
            process_dataset(
                item, dst_group, name, spatial_downsample_factor, time_downsample_factor
            )


def process_dataset(
    src_dataset: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
):
    attrs = dict(src_dataset.attrs)

    if src_dataset.shape == ():
        data = src_dataset[()]
    else:
        data = src_dataset[:]

        if name == "time":
            data = data[::time_downsample_factor]
        elif name in ["t0_fields", "t1_fields", "t2_fields"]:
            time_varying: bool = attrs["time_varying"]

            n_non_spatial_dims = {
                "t0_fields": 1,  # sample dimension
                "t1_fields": 2,  # sample and tensor component dimensions
                "t2_fields": 3,  # sample and two tensor component dimensions
            }[name] + (1 if time_varying else 0)

            n_spatial_dims = len(data.shape) - n_non_spatial_dims

            # Gaussian filter and downsample spatial dimensions
            sigma = (
                [0]  # sample
                + ([0] if time_varying else [])  # time
                + [spatial_downsample_factor / 2] * n_spatial_dims  # spatial
            )
            while len(sigma) < len(data.shape):
                sigma.append(0)  # channels

            data = gaussian_filter(data, sigma=sigma)

            # fmt: off
            slices = (
                [slice(None)]  # sample
                + ([slice(None, None, time_downsample_factor)] if time_varying else [])  # time
                + [slice(None, None, spatial_downsample_factor)] * n_spatial_dims  # spatial
            )
            while len(slices) < len(data.shape):
                slices.append(slice(None))  # channels
            # fmt: on

            data = data[tuple(slices)]
        else:
            # TODO: Is this the right behavior?
            # For other datasets, downsample all dimensions except the first by striding
            n_dims = len(data.shape)
            if n_dims > 1:
                time_varying: bool = attrs["time_varying"]
                # fmt: off
                slices = (
                    [slice(None)]  # sample
                    + ([slice(None, None, time_downsample_factor)] if time_varying else [])  # time
                    + [slice(None, None, spatial_downsample_factor)] * (n_dims - 1 - int(time_varying))  # spatial
                )
                # fmt: on
                data = data[tuple(slices)]

    dst_group.create_dataset(name, data=data)

    for key, value in attrs.items():
        dst_group[name].attrs[key] = value

    if "spatial_resolution" in attrs:
        old_resolution = attrs["spatial_resolution"]
        new_resolution = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )
        dst_group[name].attrs["spatial_resolution"] = new_resolution
