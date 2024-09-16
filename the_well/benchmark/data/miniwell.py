import copy
import os
import shutil
import warnings

import h5py
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .datasets import GenericWellDataset


def create_mini_well(
    dataset: GenericWellDataset,
    output_base_path: str,
    spatial_downsample_factor: int = 4,
    time_downsample_factor: int = 2,
    max_files: int = 10,
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

    for file_path in tqdm(dataset.files_paths[:max_files], desc="Processing files"):
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
            n_tensor_dims = {
                "t0_fields": 0,  # sample dimension
                "t1_fields": 1,  # sample and tensor component dimensions
                "t2_fields": 2,  # sample and two tensor component dimensions
            }[name]
            data = downsample_field(
                data,
                time_varying=attrs["time_varying"],
                spatial_filtering=True,
                n_tensor_dims=n_tensor_dims,
                spatial_downsample_factor=spatial_downsample_factor,
                time_downsample_factor=time_downsample_factor,
            )
        elif len(data.shape) > 1:
            if "time_varying" not in attrs:
                warnings.warn(
                    f"Dataset {name} has no time_varying attribute. Assuming time_varying=False."
                )
            data = downsample_field(
                data,
                time_varying=(
                    attrs["time_varying"] if "time_varying" in attrs else False
                ),
                spatial_filtering=False,  # No spatial filtering!
                n_tensor_dims=0,  # Assume everything is a spatial dimension past batch and time
                spatial_downsample_factor=spatial_downsample_factor,
                time_downsample_factor=time_downsample_factor,
            )
        else:
            pass

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
    n_batch_dims: int = 1,
    n_tensor_dims: int,
    spatial_downsample_factor: int,
    time_downsample_factor: int,
):
    n_time_dims = 1 if time_varying else 0
    n_spatial_dims = len(data.shape) - n_batch_dims - n_tensor_dims - n_time_dims

    # First, do time downsampling, so we can save some compute
    time_slices = (
        [slice(None)] * n_batch_dims
        + [slice(None, None, time_downsample_factor)] * n_time_dims
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
