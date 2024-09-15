import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import shutil

from .datasets import GenericWellDataset, GenericWellMetadata

def create_mini_well(
    dataset: GenericWellDataset,
    output_base_path: str,
    spatial_downsample_factor: int = 2,
    time_downsample_factor: int = 2,
    max_samples: int = 10
):
    dataset_name = dataset.metadata.dataset_name
    output_path = os.path.join(output_base_path, "datasets", dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(output_path, 'data', split)
        os.makedirs(split_path, exist_ok=True)
    
    stats_path = os.path.join(output_path, 'stats')
    os.makedirs(stats_path, exist_ok=True)

    for norm_file in ['means.pkl', 'stds.pkl']:
        src = os.path.join(dataset.normalization_path, norm_file)
        dst = os.path.join(stats_path, norm_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    mini_metadata = dataset.metadata
    mini_metadata.spatial_resolution = tuple(dim // spatial_downsample_factor for dim in mini_metadata.spatial_resolution)

    for file_idx, file_path in enumerate(tqdm(dataset.files_paths[:max_samples], desc="Processing files")):
        with h5py.File(file_path, 'r') as src_file:
            relative_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(dataset.data_path)))
            output_file_path = os.path.join(output_path, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            with h5py.File(output_file_path, 'w') as dst_file:
                process_file(src_file, dst_file, spatial_downsample_factor, time_downsample_factor)

    return output_path, mini_metadata

def process_file(src_file, dst_file, spatial_downsample_factor, time_downsample_factor):
    for key, value in src_file.attrs.items():
        dst_file.attrs[key] = value

    if 'spatial_resolution' in dst_file.attrs:
        old_resolution = dst_file.attrs['spatial_resolution']
        dst_file.attrs['spatial_resolution'] = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )

    for group_name in src_file.keys():
        process_group(src_file[group_name], dst_file.create_group(group_name),
                      spatial_downsample_factor, time_downsample_factor)

def process_group(src_group, dst_group, spatial_downsample_factor, time_downsample_factor):
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            process_group(item, dst_group.create_group(name),
                          spatial_downsample_factor, time_downsample_factor)
        elif isinstance(item, h5py.Dataset):
            process_dataset(item, dst_group, name,
                            spatial_downsample_factor, time_downsample_factor)

def process_dataset(src_dataset, dst_group, name, spatial_downsample_factor, time_downsample_factor):
    attrs = dict(src_dataset.attrs)

    if src_dataset.shape == ():
        data = src_dataset[()]
    else:
        data = src_dataset[:]

        if name == 'time':
            data = data[::time_downsample_factor]
        elif name in ['t0_fields', 't1_fields', 't2_fields']:
            # Downsample time
            data = data[:, ::time_downsample_factor, ...]

            # Gaussian filter and downsample spatial dimensions
            n_spatial_dims = len(data.shape) - 3  # Exclude sample, time, and channel dimensions
            spatial_axes = tuple(range(2, 2 + n_spatial_dims))
            sigma = [0] * 2 + [spatial_downsample_factor / 2] * n_spatial_dims + [0] * (len(data.shape) - 2 - n_spatial_dims)
            data = gaussian_filter(data, sigma=sigma)
            
            slices = [slice(None)] * 2 + [slice(None, None, spatial_downsample_factor)] * n_spatial_dims + [slice(None)] * (len(data.shape) - 2 - n_spatial_dims)
            data = data[tuple(slices)]
        else:
            # For other datasets, apply Gaussian filter and downsample all dimensions except the first
            n_dims = len(data.shape)
            if n_dims > 1:
                sigma = [0] + [spatial_downsample_factor / 2] * (n_dims - 1)
                data = gaussian_filter(data, sigma=sigma)
                
                slices = [slice(None)] + [slice(None, None, spatial_downsample_factor)] * (n_dims - 1)
                data = data[tuple(slices)]

    dst_group.create_dataset(name, data=data)
    
    for key, value in attrs.items():
        dst_group[name].attrs[key] = value

    if 'spatial_resolution' in attrs:
        old_resolution = attrs['spatial_resolution']
        new_resolution = tuple(dim // spatial_downsample_factor for dim in old_resolution)
        dst_group[name].attrs['spatial_resolution'] = new_resolution

def load_mini_well(
    well_base_path: str,
    well_dataset_name: str,
    mini_metadata: GenericWellMetadata,
    **kwargs
) -> GenericWellDataset:
    mini_dataset_path = os.path.join(well_base_path, "datasets", well_dataset_name)
    
    dataset = GenericWellDataset(
        well_base_path=well_base_path,
        well_dataset_name=well_dataset_name,
        **kwargs
    )
    dataset.metadata = mini_metadata
    return dataset