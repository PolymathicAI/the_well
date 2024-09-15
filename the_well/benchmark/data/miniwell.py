import os
import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from typing import Union
import shutil

from .datasets import GenericWellDataset, well_paths

def create_mini_well(
    dataset: GenericWellDataset,
    output_base_path: str,
    spatial_downsample_factor: int = 2,
    time_downsample_factor: int = 2,
    max_samples: int = 10
):
    dataset_name = dataset.metadata.dataset_name
    output_path = os.path.join(output_base_path, "datasets", dataset_name)
    print(f"Creating mini dataset at: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(output_path, 'data', split)
        os.makedirs(split_path, exist_ok=True)
        print(f"Created directory: {split_path}")
    
    stats_path = os.path.join(output_path, 'stats')
    os.makedirs(stats_path, exist_ok=True)
    print(f"Created stats directory: {stats_path}")

    # Copy normalization files
    for norm_file in ['means.pkl', 'stds.pkl']:
        src = os.path.join(dataset.normalization_path, norm_file)
        dst = os.path.join(stats_path, norm_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied normalization file: {src} -> {dst}")
        else:
            print(f"Warning: Normalization file not found: {src}")

    # Process each file
    for file_idx, file_path in enumerate(tqdm(dataset.files_paths[:max_samples], desc="Processing files")):
        print(f"Processing file {file_idx + 1}/{min(max_samples, len(dataset.files_paths))}: {file_path}")
        with h5py.File(file_path, 'r') as src_file:
            relative_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(dataset.data_path)))
            output_file_path = os.path.join(output_path, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            print(f"Creating mini file: {output_file_path}")
            
            with h5py.File(output_file_path, 'w') as dst_file:
                process_file(src_file, dst_file, spatial_downsample_factor, time_downsample_factor)

    print(f"Mini dataset creation completed at: {output_path}")
    return output_path

def process_file(src_file, dst_file, spatial_downsample_factor, time_downsample_factor):
    print("Processing file attributes")
    for key, value in src_file.attrs.items():
        dst_file.attrs[key] = value
        print(f"Copied attribute: {key}")

    if 'spatial_resolution' in dst_file.attrs:
        old_resolution = dst_file.attrs['spatial_resolution']
        dst_file.attrs['spatial_resolution'] = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )
        print(f"Updated spatial resolution: {old_resolution} -> {dst_file.attrs['spatial_resolution']}")

    for group_name in src_file.keys():
        print(f"Processing group: {group_name}")
        process_group(src_file[group_name], dst_file.create_group(group_name),
                      spatial_downsample_factor, time_downsample_factor)

def process_group(src_group, dst_group, spatial_downsample_factor, time_downsample_factor):
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value
        print(f"Copied group attribute: {key}")

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            print(f"Processing subgroup: {name}")
            process_group(item, dst_group.create_group(name),
                          spatial_downsample_factor, time_downsample_factor)
        elif isinstance(item, h5py.Dataset):
            print(f"Processing dataset: {name}")
            process_dataset(item, dst_group, name,
                            spatial_downsample_factor, time_downsample_factor)

def process_dataset(src_dataset, dst_group, name, spatial_downsample_factor, time_downsample_factor):
    attrs = dict(src_dataset.attrs)
    print(f"Processing dataset: {name}, shape: {src_dataset.shape}")

    if src_dataset.shape == ():
        data = src_dataset[()]
        print(f"Scalar dataset: {name}, value: {data}")
    else:
        data = src_dataset[:]

        if name == 'time':
            original_length = len(data)
            data = data[::time_downsample_factor]
            print(f"Downsampled time: {original_length} -> {len(data)}")
        elif name in ['t0_fields', 't1_fields', 't2_fields']:
            if attrs.get('time_varying', False):
                original_time_steps = data.shape[0]
                data = data[::time_downsample_factor, ...]
                print(f"Downsampled time-varying field: {original_time_steps} -> {data.shape[0]} time steps")

            n_spatial_dims = len(data.shape) - 2
            for i in range(n_spatial_dims):
                original_size = data.shape[i+1]
                slices = [slice(None)] * len(data.shape)
                slices[i + 1] = slice(None, None, spatial_downsample_factor)
                data = data[tuple(slices)]
                print(f"Downsampled spatial dimension {i}: {original_size} -> {data.shape[i+1]}")

    dst_group.create_dataset(name, data=data)
    print(f"Created downsampled dataset: {name}, new shape: {data.shape}")
    
    for key, value in attrs.items():
        dst_group[name].attrs[key] = value
        print(f"Copied dataset attribute: {key}")

    if 'spatial_resolution' in attrs:
        old_resolution = attrs['spatial_resolution']
        new_resolution = tuple(dim // spatial_downsample_factor for dim in old_resolution)
        dst_group[name].attrs['spatial_resolution'] = new_resolution
        print(f"Updated spatial resolution for {name}: {old_resolution} -> {new_resolution}")

def load_mini_well(
    well_base_path: str,
    well_dataset_name: str,
    **kwargs
) -> GenericWellDataset:
    mini_dataset_path = os.path.join(well_base_path, "datasets", well_dataset_name)
    print(f"Loading mini dataset from: {mini_dataset_path}")
    
    dataset = GenericWellDataset(
        well_base_path=well_base_path,
        well_dataset_name=well_dataset_name,
        **kwargs
    )
    print(f"Loaded mini dataset: {dataset.metadata.dataset_name}")
    print(f"Dataset path: {dataset.data_path}")
    print(f"Number of files: {len(dataset.files_paths)}")
    return dataset