import glob
import os
from enum import Enum
from typing import Any, Callable, List, Optional

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset

well_paths = {"active_matter": "2D/active_matter"}


# Boundary condition codes
class BoundaryCondition(Enum):
    WALL = 0
    OPEN = 1
    PERIODIC = 2


class GenericWellDataset(Dataset):
    """
    Generic dataset for any Well data. Returns data in B x T x H [x W [x D]] x C format.

    Train/Test/Valid is assumed to occur on a folder level.

    Takes in path to directory of HDF5 files to construct dset.

    Parameters
    ----------
    path :
        Path to directory of HDF5 files, one of path or well_base_path+well_dataset_name
          must be specified
    normalization_path:
        Path to normalization constants - assumed to be in same format as constructed data.
    well_base_path :
        Path to well dataset directory, only used with dataset_name
    well_dataset_name :
        Name of well dataset to load - overrides path if specified
    well_split_name :
        Name of split to load - options are 'train', 'valid', 'test'
    include_filters :
        Only include files whose name contains at least one of these strings
    exclude_filters :
        Exclude any files whose name contains at least one of these strings
    use_normalization:
        Whether to normalize data in the dataset
    include_normalization_in_sample: bool, default=False
        Whether to include normalization constants in the sample
    n_steps_input :
        Number of steps to include in each sample
    n_steps_output :
        Number of steps to include in y
    dt_stride :
        Minimum stride between samples
    max_dt_stride :
        Maximum stride between samples
    flatten_tensors :
        Whether to flatten tensor valued field into channels
    cache_constants :
        Whether to cache all values that do not vary in time or sample
          in memory for faster access
    max_cache_size :
        Maximum numel of constant tensor to cache
    return_grid :
        Whether to return grid coordinates
    boundary_return_type : options=['padding', 'mask', 'exact']
        How to return boundary conditions. Currently only padding supported.
    name_override :
        Override name of dataset (used for more precise logging)
    transforms :
        List of transforms to apply to data
    tensor_transformers :
        List of transforms to apply to tensor fields
    """

    def __init__(
        self,
        path: Optional[str] = None,
        normalization_path: str = "../stats/",
        well_base_path: Optional[str] = None,
        well_dataset_name: Optional[str] = None,
        well_split_name: str = "train",
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        use_normalization: bool = True,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        dt_stride: int = 1,
        max_dt_stride: int = 1,
        flatten_tensors: bool = True,
        cache_constants: bool = True,
        max_cache_size: float = 1e9,
        return_grid: bool = True,
        boundary_return_type: str = "padding",
        name_override: Optional[str] = None,
        transforms: List[Callable] = [],
        tensor_transforms: List[Callable] = [],
    ):
        super().__init__()
        assert path is not None or (
            well_base_path is not None and well_dataset_name is not None
        ), "Must specify path or well_base_path and well_dataset_name"
        if path is not None:
            path = os.path.abspath(path)
            self.data_path = path
            # Note - if the second path is absolute, this op just uses second
            self.normalization_path = os.path.abspath(
                os.path.join(self.data_path, normalization_path)
            )
        else:
            self.data_path = os.path.join(
                well_base_path, well_paths[well_dataset_name], well_split_name
            )
            self.normalization_path = os.path.abspath(
                os.path.join(self.data_path, "../stats/")
            )

        if use_normalization:
            self.means = torch.load(os.path.join(self.normalization_path, "means.pkl"))
            self.stds = torch.load(os.path.join(self.normalization_path, "stds.pkl"))

        # Input checks
        if len(transforms) > 0 or len(tensor_transforms) > 0:
            raise NotImplementedError("Transforms not yet implemented")
        if boundary_return_type not in ["padding"]:
            raise NotImplementedError("Only padding boundary conditions supported")
        if not flatten_tensors:
            raise NotImplementedError("Only flattened tensors supported right now")

        # Copy params
        self.use_normalization = use_normalization
        self.include_filters = include_filters
        self.exclude_filters = exclude_filters
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.dt_stride = dt_stride
        self.max_dt_stride = max_dt_stride
        self.flatten_tensors = flatten_tensors
        self.return_grid = return_grid
        self.boundary_return_type = boundary_return_type
        self.cache_constants = cache_constants
        self.max_cache_size = max_cache_size
        self.transforms = transforms
        self.tensor_transforms = tensor_transforms
        # Check the directory has hdf5 that meet our exclusion criteria
        sub_files = glob.glob(self.path + "/*.h5") + glob.glob(self.path + "/*.hdf5")
        # Check filters - only use file if include_filters are present and exclude_filters are not
        if len(self.include_filters) > 0:
            retain_files = []
            for include_string in self.include_filters:
                retain_files += [f for f in sub_files if include_string in f]
            sub_files = retain_files
        if len(self.exclude_filters) > 0:
            for exclude_string in self.exclude_filters:
                sub_files = [f for f in sub_files if exclude_string not in f]
        assert len(sub_files) > 0, "No HDF5 files found in path {}".format(self.path)
        self.files_paths = sub_files
        self.files_paths.sort()
        self.constant_cache = {}
        # Build multi-index
        self._build_metadata()
        # Override name if necessary for logging
        if name_override is not None:
            self.dataset_name = name_override

    def _build_metadata(self):
        """Builds multi-file indices and checks that folder contains consistent dataset"""
        self.n_files = len(self.files_paths)
        self.file_steps = []
        self.file_samples = []
        self.file_index_offsets = [0]  # Used to track where each file starts
        self.field_names = []
        # Things where we just care every file has same value
        size_tuples = set()
        names = set()
        ndims = set()
        bcs = set()
        for index, file in enumerate(self.files_paths):
            with h5.File(file, "r") as _f:
                # Run sanity checks - all files should have same ndims, size_tuple, and names
                samples = _f.attrs["n_trajectories"]
                steps = _f["dimensions"]["time"].shape[0]
                size_tuple = [
                    _f["dimensions"][d].shape[0]
                    for d in _f["dimensions"].attrs["spatial_dims"]
                ]
                ndims.add(_f.attrs["n_spatial_dims"])
                names.add(_f.attrs["dataset_name"])
                size_tuples.add(tuple(size_tuple))
                # Fast enough that I'd rather check each file rather than processing extra files before checking
                assert len(names) == 1, "Multiple dataset names found in specified path"
                assert len(ndims) == 1, "Multiple ndims found in specified path"
                assert (
                    len(size_tuples) == 1
                ), "Multiple resolutions found in specified path"
                # Check that the requested steps make sense
                assert (
                    steps - self.dt_stride * (self.n_steps_input + self.n_steps_output)
                    > 0
                ), "Not enough steps in file {} for {} input and {} output steps".format(
                    file, self.n_steps_input, self.n_steps_output
                )
                self.file_steps.append(steps)
                self.file_samples.append(samples)
                self.file_index_offsets.append(
                    self.file_index_offsets[-1]
                    + samples
                    * (
                        steps
                        - self.dt_stride
                        * (self.n_steps_input - 1 + self.n_steps_output)
                    )
                )

                # Check BCs
                for bc in _f["boundary_conditions"].keys():
                    bcs.add(_f["boundary_conditions"][bc].attrs["bc_type"])
                # Populate field names
                if index == 0:
                    self.num_fields_by_tensor_order = {}
                    self.num_constants = len(_f.attrs["simulation_parameters"])
                    for field in _f["t0_fields"].attrs["field_names"]:
                        self.field_names.append(field)
                    self.num_fields_by_tensor_order[0] = len(
                        _f["t0_fields"].attrs["field_names"]
                    )
                    for field in _f["t1_fields"].attrs["field_names"]:
                        for dim in _f["dimensions"].attrs["spatial_dims"]:
                            self.field_names.append(f"{field}_{dim}")
                    self.num_fields_by_tensor_order[1] = len(
                        _f["t1_fields"].attrs["field_names"]
                    )
                    for field in _f["t2_fields"].attrs["field_names"]:
                        for i, dim1 in enumerate(
                            _f["dimensions"].attrs["spatial_dims"]
                        ):
                            for j, dim2 in enumerate(
                                _f["dimensions"].attrs["spatial_dims"]
                            ):
                                # Commenting this out for now - need to figure out a way to
                                # actually get performance here.
                                # if _f['t2_fields'][field].attrs['symmetric']:
                                #     if i > j:
                                #         continue
                                self.field_names.append(f"{field}_{dim1}{dim2}")
                    self.num_fields_by_tensor_order[2] = len(
                        _f["t2_fields"].attrs["field_names"]
                    )

        # Just to make sure it doesn't put us in file -1
        self.file_index_offsets[0] = -1
        self.files = [
            None for _ in self.files_paths
        ]  # We open file references as they come
        # Dataset length is last number of samples
        self.len = self.file_index_offsets[-1]
        self.ndims = list(ndims)[0]  # Number of spatial dims
        self.size_tuple = list(size_tuples)[0]  # Size of spatial dims
        self.dataset_name = list(names)[0]  # Name of dataset
        # Total number of fields (flattening tensor-valued fields)
        self.num_total_fields = len(self.field_names)
        self.num_bcs = len(bcs)  # Number of boundary condition type included in data
        self.bc_types = list(bcs)  # List of boundary condition types

    def _open_file(self, file_ind: int):
        _file = h5.File(self.files_paths[file_ind], "r")
        self.files[file_ind] = _file

    def _check_cache(self, field_name: str, field_data: Any):
        if self.cache_constants:
            if field_data.numel() < self.max_cache_size:
                self.constant_cache[field_name] = field_data

    def _pad_axes(
        self,
        field_data: Any,
        use_dims,
        time_varying: bool = False,
        tensor_order: int = 0,
    ):
        """Repeats data over axes not used in storage"""
        # Look at which dimensions currently are not used and tile based on their sizes
        expand_dims = (1,) if time_varying else ()
        expand_dims = expand_dims + tuple(
            [
                self.size_tuple[i] if not use_dim else 1
                for i, use_dim in enumerate(use_dims)
            ]
        )
        expand_dims = expand_dims + (1,) * tensor_order
        return np.tile(field_data, expand_dims)

    def _reconstruct_fields(self, file, sample_idx, time_idx, n_steps, dt):
        """Reconstruct space fields starting at index sample_idx, time_idx, with
        n_steps and dt stride. Apply transformations if provided."""
        variable_fields = []
        constant_fields = []
        # Iterate through field types and apply appropriate transforms to stack them
        for i, order_fields in enumerate(["t0_fields", "t1_fields", "t2_fields"]):
            sub_fields = []
            for field_name in file[order_fields].attrs["field_names"]:
                field = file[order_fields][field_name]
                use_dims = field.attrs["dim_varying"]
                # TODO if we have slow loading, it might be better to apply both indices at once
                # If the field is constant and in the cache, use it, otherwise go through read/pad
                if field_name in self.constant_cache:
                    field_data = self.constant_cache[field_name]
                else:
                    field_data = field
                    if field.attrs["sample_varying"]:
                        field_data = field_data[sample_idx]
                    if field.attrs["time_varying"]:
                        field_data = field_data[time_idx : time_idx + n_steps * dt : dt]
                    field_data = torch.tensor(
                        self._pad_axes(
                            field_data, use_dims, time_varying=True, tensor_order=i
                        )
                    )
                    if (
                        not field.attrs["time_varying"]
                        and not field.attrs["sample_varying"]
                    ):
                        self._check_cache(
                            field_name, field_data
                        )  # If constant and processed, cache
                sub_fields.append(field_data)
            # Stack fields such that the last i dims are the tensor dims
            sub_fields = torch.stack(sub_fields, -(i + 1))
            for tensor_transform in self.tensor_transforms:
                sub_fields = tensor_transform(sub_fields, order=i)
            # If we're flattening tensors, we can then flatten last i dims
            if self.flatten_tensors:
                sub_fields = sub_fields.flatten(-(i + 1))
            if field.attrs["time_varying"]:
                variable_fields.append(sub_fields)
            else:
                constant_fields.append(sub_fields)

        return tuple(
            [
                torch.concatenate(field_group, -1)
                if len(field_group) > 0
                else torch.tensor([])
                for field_group in [
                    variable_fields,
                    constant_fields,
                ]
            ]
        )

    def _reconstruct_scalars(self, file, sample_idx, time_idx, n_steps, dt):
        """Reconstruct scalar values (not fields) starting at index sample_idx, time_idx, with
        n_steps and dt stride. """
        constant_scalars = []
        time_varying_scalars = []
        for scalar_name in file["scalars"].attrs["field_names"]:
            scalar = file["scalars"][scalar_name]
            # These shouldn't be large so the cache probably doesn't matter
            # but we'll cache them anyway since they're constant.
            if scalar_name in self.constant_cache:
                scalar_data = self.constant_cache[scalar_name]
            else:
                scalar_data = scalar
                if scalar.attrs["sample_varying"]:
                    scalar_data = torch.tensor(scalar_data[sample_idx])
                if scalar.attrs["time_varying"]:
                    scalar_data = torch.tensor(
                        scalar_data[time_idx : time_idx + n_steps * dt : dt]
                    )
                if (
                    not scalar.attrs["time_varying"]
                    and not scalar.attrs["sample_varying"]
                ):
                    scalar_data = torch.tensor(scalar_data[()]).unsqueeze(0)
                    self._check_cache(scalar_name, scalar_data)
            if scalar.attrs["time_varying"]:
                time_varying_scalars.append(scalar_data)
            else:
                constant_scalars.append(scalar_data)
        return tuple(
            [
                torch.concatenate(field_group, -1)
                if len(field_group) > 0
                else torch.tensor([])
                for field_group in [time_varying_scalars, constant_scalars]
            ]
        )

    def _reconstruct_grids(self, file, sample_idx, time_idx, n_steps, dt):
        """Reconstruct grid values starting at index sample_idx, time_idx, with
        n_steps and dt stride."""
        # Time
        if "time_grid" in self.constant_cache:
            time_grid = self.constant_cache["time_grid"]
        elif file["dimensions"]["time"].attrs["sample_varying"]:
            time_grid = file["dimensions"]["time"][
                sample_idx, time_idx : time_idx + n_steps * dt : dt
            ]
        else:
            time_grid = torch.tensor(file["dimensions"]["time"][:])
            self._check_cache("time_grid", time_grid)
        # Use actual timesteps in case we eventually decide to support non-uniform in time
        time_grid = time_grid[time_idx : time_idx + n_steps * dt : dt]
        # Nothing should depend on absolute time - might change if we add weather
        time_grid = time_grid - time_grid.min()

        # Space - TODO - support time-varying grids or non-tensor product grids
        if "space_grid" in self.constant_cache:
            space_grid = self.constant_cache["space_grid"]
        else:
            space_grid = []
            sample_invariant = True
            for i, dim in enumerate(file["dimensions"].attrs["spatial_dims"]):
                if file["dimensions"][dim].attrs["sample_varying"]:
                    sample_invariant = False
                    coords = torch.tensor(file["dimensions"][dim][sample_idx])
                else:
                    coords = torch.tensor(file["dimensions"][dim][:])
                space_grid.append(coords)
            space_grid = torch.stack(torch.meshgrid(*space_grid, indexing="ij"), -1)
            if sample_invariant:
                self._check_cache(dim, space_grid)
        return space_grid, time_grid

    def _padding_bcs(self, file, sample_idx, time_idx, n_steps, dt):
        """Handles BC case where BC corresponds to a specific padding type

        Note/TODO - currently assumes boundaries to be axis-aligned and cover the entire
        domain. This is a simplification that will need to be addressed in the future.
        """
        if "boundary_output" in self.constant_cache:
            boundary_output = self.constant_cache["boundary_output"]
        else:
            bcs = file["boundary_conditions"]
            dim_indices = {
                dim: i for i, dim in enumerate(file["dimensions"].attrs["spatial_dims"])
            }
            boundary_output = torch.zeros((2,) * self.ndims)
            for bc_name in bcs.keys():
                bc = bcs[bc_name]
                bc_type = bc.attrs["bc_type"]
                if len(bc.attrs["associated_dims"]) > 1:
                    raise NotImplementedError(
                        "Only axis-aligned boundaries supported for now"
                    )
                dim = bc.attrs["associated_dims"][0]
                mask = bc["mask"]
                if mask[0]:
                    boundary_output[dim_indices[dim]][0] = BoundaryCondition[
                        bc_type
                    ].value
                if mask[1]:
                    boundary_output[dim_indices[dim]][1] = BoundaryCondition[
                        bc_type
                    ].value
            self._check_cache("boundary_output", boundary_output)
        return boundary_output

    def _reconstruct_bcs(self, file, sample_idx, time_idx, n_steps, dt):
        """Needs work to support arbitrary BCs.

        Currently supports finite set of boundary condition types that describe
        the geometry of the domain. Implements these as mask channels. The total
        number of channels is determined by the number of BC types in the
        data.

        #TODO generalize boundary types
        """
        if self.boundary_return_type == "padding":
            return self._padding_bcs(file, sample_idx, time_idx, n_steps, dt)

    def __getitem__(self, index):
        # Find specific file and local index
        file_idx = int(
            np.searchsorted(self.file_index_offsets, index, side="right") - 1
        )  # which file we are on
        file_steps = self.file_steps[file_idx]
        local_idx = index - max(
            self.file_index_offsets[file_idx], 0
        )  # First offset is -1
        sample_idx = local_idx // file_steps
        time_idx = local_idx % file_steps

        # open hdf5 file (and cache the open object)
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        # If we gave a stride range, decide the largest size we can use given the sample location
        if self.max_dt_stride > self.dt_stride:
            sample_steps = self.n_steps_input + self.n_steps_output
            effective_max_dt = min(
                int((file_steps - time_idx) // sample_steps), self.max_dt_stride
            )
            if effective_max_dt > self.dt:
                dt = np.random.randint(self.dt, effective_max_dt)
        else:
            dt = self.dt_stride
        # Now build the data
        trajectory, constant_fields = self._reconstruct_fields(
            self.files[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + self.n_steps_output,
            dt,
        )
        time_varying_scalars, constant_scalars = self._reconstruct_scalars(
            self.files[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + self.n_steps_output,
            dt,
        )

        sample = {
            "input_fields": trajectory[
                : self.n_steps_input
            ],  # Tin x H x W x D x C tensor of input trajectory
            "output_fields": trajectory[
                self.n_steps_input :
            ],  # Tpred x H x W x D x C tensor of output trajectory
            "constant_fields": constant_fields,  # H (x W x D) x (num constant) tensor.
            "input_time_varying_scalars": time_varying_scalars[
                : self.n_steps_input
            ],  # Tin x C tensor with time varying scalars
            "output_time_varying_scalars": time_varying_scalars[self.n_steps_input :],
            "constant_scalars": constant_scalars,  # 1 x C tensor with constant values corresponding to parameters
        }

        if self.use_normalization:
            # Load normalization constants
            for k in self.means.keys():
                k = k.replace("output", "input")  # Use fields computed from input
                if k in sample:
                    sample[k] = (sample[k] - self.means[k]) / (self.stds[k] + 1e-4)

        # For complex BCs, might need to do this pre_normalization
        bcs = self._reconstruct_bcs(
            self.files[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + self.n_steps_output,
            dt,
        )
        sample["boundary_conditions"] = bcs  # Currently only mask is an option
        if self.return_grid:
            space_grid, time_grid = self._reconstruct_grids(
                self.files[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + self.n_steps_output,
                dt,
            )
            sample["space_grid"] = (
                space_grid  # H (x W x D) x (num dims) tensor with coordinate values
            )
            sample["input_time_grid"] = time_grid[
                : self.n_steps_input
            ]  # Tin x 1 tensor with time values
            sample["output_time_grid"] = time_grid[
                self.n_steps_input :
            ]  # Tpred x 1 tensor with time values

        # Return only non-empty keys - maybe change this later
        return {k: v for k, v in sample.items() if v.numel() > 0}

    def __len__(self):
        return self.len
