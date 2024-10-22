import glob
import itertools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import h5py as h5
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from the_well.utils.export import hdf5_to_xarray

well_paths = {
    "acoustic_scattering_maze": "datasets/acoustic_scattering_maze",
    "acoustic_scattering_inclusions": "datasets/acoustic_scattering_inclusions",
    "acoustic_scattering_discontinuous": "datasets/acoustic_scattering_discontinuous",
    "active_matter": "datasets/active_matter",
    "convective_envelope_rsg": "datasets/convective_envelope_rsg",
    "euler_multi_quadrants_openBC": "datasets/euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC": "datasets/euler_multi_quadrants_periodicBC",
    "helmholtz_staircase": "datasets/helmholtz_staircase",
    "MHD_256": "datasets/MHD_256",
    "MHD_64": "datasets/MHD_64",
    "gray_scott_reaction_diffusion": "datasets/gray_scott_reaction_diffusion",
    "planetswe": "datasets/planetswe",
    "post_neutron_star_merger": "datasets/post_neutron_star_merger",
    "rayleigh_benard": "datasets/rayleigh_benard",
    "rayleigh_taylor_instability": "datasets/rayleigh_taylor_instability",
    "shear_flow": "datasets/shear_flow",
    "supernova_explosion_64": "datasets/supernova_explosion_64",
    "supernova_explosion_128": "datasets/supernova_explosion_128",
    "turbulence_gravity_cooling": "datasets/turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D": "datasets/turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D": "datasets/turbulent_radiative_layer_3D",
    "viscoelastic_instability": "datasets/viscoelastic_instability",
    "dummy": "datasets/dummy_placeholder",
}


def raw_steps_to_possible_sample_t0s(
    total_steps_in_trajectory: int,
    n_steps_input: int,
    n_steps_output: int,
    dt_stride: int,
):
    """Given the total number of steps in a trajectory returns the number of samples that can be taken from the
      trajectory such that all samples have at least n_steps_input + n_steps_output steps with steps separated
      by dt_stride.

    ex1: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 1
        Possible samples are: [0, 1], [1, 2], [2, 3], [3, 4]
    ex2: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 2
        Possible samples are: [0, 2], [1, 3], [2, 4]
    ex3: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 3
        Possible samples are: [0, 3], [1, 4]
    ex4: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 1, dt_stride = 2
        Possible samples are: [0, 2, 4]

    """
    elapsed_steps_per_sample = 1 + dt_stride * (
        n_steps_input + n_steps_output - 1
    )  # Number of steps needed for sample
    return max(0, total_steps_in_trajectory - elapsed_steps_per_sample + 1)


def maximum_stride_for_initial_index(
    time_idx: int,
    total_steps_in_trajectory: int,
    n_steps_input: int,
    n_steps_output: int,
):
    """Given the total number of steps in a file and the current step returns the maximum stride
    that can be taken from the file such that all samples have at least n_steps_input + n_steps_output steps with a stride of
      dt_stride
    """
    used_steps_per_sample = n_steps_input + n_steps_output
    return max(
        0,
        int((total_steps_in_trajectory - time_idx - 1) // (used_steps_per_sample - 1)),
    )


# Boundary condition codes
class BoundaryCondition(Enum):
    WALL = 0
    OPEN = 1
    PERIODIC = 2


def flatten_field_names(metadata, include_constants=True):
    flat_field_names = itertools.chain(*metadata.field_names.values())
    flat_constant_field_names = itertools.chain(*metadata.constant_field_names.values())

    if include_constants:
        return [*flat_field_names, *flat_constant_field_names]
    else:
        return [*flat_field_names]


@dataclass
class GenericWellMetadata:
    """Dataclass to store metadata for each dataset."""

    dataset_name: str
    n_spatial_dims: int
    spatial_resolution: Tuple[int]
    scalar_names: List[str]
    constant_scalar_names: List[str]
    field_names: Dict[int, List[str]]
    constant_field_names: Dict[int, List[str]]
    boundary_condition_types: List[str]
    n_simulations: int
    n_steps_per_simulation: List[int]
    grid_type: str = "cartesian"

    @property
    def n_scalars(self) -> int:
        return len(self.scalar_names)

    @property
    def n_constant_scalars(self) -> int:
        return len(self.constant_scalar_names)

    @property
    def n_fields(self) -> int:
        return sum(map(len, self.field_names.values()))

    @property
    def n_constant_fields(self) -> int:
        return sum(map(len, self.constant_field_names.values()))

    @property
    def sample_shapes(self) -> Dict[str, List[int]]:
        return {
            "input_fields": [*self.spatial_resolution, self.n_fields],
            "output_fields": [*self.spatial_resolution, self.n_fields],
            "constant_fields": [*self.spatial_resolution, self.n_constant_fields],
            "input_scalars": [self.n_scalars],
            "output_scalars": [self.n_scalars],
            "constant_scalars": [self.n_constant_scalars],
            "space_grid": [*self.spatial_resolution, self.n_spatial_dims],
        }


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
    cache_small :
        Whether to cache small tensors in memory for faster access
    max_cache_size :
        Maximum numel of constant tensor to cache
    return_grid :
        Whether to return grid coordinates
    boundary_return_type : options=['padding', 'mask', 'exact', 'none']
        How to return boundary conditions. Currently only padding supported.
    full_trajectory_mode :
        Overrides to return full trajectory starting from t0 instead of samples
            for long run validation.
    name_override :
        Override name of dataset (used for more precise logging)
    transform :
        Transform to apply to data. Provide this in the form
        `f(data: torch.Tensor, **metadata) -> torch.Tensor`, where
        `metadata` includes `order: int`, the tensor order of the field,
        `field_names: List[str]`, the names of the fields, `field_attrs`,
        dataset-specific metadata, as well as `dataset` which is the
        dataset itself (from which one can get `dataset.metadata` to access
        more detailed metadata).
    min_std :
        Minimum standard deviation for field normalization. If a field standard
        deviation is lower than this value, it is replaced by this value.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        normalization_path: str = "../stats.yaml",
        well_base_path: Optional[str] = None,
        well_dataset_name: Optional[str] = None,
        well_split_name: str = "train",
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        use_normalization: bool = True,
        max_rollout_steps=100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        dt_stride: int = 1,
        max_dt_stride: int = 1,
        flatten_tensors: bool = True,
        cache_small: bool = True,
        max_cache_size: float = 1e9,
        return_grid: bool = True,
        boundary_return_type: str = "padding",
        full_trajectory_mode: bool = False,
        name_override: Optional[str] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        min_std: float = 1e-4,
    ):
        super().__init__()
        assert path is not None or (
            well_base_path is not None and well_dataset_name is not None
        ), "Must specify path or well_base_path and well_dataset_name"
        if path is not None:
            path = os.path.abspath(path)
            self.data_path = path
            self.normalization_path = os.path.abspath(
                os.path.join(path, normalization_path)
            )
        else:
            well_base_path = os.path.abspath(well_base_path)
            self.data_path = os.path.join(
                well_base_path, well_paths[well_dataset_name], "data", well_split_name
            )
            self.normalization_path = os.path.join(
                well_base_path, well_paths[well_dataset_name], "stats.yaml"
            )

        if use_normalization:
            with open(self.normalization_path, mode="r") as f:
                stats = yaml.safe_load(f)

            self.means = {
                field: torch.as_tensor(val) for field, val in stats["mean"].items()
            }
            self.stds = {
                field: torch.clip(torch.as_tensor(val), min=min_std)
                for field, val in stats["std"].items()
            }

        # Input checks
        if boundary_return_type is not None and boundary_return_type not in ["padding"]:
            raise NotImplementedError("Only padding boundary conditions supported")
        if not flatten_tensors:
            raise NotImplementedError("Only flattened tensors supported right now")

        # Copy params
        self.well_dataset_name = well_dataset_name
        self.use_normalization = use_normalization
        self.include_filters = include_filters
        self.exclude_filters = exclude_filters
        self.max_rollout_steps = max_rollout_steps
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output  # Gets overridden by full trajectory mode
        self.dt_stride = dt_stride
        self.max_dt_stride = max_dt_stride
        self.flatten_tensors = flatten_tensors
        self.return_grid = return_grid
        self.boundary_return_type = boundary_return_type
        self.full_trajectory_mode = full_trajectory_mode
        self.cache_small = cache_small
        self.max_cache_size = max_cache_size
        self.transform = transform
        # Check the directory has hdf5 that meet our exclusion criteria
        sub_files = glob.glob(self.data_path + "/*.h5") + glob.glob(
            self.data_path + "/*.hdf5"
        )
        # Check filters - only use file if include_filters are present and exclude_filters are not
        if len(self.include_filters) > 0:
            retain_files = []
            for include_string in self.include_filters:
                retain_files += [f for f in sub_files if include_string in f]
            sub_files = retain_files
        if len(self.exclude_filters) > 0:
            for exclude_string in self.exclude_filters:
                sub_files = [f for f in sub_files if exclude_string not in f]
        assert len(sub_files) > 0, "No HDF5 files found in path {}".format(
            self.data_path
        )
        self.files_paths = sub_files
        self.files_paths.sort()
        self.caches = [{} for _ in self.files_paths]
        # Build multi-index
        self.metadata = self._build_metadata()
        # Override name if necessary for logging
        if name_override is not None:
            self.dataset_name = name_override

    def _build_metadata(self):
        """Builds multi-file indices and checks that folder contains consistent dataset"""
        self.n_files = len(self.files_paths)
        self.total_file_steps = []  # Number of time steps in each simulation for each file
        self.available_file_steps = []  # Number of actual time steps in each simulation for each file
        self.file_samples = []  # Number of simulation per file
        self.file_index_offsets = [0]  # Used to track where each file starts
        self.field_names = {}
        self.constant_field_names = []
        # Things where we just care every file has same value
        size_tuples = set()
        names = set()
        ndims = set()
        bcs = set()
        for index, file in enumerate(self.files_paths):
            with h5.File(file, "r") as _f:
                grid_type: str = _f.attrs["grid_type"]
                # Run sanity checks - all files should have same ndims, size_tuple, and names
                samples: int = _f.attrs["n_trajectories"]
                # Number of steps is always last dim of time
                steps = _f["dimensions"]["time"].shape[-1]
                size_tuple = [
                    _f["dimensions"][d].shape[-1]
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
                # TODO - this probably bugs out if steps vary between files
                if self.full_trajectory_mode:
                    self.n_steps_output = steps - self.n_steps_input
                # Check that the requested steps make sense
                per_simulation_steps = raw_steps_to_possible_sample_t0s(
                    steps, self.n_steps_input, self.n_steps_output, self.dt_stride
                )
                assert per_simulation_steps > 0, (
                    f"Not enough steps in file {file}"
                    f"for {self.n_steps_input} input and {self.n_steps_output} output steps"
                )
                self.file_samples.append(samples)
                self.total_file_steps.append(steps)
                self.available_file_steps.append(per_simulation_steps)
                self.file_index_offsets.append(
                    self.file_index_offsets[-1] + samples * per_simulation_steps
                )

                # Check BCs
                for bc in _f["boundary_conditions"].keys():
                    bcs.add(_f["boundary_conditions"][bc].attrs["bc_type"])

                if index == 0:
                    # Populate scalar names
                    self.scalar_names = []
                    self.constant_scalar_names = []

                    for scalar in _f["scalars"].attrs["field_names"]:
                        if _f["scalars"][scalar].attrs["time_varying"]:
                            self.scalar_names.append(scalar)
                        else:
                            self.constant_scalar_names.append(scalar)

                    # Populate field names
                    self.field_names = {i: [] for i in range(3)}
                    self.constant_field_names = {i: [] for i in range(3)}

                    for i in range(3):
                        ti = f"t{i}_fields"
                        # if _f[ti][field].attrs["symmetric"]:
                        # itertools.combinations_with_replacement
                        ti_field_dims = [
                            "".join(xyz)
                            for xyz in itertools.product(
                                _f["dimensions"].attrs["spatial_dims"],
                                repeat=i,
                            )
                        ]

                        for field in _f[ti].attrs["field_names"]:
                            for dims in ti_field_dims:
                                field_name = f"{field}_{dims}" if dims else field

                                if _f[ti][field].attrs["time_varying"]:
                                    self.field_names[i].append(field_name)
                                else:
                                    self.constant_field_names[i].append(field_name)

        # Just to make sure it doesn't put us in file -1
        self.file_index_offsets[0] = -1
        self.files = [
            None for _ in self.files_paths
        ]  # We open file references as they come
        # Dataset length is last number of samples
        self.len = self.file_index_offsets[-1]
        self.n_spatial_dims = int(ndims.pop())  # Number of spatial dims
        self.size_tuple = tuple(map(int, size_tuples.pop()))  # Size of spatial dims
        self.dataset_name = names.pop()  # Name of dataset
        # BCs
        self.num_bcs = len(bcs)  # Number of boundary condition type included in data
        self.bc_types = list(bcs)  # List of boundary condition types

        return GenericWellMetadata(
            dataset_name=self.dataset_name,
            n_spatial_dims=self.n_spatial_dims,
            grid_type=grid_type,
            spatial_resolution=self.size_tuple,
            scalar_names=self.scalar_names,
            constant_scalar_names=self.constant_scalar_names,
            field_names=self.field_names,
            constant_field_names=self.constant_field_names,
            boundary_condition_types=self.bc_types,
            n_simulations=self.n_files,
            n_steps_per_simulation=self.total_file_steps,
        )

    def _open_file(self, file_ind: int):
        _file = h5.File(self.files_paths[file_ind], "r")
        self.files[file_ind] = _file

    def _check_cache(self, cache: Dict[str, Any], name: str, data: Any):
        if self.cache_small and data.numel() < self.max_cache_size:
            cache[name] = data

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
        return torch.tile(field_data, expand_dims)

    def _postprocess_field_list(self, field_list, output_list, order, tensor_metadata):
        """Postprocesses field list to apply tensor transforms"""
        if len(field_list) > 0:
            field_list = torch.stack(field_list, -(order + 1))
            if self.transform is not None:
                field_list = self.transform(field_list, order=order, **tensor_metadata)
            if self.flatten_tensors:
                field_list = field_list.flatten(-(order + 1))
            output_list.append(field_list)
        return output_list

    def _reconstruct_fields(self, file, cache, sample_idx, time_idx, n_steps, dt):
        """Reconstruct space fields starting at index sample_idx, time_idx, with
        n_steps and dt stride. Apply transformations if provided."""
        variable_fields = []
        constant_fields = []
        # Iterate through field types and apply appropriate transforms to stack them
        for i, order_fields in enumerate(["t0_fields", "t1_fields", "t2_fields"]):
            variable_subfields = []
            constant_subfields = []
            field_names = file[order_fields].attrs["field_names"]
            for field_name in field_names:
                field = file[order_fields][field_name]
                use_dims = field.attrs["dim_varying"]
                # If the field is in the cache, use it, otherwise go through read/pad
                if field_name in cache:
                    field_data = cache[field_name]
                else:
                    field_data = field
                    # Index is built gradually since there can be different numbers of leading fields
                    multi_index = ()
                    if field.attrs["sample_varying"]:
                        multi_index = multi_index + (sample_idx,)
                    if field.attrs["time_varying"]:
                        multi_index = multi_index + (
                            slice(time_idx, time_idx + n_steps * dt, dt),
                        )
                    field_data = field_data[multi_index]
                    field_data = torch.as_tensor(field_data)
                    # Expand dims
                    field_data = self._pad_axes(
                        field_data,
                        use_dims,
                        time_varying=field.attrs["time_varying"],
                        tensor_order=i,
                    )
                    # Normalize
                    if self.use_normalization:
                        if field_name in self.means:
                            field_data = field_data - self.means[field_name]
                        if field_name in self.stds:
                            field_data = field_data / self.stds[field_name]
                    # If constant, try to cache
                    if (
                        not field.attrs["time_varying"]
                        and not field.attrs["sample_varying"]
                    ):
                        self._check_cache(cache, field_name, field_data)

                if field.attrs["time_varying"]:
                    variable_subfields.append(field_data)
                else:
                    constant_subfields.append(field_data)

            tensor_metadata = {
                "dataset": self,
                "field_names": field_names,
                "field_attrs": {
                    field_name: file[order_fields][field_name].attrs
                    for field_name in file[order_fields].attrs["field_names"]
                },
            }
            # Stack fields such that the last i dims are the tensor dims
            variable_fields = self._postprocess_field_list(
                variable_subfields, variable_fields, i, tensor_metadata
            )
            constant_fields = self._postprocess_field_list(
                constant_subfields, constant_fields, i, tensor_metadata
            )

        return tuple(
            torch.concatenate(field_group, -1)
            if len(field_group) > 0
            else torch.tensor([])
            for field_group in [
                variable_fields,
                constant_fields,
            ]
        )

    def _reconstruct_scalars(self, file, cache, sample_idx, time_idx, n_steps, dt):
        """Reconstruct scalar values (not fields) starting at index sample_idx, time_idx, with
        n_steps and dt stride."""
        variable_scalars = []
        constant_scalars = []
        for scalar_name in file["scalars"].attrs["field_names"]:
            scalar = file["scalars"][scalar_name]

            if scalar_name in cache:
                scalar_data = cache[scalar_name]
            else:
                scalar_data = scalar
                # Build index gradually to account for different leading dims
                multi_index = ()
                if scalar.attrs["sample_varying"]:
                    multi_index = multi_index + (sample_idx,)
                if scalar.attrs["time_varying"]:
                    multi_index = multi_index + (
                        slice(time_idx, time_idx + n_steps * dt, dt),
                    )
                scalar_data = scalar_data[multi_index]
                scalar_data = torch.as_tensor(scalar_data)
                # If constant, try to cache
                if (
                    not scalar.attrs["time_varying"]
                    and not scalar.attrs["sample_varying"]
                ):
                    self._check_cache(cache, scalar_name, scalar_data)

            if scalar.attrs["time_varying"]:
                variable_scalars.append(scalar_data)
            else:
                constant_scalars.append(scalar_data)

        return tuple(
            torch.stack(scalar_group, -1) if len(scalar_group) > 0 else torch.tensor([])
            for scalar_group in [variable_scalars, constant_scalars]
        )

    def _reconstruct_grids(self, file, cache, sample_idx, time_idx, n_steps, dt):
        """Reconstruct grid values starting at index sample_idx, time_idx, with
        n_steps and dt stride."""
        # Time
        if "time_grid" in cache:
            time_grid = cache["time_grid"]
        elif file["dimensions"]["time"].attrs["sample_varying"]:
            time_grid = torch.tensor(file["dimensions"]["time"][sample_idx, :])
        else:
            time_grid = torch.tensor(file["dimensions"]["time"][:])
            self._check_cache(cache, "time_grid", time_grid)
        # We have already sampled leading index if it existed so timegrid should be 1D
        time_grid = time_grid[time_idx : time_idx + n_steps * dt : dt]
        # Nothing should depend on absolute time - might change if we add weather
        time_grid = time_grid - time_grid.min()

        # Space - TODO - support time-varying grids or non-tensor product grids
        if "space_grid" in cache:
            space_grid = cache["space_grid"]
        else:
            space_grid = []
            sample_invariant = True
            for dim in file["dimensions"].attrs["spatial_dims"]:
                if file["dimensions"][dim].attrs["sample_varying"]:
                    sample_invariant = False
                    coords = torch.tensor(file["dimensions"][dim][sample_idx])
                else:
                    coords = torch.tensor(file["dimensions"][dim][:])
                space_grid.append(coords)
            space_grid = torch.stack(torch.meshgrid(*space_grid, indexing="ij"), -1)
            if sample_invariant:
                self._check_cache(cache, "space_grid", space_grid)
        return space_grid, time_grid

    def _padding_bcs(self, file, cache, sample_idx, time_idx, n_steps, dt):
        """Handles BC case where BC corresponds to a specific padding type

        Note/TODO - currently assumes boundaries to be axis-aligned and cover the entire
        domain. This is a simplification that will need to be addressed in the future.
        """
        if "boundary_output" in cache:
            boundary_output = cache["boundary_output"]
        else:
            bcs = file["boundary_conditions"]
            dim_indices = {
                dim: i for i, dim in enumerate(file["dimensions"].attrs["spatial_dims"])
            }
            boundary_output = torch.zeros(self.n_spatial_dims, 2)
            for bc_name in bcs.keys():
                bc = bcs[bc_name]
                bc_type = bc.attrs["bc_type"].upper()  # Enum is in upper case
                if len(bc.attrs["associated_dims"]) > 1:
                    raise NotImplementedError(
                        "Only axis-aligned boundaries supported for now. If your code is not using BCs, consider setting `boundary_return_type` to None."
                    )
                dim = bc.attrs["associated_dims"][0]
                mask = bc["mask"]
                if mask[0]:
                    boundary_output[dim_indices[dim]][0] = BoundaryCondition[
                        bc_type
                    ].value
                if mask[-1]:
                    boundary_output[dim_indices[dim]][1] = BoundaryCondition[
                        bc_type
                    ].value
            self._check_cache(cache, "boundary_output", boundary_output)
        return boundary_output

    def _reconstruct_bcs(self, file, cache, sample_idx, time_idx, n_steps, dt):
        """Needs work to support arbitrary BCs.

        Currently supports finite set of boundary condition types that describe
        the geometry of the domain. Implements these as mask channels. The total
        number of channels is determined by the number of BC types in the
        data.

        #TODO generalize boundary types
        """
        if self.boundary_return_type == "padding":
            return self._padding_bcs(file, cache, sample_idx, time_idx, n_steps, dt)
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        # Find specific file and local index
        file_idx = int(
            np.searchsorted(self.file_index_offsets, index, side="right") - 1
        )  # which file we are on
        per_simulation_steps = self.available_file_steps[file_idx]
        local_idx = index - max(
            self.file_index_offsets[file_idx], 0
        )  # First offset is -1
        sample_idx = local_idx // per_simulation_steps
        time_idx = local_idx % per_simulation_steps

        # open hdf5 file (and cache the open object)
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        # If we gave a stride range, decide the largest size we can use given the sample location
        if self.max_dt_stride > self.dt_stride:
            effective_max_dt = maximum_stride_for_initial_index(
                time_idx,
                self.total_file_steps[file_idx],
                self.n_steps_input,
                self.n_steps_output,
            )
            if effective_max_dt > self.dt:
                dt = np.random.randint(self.dt, effective_max_dt)
        else:
            dt = self.dt_stride
        # Now build the data
        output_steps = min(self.n_steps_output, self.max_rollout_steps)
        variable_fields, constant_fields = self._reconstruct_fields(
            self.files[file_idx],
            self.caches[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + output_steps,
            dt,
        )
        variable_scalars, constant_scalars = self._reconstruct_scalars(
            self.files[file_idx],
            self.caches[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + output_steps,
            dt,
        )

        sample = {
            "input_fields": variable_fields[: self.n_steps_input],  # Ti x H x W x C
            "output_fields": variable_fields[self.n_steps_input :],  # To x H x W x C
            "constant_fields": constant_fields,  # H x W x C
            "input_scalars": variable_scalars[: self.n_steps_input],  # Ti x C
            "output_scalars": variable_scalars[self.n_steps_input :],  # To x C
            "constant_scalars": constant_scalars,  # C
        }

        # For complex BCs, might need to do this pre_normalization
        # TODO Need to generalize
        if self.boundary_return_type is not None:
            bcs = self._reconstruct_bcs(
                self.files[file_idx],
                self.caches[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )
            sample["boundary_conditions"] = bcs  # Currently only mask is an option
        if self.return_grid:
            space_grid, time_grid = self._reconstruct_grids(
                self.files[file_idx],
                self.caches[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )
            sample["space_grid"] = space_grid  # H x W x D
            sample["input_time_grid"] = time_grid[: self.n_steps_input]  # Ti
            sample["output_time_grid"] = time_grid[self.n_steps_input :]  # To

        # Return only non-empty keys - maybe change this later
        return {k: v for k, v in sample.items() if v.numel() > 0}

    def __len__(self):
        return self.len

    def to_xarray(self, backend: Literal["numpy", "dask"] = "dask"):
        """
        Export the dataset to an XArray Dataset by stacking all HDF5 files as XArray datasets
        along the existing 'sample' dimension.

        Parameters:
        - backend (str): 'numpy' for eager loading, 'dask' for lazy loading.

        Returns:
        - ds (xarray.Dataset): The stacked XArray Dataset.

        Example:

        To convert a dataset and plot the pressure for 5 different times for a single trajectory:

        >>> ds = dataset.to_xarray()
        >>> ds.pressure.isel(sample=0, time=[0, 10, 20, 30, 40]).plot(col='time', col_wrap=5)
        """
        import xarray as xr

        datasets = []
        total_samples = 0
        for file_idx in range(len(self.files_paths)):
            if self.files[file_idx] is None:
                self._open_file(file_idx)
            ds = hdf5_to_xarray(self.files[file_idx], backend=backend)
            # Ensure 'sample' dimension is always present
            if "sample" not in ds.sizes:
                ds = ds.expand_dims("sample")
            # Adjust the 'sample' coordinate
            if "sample" in ds.coords:
                n_samples = ds.sizes["sample"]
                ds = ds.assign_coords(sample=ds.coords["sample"] + total_samples)
                total_samples += n_samples
            datasets.append(ds)

        combined_ds = xr.concat(datasets, dim="sample")
        return combined_ds
