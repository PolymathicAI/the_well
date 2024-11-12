from typing import Literal

import h5py
import numpy as np


def hdf5_to_xarray(f: h5py.File, backend: Literal["numpy", "dask"] = "numpy"):
    """
    Convert an HDF5 file to an XArray Dataset, using either NumPy or Dask arrays.

    Args:
        f: HDF5 file.
        backend (str): 'numpy' for eager loading, 'dask' for lazy loading.

    Returns:
        xarray.Dataset: The resulting XArray Dataset.
    """
    if backend not in {"numpy", "dask"}:
        raise ValueError("Unsupported backend: {}".format(backend))

    import xarray as xr

    if backend == "numpy":
        array_lib = np
    else:
        import dask.array as da

        array_lib = da

    data_vars = {}
    coords = {}
    attrs = dict(f.attrs)

    # Get spatial dimensions
    dims_group = f["dimensions"]
    spatial_dims = dims_group.attrs["spatial_dims"]  # List of spatial dimension names
    n_spatial_dims = f.attrs["n_spatial_dims"]
    assert len(spatial_dims) == n_spatial_dims

    # Initialize spatial coordinates
    spatial_coords = {}
    spatial_sizes = []
    for dim in spatial_dims:
        dim_ds = dims_group[dim]
        sample_varying = dim_ds.attrs["sample_varying"]
        time_varying = dim_ds.attrs["time_varying"]
        dim_data = dim_ds[()]
        if sample_varying or time_varying:
            raise NotImplementedError(
                "Sample or time-varying spatial dimensions are not supported."
            )
        else:
            spatial_coords[dim] = dim_data
            spatial_sizes.append(len(dim_data))

    # Add spatial coordinates to coords
    coords.update(spatial_coords)

    # Get time coordinate
    time_ds = dims_group["time"]
    time_sample_varying = time_ds.attrs["sample_varying"]
    time_data = time_ds[()]
    n_samples = f.attrs["n_trajectories"]
    sample_coords = np.arange(n_samples)

    if time_sample_varying:
        # time_data is 2D: [n_samples, n_times]
        coords["sample"] = sample_coords
        # Remove 'time' from coords to avoid conflicts
        if "time" in coords:
            del coords["time"]
    else:
        # time_data is 1D: [n_times]
        time_data.shape[0]
        coords["time"] = time_data
        # 'sample' will be added to coords when needed

    # Function to process each field dataset
    def process_field_dataset(field_ds):
        sample_varying = field_ds.attrs["sample_varying"]
        time_varying = field_ds.attrs["time_varying"]
        dim_varying = field_ds.attrs["dim_varying"]  # List of bools
        tensor_order = int(field_ds.parent.name[2])  # 't0_fields' => 0

        # Read data
        if backend == "numpy":
            data = field_ds[()]
        else:
            data = array_lib.from_array(field_ds, chunks="auto")

        dims = []
        idx = 0  # index in data.shape

        # Handle sample dimension
        if sample_varying:
            dims.append("sample")
            idx += 1
            if "sample" not in coords:
                coords["sample"] = sample_coords
        # Handle time dimension
        if time_varying:
            dims.append("time")
            idx += 1

        # Handle spatial dimensions
        for i, dim_varies in enumerate(dim_varying):
            dim_name = spatial_dims[i]
            if dim_varies:
                dims.append(dim_name)
                idx += 1
            else:
                # Expand dimension
                data = array_lib.expand_dims(data, axis=idx)
                if backend == "numpy":
                    data = array_lib.repeat(data, spatial_sizes[i], axis=idx)
                else:
                    shape = data.shape
                    data = data.broadcast_to(
                        shape[:idx] + (spatial_sizes[i],) + shape[idx + 1 :]
                    )
                dims.append(dim_name)
                idx += 1  # We added a new dimension

        # Update data_shape after possible expansions
        data_shape = data.shape

        # Handle tensor components
        if tensor_order > 0:
            axes = spatial_dims  # e.g., ['x', 'y', 'z']
            n_axes = len(axes)
            component_dims = []

            # For each index in tensor order, add a new dimension
            for n in range(tensor_order):
                dim_label = chr(ord("i") + n)  # 'i', 'j', 'k', etc.
                dims.append(dim_label)
                component_dims.append(dim_label)
                coords[dim_label] = axes

            # Reshape data to include component dimensions
            expected_components_shape = (n_axes,) * tensor_order
            n_expected_components = np.prod(expected_components_shape)

            # Check if data already includes component dimensions
            if data_shape[-tensor_order:] == expected_components_shape:
                # Data already includes component dimensions
                pass
            elif data_shape[-1] == n_expected_components:
                # Data has components flattened into one dimension
                data = data.reshape(data_shape[:-1] + expected_components_shape)
                data_shape = data.shape
            else:
                # Need to reshape data to add component dimensions
                data = data.reshape(data_shape + expected_components_shape)
                data_shape = data.shape
        else:
            # Tensor order 0, no additional components
            pass

        # Create DataArray without coords
        data_var = xr.DataArray(data, dims=dims)

        # Assign coordinates
        for dim in dims:
            if dim == "sample":
                data_var.coords["sample"] = sample_coords
            elif dim == "time":
                if time_sample_varying:
                    data_var.coords["time"] = (("sample", "time"), time_data)
                else:
                    data_var.coords["time"] = coords["time"]
            else:
                data_var.coords[dim] = coords[dim]

        data_var.attrs = dict(field_ds.attrs)
        return data_var

    # Process fields
    for order in [0, 1, 2]:
        group_name = f"t{order}_fields"
        if group_name in f:
            fields_group = f[group_name]
            field_names = fields_group.attrs["field_names"]
            for field_name in field_names:
                field_ds = fields_group[field_name]
                data_var = process_field_dataset(field_ds)
                data_vars[field_name] = data_var

    # Process scalars
    if "scalars" in f:
        scalars_group = f["scalars"]
        scalar_names = scalars_group.attrs["field_names"]
        for scalar_name in scalar_names:
            scalar_ds = scalars_group[scalar_name]
            sample_varying = scalar_ds.attrs["sample_varying"]
            time_varying = scalar_ds.attrs["time_varying"]
            if backend == "numpy":
                data = scalar_ds[()]
            else:
                data = array_lib.from_array(scalar_ds, chunks="auto")
            dims = []
            if sample_varying:
                dims.append("sample")
                if "sample" not in coords:
                    coords["sample"] = sample_coords
            if time_varying:
                dims.append("time")
            # Create DataArray without coords
            data_var = xr.DataArray(data, dims=dims)

            # Assign coordinates
            for dim in dims:
                if dim == "sample":
                    data_var.coords["sample"] = sample_coords
                elif dim == "time":
                    if time_sample_varying:
                        data_var.coords["time"] = (("sample", "time"), time_data)
                    else:
                        data_var.coords["time"] = coords["time"]

            data_var.attrs = dict(scalar_ds.attrs)
            data_vars[scalar_name] = data_var

    # Process boundary conditions
    if "boundary_conditions" in f:
        bc_group = f["boundary_conditions"]
        for bc_name in bc_group.keys():
            bc_ds = bc_group[bc_name]
            mask_ds = bc_ds["mask"]
            if backend == "numpy":
                mask = mask_ds[()]
            else:
                mask = array_lib.from_array(mask_ds, chunks="auto")
            sample_varying = bc_ds.attrs["sample_varying"]
            time_varying = bc_ds.attrs["time_varying"]
            associated_dims = bc_ds.attrs["associated_dims"]  # List of dims
            dims = []
            if sample_varying:
                dims.append("sample")
                if "sample" not in coords:
                    coords["sample"] = sample_coords
            if time_varying:
                dims.append("time")
            dims.extend(associated_dims)
            # Create DataArray without coords
            data_var = xr.DataArray(mask, dims=dims)

            # Assign coordinates
            for dim in dims:
                if dim == "sample":
                    data_var.coords["sample"] = sample_coords
                elif dim == "time":
                    if time_sample_varying:
                        data_var.coords["time"] = (("sample", "time"), time_data)
                    else:
                        data_var.coords["time"] = coords["time"]
                else:
                    data_var.coords[dim] = coords[dim]

            data_var.attrs = dict(bc_ds.attrs)
            data_vars[bc_name] = data_var

    # Create Dataset without passing coords
    ds = xr.Dataset(data_vars=data_vars, attrs={})  # Clear attrs for now

    # Ensure 'sample' dimension is always present
    if "sample" not in ds.sizes:
        ds = ds.expand_dims("sample")

    # Convert per-file attributes to variables with 'sample' dimension
    for attr_name, attr_value in attrs.items():
        attr_array = np.array(attr_value)
        if attr_array.ndim == 0:
            # Scalar attribute
            data = np.full(ds.sizes["sample"], attr_value)
            da = xr.DataArray(data, dims=["sample"], name=attr_name)
            ds[attr_name] = da
        elif attr_array.ndim == 1:
            # 1D array attribute
            # Create data of shape (number of samples, attribute length)
            data = np.tile(attr_array, (ds.sizes["sample"], 1))
            dims = ["sample", f"{attr_name}_dim"]
            da = xr.DataArray(data, dims=dims, name=attr_name)
            ds[attr_name] = da
        else:
            print(f"Skipping attribute '{attr_name}' with shape {attr_array.shape}")

    return ds
