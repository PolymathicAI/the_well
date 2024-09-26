"""Create dummy data following the Well formating for testing purposes"""

import argparse

import h5py
import numpy as np


def write_dummy_data(filename: str):
    # Create dummy data
    param_a = 0.25
    param_b = 0.75
    dataset_name = "dummy_dataset"
    grid_type = "cartesian"
    n_spatial_dims = 2
    n_trajectories = 2
    dim_x = 32
    dim_y = 32
    dim_t = 10
    n_dim = 2
    x = np.linspace(0, 1, dim_x, dtype=np.float32)
    y = np.linspace(0, 1, dim_y, dtype=np.float32)
    t = np.linspace(0, 1, dim_t, dtype=np.float32)
    x_peridocity_mask = np.zeros_like(x).astype(bool)
    x_peridocity_mask[0] = x_peridocity_mask[-1]
    y_peridocity_mask = np.zeros_like(y).astype(bool)
    y_peridocity_mask[0] = y_peridocity_mask[-1]
    t1_field_values = np.random.rand(n_trajectories, dim_t, dim_x, dim_y, n_dim).astype(
        np.float32
    )
    t0_constant_field_values = np.random.rand(n_trajectories, dim_x, dim_y).astype(
        np.float32
    )
    time_varying_scalar_values = np.random.rand(dim_t)

    # Write the data in the HDF5 file
    with h5py.File(filename, "w") as file:
        # Attributes
        file.attrs["a"] = param_a
        file.attrs["b"] = param_b
        file.attrs["dataset_name"] = dataset_name
        file.attrs["grid_type"] = grid_type
        file.attrs["n_spatial_dims"] = n_spatial_dims
        file.attrs["n_trajectories"] = n_trajectories
        file.attrs["simulation_parameters"] = ["a", "b"]
        # Boundary Conditions
        group = file.create_group("boundary_conditions")
        for key, val in zip(
            ["x_periodic", "y_periodic"], [x_peridocity_mask, y_peridocity_mask]
        ):
            sub_group = group.create_group(key)
            sub_group.attrs["associated_dims"] = key[0]
            sub_group.attrs["associated_fields"] = []
            sub_group.attrs["bc_type"] = "PERIODIC"
            sub_group.attrs["sample_varying"] = False
            sub_group.attrs["time_varying"] = False
            sub_group.create_dataset("mask", data=val)
        # Dimensions
        group = file.create_group("dimensions")
        group.attrs["spatial_dims"] = ["x", "y"]
        for key, val in zip(["time", "x", "y"], [t, x, y]):
            group.create_dataset(key, data=val)
            group[key].attrs["sample_varying"] = False
        # Scalars
        group = file.create_group("scalars")
        group.attrs["field_names"] = ["a", "b", "time_varying_scalar"]
        for key, val in zip(["a", "b"], [param_a, param_b]):
            group.create_dataset(key, data=np.array(val))
            group[key].attrs["time_varying"] = False
            group[key].attrs["sample_varying"] = False
        ## Time varying
        dset = group.create_dataset(
            "time_varying_scalar", data=time_varying_scalar_values
        )
        dset.attrs["time_varying"] = True
        dset.attrs["sample_varying"] = False
        # Fields
        group = file.create_group("t0_fields")
        group.attrs["field_names"] = ["constant_field"]
        # Add a constant field regarding time
        dset = group.create_dataset("constant_field", data=t0_constant_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = False
        # Add a field varying both in time and space
        group = file.create_group("t1_fields")
        group.attrs["field_names"] = ["field"]
        dset = group.create_dataset("field", data=t1_field_values)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = True
        group = file.create_group("t2_fields")
        group.attrs["field_names"] = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Well dummy data creator")
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    filename = args.filename
    write_dummy_data(filename)
