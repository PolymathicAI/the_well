import argparse
import traceback
import warnings

import h5py as h5
import numpy as np


def check_dataset(
    f: h5.File,
    group: h5.Group,
    key: str,
    is_spatial: bool = False,
    is_field: bool = False,
    is_bc: bool = False,
    order: int = 0,
):
    assert key in group, f"{group.name} should contain '{key}' dataset"

    dataset = group[key]

    # Attrs
    for attr in ("sample_varying", "time_varying"):
        assert (
            attr in dataset.attrs
        ), f"{dataset.name} should contain '{attr}' attribute"
        assert isinstance(
            dataset.attrs[attr], (bool, np.bool_)
        ), f"attribute '{attr}' in {dataset.name} should be a boolean"

    if is_field:
        attr = "dim_varying"
        assert (
            attr in dataset.attrs
        ), f"{dataset.name} should contain '{attr}' attribute"
        assert isinstance(
            dataset.attrs[attr], (list, np.ndarray)
        ), f"attribute '{attr}' in {dataset.name} should be a list of booleans"
        assert (
            len(dataset.attrs[attr]) == f.attrs["n_spatial_dims"]
        ), f"attribute '{attr}' in {dataset.name} should be of length 'n_spatial_dims'"

    if is_bc:
        assert (
            "bc_type" in dataset.attrs
        ), f"{dataset.name} should contain 'bc_type' attribute"

        bc_type = dataset.attrs["bc_type"]

        assert isinstance(
            bc_type, str
        ), f"attribute 'bc_type' in {dataset.name} should be a string"

        if bc_type.lower() not in dataset.name:
            warnings.warn(f"{dataset.name} is not named after 'bc_type' ({bc_type})")

        for attr in ("associated_fields", "associated_dims"):
            assert (
                attr in dataset.attrs
            ), f"{dataset.name} should contain '{attr}' attribute"
            assert isinstance(
                dataset.attrs[attr], (list, np.ndarray)
            ), f"attribute '{attr}' in {dataset.name} should be a list of strings"

        associated_dims = dataset.attrs["associated_dims"]

        for dim in associated_dims:
            assert isinstance(
                dim, str
            ), f"{dim} in 'associated_dims' in {dataset.name} should be a string"
            assert (
                dim in f["dimensions"].attrs["spatial_dims"]
            ), f"{dim} in 'associated_dims' in {dataset.name} should be in 'spatial_dims'"

            if dim not in dataset.name:
                warnings.warn(
                    f"{dataset.name} is not named after 'associated_dims' ({associated_dims})"
                )

        assert "mask" in dataset, f"{dataset.name} should contain 'mask' dataset"

    # Shape
    expected = ()

    if dataset.attrs["sample_varying"]:
        expected = (*expected, f.attrs["n_trajectories"])

    if dataset.attrs["time_varying"]:
        expected = (*expected, f["dimensions"]["time"].shape[-1])

    if is_field:
        for i, dim in enumerate(f["dimensions"].attrs["spatial_dims"]):
            if dataset.attrs["dim_varying"][i]:
                expected = (*expected, f["dimensions"][dim].shape[-1])

        for _ in range(order):
            expected = (*expected, f.attrs["n_spatial_dims"])

    if is_bc:
        for i, dim in enumerate(f["dimensions"].attrs["spatial_dims"]):
            if dim in dataset.attrs["associated_dims"]:
                expected = (*expected, f["dimensions"][dim].shape[-1])

    if is_spatial:
        current = dataset.shape[:-1]
    elif is_bc:
        current = dataset["mask"].shape
    else:
        current = dataset.shape

    assert (
        current == expected
    ), f"{dataset.name} has shape {current}, expected {expected}"


def check_dimensions(f: h5.File):
    assert "dimensions" in f, "'dimensions' should be a root group"
    group = f["dimensions"]

    check_dataset(f, group, "time")

    assert (
        "spatial_dims" in group.attrs
    ), f"{group.name} should contain 'spatial_dims' attribute"
    assert isinstance(
        group.attrs["spatial_dims"], (list, np.ndarray)
    ), f"attribute 'spatial_dims' in {group.name} should be a list"
    assert (
        len(group.attrs["spatial_dims"]) == f.attrs["n_spatial_dims"]
    ), f"attribute 'spatial_dims' in {group.name} should be of length 'n_spatial_dims'"

    for key in group.attrs["spatial_dims"]:
        assert isinstance(
            key, str
        ), f"{key} in 'spatial_dims' in {group.name} should be a string"

        check_dataset(f, group, key, is_spatial=True)

    print(f"{group.name} passed!")


def check_fields(f: h5.File, i: int):
    assert f"t{i}_fields" in f, f"'t{i}_fields' should be a root group"
    group = f[f"t{i}_fields"]

    assert (
        "field_names" in group.attrs
    ), f"{group.name} should contain 'field_names' attribute"
    assert isinstance(
        group.attrs["field_names"], (list, np.ndarray)
    ), f"attribute 'field_names' in {group.name} should be a list"

    for key in group.attrs["field_names"]:
        assert isinstance(
            key, str
        ), f"{key} in 'field_names' in {group.name} should be a string"

        check_dataset(f, group, key, is_field=True, order=i)

    print(f"{group.name} passed!")


def check_scalars(f: h5.File):
    assert "scalars" in f, "'scalars' should be a root group"
    group = f["scalars"]

    assert (
        "field_names" in group.attrs
    ), f"{group.name} should contain 'field_names' attribute"
    assert isinstance(
        group.attrs["field_names"], (list, np.ndarray)
    ), f"attribute 'field_names' in {group.name} should be a list"

    for key in group.attrs["field_names"]:
        assert isinstance(
            key, str
        ), f"{key} in 'field_names' in {group.name} should be a string"

        check_dataset(f, group, key)

    print(f"{group.name} passed!")


def check_boundary_conditions(f):
    assert "boundary_conditions" in f, "'boundary_conditions' should be a root group"
    group = f["boundary_conditions"]

    for key in group:
        check_dataset(f, group, key, is_bc=True)

    print(f"{group.name} passed!")


def check_hdf5_format(path: str):
    """Check that the HDF5 file is in the correct format for the well dataset"""

    print(f"Checking the format of {path}")

    with h5.File(path, "r") as f:
        # Top level attributes
        assert "dataset_name" in f.attrs, "'dataset_name' should be a root attribute"
        assert isinstance(
            f.attrs["dataset_name"], str
        ), "attribute 'dataset_name' should be a string"

        assert (
            "n_spatial_dims" in f.attrs
        ), "'n_spatial_dims' should be a root attribute"
        assert isinstance(
            f.attrs["n_spatial_dims"], (int, np.integer)
        ), "attribute 'n_spatial_dims' should be an integer"

        assert (
            "n_trajectories" in f.attrs
        ), "'n_trajectories' should be a root attribute"
        assert isinstance(
            f.attrs["n_trajectories"], (int, np.integer)
        ), "attribute 'n_trajectories' should be an integer"

        assert "grid_type" in f.attrs, "'grid_type' should be a root attribute"
        assert isinstance(
            f.attrs["grid_type"], str
        ), "attribute 'grid_type' should be a string"

        assert (
            "simulation_parameters" in f.attrs
        ), "'simulation_parameters' should be a root attribute"
        for key in f.attrs["simulation_parameters"]:
            assert isinstance(
                key, str
            ), f"'{key}' in 'simulation_parameters' should be a string"
            assert (
                key in f.attrs
            ), f"'{key}' listed in 'simulation_parameters' should be a root attribute"
            assert (
                key in f["scalars"]
            ), f"'{key}' listed in 'simulation_parameters' should be a dataset in /scalars"

        # Groups
        check_dimensions(f)
        for i in range(3):
            check_fields(f, i)
        check_scalars(f)
        check_boundary_conditions(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Check HDF5 format validity")
    parser.add_argument("filenames", nargs="+", type=str)
    args = parser.parse_args()

    for filename in args.filenames:
        try:
            check_hdf5_format(filename)
        except AssertionError:
            print(traceback.format_exc(), end="")
        print()
