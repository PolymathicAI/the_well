import argparse
import traceback

import h5py as h5
import numpy as np


def check_dataset(
    f: h5.File,
    group: h5.Group,
    key: str,
    is_time: bool = False,
    is_field: bool = False,
    order: int = 0,
):
    assert key in group, f"{group.name} does not contain {key} dataset"

    dataset = group[key]

    # Attrs
    for attr in ("sample_varying", "time_varying"):
        if is_time and attr == "time_varying":
            continue  # TODO: replace with check that time_varying is False

        assert (
            attr in dataset.attrs
        ), f"dataset {key} in {group.name} does not contain '{attr}' attribute"
        assert isinstance(
            dataset.attrs[attr], (bool, np.bool_)
        ), f"attribute '{attr}' in dataset {key} in {group.name} is not a boolean"

    if is_field:
        attr = "dim_varying"
        assert (
            attr in dataset.attrs
        ), f"dataset {key} in {group.name} does not contain '{attr}' attribute"
        assert isinstance(
            dataset.attrs[attr], (list, np.ndarray)
        ), f"attribute '{attr}' in dataset {key} in {group.name} is not a list of booleans"
        assert (
            len(dataset.attrs[attr]) == f.attrs["n_spatial_dims"]
        ), f"attribute '{attr}' in dataset {key} in {group.name} is not of length 'n_spatial_dims'"

    # Shape
    current = dataset.shape[: dataset.ndim - order]
    expected = ()

    if dataset.attrs["sample_varying"]:
        expected = (*expected, f.attrs["n_trajectories"])

    if is_time or dataset.attrs["time_varying"]:
        expected = (*expected, f["dimensions"]["time"].shape[-1])

    if is_field:
        for i, dim in enumerate(f["dimensions"].attrs["spatial_dims"]):
            if dataset.attrs["dim_varying"][i]:
                expected = (*expected, f["dimensions"][dim].shape[-1])

    assert (
        current == expected
    ), f"dataset {key} in {group.name} has shape {current}, expected {expected}"


def check_dimensions(f: h5.File):
    group = f["dimensions"]

    check_dataset(f, group, "time", is_time=True)

    assert (
        "spatial_dims" in group.attrs
    ), f"{group.name} does not contain 'spatial_dims' attribute"
    assert isinstance(
        group.attrs["spatial_dims"], (list, np.ndarray)
    ), f"attribute 'spatial_dims' in {group.name} is not a list"
    assert (
        len(group.attrs["spatial_dims"]) == f.attrs["n_spatial_dims"]
    ), f"attribute 'spatial_dims' in {group.name} is not of length 'n_spatial_dims'"

    for key in group.attrs["spatial_dims"]:
        assert isinstance(
            key, str
        ), f"{key} in 'spatial_dims' in {group.name} is not a string"

        check_dataset(f, group, key, order=1)

    print(f"{group.name} passed!")


def check_fields(f: h5.File, i: int):
    group = f[f"t{i}_fields"]

    assert (
        "field_names" in group.attrs
    ), f"{group.name} does not contain 'field_names' attribute"
    assert isinstance(
        group.attrs["field_names"], (list, np.ndarray)
    ), f"attribute 'field_names' in {group.name} is not a list"

    for key in group.attrs["field_names"]:
        assert isinstance(
            key, str
        ), f"{key} in 'field_names' in {group.name} is not a string"

        check_dataset(f, group, key, is_field=True, order=i)

    print(f"{group.name} passed!")


def check_scalars(f: h5.File):
    group = f["scalars"]

    assert (
        "field_names" in group.attrs
    ), f"{group.name} does not contain 'field_names' attribute"
    assert isinstance(
        group.attrs["field_names"], (list, np.ndarray)
    ), f"attribute 'field_names' in {group.name} is not a list"

    for key in group.attrs["field_names"]:
        assert isinstance(
            key, str
        ), f"{key} in 'field_names' in {group.name} is not a string"

        check_dataset(f, group, key, order=0)

    print(f"{group.name} passed!")


def check_boundary_conditions(f):  # TODO: refactor when bc are datasets
    bcs = f["boundary_conditions"]
    dimensions = f["dimensions"]
    spatial_dims = dimensions.attrs["spatial_dims"]
    for key in f["boundary_conditions"]:
        bc = bcs[key]
        assert "bc_type" in bc.attrs, "Group must contain a 'bc_type' attribute"
        assert (
            "sample_varying" in bc.attrs
        ), "Group must contain a 'sample_varying' attribute"
        assert isinstance(
            bc.attrs["sample_varying"], (bool, np.bool_)
        ), "Attribute 'sample_varying' must be a boolean"
        assert (
            "time_varying" in bc.attrs
        ), "Group must contain a 'time_varying' attribute"
        assert isinstance(
            bc.attrs["time_varying"], (bool, np.bool_)
        ), "Attribute 'time_varying' must be a boolean"
        assert (
            "associated_fields" in bc.attrs
        ), "Group must contain a 'associated_fields' attribute"
        dim_count = bc.attrs["sample_varying"] + bc.attrs["time_varying"]
        assert (
            "associated_dims" in bc.attrs
        ), "Group must contain a 'associated_dims' attribute"
        assert isinstance(
            bc.attrs["associated_dims"], (list, np.ndarray)
        ), "Attribute 'associated_dims' must be a list"
        assert "mask" in bc, "Group must contain a 'mask' dataset"
        assert (
            len(bc.attrs["associated_dims"]) > 0
        ), "Attribute 'associated_dims' must have at least one entry"
        for i, dim in enumerate(bc.attrs["associated_dims"]):
            assert dim in spatial_dims, f"Dimension {dim} not found in 'spatial_dims'"
            assert (
                bc["mask"].shape[dim_count + i] == dimensions[dim].shape[-1]
            ), f"Dimension {dim} size does not match mask size"

        print(f"{bc.name} passed!")


def check_hdf5_format(path: str):
    """Check that the HDF5 file is in the correct format for the well dataset"""
    with h5.File(path, "r") as f:
        # Start by checking top level attributes
        print(f"Checking top level attributes of {path}")
        assert "n_spatial_dims" in f.attrs, "n_spatial_dims is required root attribute"
        assert isinstance(
            int(f.attrs["n_spatial_dims"]), (int, np.integer)
        ), "n_spatial_dims must be an integer"
        assert "n_trajectories" in f.attrs, "n_trajectories is required root attribute"
        assert isinstance(
            f.attrs["n_trajectories"], (int, np.integer)
        ), "n_trajectories must be an integer"
        assert "dataset_name" in f.attrs, "dataset_name is required root attribute"
        assert isinstance(f.attrs["dataset_name"], str), "dataset_name must be a string"
        assert "grid_type" in f.attrs, "grid_type is required root attribute"
        assert isinstance(f.attrs["grid_type"], str), "grid_type must be a string"
        assert (
            "simulation_parameters" in f.attrs
        ), "simulation_parameters is required root attribute"
        for sim_param in f.attrs["simulation_parameters"]:
            assert (
                sim_param in f.attrs
            ), "Every listed simulation parameter should be included at attribute"
        print("Checking groups")
        assert "dimensions" in f, "No dimensions group found in HDF5 file"
        check_dimensions(f)
        for i in range(3):
            assert f"t{i}_fields" in f, f"No t{i}_fields group found in HDF5 file"
            check_fields(f, i)
        assert "scalars" in f, "No scalars found in HDF5 file"
        check_scalars(f)
        assert "boundary_conditions" in f, "No boundary_conditions found in HDF5 file"
        check_boundary_conditions(f)
        print("HDF5 file validation passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Check HDF5 format validity")
    parser.add_argument("filenames", nargs="+", type=str)
    args = parser.parse_args()

    for filename in args.filenames:
        try:
            check_hdf5_format(filename)
        except AssertionError:
            print(traceback.format_exc())
        print()
