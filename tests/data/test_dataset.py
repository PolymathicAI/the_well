import pathlib
import random

import numpy as np
import pytest
import torch

from the_well.data.augmentation import (
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)
from the_well.data.datasets import DeltaWellDataset, WellDataset, WellMetadata
from the_well.data.normalization import RMSNormalization, ZScoreNormalization
from the_well.data.utils import (
    maximum_stride_for_initial_index,
    raw_steps_to_possible_sample_t0s,
)
from the_well.utils.dummy_data import write_dummy_data


def test_metadata():
    metadata = WellMetadata(
        dataset_name="test",
        n_spatial_dims=2,
        spatial_resolution=(256, 256),
        scalar_names=["whatever"],
        constant_scalar_names=["alpha", "beta"],
        field_names={0: ["energy"], 1: ["v_x", "v_y"]},
        constant_field_names={2: ["t_xx", "t_xy", "t_yx", "t_yy"]},
        boundary_condition_types=["periodic"],
        n_files=1,
        n_trajectories_per_file=[16],
        n_steps_per_trajectory=[100],
    )

    assert metadata.n_scalars == 1
    assert metadata.n_constant_scalars == 2
    assert metadata.n_fields == 3
    assert metadata.n_constant_fields == 4

    shapes = metadata.sample_shapes

    assert shapes["input_scalars"] == [1]
    assert shapes["output_scalars"] == [1]
    assert shapes["constant_scalars"] == [2]

    assert shapes["input_fields"] == [256, 256, 3]
    assert shapes["output_fields"] == [256, 256, 3]
    assert shapes["constant_fields"] == [256, 256, 4]


def test_local_dataset(download_dataset):
    dataset_dir: pathlib.Path = download_dataset
    base_dataset_dir = dataset_dir.parent
    dataset_name = dataset_dir.name

    dataset = WellDataset(
        well_base_path=base_dataset_dir,
        well_dataset_name=dataset_name,
        well_split_name="train",
        use_normalization=False,
    )
    assert len(dataset) > 0


def test_absolute_path_dataset(download_dataset):
    dataset_dir: pathlib.Path = download_dataset
    path = str(dataset_dir / "data" / "train")
    dataset = WellDataset(path=path, use_normalization=False)
    assert len(dataset) > 0


def test_last_time_step(download_dataset):
    dataset_dir: pathlib.Path = download_dataset
    base_dataset_dir = dataset_dir.parent
    dataset_name = dataset_dir.name

    dataset = WellDataset(
        well_base_path=base_dataset_dir,
        well_dataset_name=dataset_name,
        well_split_name="train",
        use_normalization=False,
    )
    n_time_steps = dataset.n_steps_per_trajectory[0] - 1
    data = dataset[n_time_steps]
    assert "input_fields" in data
    assert "output_fields" in data

    data = dataset[len(dataset) - 1]
    assert "input_fields" in data
    assert "output_fields" in data


@pytest.mark.parametrize("dataset_cls", [DeltaWellDataset, WellDataset])
@pytest.mark.parametrize("normalization_type", [RMSNormalization, ZScoreNormalization])
def test_dataset_with_normalization(download_dataset, dataset_cls, normalization_type):
    dataset_dir: pathlib.Path = download_dataset
    base_dataset_dir = dataset_dir.parent
    dataset_name = dataset_dir.name

    dataset = dataset_cls(
        well_base_path=base_dataset_dir,
        well_dataset_name=dataset_name,
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=normalization_type,
    )
    assert len(dataset) > 0


def test_dataset_with_augmentation(download_dataset):
    dataset_dir: pathlib.Path = download_dataset
    base_dataset_dir = dataset_dir.parent
    dataset_name = dataset_dir.name

    dataset = WellDataset(
        well_base_path=base_dataset_dir,
        well_dataset_name=dataset_name,
        well_split_name="train",
        use_normalization=False,
        transform=Compose(
            RandomAxisFlip(),
            RandomAxisPermute(),
            RandomAxisRoll(),
        ),
    )

    i = random.randrange(len(dataset))
    data = dataset[i]

    assert "input_fields" in data
    assert "output_fields" in data


def test_adjust_available_steps():
    # ex1: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 1
    #  Possible samples are: [0, 1], [1, 2], [2, 3], [3, 4], return 4
    assert raw_steps_to_possible_sample_t0s(5, 1, 1, 1) == 4
    # ex2: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 2
    #  Possible samples are: [0, 2], [1, 3], [2, 4], return 3
    assert raw_steps_to_possible_sample_t0s(5, 1, 1, 2) == 3
    # ex3: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 3
    #  Possible samples are: [0, 3], [1, 4], return 2
    assert raw_steps_to_possible_sample_t0s(5, 1, 1, 3) == 2
    # ex4: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 1, dt_stride = 2
    #  Possible samples are: [0, 2, 4], return 1
    assert raw_steps_to_possible_sample_t0s(5, 2, 1, 2) == 1
    # ex5: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 2, dt_stride = 2
    #   No possible samples, return 0
    assert raw_steps_to_possible_sample_t0s(5, 2, 2, 2) == 0
    # ex6: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 10, dt_stride = 2
    #  No possible samples, return 0
    assert raw_steps_to_possible_sample_t0s(5, 2, 10, 2) == 0


def test_maximum_stride_for_initial_index():
    # ex1: time_idx=0, total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1
    #   Maximum stride is 4 - [0, 4]
    assert maximum_stride_for_initial_index(0, 5, 1, 1) == 4
    # ex2: time_idx=2, total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1
    #   Maximum stride is 2, [2, 4]
    assert maximum_stride_for_initial_index(2, 5, 1, 1) == 2
    # ex3: time_idx=1, total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1
    #   Maximum stride is 3, [1, 4]
    assert maximum_stride_for_initial_index(1, 5, 1, 1) == 3
    # ex4: time_idx=1, total_steps_in_trajectory = 5, n_steps_input = 5, n_steps_output = 1
    #   Maximum stride is 0
    assert maximum_stride_for_initial_index(5, 5, 1, 1) == 0
    # ex5: time_idx=5, total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 2
    #   Maximum stride is 0
    assert maximum_stride_for_initial_index(5, 5, 1, 1) == 0


def test_dummy_dataset(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset = WellDataset(path=str(tmp_path), use_normalization=False, return_grid=True)
    # Dummy dataset should contain 2 trajectories of 9 valid samples each
    assert len(dataset) == 2 * 9

    data = dataset[0]

    for key in (
        "input_fields",
        "output_fields",
        "constant_fields",
        "input_scalars",
        "output_scalars",
        "constant_scalars",
    ):
        assert key in data

    for key, shape in dataset.metadata.sample_shapes.items():
        if "input" in key or "output" in key:
            assert list(data[key].shape[1:]) == shape
        else:
            assert list(data[key].shape) == shape

    data_next = dataset[1]

    for key in (
        "input_fields",
        "output_fields",
        "constant_fields",
        "input_scalars",
        "output_scalars",
        "constant_scalars",
    ):
        if "constant" in key:
            assert torch.equal(data[key], data_next[key])
        else:
            assert not torch.equal(data[key], data_next[key])


def test_restricted_traj_int(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_trajectories=1,
    )
    # Dummy dataset should contain 2 trajectories of 9 valid samples each, 1 after restriction

    assert (
        len(dataset) == 1 * 9
    ), f"Restricted dataset should contain 1 trajectories of 9 valid samples each, but found {len(dataset)}"
    # Make sure we can still pull data
    data = dataset[0]
    assert data is not None


def test_restricted_traj_float(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_trajectories=0.5,
    )
    assert (
        len(dataset) == 1 * 9
    ), f"Restricted dataset should contain 1 trajectories of 9 valid samples each, but found {len(dataset)}"
    data = dataset[0]
    assert data is not None


def test_restricted_samples_int(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_samples=5,
    )
    assert (
        len(dataset) == 5
    ), f"Restricted dataset should contain 5 total samples, but found {len(dataset)}"
    data = dataset[0]
    assert data is not None


def test_same_seed_leads_to_same_restriction_samples(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset1 = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_samples=5,
        restriction_seed=42,
    )
    dataset2 = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_samples=5,
        restriction_seed=42,
    )
    assert np.all(
        dataset1.restriction_set == dataset2.restriction_set
    ), "Restriction sets should be identical for same seed"


def test_same_seed_leads_to_same_restriction_trajectories(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset1 = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_trajectories=1,
        restriction_seed=42,
    )
    dataset2 = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_trajectories=1,
        restriction_seed=42,
    )
    assert np.all(
        dataset1.restriction_set == dataset2.restriction_set
    ), "Restriction sets should be identical for same seed"


def test_restricted_samples_float(tmp_path):
    filename = tmp_path / "dummy_well_data.hdf5"
    write_dummy_data(filename)
    dataset = WellDataset(
        path=str(tmp_path),
        use_normalization=False,
        return_grid=True,
        restrict_num_trajectories=0.5,
    )
    assert (
        len(dataset) == 1 * 9
    ), f"Restricted dataset should contain .5*18 samples, but found {len(dataset)}"
    data = dataset[0]
    assert data is not None
