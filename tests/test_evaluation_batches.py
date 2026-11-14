"""Evaluate every delivered trajectory, including partial final batches."""

import h5py
import numpy as np
import pytest
import torch

from the_well.benchmark.metrics import MSE
from the_well.benchmark.trainer.training import Trainer
from the_well.data import WellDataModule
from the_well.utils.dummy_data import write_dummy_data


@pytest.fixture
def three_trajectories(tmp_path):
    pattern = np.ones((32, 32), dtype=np.float32)
    pattern[:, 1::2] = -1
    for split in ["train", "valid", "test"]:
        directory = tmp_path / "active_matter" / "data" / split
        directory.mkdir(parents=True)
        path = directory / "sample.h5"
        write_dummy_data(str(path))
        with h5py.File(path, "r+") as handle:
            handle.attrs["n_trajectories"] = 3
            replacements = {
                "t1_fields/field": np.broadcast_to(
                    np.array([1, 2, 9], dtype=np.float32)[:, None, None, None, None]
                    * pattern[None, None, :, :, None],
                    (3, 3, 32, 32, 2),
                ),
                "t0_fields/constant_field": np.zeros((3, 32, 32), dtype=np.float32),
                "dimensions/time": np.arange(3, dtype=np.float32),
                "scalars/time_varying_scalar": np.zeros(3),
            }
            for name, values in replacements.items():
                attrs = dict(handle[name].attrs)
                del handle[name]
                handle.create_dataset(name, data=values)
                handle[name].attrs.update(attrs)
    return tmp_path


@pytest.mark.parametrize(
    "loader_name,batch_size,full",
    [
        (name, size, True)
        for name in ["val_dataloader", "test_dataloader"]
        for size in [1, 2, 4]
    ]
    + [("test_dataloader", size, False) for size in [1, 2, 4]],
)
def test_evaluation_scores_and_saved_curves(
    three_trajectories, loader_name, batch_size, full
):
    root = three_trajectories
    datamodule = WellDataModule(
        well_base_path=str(root),
        well_dataset_name="active_matter",
        batch_size=batch_size,
        data_workers=0,
        n_steps_output=2,
        max_rollout_steps=2,
    )
    model = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    trainer = Trainer(
        checkpoint_folder=str(root),
        artifact_folder=str(root),
        viz_folder=str(root),
        formatter="channels_first_default",
        model=model,
        datamodule=datamodule,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=MSE(),
        epochs=1,
        checkpoint_frequency=1,
        val_frequency=1,
        rollout_val_frequency=1,
        max_rollout_steps=2,
        short_validation_length=1,
        make_rollout_videos=False,
        num_time_intervals=1,
        device=torch.device("cpu"),
    )
    score, _ = trainer.validation_loop(getattr(datamodule, loader_name)(), full=full)
    amplitudes = np.array([1.0, 2.0, 9.0])
    if not full:
        amplitudes = amplitudes[:batch_size]
    expected_mse = np.mean(amplitudes**2)
    expected_rmse = np.full(2, amplitudes.mean())
    if not np.isclose(score, expected_mse, rtol=1e-6, atol=1e-6):
        pytest.fail(
            f"MSE {score} differs from the evaluated sample mean {expected_mse}"
        )
    curve = np.load(root / "dummy_dataset/rollout_losses/epoch_0/full_RMSE_rollout.npy")
    if not np.allclose(curve, expected_rmse, rtol=1e-6, atol=1e-6):
        pytest.fail(f"RMSE curve {curve} differs from {expected_rmse}")


def test_training_and_rollout_batch_policies(three_trajectories):
    data = WellDataModule(
        well_base_path=str(three_trajectories),
        well_dataset_name="active_matter",
        batch_size=2,
        data_workers=0,
        n_steps_output=2,
    )
    training = data.train_dataloader()
    if (
        not training.drop_last
        or sum(len(batch["output_fields"]) for batch in training) != 2
    ):
        pytest.fail("Training must retain its full-batch sampling policy")
    for loader in [data.rollout_val_dataloader(), data.rollout_test_dataloader()]:
        batches = list(loader)
        if [len(batch["output_fields"]) for batch in batches] != [1, 1, 1]:
            pytest.fail("Rollout evaluation must retain one trajectory per batch")


def test_distributed_sampler_rows_keep_their_padding(three_trajectories):
    for rank, expected in [(0, [1.0, 9.0]), (1, [2.0, 1.0])]:
        for batch_size in [1, 2, 4]:
            data = WellDataModule(
                well_base_path=str(three_trajectories),
                well_dataset_name="active_matter",
                batch_size=batch_size,
                data_workers=0,
                n_steps_output=2,
                world_size=2,
                rank=rank,
            )
            rows = [
                float(amplitude)
                for batch in data.test_dataloader()
                for amplitude in batch["output_fields"][:, 0, 0, 0, 0]
            ]
            if rows != expected:
                pytest.fail(
                    f"Rank {rank} received {rows}, expected padded rows {expected}"
                )
