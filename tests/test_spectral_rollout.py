"""Exercise spectral scores through physical fields, rollout, and saved curves."""

import h5py
import numpy as np
import pytest
import torch

from the_well.benchmark.metrics import RMSE
from the_well.benchmark.trainer.training import Trainer
from the_well.data import WellDataModule
from the_well.utils.dummy_data import write_dummy_data


def test_frequency_bands_in_saved_rollout(tmp_path):
    coordinates = np.arange(32, dtype=np.float64) / 32
    amplitudes = np.array([[1.0, 0.5, 0.25], [0.7, 0.3, 0.8]])
    waves = np.sin(2 * np.pi * coordinates[:, None] * np.array([1, 4, 12]))
    stationary = (waves @ amplitudes.T).astype(np.float32)
    for split in ["train", "valid", "test"]:
        directory = tmp_path / "active_matter" / "data" / split
        directory.mkdir(parents=True)
        path = directory / "sample.h5"
        write_dummy_data(str(path))
        with h5py.File(path, "r+") as handle:
            fields = handle["t1_fields/field"]
            fields[...] = np.broadcast_to(stationary[:, None, :], fields.shape)
            fields[1] *= 1.5

    datamodule = WellDataModule(
        well_base_path=str(tmp_path),
        well_dataset_name="active_matter",
        batch_size=1,
        data_workers=0,
        n_steps_output=2,
        max_rollout_steps=2,
    )
    model = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0, 0, 0] = 0.75
        model.weight[1, 1, 0, 0] = 0.75
    trainer = Trainer(
        checkpoint_folder=str(tmp_path),
        artifact_folder=str(tmp_path),
        viz_folder=str(tmp_path),
        formatter="channels_first_default",
        model=model,
        datamodule=datamodule,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=RMSE(),
        epochs=1,
        checkpoint_frequency=1,
        val_frequency=1,
        rollout_val_frequency=1,
        max_rollout_steps=2,
        short_validation_length=2,
        make_rollout_videos=False,
        num_time_intervals=1,
        device=torch.device("cpu"),
    )
    _, losses = trainer.validation_loop(datamodule.rollout_val_dataloader(), full=True)
    for band in range(3):
        # Each sine contributes amplitude squared / 2 over its complete periods.
        energy = (amplitudes[:, band] ** 2).mean() / 2 * (1 + 1.5**2) / 2
        expected = (0.75 ** np.array([1, 2]) - 1) ** 2 * energy
        curve = np.load(
            tmp_path
            / "dummy_dataset/rollout_losses/epoch_0"
            / f"full_spectral_error_mse_per_bin_{band}_rollout.npy"
        )
        if not np.allclose(curve, expected, rtol=3e-6, atol=1e-7):
            pytest.fail(f"Band {band} rollout {curve} differs from {expected}")
        score = losses[
            f"valid_dummy_dataset/full_spectral_error_mse_per_bin_{band}_T=all"
        ]
        if not np.isclose(score, expected.mean(), rtol=3e-6, atol=1e-7):
            pytest.fail(f"Band {band} score {score} differs from {expected.mean()}")
