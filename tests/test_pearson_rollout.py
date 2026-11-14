"""Exercise Pearson scores through the data loader, rollout, and saved curves."""

import h5py
import numpy as np
import pytest
import torch

from the_well.benchmark.metrics import RMSE
from the_well.benchmark.trainer.training import Trainer
from the_well.data import WellDataModule
from the_well.utils.dummy_data import write_dummy_data


@pytest.mark.parametrize("scale", [1.0, -1.0])
def test_stationary_field_pearson_rollout(tmp_path, scale):
    for split in ["train", "valid", "test"]:
        directory = tmp_path / "active_matter" / "data" / split
        directory.mkdir(parents=True)
        path = directory / "sample.h5"
        write_dummy_data(str(path))
        with h5py.File(path, "r+") as handle:
            fields = handle["t1_fields/field"]
            shape = fields.shape[-3:]
            stationary = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            fields[...] = np.broadcast_to(stationary, fields.shape)

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
        model.weight[0, 0, 0, 0] = scale
        model.weight[1, 1, 0, 0] = scale
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
    curve = np.load(
        tmp_path / "dummy_dataset/rollout_losses/epoch_0/full_PearsonR_rollout.npy"
    )
    expected = np.array([scale, scale * scale])
    if not np.allclose(curve, expected, rtol=0, atol=2e-6):
        pytest.fail(f"Pearson rollout curve {curve} differs from {expected}")
    score = losses["valid_dummy_dataset/full_PearsonR_T=all"]
    if not np.isclose(score, expected.mean(), rtol=0, atol=2e-6):
        pytest.fail(f"Pearson score {score} differs from {expected.mean()}")
