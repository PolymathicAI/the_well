"""Keep configured training losses and their evaluation records consistent."""

import copy
import os
import random
from functools import partial

import h5py
import numpy as np
import pytest
import torch
import wandb

from the_well.benchmark.metrics import MSE, NRMSE, VRMSE, LInfinity
from the_well.benchmark.metrics import RMSE as RootMeanSquaredError
from the_well.benchmark.metrics.common import Metric
from the_well.benchmark.trainer.training import Trainer
from the_well.data import WellDataModule
from the_well.utils.dummy_data import write_dummy_data


class OffsetMSE(Metric):
    @staticmethod
    def eval(x, y, meta):
        return MSE.eval(x, y, meta) + 7


class binned_spectral_mse(Metric):
    """A scalar loss whose class name matches a distinct dictionary-valued metric."""

    @staticmethod
    def eval(x, y, meta):
        return MSE.eval(x, y, meta)


@pytest.fixture
def validation_data(tmp_path):
    pattern = np.ones((32, 32), dtype=np.float32)
    pattern[:, 1::2] = -1
    amplitudes = np.array([1.0, 3.0], dtype=np.float32)
    for split in ["train", "valid", "test"]:
        directory = tmp_path / "active_matter" / "data" / split
        directory.mkdir(parents=True)
        path = directory / "sample.h5"
        write_dummy_data(str(path))
        with h5py.File(path, "r+") as handle:
            fields = handle["t1_fields/field"]
            fields[...] = np.broadcast_to(
                amplitudes[:, None, None, None, None] * pattern[None, None, :, :, None],
                fields.shape,
            )
    data = WellDataModule(
        well_base_path=str(tmp_path),
        well_dataset_name="active_matter",
        batch_size=1,
        data_workers=0,
        n_steps_output=2,
        max_rollout_steps=2,
    )
    return data


def test_loss_configuration_and_repeated_trainer_construction(
    tmp_path, validation_data
):
    data = validation_data
    amplitudes = np.array([1.0, 3.0], dtype=np.float32)
    configured_rmse = RootMeanSquaredError()
    configured_rmse.register_forward_hook(lambda module, inputs, output: 3 * output)
    variance = amplitudes.astype(np.float64) ** 2 * 1024 / 1023
    expected_vrmse = np.sqrt(amplitudes**2 / (variance + 1e-7)).mean()
    cases = [
        (RootMeanSquaredError(), 2.0, 2.0),
        (MSE(), 5.0, 2.0),
        (VRMSE(), expected_vrmse, 2.0),
        (MSE(), 5.0, 2.0),
        (NRMSE(), np.sqrt(amplitudes**2 / (amplitudes**2 + 1e-7)).mean(), 2.0),
        (LInfinity(), 2.0, 2.0),
        (configured_rmse, 6.0, 6.0),
        (MSE(), 5.0, 2.0),
        (OffsetMSE(), 12.0, 2.0),
        (
            partial(VRMSE(), eps=5.0),
            np.sqrt(amplitudes**2 / (variance + 5.0)).mean(),
            2.0,
        ),
        (binned_spectral_mse(), 5.0, 2.0),
    ]
    for index, (loss_fn, expected_loss, expected_rmse) in enumerate(cases):
        model = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        output = tmp_path / f"evaluation-{index}"
        trainer = Trainer(
            checkpoint_folder=str(output),
            artifact_folder=str(output),
            viz_folder=str(output),
            formatter="channels_first_default",
            model=model,
            datamodule=data,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            loss_fn=loss_fn,
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
        training_loss, _ = trainer.train_one_epoch(1, data.train_dataloader())
        validation_loss, logs = trainer.validation_loop(
            data.rollout_test_dataloader(), full=True
        )
        for phase, value in [
            ("training", training_loss),
            ("validation", validation_loss),
        ]:
            if not np.isclose(value, expected_loss, rtol=2e-6, atol=1e-6):
                pytest.fail(
                    f"Case {index} {phase} loss {value} differs from {expected_loss}"
                )
        curve = np.load(
            output / "dummy_dataset/rollout_losses/epoch_0/full_RMSE_rollout.npy"
        )
        if not np.allclose(curve, expected_rmse, rtol=2e-6, atol=1e-6):
            pytest.fail(f"Case {index} RMSE curve {curve} differs from {expected_rmse}")
        spectral = logs["valid_dummy_dataset/full_spectral_error_mse_per_bin_2_T=all"]
        if not np.isclose(spectral, 5.0, rtol=2e-6, atol=1e-6):
            pytest.fail(
                f"Case {index} changed the default spectral score to {spectral}"
            )
        if type(loss_fn) is VRMSE:
            curve = np.load(
                output / "dummy_dataset/rollout_losses/epoch_0/full_VRMSE_rollout.npy"
            )
            if not np.allclose(curve, expected_vrmse, rtol=2e-6, atol=1e-6):
                pytest.fail(f"VRMSE curve {curve} differs from {expected_vrmse}")


def resume_trainer(
    data,
    output,
    checkpoint="",
    loss_scale=1,
    epochs=2,
    evaluate_only=False,
    loss_type=RootMeanSquaredError,
):
    output.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Conv2d(3, 2, 1, bias=False)
    model.register_buffer("evaluations", torch.tensor(0), persistent=False)
    with torch.no_grad():
        model.weight.zero_()
    loss = loss_type()
    loss.register_forward_hook(lambda module, inputs, result: loss_scale * result)
    return Trainer(
        checkpoint_folder=str(output),
        artifact_folder=str(output),
        viz_folder=str(output),
        formatter="channels_first_default",
        model=model,
        datamodule=data,
        optimizer=(
            None
            if evaluate_only
            else torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9)
        ),
        loss_fn=loss,
        epochs=epochs,
        checkpoint_frequency=1,
        val_frequency=1,
        rollout_val_frequency=100,
        max_rollout_steps=1,
        short_validation_length=2,
        make_rollout_videos=False,
        num_time_intervals=1,
        device=torch.device("cpu"),
        checkpoint_path=str(checkpoint),
    )


def legacy_checkpoints(trainer, current_weight, loss_scale=1, best_score=None):
    # Legacy best.pt was saved before best_val_loss was updated.
    checkpoint = {
        "epoch": 1,
        "model_state_dict": copy.deepcopy(trainer.model.state_dict()),
        "optimizer_state_dit": trainer.optimizer.state_dict(),
        "validation_loss": 2 * (1 + loss_scale),
        "best_validation_loss": None,
    }
    output = trainer.checkpoint_folder
    torch.save(checkpoint, f"{output}/best.pt")
    with torch.no_grad():
        trainer.model.weight[0, 0, 0, 0] = current_weight
        trainer.model.weight[1, 1, 0, 0] = current_weight
    checkpoint.update(
        model_state_dict=trainer.model.state_dict(),
        validation_loss=2 * abs(1 - current_weight) * (1 + loss_scale),
        best_validation_loss=best_score,
    )
    torch.save(checkpoint, f"{output}/recent.pt")


@pytest.mark.parametrize(
    "loss_scale,current_weight,old_best_score,expected_best,replace_best",
    [(1, -0.5, 4, 2, False), (1, -0.5, None, 2, False), (3, 0.25, 8, 4.5, True)],
)
def test_resume_selects_best_with_current_loss(
    tmp_path,
    validation_data,
    loss_scale,
    current_weight,
    old_best_score,
    expected_best,
    replace_best,
):
    output = tmp_path / "resume"
    previous = resume_trainer(validation_data, output, loss_scale=loss_scale)
    legacy_checkpoints(previous, current_weight, loss_scale, old_best_score)
    original_best = (output / "best.pt").read_bytes()
    trainer = resume_trainer(
        validation_data, output, output / "recent.pt", loss_scale=loss_scale
    )
    with wandb.init(mode="disabled"):
        trainer.train()
    if not np.isclose(trainer.best_val_loss, expected_best, atol=1e-6):
        pytest.fail(f"Resumed best score {trainer.best_val_loss} != {expected_best}")
    before_validation = torch.load(output / "checkpoint_2.pt", weights_only=False)
    if before_validation["validation_loss_version"] != 0:
        pytest.fail("A pre-validation checkpoint marked legacy scores as current")
    best = torch.load(output / "best.pt", weights_only=False)
    if replace_best:
        for name in ["validation_loss", "best_validation_loss"]:
            if not np.isclose(best[name], expected_best, atol=1e-6):
                pytest.fail(f"New best checkpoint persisted {name}={best[name]}")
        if best["validation_loss_version"] != 1:
            pytest.fail("New best checkpoint has no current score definition")
        reloaded = resume_trainer(validation_data, output, output / "best.pt")
        if reloaded.best_val_loss != best["validation_loss"]:
            pytest.fail("Reloading best.pt restored a threshold for different weights")
    elif (output / "best.pt").read_bytes() != original_best:
        pytest.fail("Resuming overwrote or rewrote the better legacy best checkpoint")
    if float(trainer.model.weight[0, 0, 0, 0].detach()) != current_weight:
        pytest.fail("Recalibration replaced the model being trained")
    if any("epoch_0" in str(path) for path in output.rglob("*")):
        pytest.fail("Recalibration wrote extra epoch-zero artifacts")


@pytest.mark.parametrize("data_workers", [0, 2])
def test_resume_replays_short_validation_without_changing_training_state(
    tmp_path,
    validation_data,
    data_workers,
):
    validation_data.data_workers = data_workers
    output = tmp_path / "short-resume"
    previous = resume_trainer(validation_data, output)
    legacy_checkpoints(previous, -4.0, best_score=4.0)
    legacy_recent = torch.load(output / "recent.pt", weights_only=False)
    resumed = resume_trainer(validation_data, output, output / "recent.pt", epochs=3)
    observations = []

    def observe(module, inputs):
        if not module.training:
            module.evaluations = module.evaluations + 1
        observations.append(
            (
                module.training,
                float(module.weight[0, 0, 0, 0].detach()),
                float(inputs[0][0, 0, 0, 0]),
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
        )

    resumed.model.register_forward_pre_hook(observe)
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    with wandb.init(mode="disabled"):
        resumed.train()
    evaluated = [entry for entry in observations if not entry[0]]
    calibration = evaluated[:2]
    candidate = evaluated[2:4]
    if [entry[2:] for entry in calibration] != [entry[2:] for entry in candidate]:
        pytest.fail("Calibration changed the short-validation samples or RNG stream")
    expected_best = np.mean([abs(entry[2]) for entry in calibration])
    if not np.isclose(resumed.best_val_loss, expected_best, atol=1e-6):
        pytest.fail(
            "Legacy best was scored outside the current short-validation subset"
        )
    random_after = (random.random(), np.random.random(), float(torch.rand(())))
    current = torch.load(output / "checkpoint_3.pt", weights_only=False)
    if current["validation_loss_version"] != 1:
        pytest.fail("The next checkpoint did not persist the calibrated definition")
    if not np.isclose(current["validation_loss"], 5 * expected_best, atol=1e-5):
        pytest.fail("The next checkpoint retained the previous validation definition")
    if not np.isclose(current["best_validation_loss"], expected_best, atol=1e-6):
        pytest.fail("The next checkpoint retained the previous best definition")

    control_output = tmp_path / "current-definition"
    control_output.mkdir()
    legacy_recent.update(validation_loss_version=1, best_validation_loss=expected_best)
    torch.save(legacy_recent, control_output / "recent.pt")
    control = resume_trainer(
        validation_data, control_output, control_output / "recent.pt", epochs=3
    )
    observations.clear()
    control.model.register_forward_pre_hook(observe)
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    with wandb.init(mode="disabled"):
        control.train()
    if random_after != (random.random(), np.random.random(), float(torch.rand(()))):
        pytest.fail("Recalibration changed subsequent RNG state")
    if not torch.equal(resumed.model.evaluations, control.model.evaluations):
        pytest.fail("Recalibration leaked non-persistent buffer changes")
    for name, value in resumed.model.state_dict().items():
        if not torch.equal(value, control.model.state_dict()[name]):
            pytest.fail(f"Recalibration changed model state {name}")
    for parameter, state in resumed.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            if not torch.equal(
                value, control.optimizer.state_dict()["state"][parameter][name]
            ):
                pytest.fail(f"Recalibration changed optimizer state {name}")
    resumed_files = {path.relative_to(output) for path in output.rglob("*.npy")}
    control_files = {
        path.relative_to(control_output) for path in control_output.rglob("*.npy")
    }
    if resumed_files != control_files:
        pytest.fail("Recalibration created additional evaluation artifacts")
    for path in resumed_files:
        if not np.array_equal(
            np.load(output / path), np.load(control_output / path), equal_nan=True
        ):
            pytest.fail(f"Recalibration changed the persisted output {path}")


@pytest.mark.parametrize(
    "path_form",
    [
        "filename",
        pytest.param(
            "symlink_parent",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="POSIX symlink/.. resolution"
            ),
        ),
    ],
)
def test_relative_checkpoint_keeps_its_best_source(
    tmp_path, validation_data, path_form
):
    output = tmp_path / "relative-resume"
    previous = resume_trainer(validation_data, output)
    legacy_checkpoints(previous, -0.5, best_score=4.0)
    original_best = (output / "best.pt").read_bytes()
    source_directory = output
    checkpoint_path = "recent.pt"
    if path_form == "symlink_parent":
        source_directory = tmp_path / "request"
        source_directory.mkdir()
        child = output / "child"
        child.mkdir()
        (source_directory / "link").symlink_to(child, target_is_directory=True)
        checkpoint_path = "link/../recent.pt"
    working_directory = os.getcwd()
    try:
        os.chdir(source_directory)
        resumed = resume_trainer(validation_data, output, checkpoint_path)
    finally:
        os.chdir(working_directory)
    with wandb.init(mode="disabled"):
        resumed.train()
    if not np.isclose(resumed.best_val_loss, 2.0, atol=1e-6):
        pytest.fail("Deferred calibration changed the checkpoint's source directory")
    if (output / "best.pt").read_bytes() != original_best:
        pytest.fail("Relative checkpoint recovery replaced the better legacy model")


@pytest.mark.parametrize("checkpoint_name", ["recent.pt", "checkpoint_2.pt"])
def test_relocated_resume_preserves_best_source_across_interruption(
    tmp_path, validation_data, checkpoint_name
):
    source = tmp_path / "old-run"
    output = tmp_path / "new-run"
    previous = resume_trainer(validation_data, source)
    legacy_checkpoints(previous, -0.5, best_score=4.0)
    original_best = (source / "best.pt").read_bytes()
    resumed = resume_trainer(validation_data, output, source / "recent.pt", epochs=3)
    resumed.val_frequency = 3

    def stop_before_validation(epoch, dataloader):
        if epoch == 3:
            raise InterruptedError("Stopped before first scheduled validation")
        return Trainer.train_one_epoch(resumed, epoch, dataloader)

    resumed.train_one_epoch = stop_before_validation
    with wandb.init(mode="disabled"):
        with pytest.raises(InterruptedError, match="first scheduled validation"):
            resumed.train()
    if (output / "best.pt").exists():
        pytest.fail("The interrupted run unexpectedly selected a best model")

    restarted = resume_trainer(
        validation_data, output, output / checkpoint_name, epochs=4
    )
    restarted.short_validation_length = len(validation_data.val_dataloader())
    with wandb.init(mode="disabled"):
        restarted.train()
    if not np.isclose(restarted.best_val_loss, 2.0, atol=1e-6):
        pytest.fail("Restart lost the better model in the original output folder")
    if (source / "best.pt").read_bytes() != original_best:
        pytest.fail("Restart modified the original best model")
    current = torch.load(output / "recent.pt", weights_only=False)
    if current["validation_loss_version"] != 1 or current.get("legacy_best_checkpoint"):
        pytest.fail("Completed calibration retained a legacy dependency")

    (source / "best.pt").unlink()
    current_resume = resume_trainer(
        validation_data, output, output / "recent.pt", epochs=5
    )
    with wandb.init(mode="disabled"):
        current_resume.train()
    if not np.isclose(current_resume.best_val_loss, 2.0, atol=1e-6):
        pytest.fail("Current-format resume changed the calibrated best score")


@pytest.mark.parametrize("checkpoint_name", ["recent.pt", "checkpoint_2.pt"])
def test_pending_resume_moves_with_its_run_directory(
    tmp_path, validation_data, checkpoint_name
):
    source = tmp_path / "original-run"
    output = tmp_path / "relocated-run"
    previous = resume_trainer(validation_data, source)
    legacy_checkpoints(previous, -0.5, best_score=4.0)
    original_best = (source / "best.pt").read_bytes()
    resumed = resume_trainer(validation_data, source, source / "recent.pt", epochs=3)
    resumed.val_frequency = 3

    def stop_before_validation(epoch, dataloader):
        if epoch == 3:
            raise InterruptedError("Stopped before moving the run directory")
        return Trainer.train_one_epoch(resumed, epoch, dataloader)

    resumed.train_one_epoch = stop_before_validation
    with wandb.init(mode="disabled"):
        with pytest.raises(InterruptedError, match="moving the run directory"):
            resumed.train()
    source.rename(output)
    restarted = resume_trainer(
        validation_data, output, output / checkpoint_name, epochs=3
    )
    with wandb.init(mode="disabled"):
        restarted.train()
    if not np.isclose(restarted.best_val_loss, 2.0, atol=1e-6):
        pytest.fail("Moving the run directory changed the best model score")
    if (output / "best.pt").read_bytes() != original_best:
        pytest.fail("Moving the run directory replaced the better legacy model")


def test_missing_legacy_best_and_evaluation_only(tmp_path, validation_data):
    output = tmp_path / "missing-best"
    previous = resume_trainer(validation_data, output)
    legacy_checkpoints(previous, -0.5, best_score=4.0)
    (output / "best.pt").unlink()
    trainer = resume_trainer(validation_data, output, output / "recent.pt")
    with wandb.init(mode="disabled"):
        with pytest.raises(FileNotFoundError, match="Restore this file"):
            trainer.train()
    if (output / "best.pt").exists():
        pytest.fail("A missing legacy best was silently replaced")
    evaluation = resume_trainer(
        validation_data, output, output / "recent.pt", evaluate_only=True
    )
    with wandb.init(mode="disabled"):
        evaluation.validate()
    if (output / "best.pt").exists():
        pytest.fail("Evaluation-only wrote a checkpoint")


def test_resume_before_first_validation_can_choose_first_best(
    tmp_path, validation_data
):
    output = tmp_path / "first-validation"
    interrupted = resume_trainer(validation_data, output)

    def stop_before_validation(module, inputs):
        if not module.training:
            raise InterruptedError("Stopped after recent.pt, before first validation")

    interrupted.model.register_forward_pre_hook(stop_before_validation)
    with wandb.init(mode="disabled"):
        with pytest.raises(InterruptedError, match="before first validation"):
            interrupted.train()
    checkpoint = torch.load(output / "recent.pt", weights_only=False)
    if checkpoint["best_validation_loss"] is not None or (output / "best.pt").exists():
        pytest.fail("The interruption did not precede selection of the first best")
    checkpoint.pop("validation_loss_version")
    torch.save(checkpoint, output / "recent.pt")
    resumed = resume_trainer(validation_data, output, output / "recent.pt")
    with wandb.init(mode="disabled"):
        resumed.train()
    best = torch.load(output / "best.pt", weights_only=False)
    if not np.isclose(best["validation_loss"], 2.0, atol=1e-6):
        pytest.fail(
            "Early resume failed to score the first best under the current loss"
        )
    if best["best_validation_loss"] != best["validation_loss"]:
        pytest.fail("The first best checkpoint stored an unset best threshold")


@pytest.mark.parametrize(
    "buffer_mode", ["inplace", "rebound", "none", "alias", "persistent"]
)
def test_failed_recalibration_restores_resumed_state(
    tmp_path, validation_data, buffer_mode
):
    def configure_buffer(model):
        if buffer_mode == "none":
            model.register_buffer("evaluations", None, persistent=False)
        elif buffer_mode == "alias":
            model.register_buffer(
                "evaluation_alias", model.evaluations, persistent=False
            )
        elif buffer_mode == "persistent":
            model.register_buffer("evaluations", torch.tensor(0), persistent=True)

    output = tmp_path / "failed-calibration"
    previous = resume_trainer(validation_data, output)
    configure_buffer(previous.model)
    legacy_checkpoints(previous, -0.5, best_score=4.0)
    original_best = (output / "best.pt").read_bytes()
    trainer = resume_trainer(validation_data, output)
    configure_buffer(trainer.model)
    trainer.load_checkpoint(output / "recent.pt")
    original_buffers = {"evaluations": trainer.model.evaluations}
    if buffer_mode == "alias":
        original_buffers["evaluation_alias"] = trainer.model.evaluation_alias
    before = {}

    def record_state(module, state_dict, prefix, *args):
        if float(state_dict["weight"][0, 0, 0, 0]) == 0:
            before.update(
                modes=[child.training for child in module.modules()],
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
            )

    def fail_on_old_best(module, inputs):
        if not module.training and float(module.weight[0, 0, 0, 0]) == 0:
            if buffer_mode == "none":
                module.evaluations = torch.tensor(1)
            elif buffer_mode == "inplace":
                module.evaluations.add_(1)
            else:
                module.evaluations = module.evaluations + 1
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError("Old-best evaluation interrupted")

    trainer.model.register_load_state_dict_pre_hook(record_state)
    trainer.model.register_forward_pre_hook(fail_on_old_best)
    with wandb.init(mode="disabled"):
        with pytest.raises(RuntimeError, match="Old-best evaluation interrupted"):
            trainer.train()
    saved = torch.load(output / "recent.pt", weights_only=False)
    if saved["validation_loss_version"] != 0 or trainer.best_val_loss != 4.0:
        pytest.fail("Failed calibration changed the legacy score definition")
    for name, value in trainer.model.state_dict().items():
        if not torch.equal(value, saved["model_state_dict"][name]):
            pytest.fail(f"Failed calibration changed current model state {name}")
    for parameter, state in trainer.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            if not torch.equal(
                value, saved["optimizer_state_dit"]["state"][parameter][name]
            ):
                pytest.fail(f"Failed calibration changed optimizer state {name}")
    for name, original in original_buffers.items():
        if getattr(trainer.model, name) is not original:
            pytest.fail(f"Failed calibration changed registered buffer binding {name}")
        if original is not None and int(original) != 0:
            pytest.fail(f"Failed calibration retained buffer contents for {name}")
    if before["modes"] != [module.training for module in trainer.model.modules()]:
        pytest.fail("Failed calibration changed model training modes")
    if before["python"] != random.getstate():
        pytest.fail("Failed calibration advanced Python RNG")
    numpy_state = np.random.get_state()
    if (
        before["numpy"][0] != numpy_state[0]
        or not np.array_equal(before["numpy"][1], numpy_state[1])
        or before["numpy"][2:] != numpy_state[2:]
    ):
        pytest.fail("Failed calibration advanced NumPy RNG")
    if not torch.equal(before["torch"], torch.get_rng_state()):
        pytest.fail("Failed calibration advanced Torch RNG")
    if (output / "best.pt").read_bytes() != original_best or list(
        output.rglob("*.npy")
    ):
        pytest.fail("Failed calibration wrote a best checkpoint or artifacts")


def test_legacy_mse_resume_keeps_its_existing_score_definition(
    tmp_path, validation_data
):
    output = tmp_path / "mse-resume"
    previous = resume_trainer(validation_data, output, loss_type=MSE)
    legacy_checkpoints(previous, -0.5)
    checkpoint = torch.load(output / "recent.pt", weights_only=False)
    checkpoint.update(validation_loss=11.25, best_validation_loss=5.0)
    torch.save(checkpoint, output / "recent.pt")
    (output / "best.pt").unlink()
    resumed = resume_trainer(
        validation_data, output, output / "recent.pt", loss_type=MSE
    )
    with wandb.init(mode="disabled"):
        resumed.train()
    if resumed.best_val_loss != 5.0 or (output / "best.pt").exists():
        pytest.fail("Legacy MSE resume changed its unaffected best-score threshold")
    saved = torch.load(output / "checkpoint_2.pt", weights_only=False)
    if saved["validation_loss_version"] != 1 or saved["validation_loss"] != 11.25:
        pytest.fail("Unchanged legacy MSE scores received an incorrect definition")
