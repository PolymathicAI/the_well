import torch

from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.trainer import training
from the_well.data.datasets import WellMetadata


def test_validation_loop_averages_time_logs_over_batches(monkeypatch, tmp_path):
    """The time curves logged for long_time_metrics must be averaged over all
    validation batches, mirroring how loss_dict is accumulated, instead of
    reflecting only the last batch."""
    meta = WellMetadata(
        dataset_name="test",
        n_spatial_dims=1,
        spatial_resolution=(8,),
        scalar_names=[],
        constant_scalar_names=[],
        field_names={0: ["test"]},
        constant_field_names={},
        boundary_condition_types=["periodic"],
        n_files=1,
        n_trajectories_per_file=[10],
        n_steps_per_trajectory=[100],
    )
    metric = VRMSE()

    # Build a Trainer without running __init__ (which requires a full
    # model/datamodule stack) and set only the attributes validation_loop uses.
    trainer = training.Trainer.__new__(training.Trainer)
    trainer.model = torch.nn.Identity()
    trainer.device = torch.device("cpu")
    trainer.enable_amp = False
    trainer.amp_type = torch.float16
    trainer.formatter = None
    trainer.dset_metadata = meta
    trainer.validation_suite = [metric]
    trainer.loss_fn = metric
    trainer.num_time_intervals = 2
    trainer.short_validation_length = 10
    trainer.viz_folder = str(tmp_path)
    trainer.make_rollout_videos = False
    trainer.is_distributed = False
    # The rollout itself is not under test: each "batch" already holds
    # a (y_pred, y_ref) pair with shape (batch, time, space, field).
    trainer.rollout_model = lambda model, batch, formatter, train=False: batch

    torch.manual_seed(0)
    batches = []
    for scale in (1.0, 10.0):
        y_ref = torch.randn(2, 4, 8, 1)
        y_pred = y_ref + scale * torch.randn(2, 4, 8, 1)
        batches.append((y_pred, y_ref))

    captured = {}

    def capture_time_logs(time_logs, *args, **kwargs):
        captured.update(time_logs)

    monkeypatch.setattr(training, "validation_plots", [])
    monkeypatch.setattr(training, "plot_all_time_metrics", capture_time_logs)

    trainer.validation_loop(batches, full=True)

    # Expected: the average over batches of the per-batch rollout curves.
    per_batch_curves = []
    for y_pred, y_ref in batches:
        sub_loss = metric(y_pred, y_ref, meta).mean(0)
        _, batch_time_logs = trainer.split_up_losses(
            sub_loss, "VRMSE", meta.dataset_name, ["test"]
        )
        per_batch_curves.append(batch_time_logs)

    key = f"{meta.dataset_name}/full_VRMSE_rollout"
    assert key in captured
    expected = (per_batch_curves[0][key] + per_batch_curves[1][key]) / 2
    last_batch_only = per_batch_curves[1][key]
    assert not torch.allclose(expected, last_batch_only)
    assert torch.allclose(captured[key], expected)
