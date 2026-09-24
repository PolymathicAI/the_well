import pytest
import torch

pytest.importorskip("wandb")
from the_well.benchmark.trainer.training import Trainer


def _make_trainer(lr_scheduler_steps: int = 0) -> Trainer:
    # Bypass __init__, which requires a full datamodule, and set only what the
    # checkpointing methods use.
    trainer = object.__new__(Trainer)
    trainer.model = torch.nn.Linear(3, 3)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1.0)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.5
    )
    trainer.grad_scaler = torch.GradScaler("cpu", init_scale=1024.0)
    trainer.best_val_loss = 0.25
    for _ in range(lr_scheduler_steps):
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
    return trainer


def test_checkpoint_restores_scheduler_and_grad_scaler(tmp_path):
    trainer = _make_trainer(lr_scheduler_steps=3)
    trainer.grad_scaler.scale(torch.ones(()))  # initializes the scale
    trainer.grad_scaler.update(new_scale=64.0)
    path = tmp_path / "recent.pt"
    trainer.save_model(epoch=3, validation_loss=0.5, output_path=path)

    resumed = _make_trainer()
    resumed.load_checkpoint(path)

    assert resumed.starting_epoch == 4
    assert resumed.lr_scheduler.last_epoch == 3
    assert resumed.lr_scheduler.get_last_lr() == trainer.lr_scheduler.get_last_lr()
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.125)
    assert resumed.grad_scaler.get_scale() == 64.0


def test_load_checkpoint_with_legacy_optimizer_key(tmp_path):
    trainer = _make_trainer(lr_scheduler_steps=2)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dit": trainer.optimizer.state_dict(),
            "validation_loss": 0.5,
            "best_validation_loss": 0.25,
        },
        path,
    )

    resumed = _make_trainer()
    resumed.load_checkpoint(path)

    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
    assert resumed.lr_scheduler.last_epoch == 0
