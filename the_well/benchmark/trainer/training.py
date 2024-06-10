import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.utils.data import DataLoader

from ..data.data_formatter import (
    DefaultChannelsFirstFormatter,
    DefaultChannelsLastFormatter,
)
from ..data.datamodule import AbstractDataModule
from ..metrics import (
    long_time_metrics,
    plot_all_time_metrics,
    validation_metric_suite,
    validation_plots,
)

logger = logging.getLogger(__name__)


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5


class Trainer:
    def __init__(
        self,
        experiment_name: str,
        formatter: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        # validation_suite: list,
        epochs: int,
        val_frequency: int,
        rollout_val_frequency: int,
        max_rollout_steps: int,
        short_validation_length: int,
        num_time_intervals: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        is_distributed: bool = False,
        enable_amp: bool = False,
        amp_type: str = "float16",  # bfloat not supported in FFT
    ):
        """
        Class in charge of the training loop. It performs train, validation and test.

        Parameters
        ----------
        experiment_name:
            The name of the training experiment to be run
        model:
            The Pytorch model to train
        datamodule:
            A datamodule that provides dataloaders for each split (train, valid, and test)
        optimizer:
            A Pytorch optimizer to perform the backprop (e.g. Adam)
        loss_fn:
            A loss function that evaluates the model predictions to be used for training
        epochs:
            Number of epochs to train the model.
            One epoch correspond to a full loop over the datamodule's training dataloader
        val_frequency:
            The frequency in terms of number of epochs to perform the validation
        rollout_val_frequency:
            The frequency in terms of number of epochs to perform the rollout validation
        max_rollout_steps:
            The maximum number of timesteps to rollout the model
        num_time_intervals:
            The number of time intervals to split the loss over
        lr_scheduler:
            A Pytorch learning rate scheduler to update the learning rate during training
        device:
            A Pytorch device (e.g. "cuda" or "cpu")
        is_distributed:
            A boolean flag to trigger DDP training

        """
        self.experiment_name = experiment_name
        self.device = device
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.validation_suite = validation_metric_suite + [self.loss_fn]
        self.max_epoch = epochs
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps
        self.short_validation_length = short_validation_length
        self.num_time_intervals = num_time_intervals
        self.enable_amp = enable_amp
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=enable_amp and amp_type != "bfloat16"
        )
        self.is_distributed = is_distributed
        self.best_val_loss = None
        self.dset_metadata = self.datamodule.train_dataset.metadata
        if formatter == "channels_first_default":
            self.formatter = DefaultChannelsFirstFormatter(self.dset_metadata)
        elif formatter == "channels_last_default":
            self.formatter = DefaultChannelsLastFormatter(self.dset_metadata)

    def save_model(self, epoch: int, validation_loss: float, output_path: str):
        """Save the model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dit": self.optimizer.state_dict(),
                "validation_loss": validation_loss,
            },
            output_path,
        )

    def rollout_model(self, model, batch, formatter):
        """Rollout the model for as many steps as we have data for."""
        inputs, y_ref = formatter.process_input(batch)
        rollout_steps = min(
            y_ref.shape[1], self.max_rollout_steps
        )  # Number of timesteps in target
        y_ref = y_ref[:, :rollout_steps]
        # Create a moving batch of one step at a time
        moving_batch = batch
        moving_batch["input_fields"] = moving_batch["input_fields"].to(self.device)
        if "constant_fields" in moving_batch:
            moving_batch["constant_fields"] = moving_batch["constant_fields"].to(
                self.device
            )
        y_preds = []
        for i in range(rollout_steps):
            inputs, _ = formatter.process_input(moving_batch)
            inputs = map(lambda x: x.to(self.device), inputs)
            y_pred = model(*inputs)
            y_pred = formatter.process_output(y_pred)
            # If not last step, update moving batch for autoregressive prediction
            if i != rollout_steps - 1:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, 1:], y_pred], dim=1
                )
            y_preds.append(y_pred)
        y_pred_out = torch.cat(y_preds, dim=1)
        y_ref = y_ref.to(self.device)
        return y_pred_out, y_ref

    def temporal_split_losses(
        self, loss_values, temporal_loss_intervals, loss_name, dset_name, fname="full"
    ):
        new_losses = {}
        # Average over time interval
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_values.mean()
        # Don't compute sublosses if we only have one interval
        if len(temporal_loss_intervals) == 2:
            return new_losses
        # Break it down by time interval
        for k in range(len(temporal_loss_intervals) - 1):
            start_ind = temporal_loss_intervals[k]
            end_ind = temporal_loss_intervals[k + 1]
            time_str = f"{start_ind}:{end_ind}"
            loss_subset = loss_values[start_ind:end_ind].mean()
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={time_str}"] = loss_subset
        return new_losses

    def split_up_losses(self, loss_values, loss_name, dset_name, field_names):
        new_losses = {}
        time_logs = {}
        time_steps = loss_values.shape[0]  # we already average over batch
        num_time_intervals = min(time_steps, self.num_time_intervals)
        temporal_loss_intervals = np.linspace(0, np.log(time_steps), num_time_intervals)
        temporal_loss_intervals = [0] + [
            int(np.exp(x)) for x in temporal_loss_intervals
        ]
        # Split up losses by field
        for i, fname in enumerate(field_names):
            time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                :, i
            ].cpu()
            new_losses |= self.temporal_split_losses(
                loss_values[:, i], temporal_loss_intervals, loss_name, dset_name, fname
            )
        # Compute average over all fields
        new_losses |= self.temporal_split_losses(
            loss_values.mean(1), temporal_loss_intervals, loss_name, dset_name, "full"
        )
        time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(1).cpu()
        return new_losses, time_logs

    @torch.inference_mode()
    def validation_loop(
        self, dataloader: DataLoader, valid_or_test: str = "valid", full=False
    ) -> float:
        """Run validation by looping over the dataloader."""
        self.model.eval()
        validation_loss = 0.0
        field_names = self.dset_metadata.field_names
        dset_name = self.dset_metadata.dataset_name
        loss_dict = {}
        time_logs = {}
        count = 0
        denom = len(dataloader) if full else self.short_validation_length
        with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_type):
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                # Rollout for length of target
                y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter)
                assert (
                    y_ref.shape == y_pred.shape
                ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                # Go through losses
                for loss_fn in self.validation_suite:
                    # Mean over batch and time per field
                    loss = loss_fn(y_pred, y_ref, self.dset_metadata)
                    # Some losses return multiple values for efficiency
                    if not isinstance(loss, dict):
                        loss = {loss_fn.__class__.__name__: loss}
                    # Split the losses and update the logging dictionary
                    for k, v in loss.items():
                        sub_loss = v.mean(0)
                        new_losses, new_time_logs = self.split_up_losses(
                            sub_loss, k, dset_name, field_names
                        )
                        # TODO get better way to include spectral error.
                        if k in long_time_metrics or "spectral_error" in k:
                            time_logs |= new_time_logs
                        for loss_name, loss_value in new_losses.items():
                            loss_dict[loss_name] = (
                                loss_dict.get(loss_name, 0.0) + loss_value / denom
                            )
                count += 1
                if not full and count >= self.short_validation_length:
                    break

        # Last batch plots - too much work to combine from batches
        plot_dicts = {}
        for plot_fn in validation_plots:
            plot_dicts |= plot_fn(y_pred, y_ref, self.dset_metadata)
        if y_ref.shape[1] > 1:
            # Only plot if we have more than one timestep, but then track loss over timesteps
            plot_dicts |= plot_all_time_metrics(time_logs)

        if self.is_distributed:
            for k, v in loss_dict.items():
                dist.all_reduce(loss_dict[k], op=dist.ReduceOp.AVG)
        validation_loss = loss_dict[
            f"{dset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
        ].item()
        loss_dict = {f"{valid_or_test}_{k}": v.item() for k, v in loss_dict.items()}
        loss_dict |= plot_dicts
        # Misc metrics
        loss_dict["param_norm"] = param_norm(self.model.parameters())
        return validation_loss, loss_dict

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        """Train the model for one epoch by looping over the dataloader."""
        self.model.train()
        epoch_loss = 0.0
        train_logs = {}
        start_time = time.time()  # Don't need to sync this.
        batch_start = time.time()
        for i, batch in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_type):
                batch_time = time.time() - batch_start
                y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter)
                forward_time = time.time() - batch_start - batch_time
                assert (
                    y_ref.shape == y_pred.shape
                ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                loss = self.loss_fn(y_pred, y_ref, self.dset_metadata).mean()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # Syncing for all reduce anyway so may as well compute synchornous metrics
            epoch_loss += loss.item() / len(dataloader)
            backward_time = time.time() - batch_start - forward_time - batch_time
            total_time = time.time() - batch_start
            logger.info(
                f"Epoch {epoch}, Batch {i+1}/{len(dataloader)}: loss {loss.item()}, total_time {total_time}, batch time {batch_time}, forward time {forward_time}, backward time {backward_time}"
            )
            batch_start = time.time()
        train_logs["time_per_train_iter"] = (time.time() - start_time) / len(dataloader)
        train_logs["train_loss"] = epoch_loss
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        rollout_val_dataloader = self.datamodule.rollout_val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()
        rollout_test_dataloader = self.datamodule.rollout_test_dataloader()
        os.makedirs(
            "checkpoints", exist_ok=True
        )  # Quick fix here - should parameterize.
        for epoch in range(self.max_epoch):
            if epoch % self.val_frequency == 0:
                logger.info(f"Epoch {epoch+1}/{self.max_epoch}: starting validation")
                val_loss, val_loss_dict = self.validation_loop(
                    val_dataloder, full=epoch == self.max_epoch - 1
                )
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}"
                )
                val_loss_dict |= {"valid": val_loss, "epoch": epoch}
                wandb.log(val_loss_dict)
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.save_model(
                        epoch, val_loss, f"checkpoints/{self.experiment_name}-best.pt"
                    )
            if epoch % self.rollout_val_frequency == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epoch}: starting rollout validation"
                )
                rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                    rollout_val_dataloader,
                    valid_or_test="rollout_valid",
                    full=epoch == self.max_epoch - 1,
                )
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epoch}: rollout validation loss {rollout_val_loss}"
                )
                rollout_val_loss_dict |= {
                    "rollout_valid": rollout_val_loss,
                    "epoch": epoch,
                }
                wandb.log(rollout_val_loss_dict)

            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: starting training")
            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: training loss {train_loss}")
            train_logs |= {"train": train_loss, "epoch": epoch}
            wandb.log(train_logs)
            self.save_model(
                epoch, val_loss, f"checkpoints/{self.experiment_name}-recent.pt"
            )
        # Run validation on last epoch if not already run
        if epoch % self.val_frequency != 0:
            val_loss, val_loss_dict = self.validation_loop(val_dataloder, full=True)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}")
            val_loss_dict |= {"valid": val_loss, "epoch": epoch}
            wandb.log(val_loss_dict)
        if epoch % self.rollout_val_frequency != 0:
            rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                rollout_val_dataloader, valid_or_test="rollout_valid", full=True
            )
            logger.info(
                f"Epoch {epoch+1}/{self.max_epoch}: rollout validation loss {rollout_val_loss}"
            )
            rollout_val_loss_dict |= {"rollout_valid": rollout_val_loss, "epoch": epoch}
            wandb.log(rollout_val_loss_dict)

        test_loss, test_logs = self.validation_loop(
            test_dataloader, valid_or_test="test", full=True
        )
        rollout_test_loss, rollout_test_logs = self.validation_loop(
            rollout_test_dataloader, valid_or_test="rollout_test", full=True
        )
        test_logs |= rollout_test_logs
        logger.info(f"Test loss {test_loss}")
        test_logs |= {
            "test": test_loss,
            "rollout_test": rollout_test_loss,
            "epoch": epoch,
        }
        wandb.log(test_logs)
        self.save_model(epoch, val_loss, f"checkpoints/{self.experiment_name}-last.pt")
