import logging
import time
from typing import Callable, Optional

import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.utils.data import DataLoader

from ..data.data_formatter import DefaultChannelsFirstFormatter
from ..data.datamodule import AbstractDataModule
from ..metrics import validation_metric_suite

logger = logging.getLogger(__name__)


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
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        is_distributed: bool = False,
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
        self.is_distributed = is_distributed
        self.best_val_loss = None
        self.dset_metadata = self.datamodule.train_dataset.metadata
        if formatter == "channels_first_default":
            self.formatter = DefaultChannelsFirstFormatter(self.dset_metadata)

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

    @torch.no_grad()
    def validation_loop(self, dataloader: DataLoader, valid_or_test: str="valid") -> float:
        """Run validation by looping over the dataloader."""
        self.model.eval()
        validation_loss = 0.0
        field_names = self.dset_metadata.field_names
        dset_name = self.dset_metadata.dataset_name
        loss_dict = {}
        for batch in tqdm.tqdm(dataloader):
            inputs, y_ref = self.formatter.process_input(batch)
            inputs = map(lambda x: x.to(self.device), inputs)
            y_ref = y_ref.to(self.device)
            y_pred = self.model(*inputs)
            y_pred = self.formatter.process_output(y_pred)
            assert (
                y_ref.shape == y_pred.shape
            ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
            for loss_fn in self.validation_suite:
                # Mean over batch and time per field
                loss = loss_fn(y_ref, y_pred, self.dset_metadata)
                # Some losses return multiple values for efficiency
                if not isinstance(loss, dict):
                    loss = {loss_fn.__class__.__name__: loss}
                for k, v in loss.items():
                    sub_loss = v.mean(0).mean(0)
                    for i, fname in enumerate(field_names):
                        loss_dict[f"{dset_name}/{fname}_{k}"] = (
                            loss_dict.get(f"{dset_name}/{fname}_{k}", 0.0)
                             + sub_loss[i] / len(dataloader)
                        )
                    # Now mean over field too
                    loss_dict[f"{dset_name}/full_{k}"] = (
                        loss_dict.get(f"{dset_name}/full_{k}", 0.0)
                        + sub_loss.mean() / len(dataloader)
                )
            else: # Last batch plots - too much work to combine from batches
                pass
            # break
        if self.is_distributed:
            for k, v in loss_dict.items():
                dist.all_reduce(loss_dict[k], op=dist.ReduceOp.AVG)
        validation_loss = loss_dict[
            f"{dset_name}/full_{self.loss_fn.__class__.__name__}"
        ].item()
        loss_dict = {f"{valid_or_test}_{k}": v.item() for k, v in loss_dict.items()}

        return validation_loss, loss_dict

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """Train the model for one epoch by looping over the dataloader."""
        self.model.train()
        epoch_loss = 0.0
        train_logs = {}
        start_time = time.time() # Don't need to sync this. 
        for batch in tqdm.tqdm(dataloader):
            inputs, y_ref = self.formatter.process_input(batch)
            inputs = map(lambda x: x.to(self.device), inputs)
            y_ref = y_ref.to(self.device)
            y_pred = self.model(*inputs)
            y_pred = self.formatter.process_output(y_pred)
            assert (
                y_ref.shape == y_pred.shape
            ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
            loss = self.loss_fn(y_ref, y_pred, self.dset_metadata).mean()
            epoch_loss += loss.item() / len(dataloader)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        train_logs["time_per_train_iter"] = (time.time() - start_time) / len(dataloader)
        train_logs["train_loss"] = epoch_loss
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_lr()
        return epoch_loss, train_logs

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()

        for epoch in range(self.max_epoch):
            if epoch % self.val_frequency == 0:
                val_loss, val_loss_dict = self.validation_loop(val_dataloder)
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}"
                )
                val_loss_dict |= {"valid": val_loss, "epoch": epoch}
                wandb.log(val_loss_dict)
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.save_model(epoch, val_loss, f"{self.experiment_name}-best.pt")
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            train_loss, train_logs = self.train_one_epoch(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: training loss {train_loss}")
            train_logs |= {"train": train_loss, "epoch": epoch}
            wandb.log(train_logs)
        # Run validation on last epoch if not already run
        if epoch % self.val_frequency != 0:
            val_loss, val_loss_dict = self.validation_loop(val_dataloder)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}")
            val_loss_dict |= {"valid": val_loss, "epoch": epoch}
            wandb.log(val_loss_dict)
        test_loss, test_logs = self.validation_loop(test_dataloader, valid_or_test="test")
        logger.info(f"Test loss {test_loss}")
        test_logs |= {"test": test_loss, "epoch": epoch}
        wandb.log(test_logs)
        self.save_model(epoch, val_loss, f"{self.experiment_name}-last.pt")
