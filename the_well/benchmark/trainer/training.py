import logging
from typing import Callable, Optional

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader

from ..data.datamodule import AbstractDataModule
from ..data.utils import preprocess_batch

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        experiment_name: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
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
        self.max_epoch = epochs
        self.val_frequency = val_frequency
        self.is_distributed = is_distributed
        self.best_val_loss = None

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
    def validation_loop(self, dataloader: DataLoader) -> float:
        """Run validation by looping over the dataloader."""
        self.model.eval()
        validation_loss = 0.0
        for batch in dataloader:
            x, y_ref = preprocess_batch(batch)
            for key, val in x.items():
                x[key] = val.to(self.device)
            y_ref = y_ref.to(self.device)
            y_pred = self.model(x)
            assert (
                y_ref.shape == y_pred.shape
            ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
            loss = self.loss_fn(y_ref, y_pred)
            validation_loss += loss * y_ref.size(0) / len(dataloader.dataset)
        if self.is_distributed:
            dist.all_reduce(validation_loss, op=dist.ReduceOp.AVG)
        validation_loss = validation_loss.item()

        return validation_loss

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """Train the model for one epoch by looping over the dataloader."""
        self.model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            x, y_ref = preprocess_batch(batch)
            for key, val in x.items():
                x[key] = val.to(self.device)
            y_ref = y_ref.to(self.device)
            y_pred = self.model(x)
            print('This is actually happening, right?')
            assert (
                y_ref.shape == y_pred.shape
            ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
            loss = self.loss_fn(y_ref, y_pred)
            epoch_loss += loss.item() * y_ref.size(0) / len(dataloader.dataset)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()
            wandb.log({"lr": self.lr_scheduler.get_lr()})
        return epoch_loss

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()

        for epoch in range(self.max_epoch):
            if epoch % self.val_frequency == 0:
                val_loss = self.validation_loop(val_dataloder)
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}"
                )
                wandb.log({"valid": val_loss, "epoch": epoch})
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.save_model(epoch, val_loss, f"{self.experiment_name}-best.pt")
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: training loss {train_loss}")
            wandb.log({"train": train_loss, "epoch": epoch})
        # Run validation on last epoch if not already run
        if epoch % self.val_frequency != 0:
            val_loss = self.validation_loop(val_dataloder)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}")
            wandb.log({"valid": val_loss, "epoch": epoch})
        test_loss = self.validation_loop(test_dataloader)
        logger.info(f"Test loss {test_loss}")
        wandb.log({"test": test_loss, "epoch": epoch})
        self.save_model(epoch, val_loss, f"{self.experiment_name}-last.pt")
