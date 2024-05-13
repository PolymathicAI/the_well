from typing import Optional, Callable
import logging
import torch
from torch.utils.data import DataLoader

from ..data.datamodule import AbstractDataModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int,
        val_frequency: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
    ):
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.max_epoch = epochs
        self.val_frequency = val_frequency

    @torch.no_grad()
    def validation_loop(self, dataloader: DataLoader) -> float:
        self.model.eval()
        validation_loss = 0.0
        for batch in dataloader:
            x, y_ref = batch
            x = x.to(self.device)
            y_ref = y_ref.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_fn(y_ref, y_pred)
            validation_loss += (
                loss.item() * dataloader.batch_size / len(dataloader.dataset)
            )
        return loss

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            x, y_ref = batch
            x = x.to(self.device)
            y_ref = y_ref.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_fn(y_ref, y_pred)
            epoch_loss += loss.item() * dataloader.batch_size / len(dataloader.dataset)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return epoch_loss

    def train(self):
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()

        for epoch in range(self.max_epoch):
            if epoch % self.val_frequency == 0:
                val_loss = self.validation_loop(val_dataloder)
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}"
                )
            train_loss = self.train_one_epoch(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: training loss {train_loss}")
        # Run validation on last epoch if not already run
        if epoch % self.val_frequency != 0:
            val_loss = self.validation_loop(val_dataloder)
            logger.info(f"Epoch {epoch+1}/{self.max_epoch}: validation loss {val_loss}")
        test_loss = self.validation_loop(test_dataloader)
        logger.info(f"Test loss {test_loss}")
