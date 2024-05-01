import logging

import torch

# from benchmarking.data_utils.datasets import GenericWellDataset
from modulus.models.fno import FNO
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def validation_loop(
    model: torch.nn.Module, dataloader: DataLoader, loss_fn, device: torch.device
) -> float:
    model.eval()
    validation_loss = 0.0
    for batch in dataloader:
        x, y_ref = batch
        x = x.to(device)
        y_ref = y_ref.to(device)
        y_pred = model(x)
        loss = loss_fn(y_ref, y_pred)
        validation_loss += loss.item() * batch_size / len(dataloader.dataset)
    return loss


def training_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> float:
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        x, y_ref = batch
        x = x.to(device)
        y_ref = y_ref.to(device)
        y_pred = model(x)
        loss = loss_fn(y_ref, y_pred)
        epoch_loss += loss.item() * batch_size / len(dataloader.dataset)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    return epoch_loss


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn,
    validation_frequency: int,
):
    for epoch in range(epochs):
        if epoch % validation_frequency == 0:
            val_loss = validation_loop(model, valid_dataloader, loss_fn, device)
            logger.info(f"Epoch {epoch+1}/{epochs}: validation loss {val_loss}")
        train_loss = training_loop(
            model, train_dataloader, loss_fn, device, optimizer, scheduler
        )
        logger.info(f"Epoch {epoch+1}/{epochs}: training loss {train_loss}")
    # Run validation on last epoch if not already run
    if epoch % validation_frequency != 0:
        val_loss = validation_loop(model, valid_dataloader, loss_fn, device)
        logger.info(f"Epoch {epoch+1}/{epochs}: validation loss {val_loss}")
    test_loss = validation_loop(model, test_dataloader, loss_fn, device)
    logger.info(f"Test loss {test_loss}")


if __name__ == "__main__":
    device = torch.device("cuda")

    logger.info(f"Running on device {device}")

    # Model
    in_channels = 2
    out_channels = 2
    num_fno_modes = 12
    model = FNO(
        in_channels=in_channels, out_channels=out_channels, num_fno_modes=num_fno_modes
    )
    logger.info(f"Training module {model.__class__.__name__}")
    model = model.to(device)

    # Data
    path = "..."
    # train_dataset = GenericWellDataset(path, well_split_name="train")
    fake_inputs = torch.rand(10, in_channels, 32, 32)
    fake_outputs = torch.rand(10, out_channels, 32, 32)
    train_dataset = TensorDataset(fake_inputs, fake_outputs)
    batch_size = 4
    shuffle = True
    pin_memory = False
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True,
    )
    # validation
    validation_frequency = 100
    fake_inputs = torch.rand(10, in_channels, 32, 32)
    fake_outputs = torch.rand(10, out_channels, 32, 32)
    validation_dataset = TensorDataset(fake_inputs, fake_outputs)
    valid_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    # test
    fake_inputs = torch.rand(10, in_channels, 32, 32)
    fake_outputs = torch.rand(10, out_channels, 32, 32)
    validation_dataset = TensorDataset(fake_inputs, fake_outputs)
    test_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    logger.info(f"Training on dataset {train_dataset}")

    # Trainer
    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)
    step_size = 100
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = MSELoss()
    # Loop
    max_epochs = 500

    train(
        model,
        train_dataloader,
        valid_dataloader,
        valid_dataloader,
        max_epochs,
        optimizer,
        scheduler,
        loss_fn,
        validation_frequency,
    )
