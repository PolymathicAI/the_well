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
    train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
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
model.train()
for epoch in range(max_epochs):
    epoch_loss = 0.0
    for batch in train_dataloader:
        x, y_ref = batch
        x = x.to(device)
        y_ref = y_ref.to(device)
        y_pred = model(x)
        loss = loss_fn(y_ref, y_pred)
        epoch_loss += loss.item() * batch_size / len(train_dataset)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    logger.info(f"Epoch {epoch+1}/{max_epochs}: training loss {epoch_loss}")
