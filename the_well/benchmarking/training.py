import logging

import torch
from benchmarking.data_utils.datasets import GenericWellDataset
from modulus.models.fno import FNO
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

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
train_dataset = GenericWellDataset(path, well_split_name="train")
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
    for batch in train_dataloader:
        x = batch.to(device)
        y = model(x)
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
