import os.path as osp

from torch.utils.data import DataLoader

from .datasets import GenericWellDataset


class WellDataModule:
    def __init__(self, path: str, batch_size: int):
        """Data module class to yield batches of samples.

        Parameters
        ----------
        path:
            Path to the data folder containing the splits (train, validation, and test).
        batch_size:
            Size of the batches yielded by the dataloaders

        """
        train_data_path = osp.join(path, "train")
        val_data_path = osp.join(path, "validation")
        test_data_path = osp.join(path, "test")
        self.train_dataset = GenericWellDataset(path=train_data_path)
        self.val_dataset = GenericWellDataset(path=val_data_path)
        self.test_dataset = GenericWellDataset(path=test_data_path)
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
