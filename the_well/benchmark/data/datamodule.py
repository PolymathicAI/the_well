from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from .datasets import GenericWellDataset


class AbstractDataModule(ABC):
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError


class WellDataModule(AbstractDataModule):
    def __init__(self, well_base_path: str, well_dataset_name: str, batch_size: int):
        """Data module class to yield batches of samples.

        Parameters
        ----------
        path:
            Path to the data folder containing the splits (train, validation, and test).
        batch_size:
            Size of the batches yielded by the dataloaders

        """
        self.train_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="train",
        )
        self.val_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="valid",
        )
        self.test_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="test",
        )
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
