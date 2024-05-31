import logging
from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import DataLoader, DistributedSampler

from .datasets import GenericWellDataset

logger = logging.getLogger(__name__)


class AbstractDataModule(ABC):
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def rollout_val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def rollout_test_dataloader(self) -> DataLoader:
        raise NotImplementedError


class WellDataModule(AbstractDataModule):
    def __init__(
        self,
        well_base_path: str,
        well_dataset_name: str,
        batch_size: int,
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        dt_stride: int = 1,
        world_size: int = 1,
        data_workers: int = 4,
        rank: int = 1,
    ):
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
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
        )
        self.val_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="valid",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
        )
        self.rollout_val_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="valid",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            max_rollout_steps=max_rollout_steps,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            full_trajectory_mode=True,
            dt_stride=dt_stride,
        )
        self.test_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="test",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=dt_stride,
        )
        self.rollout_test_dataset = GenericWellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="test",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            max_rollout_steps=max_rollout_steps,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            full_trajectory_mode=True,
            dt_stride=dt_stride,
        )
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for training data"
            )
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for validation data"
            )
        shuffle = sampler is None # Most valid epochs are short
        return DataLoader(
            self.val_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def rollout_val_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.rollout_val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True, # Since we're subsampling, don't want continuous
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for rollout validation data"
            )
        shuffle = sampler is None # Most valid epochs are short
        return DataLoader(
            self.rollout_val_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=1,  
            shuffle=shuffle, # Shuffling because most batches we take a small subsample
            drop_last=True,
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for test data"
            )
        return DataLoader(
            self.test_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            sampler=sampler,
        )

    def rollout_test_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.rollout_test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for rollout test data"
            )
        return DataLoader(
            self.rollout_test_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=1,  # min(self.batch_size, len(self.rollout_test_dataset)),
            shuffle=False,
            drop_last=True,
            sampler=sampler,
        )
