# API

## Data Classes

The Well provides two main class `WellDataset` and `WellDataModule` to handle the raw data that are stored in `.hdf5` files. The `WellDataset` implements a map-style PyTorch Dataset. The `WellDataModule` provides dataloaders for training, validation, and test. The [tutorial](tutorials/dataset.ipynb) provides a guide on how to use these classes in a training pipeline.

### Dataset

The `WellDataset` is a map-style dataset. It converts the `.hdf5` file structure expected by the Well into `torch.Tensor` data. It first processes metadata from the `.hdf5` attributes to allow for retrieval of individual samples.

::: the_well.data.WellDataset
    options:
        show_root_heading: true
        heading_level: 4

### DataModule

The `WellDataModule` provides the different dataloaders required for training, validation, and testing. It has two kinds of dataloaders: the default one that yields batches of a fixed time horizon, and rollout ones that yields batches to evaluate rollout performances.

::: the_well.data.WellDataModule
    options:
        show_root_heading: true
        heading_level: 4

## Metrics

The Well package implements a series of metrics to assess the performances of a trained model.

::: the_well.benchmark.metrics
    options:
        show_root_heading: true
        heading_level: 3
