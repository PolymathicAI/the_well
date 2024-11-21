
# Reference

## Data format

The raw data of the Well are stored as `.hdf5` files, internally organized following a shared specification. The data stored in these files have been generated on uniform grids and sampled at constant time intervals. These
files include all available state variables or spatially varying coefficients associated with a given set of
dynamics in numpy arrays of shape (`n_traj`, `n_steps`, `coord1`, `coord2`, `(coord3)`) in
single precision fp32. We distinguish between scalar, vector, and tensor-valued fields due to their different
transformation properties.

The specification is described below with example entries for a hypothetical 2D ($D=2$) simulation with dimension B x T x H x W. Note that this uses HDF5 Groups, Datasets, and attributes (denoted by "@"):

```python
root: Group
  @simulation_parameters: list[str] = ['ParamA', ...]
  @ParamA: float = 1.0
  ... # Additional listed parameters
  @dataset_name: str = 'ExampleDSet'
  @grid_type: str = 'cartesian' # "cartesian/spherical currently supported"
  @n_spatial_dims: int = 2 # Should match number of provided spatial dimensions.
  @n_trajectories: int = B # "Batch" dimension of dataset

  -dimensions: Group
    @spatial_dims: list[str] = ['x', 'y'] # Names match datasets below.
    time: Dataset = float32(T)
      @sample_varying = False # Does this value vary between trajectories?
    -x: Dataset = float32(W) # Grid coordinates in x
      @sample_varying = False
      @time_varying = False # True not currently supported
    -y = float32(H) # Grid coordinates in y
      @sample_varying = False
      @time_varying = False

  -boundary_conditions: Group # Internal and external boundary conditions
    -X_boundary: Group
      @associated_dims: list[str] = ['x'] # Defined on x
      # If associated with set values for given field.
      @associated_fields: list[str] = []
      # Geometric description of BC. Currently support periodic/wall/open
      @bc_type = 'periodic'
      @sample_varying = False
      @time_varying = False
      -mask: Dataset = bool(W) # True on coordinates where boundary is defined.
      -values: Dataset = float32(NumTrue(mask)) # Values defined on mask points

  scalars: Group # Non-spatially varying scalars.
    @field_names: list[str] = ['ParamA', 'OtherScalar', ...]
    ParamA: Dataset = float32(1)
      @sample_varying = False # Does this vary between trajectories?
      @time_varying = False # Does this vary over time?
    OtherScalar: Dataset = float32(T)
      @sample_varying = False
      @time_varying = True

  t0_fields: Group
    # field_names should list all datasets in this category
    @field_names: list[str] = ['FieldA', 'FieldB', 'FieldC', ...]
    -FieldA: Dataset = float32(BxTxHxW)
      @dim_varying = [ True  True]
      @sample_varying = True
      @time_varying = True
    -FieldB: Dataset = float32(TxHxW)
      @dim_varying = [ True  True]
      @sample_varying = True
      @time_varying = False
    -FieldC: Dataset = float32(BxTxH)
      @dim_varying = [ True  False]
      @sample_varying = True
      @time_varying = True
    ... # Additional fields

  -t1_fields: Group
    @field_names = ['VFieldA', ...]
    -VFieldA: Dataset = float32(BxTxHxWxD)
      @dim_varying = [ True  True]
      @sample_varying = True
      @time_varying = True
    ... # Additional fields

  -t2_fields: Group
    @field_names: list[str] = ['TFieldA', ...]
    - TFieldA: Dataset = float32(BxTxHxWxD^2)
      @antisymmetric = False
      @dim_varying = [ True  True]
      @sample_varying = True
      @symmetric = True # Whether tensor is symmetric
      @time_varying = True
    ... # Additional fields
```

## Data Classes

The Well provides two main class `WellDataset` and `WellDataModule` to handle the raw data that are stored in `.hdf5` files. The [tutorial](tutorials/dataset.ipynb) provides a guide on how to use these classes in a training pipeline.

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

Discuss metrics available in the package ...

::: the_well.benchmark.metrics
    options:
        show_root_heading: true
        heading_level: 3

## Contribute

Explain the contributions procedure ...
