# Data format

The raw data of the Well are stored as [HDF5 files](https://www.hdfgroup.org/solutions/hdf5/), internally organized following a shared specification. The data stored in these files have been generated on uniform grids and sampled at constant time intervals. These
files include all available state variables or spatially varying coefficients associated with a given set of
dynamics in numpy arrays of shape (`n_traj`, `n_steps`, `coord1`, `coord2`, `(coord3)`) in
single precision fp32. We distinguish between scalar, vector, and tensor-valued fields due to their different
transformation properties.

The specification is described below with example entries for a hypothetical 2D ($D=2$) simulation with dimension B x T x W x H. Note that this uses HDF5 Groups, Datasets, and attributes (denoted by "@"):

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
    -FieldA: Dataset = float32(BxTxWxH)
      @dim_varying = [ True  True]
      @sample_varying = True
      @time_varying = True
    -FieldB: Dataset = float32(TxWxH)
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
    -VFieldA: Dataset = float32(BxTxWxHxD)
      @dim_varying = [ True  True]
      @sample_varying = True
      @time_varying = True
    ... # Additional fields

  -t2_fields: Group
    @field_names: list[str] = ['TFieldA', ...]
    - TFieldA: Dataset = float32(BxTxWxHxD^2)
      @antisymmetric = False
      @dim_varying = [ True  True]
      @sample_varying = True
      @symmetric = True # Whether tensor is symmetric
      @time_varying = True
    ... # Additional fields
```
