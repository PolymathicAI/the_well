import torch
import h5py as h5
import glob
import numpy as np
from torch.utils.data import Dataset


well_paths = {'active_matter': '/2D/active_matter'}



class GenericWellDataset(Dataset):
    """
    Generic dataset for any Well data. Returns data in B x T x C x H ...(x W) (x D) format.

    Note - doesn't currently normalize internally

    Train/Test/Valid is assumed to occur on a folder level.

    Takes in path to directory of HDF5 files to construct dset. 

    Parameters
    ----------
    path : str, default=None
        Path to directory of HDF5 files, one of path or well_base_path+well_dataset_name must be specified
    well_base_path : str, default=None
        Path to well dataset directory, only used with dataset_name
    well_dataset_name : str, default=None
        Name of well dataset to load - overrides path if specified
    include_string : str, default=None
        Only include files with this string in name
    exclude_string : str, default=None
        Exclude files with this string in name
    n_steps_input : int, default=1
        Number of steps to include in each sample
    n_steps_output : int, default=1
        Number of steps to include in y
    dt_stride : int, default=1
        Minimum stride between samples
    max_dt_stride : int, default=1
        Maximum stride between samples
    flatten_tensors : bool, default=True
        Whether to flatten tensor valued field into channels
    return_grid : bool, default=False
        Whether to return grid coordinates
    name_override : str, default=None
        Override name of dataset (used for more precise logging)
    transforms : List[function], default=[]
        List of transforms to apply to data
    tensor_transformers : List[function], default=[]
        List of transforms to apply to tensor fields
    """
    def __init__(self, path=None, well_base_path=None, well_dataset_name=None, 
                 include_string=None, exclude_string=None,
                    n_steps_input=1, n_steps_output=1, dt_stride=1, max_dt_stride=1,
                    flatten_tensors=True, return_grid=False, 
                    name_override=None, transforms=[], tensor_transforms=[]):
        super().__init__()
        assert path is not None or (well_base_path is not None and well_dataset_name is not None), \
                 'Must specify path or well_base_path and well_dataset_name'
        if path is not None:
            self.path = path
        else:
            self.path = well_base_path + well_paths[well_dataset_name]
        # Copy params
        self.include_string = include_string
        self.exclude_string = exclude_string
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.dt_stride = dt_stride
        self.max_dt_stride = max_dt_stride
        self.flatten_tensors = flatten_tensors
        self.return_grid = return_grid
        self.transforms = transforms
        self.tensor_transforms = tensor_transforms
        # Check the directory has hdf5 that meet our exclusion criteria
        sub_files = glob.glob(self.path + '/*.h5') + glob.glob(self.path + '/*.hdf5')
        if include_string is not None and len(include_string) > 0:
            sub_files = [f for f in sub_files if include_string in f]
        if exclude_string is not None and len(exclude_string) > 0:
            sub_files = [f for f in sub_files if exclude_string not in f]
        assert len(sub_files) > 0, 'No HDF5 files found in path {}'.format(self.path)
        self.files_paths = sub_files
        self.files_paths.sort()
        # Build multi-index
        self._build_metadata()
        # Override name if necessary for logging
        if name_override is not None:
            self.dataset_name = name_override
    
    def get_per_file_dsets(self):
        # TODO Not implemented for real yet.
        if self.split_level == 'file' or len(self.files_paths) == 1 or 'pa' in self.type:
            return [self]
        else:
            sub_dsets = []
            for file in self.files_paths:
                subd = self.__class__(self.path, file, n_steps=self.n_steps, dt=self.dt, split=self.split,
                               train_val_test=self.train_val_test, subname=self.subname,
                                 extra_specific=True)
                sub_dsets.append(subd)
            return sub_dsets
    
    def _get_specific_bcs(self, f):
        raise NotImplementedError # Per dset
    
    def _get_space_grid(self, file):
        raise NotImplementedError

    
    def _build_metadata(self):
        """ Builds multi-file indices and checks that folder contains consistent dataset
        """
        self.n_files = len(self.files_paths)
        self.file_steps = []
        self.file_nsteps = []
        self.file_samples = []
        self.offsets = [0]
        self.field_names = []
        # Things where we just care every file has same value
        size_tuples = set()
        names = set()
        ndims = set()
        for index, file in enumerate(self.files_paths):
            with h5.File(file, 'r') as _f:
                # Run sanity checks - all files should have same ndims, size_tuple, and names
                samples = _f.attrs['n_trajectories']
                steps = _f['dimensions']['time'].shape[0]
                size_tuple = [_f['dimensions'][d].shape[0] 
                                for d in _f['dimensions'].attrs['spatial_dims']]
                ndims.add(_f.attrs['n_spatial_dims'])
                names.add(_f.attrs['dataset_name'])
                size_tuples.add(tuple(size_tuple))
                # Fast enough that I'd rather check each file rather than processing extra files before checking
                assert len(names) == 1, 'Multiple dataset names found in specified path'
                assert len(ndims) == 1, 'Multiple ndims found in specified path'
                assert len(size_tuples) == 1, 'Multiple resolutions found in specified path'
                # Check that the requested steps make sense
                assert steps - self.dt_stride*(self.n_steps_input + self.n_steps_output) > 0, \
                    'Not enough steps in file {} for {} input and {} output steps'.format(file,
                                                                                          self.n_steps_input,
                                                                                          self.n_steps_output)
                self.file_steps.append(steps)
                self.file_samples.append(samples)
                self.offsets.append(self.offsets[-1] +  samples * (steps 
                                    - self.dt_stride*(self.n_steps_input-1 + self.n_steps_output)))
                # Populate field names
                if index == 0:
                    self.num_fields_by_tensor_order = {}
                    self.num_constants = len(_f.attrs['simulation_parameters'])
                    for field in _f['t0_fields'].attrs['field_names']:
                        self.field_names.append(field)
                    self.num_fields_by_tensor_order[0] = len(_f['t0_fields'].attrs['field_names'])
                    for field in _f['t1_fields'].attrs['field_names']:
                        for dim in _f['dimensions'].attrs['spatial_dims']:
                            self.field_names.append(f'{field}_{dim}')
                    self.num_fields_by_tensor_order[1] = len(_f['t1_fields'].attrs['field_names'])
                    for field in _f['t2_fields'].attrs['field_names']:
                        for i, dim1 in enumerate(_f['dimensions'].attrs['spatial_dims']):
                            for j, dim2 in enumerate(_f['dimensions'].attrs['spatial_dims']):
                                # Commenting this out for now - need to figure out a way to
                                # actually get performance here. 
                                # if _f['t2_fields'][field].attrs['symmetric']:
                                #     if i > j:
                                #         continue
                                self.field_names.append(f'{field}_{dim1}{dim2}')
                    self.num_fields_by_tensor_order[2] = len(_f['t2_fields'].attrs['field_names'])
                                
        self.offsets[0] = -1 # Just to make sure it doesn't put us in file -1
        self.files = [None for _ in self.files_paths] # We open file references as they come
        self.len = self.offsets[-1]
        self.ndims = list(ndims)[0]
        self.size_tuple = list(size_tuples)[0]
        self.dataset_name = list(names)[0]
        self.num_total_fields = len(self.field_names)
                
    def _open_file(self, file_ind):
        _file = h5.File(self.files_paths[file_ind], 'r')
        self.files[file_ind] = _file

    def _pad_axes(self, field_data, use_dims, time_varying=False, tensor_order=0):
        """Repeats data over axes not used in storage"""
        # Look at which dimensions currently are not used and tile based on their sizes
        expand_dims = (1,) if time_varying else ()
        expand_dims = expand_dims + tuple([self.size_tuple[i] if not use_dim else 1 
                                           for i, use_dim in enumerate(use_dims)])
        expand_dims = expand_dims + (1,)*tensor_order
        return np.tile(field_data, expand_dims)

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps, dt):
        """ Reconstruct sample starting at index sample_idx, time_idx, with 
        n_steps and dt stride. Apply transformations if provided."""
        variable_fields = []
        constant_fields = []
        # Iterate through field types and apply appropriate transforms to stack them
        for i, order_fields in enumerate(['t0_fields', 't1_fields', 't2_fields']):
            sub_fields = []
            for field_name in file[order_fields].attrs['field_names']:
                field = file[order_fields][field_name]
                use_dims = field.attrs['dim_varying']
                # TODO if we have slow loading, it might be better to apply both indices at once
                field_data = field
                if field.attrs['sample_varying']:
                    field_data = field_data[sample_idx]
                if field.attrs['time_varying']:
                    field_data = field_data[time_idx:time_idx+n_steps*dt:dt]
                field_data = torch.tensor(
                    self._pad_axes(field_data, use_dims, time_varying=True, tensor_order=i)
                )
                sub_fields.append(field_data)
            # Stack fields such that the last i dims are the tensor dims
            sub_fields = torch.stack(sub_fields, -(i+1))
            for tensor_transform in self.tensor_transforms:
                sub_fields = tensor_transform(sub_fields, order=i)
            # If we're flattening tensors, we can then flatten last i dims
            if self.flatten_tensors:
                sub_fields = sub_fields.flatten(-(i+1))
            if field.attrs['time_varying']:
                variable_fields.append(sub_fields)
            else:
                constant_fields.append(sub_fields)

        constant_scalars = []
        return tuple([torch.concatenate(field_group, -1) 
                      if len(field_group) > 0 else None 
                      for field_group in [variable_fields, constant_fields, 
                                          constant_scalars]])

    def __getitem__(self, index):
        # Find specific file and local index
        file_idx = int(np.searchsorted(self.offsets, index, side='right')-1) #which file we are on
        file_steps = self.file_steps[file_idx]
        local_idx = index - max(self.offsets[file_idx], 0) # First offset is -1
        sample_idx = local_idx // file_steps
        time_idx = local_idx % file_steps

        #open hdf5 file (and cache the open object)
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        #if we overflow into the next sample/file then shift backward. Double counting until we bother fixing this.
        if self.max_dt_stride > self.dt_stride:
            sample_steps = self.n_steps_input + self.n_steps_output
            effective_max_dt = min(int((file_steps - time_idx) // sample_steps), self.max_dt_stride)
            if effective_max_dt > self.dt:
                dt = np.random.randint(self.dt, effective_max_dt)
        else:
            dt = self.dt_stride
        field_list = []
        t0 = self.files[file_idx]['t0_fields']
        for field in t0.attrs['field_names']:
            field_list.append(t0[field][sample_idx, time_idx])
        self.files[file_idx]
        trajectory, constant_fields, constant_scalars = self._reconstruct_sample(self.files[file_idx], sample_idx, 
                                                time_idx, self.n_steps_input + self.n_steps_output, dt)
        # TODO - Add BCS for real
        bcs = [] 
        return {'input_state': trajectory[:self.n_steps_input], # Tin x H x W x D x C tensor of input trajectory
                'constant_fields': constant_fields, # H (x W x D) x (num constant) tensor. 
                'space_grid': None, # H (x W x D) x (num dims) tensor with coordinate values
                'time_grid': None, # T x 1 tensor with time values
                'constant_scalars': None, # 1 x C tensor with constant values corresponding to parameters
                'time_varying_scalars': None, # Tin x C tensor with time varying scalars
                 'output_state': trajectory[self.n_steps_input:], # Tpred x H x W x D x C tensor of output trajectory
                   'boundary_conditions': torch.tensor(bcs) # WIP - currently ()
                   }

    def __len__(self):
        return self.len