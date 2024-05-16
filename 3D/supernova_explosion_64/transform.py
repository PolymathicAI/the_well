#Template for tranforming datasets

import numpy as np
import h5py as h5
import os
import argparse
import torch
import sys
import glob
original_sys_path = sys.path[:]
sys.path.append('../..')
from the_well.benchmark.data.datasets import GenericWellDataset  
sys.path = original_sys_path


def populate_empty_file(file):
    create_dimensions(file)
    create_base_attributes(file)
    create_field_types(file)
    

def create_boundary_conditions(file):
    bcs = file.create_group('boundary_conditions')
    x = bcs.create_group('x_open')
    x.attrs['associated_dims'] = ['x']
    x.attrs['bc_type'] = 'open'
    x.attrs['associated_fields'] = []
    x.attrs['sample_varying'] = False
    x.attrs['time_varying'] = False
    mask_x = np.zeros_like(file['dimensions']['x'], dtype=bool)
    mask_x[0] = True
    mask_x[-1] = True
    xds = x.create_dataset('mask', data=mask_x, dtype=bool)

    y = bcs.create_group('y_open')
    y.attrs['associated_dims'] = ['y']
    y.attrs['bc_type'] = 'open'
    y.attrs['associated_fields'] = []
    y.attrs['sample_varying'] = False
    y.attrs['time_varying'] = False
    mask_y = np.zeros_like(file['dimensions']['y'], dtype=bool)
    mask_y[0] = True
    mask_y[-1] = True
    yds = y.create_dataset('mask', data=mask_y, dtype=bool)

    z = bcs.create_group('z_open')
    z.attrs['associated_dims'] = ['z']
    z.attrs['bc_type'] = 'open'
    z.attrs['associated_fields'] = []
    z.attrs['sample_varying'] = False
    z.attrs['time_varying'] = False
    mask_z = np.zeros_like(file['dimensions']['z'], dtype=bool)
    mask_z[0] = True
    mask_z[-1] = True
    zds = z.create_dataset('mask', data=mask_z, dtype=bool)


def create_base_attributes(file):
    file.attrs['dataset_name'] = 'dataset'
    file.attrs['n_spatial_dims'] = 3
    file.attrs['simulation_parameters'] = []
    file.attrs['grid_type'] = 'cartesian'

def create_field_types(file):
    field_types = ['t0_fields', 't1_fields', 't2_fields', 'scalars']
    for field_type in field_types:
        gr = file.create_group(field_type)
        gr.attrs['field_names'] = []

def create_dimensions(file):
    file.create_group('dimensions')
    file['dimensions'].attrs['spatial_dims'] = ['x', 'y', 'z']
    # file['dimensions'].create_dataset('time', data=np.array([0]))


def dataset_to_well(in_path, replacing_path=None):
    out_path = in_path.replace('data/', replacing_path)
    print('in_path', in_path)
    orig_file = h5.File(in_path, 'r')

    print('orig keys', list(orig_file.keys()))
    # Lxmin = 0
    # Lymin = 0
    # Lxmax = 1
    # Lymax = 1
    coordinates_x = orig_file['x-coordinate'][:]
    coordinates_y = orig_file['y-coordinate'][:]
    coordinates_z = orig_file['z-coordinate'][:]
    #coordinates_y = np.linspace(Lymin, Lymax, orig_file['u'].shape[-2]) #CHECK SHAPE
    if os.path.exists(out_path):
        os.remove(out_path)
    with h5.File(out_path, 'w-') as new_file:
        populate_empty_file(new_file)
        ## First populate the attributes
        new_file.attrs['dataset_name'] = 'supernova_explosion'
        new_file.attrs['n_spatial_dims'] = 3
        new_file.attrs['simulation_parameters'] = ['Msun','rho0', 'Z', 'T0']
        new_file.attrs['grid_type'] = 'cartesian'
        print('orig_file', orig_file.keys())
        new_file.attrs['n_trajectories'] = orig_file['density'].shape[0]
        # Make attributes for each simulation parameter
        parameter_string = in_path.split('/')[-1][:-5].split('_')
        print(parameter_string)
        new_file['scalars'].attrs['field_names'] = new_file.attrs['simulation_parameters']
        #for i, param in enumerate(new_file.attrs['simulation_parameters']):
        new_file.attrs['Msun'] = float(parameter_string[1]) #changed here to 2* +2 as there is the first word
        f = new_file['scalars'].create_dataset('Msun', data=np.array(float(parameter_string[1])), 
                                            dtype='f4')
        
        f.attrs['time_varying'] = False
        f.attrs['sample_varying'] = False        
        f = new_file['scalars'].create_dataset('rho0', data=float(44.5), 
                                            dtype='f4')
        f.attrs['time_varying'] = False
        f.attrs['sample_varying'] = False
        f = new_file['scalars'].create_dataset('Z', data=float(1.0),
                                            dtype='f4')
        f.attrs['time_varying'] = False
        f.attrs['sample_varying'] = False
        f = new_file['scalars'].create_dataset('T0', data=float(100.0),
                                            dtype='f4')
        f.attrs['time_varying'] = False
        f.attrs['sample_varying'] = False
        # Now let's populate the dimensions
        new_file['dimensions'].attrs['spatial_dims'] = ['x', 'y', 'z']
        time = new_file['dimensions'].create_dataset('time', data=(orig_file['t-coordinate'][:]*0.4), dtype='f4')
        time.attrs['sample_varying'] = False
        # Same coordinates for x and y in this specific data
        # for dim in new_file['dimensions'].attrs['spatial_dims']:
        dim = 'x'
        d = new_file['dimensions'].create_dataset(dim, data=coordinates_x, dtype='f4')
        d.attrs['time_varying'] = False
        d.attrs['sample_varying'] = False

        dim = 'y'
        d = new_file['dimensions'].create_dataset(dim, data=coordinates_y, dtype='f4')
        d.attrs['time_varying'] = False
        d.attrs['sample_varying'] = False

        dim = 'z'
        d = new_file['dimensions'].create_dataset(dim, data=coordinates_z, dtype='f4')
        d.attrs['time_varying'] = False
        d.attrs['sample_varying'] = False

        # T0 Data        
        new_file['t0_fields'].attrs['field_names'] = ['density', 'pressure', 'temperature']
        f = new_file['t0_fields'].create_dataset('density', data=orig_file['density'], dtype='f4')
        f.attrs['time_varying'] = True
        f.attrs['sample_varying'] = True
        f.attrs['dim_varying'] = [True, True, True]

        f = new_file['t0_fields'].create_dataset('pressure', data=orig_file['pressure'], dtype='f4')
        f.attrs['time_varying'] = True
        f.attrs['sample_varying'] = True
        f.attrs['dim_varying'] = [True, True,True]

        f = new_file['t0_fields'].create_dataset('temperature', data=orig_file['temperature'], dtype='f4')
        f.attrs['time_varying'] = True
        f.attrs['sample_varying'] = True
        f.attrs['dim_varying'] = [True, True,True]

        # new_file['t0_fields'].attrs['field_names'] = ['pressure']
        # f = new_file['t0_fields'].create_dataset('pressure', data=orig_file['pressure'], dtype='f4')
        # f.attrs['time_varying'] = True
        # f.attrs['sample_varying'] = True
        # f.attrs['dim_varying'] = [True, True]
    
        # T1 Data
        new_file['t1_fields'].attrs['field_names'] = ['velocity']
        f = new_file['t1_fields'].create_dataset('velocity', data=np.stack([orig_file['Vx'], orig_file['Vy'], orig_file['Vz']], axis=-1), dtype='f4')
        f.attrs['time_varying'] = True
        f.attrs['sample_varying'] = True
        f.attrs['dim_varying'] = [True, True, True]

        # T2 Data
        # new_file['t2_fields'].attrs['field_names'] = ['D', 'E']
        # for field in new_file['t2_fields'].attrs['field_names']:
        #     new_data = np.zeros(orig_file[f'{field}_xx'].shape 
        #                         + len(new_file['dimensions'].attrs['spatial_dims'])
        #                               *(len(new_file['dimensions'].attrs['spatial_dims'],), 
        #                                ))
        #     new_data[..., 0, 0] = orig_file[f'{field}_xx']
        #     new_data[..., 0, 1] = orig_file[f'{field}_xy']
        #     new_data[..., 1, 0] = orig_file[f'{field}_xy']
        #     new_data[..., 1, 1] = orig_file[f'{field}_yy']
        #     f = new_file['t2_fields'].create_dataset(field, data=new_data, dtype='f4')
        #     f.attrs['symmetric'] = True
        #     f.attrs['antisymmetric'] = False
        #     f.attrs['time_varying'] = True
        #     f.attrs['sample_varying'] = True
        #     f.attrs['dim_varying'] = [True, True]

        create_boundary_conditions(new_file)


#split into train/val/test:
def copy_subset(in_file, out_file, subset):
    for key in in_file.attrs.keys():
        if key == 'n_trajectories':
            out_file.attrs[key] = sum(subset)
        else:
            out_file.attrs[key] = in_file.attrs[key]
    if isinstance(in_file, h5.Group):
        for key in in_file.keys():
            if isinstance(in_file[key], h5.Group):
                out_subfile = out_file.create_group(key)
                in_subfile = in_file[key]
                
            elif isinstance(in_file[key], h5.Dataset):
                if 'sample_varying' in in_file[key].attrs and in_file[key].attrs['sample_varying']:
                    out_subfile = out_file.create_dataset(key, data=in_file[key][subset])
                else:
                    out_subfile = out_file.create_dataset(key, data=in_file[key])
                in_subfile = in_file[key]
            copy_subset(in_subfile, out_subfile, subset)
            
        

def split_hdf5_file(in_path, split_ratios=(0.8, .1, 0.1)):
    
    # Check if the folder containing "in_path" has train and test folders
    # If not, create them
    folder = os.path.dirname(in_path)
    base_object_name = in_path.split('/')[-1]
    train_folder = os.path.join(folder, 'train')
    test_folder = os.path.join(folder, 'test')
    valid_folder = os.path.join(folder, 'valid')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    train = None
    valid = None
    test = None
    # Open the original file    
    with h5.File(in_path, 'r') as f:
        n_trajectories = f.attrs['n_trajectories']

        assignments = np.zeros(n_trajectories)
        train_idx = int(split_ratios[0]*n_trajectories)
        valid_idx = int(split_ratios[1]*n_trajectories)
        test_idx = int(split_ratios[2]*n_trajectories)
        assignments[0:train_idx] = 0
        assignments[train_idx:train_idx+valid_idx] = 1
        assignments[train_idx+valid_idx:] = 2
        assignments = np.random.permutation(assignments)
        # print(assignments)
        if sum(assignments == 0) > 0:
            with h5.File(os.path.join(train_folder, base_object_name), 'w-') as train:
                copy_subset(f, train, assignments == 0)
        if sum(assignments == 1) > 0:
            with h5.File(os.path.join(valid_folder, base_object_name), 'w-') as valid:
                copy_subset(f, valid, assignments == 1)
        if sum(assignments == 2) > 0:
            with h5.File(os.path.join(test_folder, base_object_name), 'w-') as test:
                copy_subset(f, test, assignments == 2)

def compute_statistics(train_path):
    ds = GenericWellDataset(train_path, use_normalization=False)
    paths = ds.files_paths
    means = {}
    counts = {}
    fields = ['t0_fields', 't1_fields', 't2_fields']
    for p in paths:
        with h5.File(p, 'r') as f:
            print('\n',p)
            print(f)
            for fi in fields:
                print('\n',fi)
                for field in f[fi].attrs['field_names']:
                    print('\t',field)
                    data = f[fi][field][:]
                    if field not in means:
                        means[field] = data.sum()
                        counts[field] = np.prod(data.shape)
                    else:
                        means[field] += data.sum()
                        counts[field] += np.prod(data.shape)


    # Compute means from sum and counts
    for field in means:
        means[field] /= counts[field]
    print(means)

    # Now let's compute the variance
    variances = {}
    for p in paths:
        with h5.File(p, 'r') as f:
            print('\n',p)
            for fi in fields:
                print('\n',fi)
                for field in f[fi].attrs['field_names']:
                    print('\t',field)
                    data = f[fi][field][:]
                    if field not in variances:
                        variances[field] = ((data - means[field])**2).sum()
                    else:
                        variances[field] += ((data - means[field])**2).sum()

    # Compute vars from sum and counts
    for field in variances:
        variances[field] /= counts[field] - 1 

    stds = {k: np.sqrt(v) for k, v in variances.items()}
    print('saving')
    if not os.path.exists('stats'):
        os.makedirs('stats')
    with open('stats/means.pkl', 'wb') as f:
        torch.save(means, f)

    with open('stats/stds.pkl', 'wb') as f:
        torch.save(stds, f)


def launch_transform(subset=None):

    paths = sorted(glob.glob('data/*.hdf5'))
    for in_path in paths[40:]:
        print(f'transforming {in_path}')
        replacing_path = 'data_processed/supernova_explosion_'
        dataset_to_well(in_path, replacing_path)
        in_path = in_path.replace('data/', replacing_path)
        print(f'splitting {in_path}')
        split_hdf5_file(in_path)

    

if __name__ == '__main__':

    #launch_transform()
    compute_statistics('data_processed/train')
