### Visualize the `acoustic_scattering_maze` dataset


```python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
```


```python
#print the list of paths of files in the training set
set_path = 'train'
paths = sorted(glob.glob(f'data/{set_path}/*.hdf5'))
print(paths)
```

    ['data/train/acoustic_scattering_maze_2d_chunk_0.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_1.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_10.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_11.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_12.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_13.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_14.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_15.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_2.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_3.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_4.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_5.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_6.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_7.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_8.hdf5', 'data/train/acoustic_scattering_maze_2d_chunk_9.hdf5']



```python
#select the first path (arbitrary choice)
p = paths[0]

#print the first layer of keys
with h5py.File(p,'r') as f:
    print(f.keys())
```

    <KeysViewHDF5 ['boundary_conditions', 'dimensions', 'scalars', 't0_fields', 't1_fields', 't2_fields']>



```python
# In 'boundary_conditions' is stored the information about the boundary conditions:
with h5py.File(p,'r') as f:
    print('print bc available:', f['boundary_conditions'].keys())
    print('print attributes of the bc:', f['boundary_conditions']['x_wall'].attrs.keys())
    print('get the bc type:', f['boundary_conditions']['x_wall'].attrs['bc_type'])
```

    print bc available: <KeysViewHDF5 ['x_wall', 'y_open']>
    print attributes of the bc: <KeysViewHDF5 ['associated_dims', 'associated_fields', 'bc_type', 'sample_varying', 'time_varying']>
    get the bc type: WALL



```python
#Reminder: 't0_fields', 't1_fields', 't2_fields' are respectively scalar fields, vector fields and tensor fields
#print the different fields available in the dataset
with h5py.File(p,'r') as f:
    print('t0_fields:', f['t0_fields'].keys())
    print('t1_fields:', f['t1_fields'].keys())
    print('t2_fields:', f['t2_fields'].keys())
```

    t0_fields: <KeysViewHDF5 ['bulk_modulus', 'density', 'pressure']>
    t1_fields: <KeysViewHDF5 ['velocity']>
    t2_fields: <KeysViewHDF5 []>



```python
#The data is of shape (n_trajectories, n_timesteps, x, y)
#Get the first t0_field and save it as a numpy array
with h5py.File(p,'r') as f:
    pressure = f['t0_fields']['pressure'][:] #HDF5 datasets can be sliced like a numpy array
    print(f'shape of the selected t0_field: ', pressure.shape)   

    #you can directly slice the selected field without reading the whole dataset by doing:
    #traj = 0 #select the trajectory
    # field = f['t0_fields']['concentration'][traj, :] 
```

    shape of the selected t0_field:  (100, 202, 256, 256)



```python
#field is now of shape (n_timesteps, x, y). 
traj = 1
traj_toplot = pressure[traj,...] 
# Let's do a subplot to plot it at t= 0, t= T/3, t= 2T/3 and t= T:
fig, axs = plt.subplots(1, 4, figsize=(20,5))
T = traj_toplot.shape[0]

#same colorbar for all subplots:
normalize_plots = False
cmap =  'RdBu_r'

if normalize_plots:
    vmin = np.nanmin(traj_toplot)
    vmax = np.nanmax(traj_toplot)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i, t in enumerate([0, T//3, (2*T)//3, T-1]):
        axs[i].imshow(traj_toplot[t], cmap=cmap, norm=norm)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(f't={t}')
else:
    for i, t in enumerate([0, T//3, (2*T)//3, T-1]):
        axs[i].imshow(traj_toplot[t], cmap=cmap)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(f't={t}')
plt.tight_layout()


```


    
![png](visualization_acoustic_scattering_maze_files/visualization_acoustic_scattering_maze_7_0.png)
    

