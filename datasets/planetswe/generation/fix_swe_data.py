import h5py
import numpy as np
import dedalus.public as d3
import os
import glob
import multiprocessing as mp
import argparse

# Sims that we know have issues from individual inspection
bad_sims = ['IC_00_s2.h5',
      'IC_01_s1.h5', 'IC_01_s2.h5',
      'IC_08_s1.h5', 'IC_09_s1.h5',
      'IC_10_s1.h5', 'IC_13_s2.h5',
      'IC_13_s3.h5',
      'IC_15_s1.h5', 'IC_16_s1.h5',
      'IC_17_s1.h5', 'IC_18_s1.h5',
      'IC_23_s2.h5', 'IC_24_s1.h5'
      'IC_25_s1.h5', 'IC_26_s1.h5']

val_cutoff = 25
test_cutoff = 28

def process_files(args):
    filename, dset = args
    last_part = filename.split('/')[-1]
    os.makedirs(filename.replace(last_part, dset), exist_ok=True)
    new_filename = filename.replace(last_part, dset + '/' + last_part)
    print(new_filename)

    meter = 1 / 6.37122e6
    hour = 1
    second = hour / 3600
    g = 9.80616 * meter / second**2

    H = 5960 * meter

    Nphi = 256
    Ntheta = 128
    dtype = np.float64
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=1, dealias=1, dtype=dtype)
    h3 = dist.Field(name='h', bases=basis)
    u3 = dist.VectorField(coords, name='u', bases=basis)

    #filecount += 1
    with h5py.File(new_filename, mode='w') as new_file:
        with h5py.File(filename, mode='r+') as file:
            try:
                del(file['fields'])
            except:
                pass
            u = file['tasks']['u'][:]
            h = file['tasks']['h'][:]
            nphi = h.shape[1]
            ntheta = h.shape[2]
            out_fields = np.zeros((u.shape[0], 3, ntheta, nphi))
            delta = np.pi/(ntheta+1)
            for j in range(u.shape[0]):
                if j % 50 == 0:
                    print('row', j)
                u3['g'] = u[j]
                h3['g'] = h[j]

                for i, pt in enumerate(np.linspace(np.pi-delta/2, delta/2, ntheta)):
                    u_interp = np.swapaxes(d3.Interpolate(u3, 'theta', pt).evaluate()['g'], 1, 2)
                    h_interp = np.swapaxes(d3.Interpolate(h3, 'theta', pt).evaluate()['g'], 0, 1)
                    out_fields[j, :2, i:i+1] = u_interp * second / meter
                    out_fields[j, 2, i:i+1] = h_interp / meter

            new_file.create_dataset('fields', data=out_fields)
        print(filename, 'complete')
        return filename

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.')
    args = parser.parse_args()
    ic_file = args.ic_file

    file_path = parser.data_root 
    files = glob.glob(file_path + '/*.h5')
    files = [f for f in files if f.split('/')[-1] not in bad_sims]

    train_files = [f for f in files if int(f.split('/')[-1].split('_')[1]) < val_cutoff]
    val_files = [f for f in files if int(f.split('/')[-1].split('_')[1]) >= val_cutoff 
                 and int(f.split('/')[-1].split('_')[1]) < test_cutoff]
    test_files = [f for f in files if int(f.split('/')[-1].split('_')[1]) >= test_cutoff]

    all_files = [(f, 'train') for f in train_files] \
            + [(f, 'val') for f in val_files] \
                + [(f, 'test') for f in test_files] 

    cpus = mp.cpu_count()
    print('CPUs', cpus)
    filecount = 0

    pool = mp.Pool(16)
    returns = pool.map(process_files, all_files)
    print('Processing complete. Feel free to delete the raw files.')