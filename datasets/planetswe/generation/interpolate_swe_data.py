import argparse
import glob
import os

import dedalus.public as d3
import h5py as h5
import numpy as np


def populate_empty_file(file):
    create_dimensions(file)
    create_base_attributes(file)
    create_field_types(file)


def create_boundary_conditions(file):
    bcs = file.create_group("boundary_conditions")
    x = bcs.create_group("x_periodic")
    x.attrs["associated_dims"] = ["phi"]
    x.attrs["bc_type"] = "PERIODIC"
    x.attrs["associated_fields"] = []
    x.attrs["sample_varying"] = False
    x.attrs["time_varying"] = False
    mask = np.zeros_like(file["dimensions"]["phi"], dtype=bool)
    mask[0] = True
    mask[-1] = True
    xds = x.create_dataset("mask", data=mask, dtype=bool)

    y = bcs.create_group("y_open")
    y.attrs["associated_dims"] = ["theta"]
    y.attrs["bc_type"] = "OPEN"
    y.attrs["associated_fields"] = []
    mask = np.zeros_like(file["dimensions"]["theta"], dtype=bool)
    mask[0] = True
    mask[-1] = True
    yds = y.create_dataset("mask", data=mask, dtype=bool)
    y.attrs["sample_varying"] = False
    y.attrs["time_varying"] = False


def create_base_attributes(file):
    file.attrs["dataset_name"] = "dataset"
    file.attrs["n_spatial_dims"] = 3
    file.attrs["simulation_parameters"] = []
    file.attrs["grid_type"] = "cartesian"


def create_field_types(file):
    field_types = ["t0_fields", "t1_fields", "t2_fields", "scalars"]
    for field_type in field_types:
        gr = file.create_group(field_type)
        gr.attrs["field_names"] = []


def create_dimensions(file):
    file.create_group("dimensions")
    file["dimensions"].attrs["spatial_dims"] = ["phi", "theta"]
    # file['dimensions'].create_dataset('time', data=np.array([0]))


def earthswe_to_well(in_path, out_path):
    print("Starting file copy!")
    orig_file = h5.File(in_path, "r")

    print("orig keys", list(orig_file.keys()))
    if os.path.exists(out_path):
        os.remove(out_path)
    with h5.File(out_path, "w") as new_file:
        populate_empty_file(new_file)
        ## First populate the attributes
        new_file.attrs["dataset_name"] = "planetswe"
        new_file.attrs["n_spatial_dims"] = 2
        new_file.attrs["simulation_parameters"] = []
        new_file.attrs["grid_type"] = "equiangular"
        print("orig_file", orig_file.keys())
        new_file.attrs["n_trajectories"] = 1  # orig_file['c'].shape[0]
        # Make attributes for each simulation parameter
        # parameter_string = in_path.split('/')[-1][:-5].split('_')
        # print(parameter_string)
        new_file["scalars"].attrs["field_names"] = new_file.attrs[
            "simulation_parameters"
        ]
        # Now let's populate the dimensions
        new_file["dimensions"].attrs["spatial_dims"] = ["theta", "phi"]
        time = new_file["dimensions"].create_dataset(
            "time", data=orig_file["scales"]["sim_time"], dtype="f4"
        )
        time.attrs["sample_varying"] = False
        # Same coordinates for x and y in this specific data
        print(orig_file["scales"].keys())
        d = new_file["dimensions"].create_dataset(
            "phi",
            data=orig_file["scales"][
                "phi_hash_7b8ec7cabc40ac4b596a5ef833e9eab019f07d46"
            ],
            dtype="f4",
        )
        d.attrs["time_varying"] = False
        d.attrs["sample_varying"] = False
        d = new_file["dimensions"].create_dataset(
            "theta",
            data=orig_file["scales"][
                "theta_hash_47f1a1c5acad69381fef2149e23fb804716211f6"
            ],
            dtype="f4",
        )
        d.attrs["time_varying"] = False
        d.attrs["sample_varying"] = False

        h, u = dedalus_interpolate(
            orig_file["tasks"]["h"][:], orig_file["tasks"]["u"][:]
        )
        # T0 Data
        new_file["t0_fields"].attrs["field_names"] = ["height"]
        f = new_file["t0_fields"].create_dataset(
            "height", data=np.transpose(h[np.newaxis, ...], (0, 1, 3, 2)), dtype="f4"
        )
        f.attrs["time_varying"] = True
        f.attrs["sample_varying"] = True
        f.attrs["dim_varying"] = [True, True]

        # T1 Data
        new_file["t1_fields"].attrs["field_names"] = ["velocity"]
        f = new_file["t1_fields"].create_dataset(
            "velocity",
            data=np.transpose(u[np.newaxis, ...], (0, 1, 4, 3, 2)),
            dtype="f4",
        )
        f.attrs["time_varying"] = True
        f.attrs["sample_varying"] = True
        f.attrs["dim_varying"] = [True, True]
        # T2 Data
        new_file["t2_fields"].attrs["field_names"] = []

        create_boundary_conditions(new_file)


def dedalus_interpolate(h, u):
    meter = 1 / 6.37122e6
    hour = 1
    second = hour / 3600
    g = 9.80616 * meter / second**2
    Nphi = 512
    Ntheta = 256
    dtype = np.float64
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=1, dealias=1, dtype=dtype)
    h3 = dist.Field(name="h", bases=basis)
    u3 = dist.VectorField(coords, name="u", bases=basis)

    nphi = h.shape[1]
    ntheta = h.shape[2]
    u_out = np.zeros(u.shape)
    h_out = np.zeros(h.shape)
    delta = np.pi / (ntheta + 1)
    for j in range(u.shape[0]):
        if j % 50 == 0:
            print("row", j)
        u3["g"] = u[j]
        h3["g"] = h[j]
        print("field shape!", u3["g"].shape)

        for i, pt in enumerate(np.linspace(np.pi - delta / 2, delta / 2, ntheta)):
            u_interp = d3.Interpolate(u3, "theta", pt).evaluate()["g"]
            h_interp = d3.Interpolate(h3, "theta", pt).evaluate()["g"]
            u_out[j, ..., i : i + 1] = u_interp * second / meter
            h_out[j, ..., i : i + 1] = h_interp / meter

    return h_out, u_out


if __name__ == "__main__":
    print("HAVE WE EVEN STARTED CODE YET?")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/mnt/home/polymathic/ceph/the_well/testing_before_adding/earthswe",
    )
    parser.add_argument(
        "--dest", default="/mnt/home/polymathic/ceph/the_well/datasets/planetswe/data"
    )
    parser.add_argument("--index", default="0")
    args = parser.parse_args()

    current_path = args.source
    write_path = args.dest
    ic_file = int(args.index)
    max_ic_train = 32
    max_ic_valid = 36

    ic_folders = sorted(glob.glob(f"{current_path}/IC*"))
    target_ic = ic_folders[ic_file]
    print("picked source", target_ic)
    # for i, folder in enumerate(ic_folders):
    ic_num = int(target_ic.split("_")[-1])
    if ic_num < max_ic_train:
        split = "train"
    elif ic_num < max_ic_valid:
        split = "valid"
    else:
        split = "test"
    for i in range(10):
        print(i)
    for file in glob.glob(f"{target_ic}/*.h5"):
        file_idx = file.split("_")[-1][:-3]
        target_path = f"{write_path}/{split}/planetswe_IC{ic_num:02d}_{file_idx}.h5"
        print(file, target_path)
        earthswe_to_well(file, target_path)
