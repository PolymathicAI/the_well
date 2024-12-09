"""Script based on streamlit to visualize data from the Well that are hosted on Hugging Face hub.

Any time the state change (due to UI interaction and callbacks),
the script is evaluated again.
Based on the state attributes some UI component are rendered
(e.g. slider for field time step).

"""

import pathlib

import fsspec
import h5py
import numpy as np
import pyvista as pv
import streamlit as st
from stpyvista.trame_backend import stpyvista

# Dataset whose data will be visualized
DATASET_NAMES = [
    "acoustic_scattering_inclusions",
    "active_matter",
    "helmholtz_staircase",
    "MHD_64",
    "shear_flow",
]

DIM_SUFFIXES = ["x", "y", "z"]

# Options for HDF5 cloud optimized reads
IO_PARAMS = {
    "fsspec_params": {
        # "skip_instance_cache": True
        "cache_type": "blockcache",  # or "first" with enough space
        "block_size": 2 * 1024 * 1024,  # could be bigger
        "token": st.secrets["HF_TOKEN"],
    },
    "h5py_params": {
        "driver_kwds": {  # only recent versions of xarray and h5netcdf allow this correctly
            "page_buf_size": 2 * 1024 * 1024,  # this one only works in repacked files
            "rdcc_nbytes": 2 * 1024 * 1024,  # this one is to read the chunks
        }
    },
}


# Instantiate streamlit state attributes
for key in ["file", "files", "field_names", "spatial_dim", "data"]:
    if key not in st.session_state:
        st.session_state[key] = None


def reset_state(key: str):
    if key in st.session_state:
        st.session_state[key] = None
        del st.session_state[key]


@st.cache_data
def get_dataset_path(dataset_name: str) -> str:
    """Compose the path to the dataset on HF hub."""
    repo_id = "polymathic-ai"
    dataset_path = f"hf://datasets/{repo_id}/{dataset_name}"
    return dataset_path


@st.cache_data
def get_dataset_files(dataset_name: str):
    """Get the list of files in the dataset."""
    dataset_path = get_dataset_path(dataset_name)
    fs, _ = fsspec.url_to_fs(dataset_path)
    dataset_files = fs.glob(f"{dataset_path}/**/*.hdf5")
    return dataset_files


@st.cache_data
def get_dataset_info(file_path: str) -> tuple([int, list[str]]):
    """Retrive spatial dimension and field names from the dataset."""
    file_path = f"hf://{file_path}"
    with fsspec.open(file_path, "rb") as f, h5py.File(f, "r") as file:
        spatial_dim = file.attrs["n_spatial_dims"]
        field_names = []
        for field in file["t0_fields"].keys():
            field_names.append((field, "t0_fields"))
        for field in file["t1_fields"].keys():
            for _, dim_suffix in zip(range(spatial_dim), DIM_SUFFIXES):
                field_names.append((f"{field}_{dim_suffix}", "t1_fields"))

        return spatial_dim, field_names


def dataset_info_callback():
    dataset_name = st.session_state.name
    dataset_files = get_dataset_files(dataset_name)
    st.session_state.files = dataset_files
    spatial_dim, field_names = get_dataset_info(dataset_files[0])
    st.session_state.spatial_dim = spatial_dim
    st.session_state.field_names = field_names
    # Field data for previous dataset must be cleared
    reset_state(key="data")


@st.cache_data
def get_field(file_path: str, field: tuple[str, str], spatial_dim: int) -> np.ndarray:
    """Load the first trajectory of a field in a given file."""
    file_path = f"hf://{file_path}"
    field_name, field_tensor_order = field
    if field_tensor_order == "t1_fields":
        field_name_splits = field_name.split("_")
        dim_suffix = field_name_splits[-1]
        dim_index = DIM_SUFFIXES.index(dim_suffix)
        field_name = "_".join(field_name_splits[:-1])
    else:
        dim_index = None
    with (
        fsspec.open(file_path, "rb", **IO_PARAMS["fsspec_params"]) as f,
        h5py.File(f, "r", **IO_PARAMS["h5py_params"]) as file,
    ):
        # Get the first trajectory of the file
        # For tensor of order 1 take the relevant spatial dimension
        if dim_index is not None:
            take_indices = (0, ..., dim_index)
        else:
            take_indices = 0
        field_data = np.array(file[field_tensor_order][field_name][take_indices])

        return field_data


def field_callback():
    """Callback to retrieve field data given file and field name state."""
    file = st.session_state.get("file", None)
    if file:
        field = st.session_state.field
        spatial_dim = st.session_state.spatial_dim
        field_data = get_field(file, field, spatial_dim)
        st.session_state.data = field_data
        # The field is constant
        if st.session_state.data.ndim <= 2:
            reset_state(key="time_step")


def create_plotter() -> pv.Plotter:
    """Create a pyvista.Plotter of the field in state."""
    # Check wether the field is dynamic
    # to account for time in spatial dimension retrieval
    time_step = st.session_state.get("time_step", None)
    position_offset = 0 if time_step is None else 1
    # Create 2D or 3D grid
    spatial_dim = st.session_state.spatial_dim
    if spatial_dim == 2:
        nx, ny = st.session_state.data.shape[position_offset:]
        xrng = np.arange(0, nx)
        yrng = np.arange(0, ny)
        grid = pv.RectilinearGrid(xrng, yrng)
    elif spatial_dim == 3:
        nx, ny, nz = st.session_state.data.shape[position_offset:]
        xrng = np.arange(0, nx)
        yrng = np.arange(0, ny)
        zrng = np.arange(0, nz)
        grid = pv.RectilinearGrid(xrng, yrng, zrng)
    # Set the grid scalar field
    # If no time step is set the field is assumed to be constant
    field_name = st.session_state.field[0]
    if time_step is None:
        grid[field_name] = st.session_state.data.ravel()
    else:
        grid[field_name] = st.session_state.data[time_step].ravel()

    plotter = pv.Plotter(window_size=[400, 400])
    plotter.add_mesh(grid, scalars=field_name)
    if spatial_dim == 2:
        plotter.view_xy()
    elif spatial_dim == 3:
        plotter.view_isometric()
    plotter.background_color = "white"
    return plotter


st.set_page_config(
    page_title="Tap into the Well", page_icon="assets/the_well_color_icon.svg"
)
st.image("assets/the_well_logo.png")
st.markdown("""
    [The Well](https://arxiv.org/abs/2412.00568) is a collection of 15TB datasets of physics simulations.

    This space allows you to tap into the Well by visualizing different datasets hosted on the [Hugging Face Hub](https://huggingface.co/polymathic-ai).
    - Select a dataset
    - Select a field
    - Select a file
    - Visualize different time steps

    For field corresponding of higher tensor order (e.g. velocity) loading the data may be slow.
    For this reason, we recommend downloading the data to work on the Well.
    Check the [documentation](polymathic-ai.org/the_well) for more information.

""")
# The order of the following widget matters
# Field data is updated whenever a file or a field is selected

# Dataset selection
dataset = st.selectbox(
    "Select a Dataset",
    options=DATASET_NAMES,
    index=None,
    key="name",
    on_change=dataset_info_callback,
)

# File selection
if st.session_state.name:
    field_selector = st.selectbox(
        "Select a field",
        key="field",
        options=st.session_state.field_names,
        format_func=lambda option: option[0],  # Fields are (name, tensor_order)
        on_change=field_callback,
    )
    file_selector = st.selectbox(
        "Select a file",
        options=st.session_state.files,
        key="file",
        index=None,
        format_func=lambda option: pathlib.Path(option).name,
        on_change=field_callback,
    )
    if st.session_state.data is not None:
        # Add a time step slider for dynamic fields
        if st.session_state.data.ndim > 2:
            time_step_slider = st.slider(
                "Time step",
                min_value=0,
                value=0,
                max_value=st.session_state.data.shape[0] - 1,
                key="time_step",
            )

if st.session_state.data is not None:
    plotter = create_plotter()
    stpyvista(plotter)
