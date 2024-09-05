The folder contains a collection of scripts whose goals are described below

# `generate_metadata.py`

Go through the different datasets that compose the Well to generate YAML files containing the metadata of the corresponding dataset.

# `compute_statistics.py`

Go through the different datasets that compose the Well to compute their statistics in terms of mean and std for each tensor field.

# `check_thewell_data.py`

Check for all the HDF5 files that compose the dataset if:
- the name of boundary conditions is consistant;
- the different tensor fields contain NAN values;
- the different tensor fields contain constant frames;
- the different tensor fields contain outliers compared to the mean and std computed on the fly. A default value of $5\sigma$ serves as threshold for characterizing outliers.

# `check_thewell_formatting.py`

Check that a HDF5 file follows the format expected by the Well.

# `plot_velocity.py`

For each dataset, plot the velocity field, if it exists, at four times of the first validation $0$, $T/3$, $2T/3$ and $T$. Where $T$ is the original length of the simulation.

# `create_gif.py`

Create a gif from a time series input. The script must be edited to first load the time series.

