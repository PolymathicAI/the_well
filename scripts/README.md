The folder contains a collection of scripts whose goals are described below

- `generate_metadata.py` goes through the different datasets that compose the Well to generate YAML files containing the metadata of the corresponding dataset.

- `compute_statistics.py` goes through the training set of the different datasets that compose the Well to compute their statistics in terms of mean and std for each tensor field.

- `check_thewell_data.py` checks for all the HDF5 files that compose the dataset if:
- the name of boundary conditions is consistent;
- the different tensor fields contain NAN values;
- the different tensor fields contain constant frames;
- the different tensor fields contain outliers compared to the mean and std computed on the fly. A default value of $5\sigma$ serves as threshold for characterizing outliers.

- `check_thewell_formatting.py` checks that a HDF5 file follows the format expected by the Well.

- `plot_velocity.py` for each dataset, plots the velocity field, if it exists, at four times of the first validation file $0$, $T/3$, $2T/3$ and $T$. Where $T$ is the original length of the simulation.

- `create_gif.py` creates a gif from a time series input. The script must be edited to first load the time series.

- `huggingface` folder contains a script to upload data to the Hugging Face hub (`upload.py`). It also contains a standalone application for visualizing some data of the Well that are hosted on the Hugging Face hub.
