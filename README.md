![The Well logo](assets/the_well_logo.png)
# Tap into the Well

The Well is a large-scale collection of machine learning datasets containing numerical simulations of a wide variety of spatiotemporal physical systems. The Well draws from domain scientists and numerical software developers to provide 15TB of data across 16 datasets covering diverse domains such as biological systems, fluid dynamics, acoustic scattering, as well as magneto-hydrodynamic simulations of extra-galactic fluids or supernova explosions. These datasets can be used individually or as part of a broader benchmark suite.

## Getting Started

This repository contains the code to **download the datasets** that compose the Well and to **run a benchmark** on each dataset. 

### Installation

The package is not yet distributed on PyPi. To install the Well, you must first clone the repository. 

```bash
git clone git@github.com:PolymathicAI/the_well.git
cd the_well
```

Then you must install the package manually. If you are interested only in downloading the data, you can install a version of the package that has limited dependencies.

```bash
pip install .
```

Otherwise, if you want to run the benchmark, you must install the full package with extended dependencies.

```bash
pip install ".[benchmark]"
```

### Data Download

Once installed, you can use the `the-well-download` executable to download any dataset of the Well. By default, the dataset will be downloaded in the corresponding subfolder of`datasets` as provided with this repository. However, it is possible to override the download location with the `--output_dir` option.

For instance, run the following to download only samples of the `active_matter` dataset.

```bash
the-well-download --dataset active_matter --sample-only
```

### Benchmark

The repository allows benchmarking surrogate models on the different datasets that compose the Well. Some state-of-the-art models are already implemented in [`models`](the_well/benchmark/models/), while [dataset classes](the_well/benchmark/data/) handle the raw data of the Well. 
The benchmark relies on [a training script](the_well/benchmark/train.py) that uses [hydra](https://hydra.cc/) to instantiate various instances (e.g. dataset, model, optimizer) from [configuration files](the_well/benchmark/configs/).

For instance, to run the training script of default FNO architecture on the active matter dataset, launch the following:

```bash
python train.py experiment=fno server=local data=active_matter 
```

Each argument corresponds to a specific configuration file. In the command above `server=local` indicates the training script to use [`local.yaml`](the_well/benchmark/configs/server/local.yaml), which just declares the relative path to the data. The configuration can be overridden directly or edited with new YAML files. Please refer to [hydra documentation](https://hydra.cc/) for editing configuration.

You can use this command within a sbatch script to launch the training with Slurm.

## How To Contribute

Contributions are welcome. To contribute please open a dedicated issue describing the problem you are facing or the feature you would like to add.

### Code Style

The code should follow the standard [PEP8 Style Guide for Python Code](https://peps.python.org/pep-0008/). This is enforced by using the [ruff](https://docs.astral.sh/ruff/) code linter and formatter. 

To help with maintaining compliance with the format, we provide a `pre-commit` hook that automatically runs `ruff` on the modified code. To install and learn more about `pre-commit`, please refer to [the official documentation](https://pre-commit.com/).
