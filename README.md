<div align="center">
    <img src="https://raw.githubusercontent.com/PolymathicAI/the_well/master/docs/assets/images/the_well_color.svg" width="60%"/>
</div>

<br>

<div align="center">

![Test Workflow](https://github.com/PolymathicAI/the_well/actions/workflows/tests.yaml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/the_well)](https://pypi.org/project/the-well/)
[![Docs](https://img.shields.io/badge/docs-latest---?color=25005a&labelColor=grey)](https://polymathic-ai.org/the_well/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.00568---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2412.00568)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024---?logo=https%3A%2F%2Fneurips.cc%2Fstatic%2Fcore%2Fimg%2FNeurIPS-logo.svg&labelColor=68448B&color=b3b3b3)](https://openreview.net/forum?id=00Sx577BT3)

</div>

# The Well: 15TB of Physics Simulations


Welcome to the Well, a large-scale collection of machine learning datasets containing numerical simulations of a wide variety of spatiotemporal physical systems. The Well draws from domain scientists and numerical software developers to provide 15TB of data across 16 datasets covering diverse domains such as biological systems, fluid dynamics, acoustic scattering, as well as magneto-hydrodynamic simulations of extra-galactic fluids or supernova explosions. These datasets can be used individually or as part of a broader benchmark suite for accelerating research in machine learning and computational sciences.

## Tap into the Well

Once the Well package installed and the data downloaded you can use them in your training pipeline.

```python
from the_well.data import WellDataset
from torch.utils.data import DataLoader

trainset = WellDataset(
    well_base_path="path/to/base",
    well_dataset_name="name_of_the_dataset",
    well_split_name="train"
)
train_loader = DataLoader(trainset)

for batch in train_loader:
    ...
```

For more information regarding the interface, please refer to the [API](https://github.com/PolymathicAI/the_well/tree/master/docs/api.md) and the [tutorials](https://github.com/PolymathicAI/the_well/blob/master/docs/tutorials/dataset.ipynb).

### Installation

If you plan to use The Well datasets to train or evaluate deep learning models, we recommend to use a machine with enough computing resources.
We also recommend creating a new Python (>=3.10) environment to install the Well. For instance, with [venv](https://docs.python.org/3/library/venv.html):

```
python -m venv path/to/env
source path/to/env/activate/bin
```

#### From PyPI

The Well package can be installed directly from PyPI.

```
pip install the_well
```

#### From Source

It can also be installed from source. For this, clone the [repository](https://github.com/PolymathicAI/the_well) and install the package with its dependencies.

```
git clone https://github.com/PolymathicAI/the_well
cd the_well
pip install .
```

Depending on your acceleration hardware, you can specify `--extra-index-url` to install the relevant PyTorch version. For example, use

```
pip install . --extra-index-url https://download.pytorch.org/whl/cu121
```

to install the dependencies built for CUDA 12.1.

#### Benchmark Dependencies

If you want to run the benchmarks, you should install additional dependencies.

```
pip install the_well[benchmark]
```

### Downloading the Data

The Well datasets range between 6.9GB and 5.1TB of data each, for a total of 15TB for the full collection. Ensure that your system has enough free disk space to accomodate the datasets you wish to download.

Once `the_well` is installed, you can use the `the-well-download` command to download any dataset of The Well.

```
the-well-download --base-path path/to/base --dataset active_matter --split train
```

If `--dataset` and `--split` are omitted, all datasets and splits will be downloaded. This could take a while!

### Streaming from Hugging Face

Most of the Well datasets are also hosted on [Hugging Face](https://huggingface.co/polymathic-ai). Data can be streamed directly from the hub using the following code.

```python
from the_well.data import WellDataset
from torch.utils.data import DataLoader

# The following line may take a couple of minutes to instantiate the datamodule
trainset = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",  # access from HF hub
    well_dataset_name="active_matter",
    well_split_name="train",
)
train_loader = DataLoader(trainset)

for batch in train_loader:
    ...
```

For better performance in large training, we advise [downloading the data locally](#downloading-the-data) instead of streaming it over the network.

## Benchmark

The repository allows benchmarking surrogate models on the different datasets that compose the Well. Some state-of-the-art models are already implemented in [`models`](https://github.com/PolymathicAI/the_well/tree/master/the_well/benchmark/models), while [dataset classes](https://github.com/PolymathicAI/the_well/tree/master/the_well/data) handle the raw data of the Well.
The benchmark relies on [a training script](https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/train.py) that uses [hydra](https://hydra.cc/) to instantiate various classes (e.g. dataset, model, optimizer) from [configuration files](https://github.com/PolymathicAI/the_well/tree/master/the_well/benchmark/configs).

For instance, to run the training script of default FNO architecture on the active matter dataset, launch the following commands:

```bash
cd the_well/benchmark
python train.py experiment=fno server=local data=active_matter
```

Each argument corresponds to a specific configuration file. In the command above `server=local` indicates the training script to use [`local.yaml`](https://github.com/PolymathicAI/the_well/tree/master/the_well/benchmark/configs/server/local.yaml), which just declares the relative path to the data. The configuration can be overridden directly or edited with new YAML files. Please refer to [hydra documentation](https://hydra.cc/) for editing configuration.

You can use this command within a sbatch script to launch the training with Slurm.

## Citation

This project has been led by the <a href="https://polymathic-ai.org/">Polymathic AI</a> organization, in collaboration with researchers from the Flatiron Institute, University of Colorado Boulder, University of Cambridge, New York University, Rutgers University, Cornell University, University of Tokyo, Los Alamos Natioinal Laboratory, University of Califronia, Berkeley, Princeton University, CEA DAM, and University of Liège.

If you find this project useful for your research, please consider citing

```
@inproceedings{ohana2024thewell,
  title={The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning},
  author={Ruben Ohana and Michael McCabe and Lucas Thibaut Meyer and Rudy Morel and Fruzsina Julia Agocs and Miguel Beneitez and Marsha Berger and Blakesley Burkhart and Stuart B. Dalziel and Drummond Buschman Fielding and Daniel Fortunato and Jared A. Goldberg and Keiya Hirashima and Yan-Fei Jiang and Rich Kerswell and Suryanarayana Maddu and Jonah M. Miller and Payel Mukhopadhyay and Stefan S. Nixon and Jeff Shen and Romain Watteaux and Bruno R{\'e}galdo-Saint Blancard and Fran{\c{c}}ois Rozet and Liam Holden Parker and Miles Cranmer and Shirley Ho},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024},
  url={https://openreview.net/forum?id=00Sx577BT3}
}
```

## Contact

For questions regarding this project, please contact [Ruben Ohana](https://rubenohana.github.io/) and [Michael McCabe](https://mikemccabe210.github.io/) at {rohana,mmccabe}@flatironinstitute.org.

## Bug Reports and Feature Requests

To report a bug (in the data or the code), request a feature or simply ask a question, you can [open an issue](https://github.com/PolymathicAI/the_well/issues) on the [repository](https://github.com/PolymathicAI/the_well).
