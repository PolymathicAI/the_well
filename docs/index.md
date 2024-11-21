# The Well: a collection of 15TB datasets of physics numerical simulations

<figure class="video_container">
  <video allowfullscreen="true" autoplay loop>
    <source src="/assets/videos/background.mp4" type="video/mp4">
  </video>
</figure>

## Dip into the Well

``` py
from the_well.benchmark.data import WellDataModule

# The following line may take a couple of minutes to instantiate the datamodule
datamodule = WellDataModule(
    "hf://datasets/polymathic-ai/",  # Access data from HF hub
    "active_matter",  # Dataset name
)
train_dataloader = datamodule.train_dataloader()

for batch in dataloader:
    # Process training batch
    ...
```

## Introduction

The Well is a large-scale collection of machine learning datasets containing numerical simulations of a wide variety of spatiotemporal physical systems. The Well draws from domain scientists and numerical software developers to provide 15TB of data across 16 datasets covering diverse domains such as biological systems, fluid dynamics, acoustic scattering, as well as magneto-hydrodynamic simulations of extra-galactic fluids or supernova explosions. These datasets can be used individually or as part of a broader benchmark suite.
