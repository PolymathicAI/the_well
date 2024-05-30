# from typing import Dict, Tuple

import torch
import torch.nn as nn
from neuralop.models import FNO as neuralop_FNO

from the_well.benchmark.data.datasets import GenericWellMetadata


class FNO(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dset_metadata: GenericWellMetadata,
        modes1: int,
        modes2: int,
        modes3: int = 16,
        hidden_channels: int = 64,
    ):
        super(FNO, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.hidden_channels = hidden_channels
        self.model = None
        self.initialized = False
        self.n_spatial_dims = dset_metadata.n_spatial_dims

        if self.n_spatial_dims == 2:
            self.n_modes = (self.modes1, self.modes2)
        elif self.n_spatial_dims == 3:
            self.n_modes = (self.modes1, self.modes2, self.modes3)

        self.model = neuralop_FNO(
            n_modes=self.n_modes,
            in_channels=self.dim_in,
            out_channels=self.dim_out,
            hidden_channels=self.hidden_channels,
        )

    def forward(self, input) -> torch.Tensor:
        return self.model(input)
