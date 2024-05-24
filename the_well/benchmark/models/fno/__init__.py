from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from neuralop.models import FNO as neuralop_FNO


class FNO(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
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
        self.n_spatial_dims = n_spatial_dims

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

    # def preprocess_input(
    #     self,
    #     input: Dict[str, torch.Tensor],
    # ) -> Tuple[torch.Tensor, torch.Size, torch.Tensor, torch.Tensor]:
    #     """Retrieve input fields, time and parameters from passed input."""
    #     time = input["time"].view(-1)
    #     param = input["parameters"]
    #     x = input["x"]
    #     assert x.dim() == 5
    #     original_shape = x.shape
    #     x = rearrange(x, "B T W H C -> B (T C) W H")
    #     return x, original_shape, time, param

    def forward(self, input) -> torch.Tensor:
        # x, original_shape, time, z = self.preprocess_input(input)
        return self.model(input)
        # return x #x.reshape(*original_shape)
