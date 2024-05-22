from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from neuralop.models import TFNO as neuralop_TFNO


class TFNO(nn.Module):
    def __init__(
        self,
        n_spatial_dim: int,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        modes1: int,
        modes2: int,
        modes3: int = 16,
        hidden_channels: int = 64,
        n_param_conditioning: int = 1,
        time_history: int = 1,
        time_future: int = 1,
    ):
        super(TFNO, self).__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.hidden_channels = hidden_channels
        self.model = None
        self.initialized = False
        self.n_spatial_dim = n_spatial_dim

        if self.n_spatial_dim == 2:
            self.n_modes = (self.modes1, self.modes2)
        elif self.n_spatial_dim == 3:
            self.n_modes = (self.modes1, self.modes2, self.modes3)

        self.in_channels = time_history * (
            self.n_input_scalar_components
            + (self.n_input_vector_components * self.n_spatial_dim)
        )
        self.out_channels = time_future * (
            self.n_output_scalar_components
            + (self.n_output_vector_components * self.n_spatial_dim)
        )

        self.model = neuralop_TFNO(
            n_modes=self.n_modes,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
        )

    def preprocess_input(
        self,
        input: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Size, torch.Tensor, torch.Tensor]:
        """Retrieve input fields, time and parameters from passed input."""
        time = input["time"].view(-1)
        param = input["parameters"]
        x = input["x"]
        assert x.dim() == 5
        original_shape = x.shape
        x = rearrange(x, "B T W H C -> B (T C) W H")
        return x, original_shape, time, param

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, original_shape, time, z = self.preprocess_input(input)
        x = self.model(x)
        return x.reshape(*original_shape)
