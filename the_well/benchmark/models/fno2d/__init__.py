from re import S
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neuralop.models import FNO2d

class FNO2D(nn.Module):
    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int = 1,
        time_future: int = 1,
        modes1: int,
        modes2: int,
        hidden_channels: int = 64,
        n_param_conditioning: int = 1,

    ):
        super(FNO2D, self).__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.modes1 = modes1
        self.modes2 = modes2
        self.hidden_channels = hidden_channels
        self.model = None
        self.initialized = False

        self.insize = time_history * (
            self.n_input_scalar_components + self.n_input_vector_components * 2
        )
        self.outsize = time_future
            * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        self.model = FNO2d(n_modes_height=self.modes1, n_modes_width=self.modes2, in_channels=self.insize, out_channels = self.outsize, hidden_channels=self.hidden_channels)
        

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
        in_channels = x.shape[1]
        # if not self.initialized:
        #     self.model = FNO2d(n_modes_height=self.modes1, n_modes_width=self.modes2, in_channels=in_channels, out_channels = in_channels, hidden_channels=self.hidden_channels)
        #     self.initialized = True
        return x, original_shape, time, param
    
    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, original_shape, time, z = self.preprocess_input(input)
        x = self.model(x)

        return x.reshape(*original_shape)



# class FNO(ResNet):
#     """Reimplementation of the base class from PDEArena to include multi-parameters inputs.

#     Original code: https://github.com/pdearena/pdearena/blob/main/pdearena/modules/conditioned/twod_resnet.py
#     """

#     def __init__(
#         self,
#         n_input_scalar_components: int,
#         n_input_vector_components: int,
#         n_output_scalar_components: int,
#         n_output_vector_components: int,
#         num_blocks: list,
#         modes1: int = 16,
#         modes2: int = 16,
#         hidden_channels: int = 64,
#         activation: str = "gelu",
#         n_param_conditioning: int = 1,
#     ):
#         block = partialclass(
#             "FourierBasicBlock", FourierBlock, modes1=modes1, modes2=modes2
#         )
#         super().__init__(
#             n_input_scalar_components,
#             n_input_vector_components,
#             n_output_scalar_components,
#             n_output_vector_components,
#             block=block,
#             num_blocks=num_blocks,
#             time_history=1,
#             time_future=1,
#             hidden_channels=hidden_channels,
#             activation=activation,
#             norm=False,
#             diffmode=False,
#             usegrid=False,
#             param_conditioning="scalar",
#         )
#         self.n_param_conditioning = n_param_conditioning
#         time_embed_dim = hidden_channels * 4
#         self.pde_emb = torch.nn.ModuleList(
#             [
#                 torch.nn.Sequential(
#                     torch.nn.Linear(hidden_channels, time_embed_dim),
#                     self.activation,
#                     torch.nn.Linear(time_embed_dim, time_embed_dim),
#                 )
#                 for _ in range(self.n_param_conditioning)
#             ]
#         )

#     def preprocess_input(
#         self,
#         input: Dict[str, torch.Tensor],
#     ) -> Tuple[torch.Tensor, torch.Size, torch.Tensor, torch.Tensor]:
#         """Retrieve input fields, time and parameters from passed input."""
#         time = input["time"].view(-1)
#         param = input["parameters"]
#         x = input["x"]
#         assert x.dim() == 5
#         original_shape = x.shape
#         x = rearrange(x, "B T W H C -> B (T C) W H")
#         return x, original_shape, time, param

#     def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
#         x, original_shape, time, z = self.preprocess_input(input)

#         emb = self.time_embed(fourier_embedding(time, self.in_planes))
#         if z is not None:
#             if self.param_conditioning == "scalar":
#                 assert z.dim() == 2
#                 assert (
#                     z.shape[-1] == self.n_param_conditioning
#                 ), f"Expected Nx{self.n_param_conditioning} parameters but received {z.shape}"
#                 for p_dim in range(z.shape[-1]):
#                     p_embedding = self.pde_emb[p_dim](
#                         fourier_embedding(z[:, p_dim], self.in_planes)
#                     )
#                     emb = emb + p_embedding
#         x = self.activation(self.conv_in1(x.float()))
#         x = self.activation(self.conv_in2(x.float()))

#         if self.padding > 0:
#             x = F.pad(x, [0, self.padding, 0, self.padding])
#         for layer in self.layers:
#             x = layer(x, emb)

#         if self.padding > 0:
#             x = x[..., : -self.padding, : -self.padding]

#         x = self.activation(self.conv_out1(x))
#         x = self.conv_out2(x)

#         if self.diffmode:
#             raise NotImplementedError("diffmode")
#         return x.reshape(*original_shape)
