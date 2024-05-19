"""Model and utility functions copied from PDEArena implemenation

https://github.com/pdearena/pdearena
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""

import math
import sys
from abc import abstractmethod
from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    r"""Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def batchmul2d(input, weights, emb):
    temp = input * emb.unsqueeze(1)
    out = torch.einsum("bixy,ioxy->boxy", temp, weights)
    return out


class FreqLinear(torch.nn.Module):
    def __init__(self, in_channel, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = torch.nn.Parameter(
            scale * torch.randn(in_channel, 4 * modes1 * modes2, dtype=torch.float32)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32)
        )

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)


class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        @author: Zongyi Li
        [paper](https://arxiv.org/pdf/2010.08895.pdf)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = torch.nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                2,
                dtype=torch.float32,
            )
        )
        self.weights2 = torch.nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                2,
                dtype=torch.float32,
            )
        )
        self.cond_emb = FreqLinear(cond_channels, self.modes1, self.modes2)

    def forward(self, x, emb):
        emb12 = self.cond_emb(emb)
        emb1 = emb12[..., 0]
        emb2 = emb12[..., 1]
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            torch.view_as_complex(self.weights1),
            emb1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            torch.view_as_complex(self.weights2),
            emb2,
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class ConditionedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` embdding of time or others."""


class EmbedSequential(nn.Sequential, ConditionedBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ConditionedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)

        return x


class FourierBlock(ConditionedBlock):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        stride: int = 1,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        if activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        assert not norm
        self.fourier1 = SpectralConv2d(
            in_planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2
        )
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True
        )
        self.fourier2 = SpectralConv2d(
            planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2
        )
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True
        )
        self.cond_emb = torch.nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x1 = self.fourier1(x, emb)
        x2 = self.conv1(x)
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(x2.shape):
            emb_out = emb_out[..., None]

        out = self.activation(x1 + x2 + emb_out)
        x1 = self.fourier2(out, emb)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block: nn.Module,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = False,
        diffmode: bool = False,
        usegrid: bool = False,
        param_conditioning: Optional[str] = None,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        self.param_conditioning = param_conditioning
        insize = time_history * (
            self.n_input_scalar_components + self.n_input_vector_components * 2
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.param_conditioning is not None:
            if self.param_conditioning == "scalar":
                self.pde_emb = nn.Sequential(
                    nn.Linear(hidden_channels, time_embed_dim),
                    self.activation,
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            else:
                raise NotImplementedError(
                    f"Param conditioning {self.param_conditioning} not implemented"
                )

        if self.usegrid:
            insize += 2
        self.conv_in1 = nn.Conv2d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            time_future
            * (self.n_output_scalar_components + self.n_output_vector_components * 2),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    time_embed_dim,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        cond_channels: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    cond_channels=cond_channels,
                    stride=stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return EmbedSequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor, time, z=None) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = self.time_embed(fourier_embedding(time, self.in_planes))
        if z is not None:
            if self.param_conditioning == "scalar":
                emb = emb + self.pde_emb(fourier_embedding(z, self.in_planes))
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x, emb)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
            # x = x + prev[:, -1:, ...].detach()
        return x.reshape(
            orig_shape[0],
            -1,
            (self.n_output_scalar_components + self.n_output_vector_components * 2),
            *orig_shape[3:],
        )


# From https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
def partialclass(name, cls, *args, **kwds):
    new_cls = type(
        name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwds)}
    )

    # The following is copied nearly ad verbatim from `namedtuple's` source.

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).

    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass

    return new_cls
