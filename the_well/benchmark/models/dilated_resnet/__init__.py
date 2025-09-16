"""
From Stachenfeld et al, 2022 - LEARNED COARSE MODELS FOR
EFFICIENT TURBULENCE SIMULATION (https://openreview.net/forum?id=msRBojTz-Nh)

Thanks to Kim Stachenfeld for providing the TF version of the code.
"""

import torch.nn as nn

from the_well.benchmark.models.common import BaseModel


class DilatedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_spatial_dims,
        kernel_size,
        num_levels,
        padding_type="zero",
    ):
        super(DilatedBlock, self).__init__()
        if n_spatial_dims == 1:
            conv = nn.Conv1d
        elif n_spatial_dims == 2:
            conv = nn.Conv2d
        elif n_spatial_dims == 3:
            conv = nn.Conv3d
        self.act = nn.GELU()
        conv_layers = []
        dilation_rates = [2**i for i in range(num_levels)]
        dilation_rates = dilation_rates + dilation_rates[:-1][::-1]
        self.padding_type = padding_type
        self.pad_in_func = []
        for rate in dilation_rates:
            if self.padding_type == "zero":
                padding = rate
            else:
                padding = 0
                # TODO implement variable padding types
            conv_layers.append(
                conv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=rate,
                    padding=padding,
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = self.act(layer(x))
        return x


class DilatedResNet(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        kernel_size: int = 3,
        blocks: int = 4,
        levels_per_block: int = 4,
        hidden_dim: int = 32,
        padding_type: str = "zero",
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.levels_per_block = levels_per_block
        self.hidden_dim = hidden_dim
        self.padding_type = padding_type

        if self.n_spatial_dims == 1:
            conv = nn.Conv1d
        elif self.n_spatial_dims == 2:
            conv = nn.Conv2d
        elif self.n_spatial_dims == 3:
            conv = nn.Conv3d

        if self.padding_type == "zero":
            self.input_conv = conv(
                dim_in, hidden_dim, kernel_size=kernel_size, padding=1
            )
        else:
            self.input_conv = conv(
                dim_in, hidden_dim, kernel_size=kernel_size, padding=0
            )

        self.processors = nn.ModuleList(
            [
                DilatedBlock(
                    hidden_dim,
                    hidden_dim,
                    self.n_spatial_dims,
                    kernel_size,
                    levels_per_block,
                    padding_type,
                )
                for _ in range(blocks)
            ]
        )

        if self.padding_type == "zero":
            self.output_conv = conv(
                hidden_dim, dim_out, kernel_size=kernel_size, padding=1
            )
        else:
            self.output_conv = conv(
                hidden_dim, dim_out, kernel_size=kernel_size, padding=0
            )

    def forward(self, x):
        x = self.input_conv(x)
        for processor in self.processors:
            x = x + processor(x)
        x = self.output_conv(x)
        return x
