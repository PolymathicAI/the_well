import torch
import numpy as np
import torch.nn as nn


from ..common import SN_MLP, SigmaNormLinear

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ComplexLN(nn.Module):
    def __init__(self, tokens, channels, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.view_as_real(torch.ones(1, tokens, channels, dtype=torch.cfloat)))
        self.bias = nn.Parameter(torch.view_as_real(torch.zeros(1, tokens, channels, dtype=torch.cfloat)))

    def forward(self, x):
        if self.training:
            mean, std = x.mean((0, 1), keepdims=True), x.std((0, 1), keepdims=True)
            with torch.no_grad():
                self.mean.copy_((1 - self.momentum) * self.mean + self.momentum * torch.view_as_real(mean))
                self.std.copy_((1 - self.momentum) * self.std + self.momentum * std)
        else:
            mean, std = torch.view_as_complex(self.mean), self.std
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        x = (x - mean) / (self.eps + std)
        x = x * torch.view_as_complex(self.mags) + torch.view_as_complex(self.bias)
        return x
    
class ComplexLinearDDP(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        temp = nn.Linear(in_features, out_features, bias=bias, dtype=torch.complex)
        self.weights = nn.Parameter(torch.view_as_real(temp.weight))
        self.bias = nn.Parameter(torch.view_as_real(temp.bias))

    def forward(self, input):
        return F.linear(input, torch.view_as_complex(self.weights), torch.view_as_complex(self.bias))
    
class ModGELU(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(.02 * torch.randn(channels))

    def forward(self, x):
        return torch.polar(F.gelu(torch.abs(x) + self.b), x.angle())
    
class DSConvSpectralNd(nn.Module):
    def __init__(self, hidden_dim, resolution, ratio=1.):
        super(DSConvSpectralNd, self).__init__()

        self.filter_generator = nn.Sequential(
                                ComplexLN(hidden_dim, tokens),
                                ComplexLinearDDP(hidden_dim, hidden_dim),
                                 ModGELU(hidden_dim),
                                 ComplexLinearDDP(hidden_dim, hidden_dim)
                                 )
        self.channel_mix = SigmaNormLinear(hidden_dim, hidden_dim)
        self.resolution = resolution 

        delta = 1 / (n + 1) / 2
        k = torch.fft.rfftfreq(m) * m
        rel_circ = torch.fft.fftfreq(n) * n
        waves = (k[None, :] ** 2 + rel_circ[:, None] ** 2) ** .5

        # inds = torch.searchsorted(k, waves.flatten()).reshape(n, m//2+1)
        useable_inds = waves <= ((m // 2) * ratio)
        used_inds = torch.masked_select(waves, useable_inds)  # This is a dummy - just using size
        self.register_buffer('useable_inds', useable_inds)
        conv = torch.view_as_real(torch.randn(1, in_channels, used_inds.shape[0], dtype=torch.cfloat))
        self.mags = nn.Parameter(conv[..., 0])
        self.phases = nn.Parameter(conv[..., 1])
        self.spectral_rescale = SpectralRescale(used_inds.shape[0], in_channels)

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        x_linear = self.mlp(self.spectral_rescale(x_masked))
        conv = complex_glu(x_linear, self.mags, self.phases)
        x_masked = x_masked * conv

        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
        x = torch.fft.irfft2(x_ft, norm='ortho')
        return self.channel_mix(x)

class ReFNOBlock(nn.Module):
    def __init__(self, dim, grid_size, ratio=1.):
        super(ReFNOBlock, self).__init__()
        self.mlp = SN_MLP(dim)
        self.spectral_conv = DSConvSpectralND(dim, dim, grid_size, 1, 1, ratio=ratio)

    def forward(self, x):
        return self.spectral_conv(self.mlp(x))

class ReFNN2d(nn.Module):
    def __init__(self, 
                 dim_in: int,
                 dim_out: int,
                 hidden_dim:int =128,
                 blocks: int=4,
                 resolution: tuple=(64, 64),
                 ratio: float=1.):
        super(ReFNN2d, self).__init__()
        """

        """
        self.encoder = SigmaNormLinear(dim_in, hidden_dim) 
        self.pos_embedding = nn.Parameter(torch.randn((1, hidden_dim)+resolution) * .02)
        self.processor_blocks = nn.ModuleList([ReFNOBlock(hidden_dim, resolution, ratio) for _ in range(blocks)])
        self.decoder = SigmaNormLinear(hidden_dim, dim_out)


    def forward(self, x, *args, **kwargs):
        '''
        (b,c,h,w) -> (b,1,h,w)
        '''
        x = self.encoder(x) * self.embed_ratio ** .5  # project
        x = x + self.pos_embedding
        for process in self.processor_blocks:
            x = x + process(x)

        x = self.decoder(x) 
        return x