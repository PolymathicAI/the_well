import torch
import numpy as np
import torch.nn as nn


from ..common import SN_MLP

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DSConvSpectralND(nn.Module):
    def __init__(self, dim, resolution, ratio=1.):
        super(DSConvSpectralND, self).__init__()
        self.dim = dim
        self.resolution = resolution
        self.ratio = ratio
        self.weights = nn.Parameter(torch.randn(dim, dim, 2) * self.scale)

class ReFNOBlock(nn.Module):
    def __init__(self, dim, grid_size, ratio=1.):
        super(ReFNOBlock, self).__init__()
        self.mlp = SN_MLP(dim)
        self.spectral_conv = DSConvSpectral2d(dim, dim, grid_size, 1, 1, ratio=ratio)

    def forward(self, x):
        return self.spectral_conv(self.mlp(x))

class ReFNN2d(nn.Module):
    def __init__(self, dim=128,
                 in_dim=2,
                 layers=4,
                 grid_size=(64, 64),
                 nonlinear=False,
                 ratio=1.):
        super(ReFNN2d, self).__init__()
        """

        """
        conv = DSConvSpectral2d 
        self.encoder = spectral_norm(nn.Conv2d(in_dim, dim, 1))
        self.spectral_blocks = nn.ModuleList([conv(dim, dim, grid_size, 1, 1, ratio=ratio) for _ in range(layers)])
        self.mlp_blocks = nn.ModuleList([nn.Sequential(spectral_norm(nn.Conv2d(dim, 4*dim, 1)),
                                                       nn.SiLU(),
                                                       spectral_norm(nn.Conv2d(4*dim, dim, 1))) for _ in range(layers)])
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(0.)) for _ in range(layers)])
        self.final_gate = nn.Parameter(torch.tensor(0.))
        self.decoder = spectral_norm(nn.Conv2d(dim, in_dim, 1))
        self.embed_ratio = dim/in_dim
        self.pos_embedding = nn.Parameter(torch.randn((1, dim)+grid_size) * .02)


    def forward(self, x, *args, **kwargs):
        '''
        (b,c,h,w) -> (b,1,h,w)
        '''
        x = self.encoder(x) * self.embed_ratio ** .5  # project
        x = x + self.pos_embedding
        for i, (speconv, mlp, gate) in enumerate(zip(self.spectral_blocks, self.mlp_blocks, self.gates)):
            x1 = mlp(x)
            x1 = speconv(x1)
            x = x1 + x

        x = self.decoder(x) / self.embed_ratio ** .5
        return x