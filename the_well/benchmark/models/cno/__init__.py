

import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# CNO LReLu activation fucntion
# CNO building block (CNOBlock) → Conv2d - BatchNorm - Activation
# Lift/Project Block (Important for embeddings)
# Residual Block → Conv2d - BatchNorm - Activation - Conv2d - BatchNorm - Skip Connection
# ResNet → Stacked ResidualBlocks (several blocks applied iteratively)


#---------------------
# Activation Function:
#---------------------

class CNO_LReLu(nn.Module):
    def __init__(self,
                 in_factor=2,
                 out_factor=1.,
                n_spatial_dims = 2
                ):
        super(CNO_LReLu, self).__init__()
        self.in_factor = in_factor
        self.out_factor = out_factor
        self.n_spatial_dims = n_spatial_dims
        if n_spatial_dims == 2:
            self.interp_type = 'bilinear'
        elif n_spatial_dims == 3:
            self.interp_type = 'trilinear'
        self.act = nn.LeakyReLU()

    def forward(self, x, out_size_override = None):
        # print('its here')
        # return checkpoint(self.inner_forward, x, out_size_override)
        return self.inner_forward(x, out_size_override)

    def inner_forward(self, x, out_size_override = None):
        # print('pre_interpolate', x.shape)
        # x = F.interpolate(x, scale_factor=self.in_factor, mode = self.interp_type, antialias = False)
        x = self.act(x)
        # print('post_act', x.shape)
        # if out_size_override is not None:
        #     x = F.interpolate(x, size = out_size_override, antialias = False)
        # else:
        #     x = F.interpolate(x, scale_factor=1/self.in_factor * self.out_factor, antialias = False)
        # print('post_interpolate', x.shape)
        return x

#--------------------
# CNO Block:
#--------------------

class CNOBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                out_factor=1.,
                use_bn = True,
                conv = nn.Conv2d,
                bn_module = nn.BatchNorm2d,
                n_spatial_dims=2
                ):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = conv(in_channels = self.in_channels,
                                            out_channels= self.out_channels,
                                            kernel_size = 3,
                                            padding     = 1)

        if use_bn:
            self.batch_norm  = bn_module(self.out_channels)
        else:
            self.batch_norm  = nn.Identity()
        self.act           = CNO_LReLu(n_spatial_dims=n_spatial_dims, out_factor=out_factor)

    def forward(self, x, out_size_override=None):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x, out_size_override)
    
#--------------------
# Lift/Project Block:
#--------------------

class LiftProjectBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                size,
                latent_dim = 64,
                conv=nn.Conv2d,
                bn_module = nn.BatchNorm2d,
                n_spatial_dims = 2
                ):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_dim,
                                        use_bn           = False,
                                        conv             = conv,
                                        bn_module        = bn_module,
                                        n_spatial_dims   = n_spatial_dims)

        self.convolution = conv(in_channels  = latent_dim,
                                            out_channels = out_channels,
                                            kernel_size  = 3,
                                            padding      = 1)


    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x

#--------------------
# Residual Block:
#--------------------

class ResidualBlock(nn.Module):
    def __init__(self,
                channels,
                size,
                use_bn = True,
                conv = nn.Conv2d,
                bn_module = nn.BatchNorm2d,
                n_spatial_dims = 2
                ):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size     = size

        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation

        self.convolution1 = conv(in_channels = self.channels,
                                            out_channels= self.channels,
                                            kernel_size = 3,
                                            padding     = 1)
        self.convolution2 = conv(in_channels = self.channels,
                                            out_channels= self.channels,
                                            kernel_size = 3,
                                            padding     = 1)

        if use_bn:
            self.batch_norm1  = bn_module(self.channels)
            self.batch_norm2  = bn_module(self.channels)

        else:
            self.batch_norm1  = nn.Identity()
            self.batch_norm2  = nn.Identity()

        self.act           = CNO_LReLu(n_spatial_dims=n_spatial_dims,
                                       )


    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out

#--------------------
# ResNet:
#--------------------

class ResNet(nn.Module):
    def __init__(self,
                channels,
                size,
                num_blocks,
                use_bn = True,
                conv = nn.Conv2d,
                bn_module = nn.BatchNorm2d,
                n_spatial_dims = 2
                ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels = channels,
                                                size = size,
                                                use_bn = use_bn,
                                                conv=conv,
                                                bn_module=bn_module,
                                                n_spatial_dims=n_spatial_dims))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        # for i in range(self.num_blocks):
        x = self.res_nets[0](x)
        return x
    
    
#--------------------
# CNO:
#--------------------

class CNO(nn.Module):
    def __init__(self,
                in_dim,                    # Number of input channels.
                out_dim,                   # Number of input channels.
                n_spatial_dims,            # Number of spatial dimensions.
                in_size,                      # Input and Output spatial size (required )
                N_layers,                  # Number of (D) or (U) blocks in the network
                N_res = 1,                 # Number of (R) blocks per level (except the neck)
                N_res_neck = 6,            # Number of (R) blocks in the neck
                channel_multiplier = 32,   # How the number of channels evolve?
                use_bn = True,             # Add BN? We do not add BN in lifting/projection layer
                ):

        super(CNO, self).__init__()

        if n_spatial_dims == 2:
            conv = nn.Conv2d
            bn_module = nn.BatchNorm2d
            self.interpolate_mode = 'bicubic'
        elif n_spatial_dims == 3:
            conv = nn.Conv3d
            bn_module = nn.BatchNorm3d
            self.interpolate_mode = 'trilinear'
        self.n_spatial_dims = n_spatial_dims

        self.N_layers = int(N_layers)         # Number od (D) & (U) Blocks
        self.lift_dim = channel_multiplier//2 # Input is lifted to the half of channel_multiplier dimension
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.channel_multiplier = channel_multiplier  # The growth of the channels

        ######## Num of channels/features - evolution ########

        self.encoder_features = [self.lift_dim] # How the features in Encoder evolve (number of features)
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:] # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets (we must multiply by 2 then)

        ######## Spatial sizes of channels - evolution ########

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(in_size // 2 ** i)
            self.decoder_sizes.append(in_size // 2 ** (self.N_layers - i))


        ######## Define Lift and Project blocks ########

        self.lift   = LiftProjectBlock(in_channels = in_dim,
                                        out_channels = self.encoder_features[0],
                                        size = in_size,
                                        conv=conv,
                                        bn_module=bn_module,
                                        n_spatial_dims=n_spatial_dims)

        self.project   = LiftProjectBlock(in_channels = self.encoder_features[0] + self.decoder_features_out[-1],
                                            out_channels = out_dim,
                                            size = in_size,
                                            conv=conv,
                                            bn_module=bn_module,
                                            n_spatial_dims=n_spatial_dims)

        ######## Define Encoder, ED Linker and Decoder networks ########

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        out_factor      = self.encoder_sizes[i],
                                                        use_bn       = use_bn,
                                                        conv=conv,
                                                        bn_module=bn_module,
                                                        n_spatial_dims=n_spatial_dims))
                                                for i in range(self.N_layers)])

        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        # print('ED FACTORS!!!!')
        # for i in range(self.N_layers + 1):
        #     print(f'ED FACTOR {i}: {self.encoder_sizes[i]} {self.decoder_sizes[self.N_layers - i]}')
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        # in_size      = self.encoder_sizes[i],
                                                        # out_size     = self.decoder_sizes[self.N_layers - i],
                                                        use_bn       = use_bn,
                                                        conv=conv,
                                                        bn_module=bn_module,
                                                        n_spatial_dims=n_spatial_dims))
                                                for i in range(self.N_layers + 1)])

        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        out_factor   = 2,
                                                        use_bn       = use_bn,
                                                        conv=conv,
                                                        bn_module=bn_module,
                                                        n_spatial_dims=n_spatial_dims))
                                                for i in range(self.N_layers)])
        self.dummy_parm = nn.Parameter(torch.ones(1))
        #### Define ResNets Blocks 

        # Here, we define ResNet Blocks.

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet networks (before the neck)
        for l in range(self.N_layers):
            self.res_nets.append(ResNet(channels = self.encoder_features[l],
                                        size = self.encoder_sizes[l],
                                        num_blocks = self.N_res,
                                        use_bn = use_bn,
                                        conv=conv,
                                        bn_module=bn_module,
                                        n_spatial_dims=n_spatial_dims))

        self.res_net_neck = ResNet(channels = self.encoder_features[self.N_layers],
                                    size = self.encoder_sizes[self.N_layers],
                                    num_blocks = self.N_res_neck,
                                    use_bn = use_bn,
                                    conv = conv,
                                    bn_module = bn_module,
                                    n_spatial_dims = n_spatial_dims)

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        x = x['x']
        inn = x.clone()
        x = x.permute(0, 4, 2, 3, 1).squeeze(-1)
        x = self.lift(x) #Execute Lift
        skip = []
       
        # Execute Encoder
        for i in range(self.N_layers):

        #     #Apply ResNet & save the result
            # y = self.res_nets[i](x)
            y = x.clone()
            skip.append(y)

        #     # Apply (D) block
            x = self.encoder[i](x)
        
        # Apply the deepest ResNet (bottle neck)
        # x = self.res_net_neck(x)

        # Execute Decode
        for i in range(self.N_layers):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])),1)

            # Apply (U) block
            x = self.decoder[i](x)

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0])),1)
        x = self.project(x)
        # del skip
        # del y
        
        
        x = x.unsqueeze(-1).permute(0, 4, 2, 3, 1)
        return x #self.dummy_parm * inn #x