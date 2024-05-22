"""
Copied from https://github.com/bogdanraonic3/ConvolutionalNeuralOperator/blob/main/CNOModule.py
"""

import torch.nn as nn

import torch
#from debug_tools import *
from .training.filtered_networks import LReLu, LReLu_regular, LReLu_torch #Either "filtered LReLU" or regular LReLu
from .debug_tools import format_tensor_size

#------------------------------------------------------------------------------

#Depending on in_size, out_size, the CNOBlock can be:
#   -- (D) Block
#   -- (U) Block
#   -- (I) Block

class CNOBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(CNOBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size
        self.out_size = out_size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.in_cutoff  = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den
        
        self.in_halfwidth =  half_width_mult*self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den
        
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation
        
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels, 
                                           kernel_size=self.conv_kernel, 
                                           padding = pad)
    
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(self.out_channels)
        
        if activation == "cno_lrelu":
            self.activation  = LReLu(in_channels           = self.in_channels, #In _channels is not used in these settings
                                     out_channels          = self.out_channels,                   
                                     in_size               = self.in_size,                       
                                     out_size              = self.out_size,                       
                                     in_sampling_rate      = self.in_size,               
                                     out_sampling_rate     = self.out_size,             
                                     in_cutoff             = self.in_cutoff,                     
                                     out_cutoff            = self.out_cutoff,                  
                                     in_half_width         = self.in_halfwidth,                
                                     out_half_width        = self.out_halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.out_channels, #In _channels is not used in these settings
                                            out_channels          = self.out_channels,                   
                                            in_size               = self.in_size,                       
                                            out_size              = self.out_size,                       
                                            in_sampling_rate      = self.in_size,               
                                            out_sampling_rate     = self.out_size)
        elif activation == "lrelu":
            self.activation  = LReLu_regular(in_channels           = self.in_channels, #In _channels is not used in these settings
                                             out_channels          = self.out_channels,                   
                                             in_size               = self.in_size,                       
                                             out_size              = self.out_size,                       
                                             in_sampling_rate      = self.in_size,               
                                             out_sampling_rate     = self.out_size)
        else:
            raise ValueError("Please specify different activation function")
        
        
    def forward(self, x):
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.activation(x)

#------------------------------------------------------------------------------

# Contains CNOBlock -> Convolution -> BN

class LiftProjectBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 latent_dim = 64,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(LiftProjectBlock, self).__init__()
    
        self.inter_CNOBlock = CNOBlock(in_channels = in_channels,
                                    out_channels = latent_dim,
                                    in_size = in_size,
                                    out_size = out_size,
                                    cutoff_den = cutoff_den,
                                    conv_kernel = conv_kernel,
                                    filter_size = filter_size,
                                    lrelu_upsampling = lrelu_upsampling,
                                    half_width_mult  = half_width_mult,
                                    radial = radial,
                                    batch_norm = batch_norm,
                                    activation = activation)
        
        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = latent_dim, out_channels=out_channels, 
                                           kernel_size=conv_kernel, stride = 1, 
                                           padding = pad)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.inter_CNOBlock(x)
        
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x
        
#------------------------------------------------------------------------------

# Residual Block containts:
    # Convolution -> BN -> Activation -> Convolution -> BN -> SKIP CONNECTION

class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm

        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den        
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den
        
        #-----------------------------------------
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        
        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)
        
        if activation == "cno_lrelu":

            self.activation  = LReLu(in_channels           = self.channels, #In _channels is not used in these settings
                                     out_channels          = self.channels,                   
                                     in_size               = self.size,                       
                                     out_size              = self.size,                       
                                     in_sampling_rate      = self.size,               
                                     out_sampling_rate     = self.size,             
                                     in_cutoff             = self.cutoff,                     
                                     out_cutoff            = self.cutoff,                  
                                     in_half_width         = self.halfwidth,                
                                     out_half_width        = self.halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        elif activation == "lrelu":

            self.activation = LReLu_regular(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        else:
            raise ValueError("Please specify different activation function")
            

    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        
        return x + out
#------------------------------------------------------------------------------

#CNO NETWORK:
class CNO(nn.Module):
    def __init__(self,  
                in_dim,
                out_dim,
                n_spatial_dims,
                 in_size,                   # Input spatial size
                 N_layers,                  # Number of (D) or (U) blocks in the network
                 N_res = 1,                 # Number of (R) blocks per level (except the neck)
                n_param_conditioning=0,
                 N_res_neck = 6,            # Number of (R) blocks in the neck
                 channel_multiplier = 32,   # How the number of channels evolve?
                 conv_kernel=3,             # Size of all the kernels
                 cutoff_den = 2.0001,       # Filter property 1.
                 filter_size=6,             # Filter property 2.
                 lrelu_upsampling = 2,      # Filter property 3.
                 half_width_mult  = 0.8,    # Filter property 4.
                 radial = False,            # Filter property 5. Is filter radial?
                 batch_norm = True,         # Add BN? We do not add BN in lifting/projection layer
                 out_size = 1,              # If out_size is 1, Then out_size = in_size. Else must be int
                 expand_input = False,      # Start with original in_size, or expand it (pad zeros in the spectrum)
                 latent_lift_proj_dim = 64, # Intermediate latent dimension in the lifting/projection layer
                 add_inv = True,            # Add invariant block (I) after the intermediate connections?
                 activation = 'cno_lrelu'   # Activation function can be 'cno_lrelu' or 'lrelu'
                ):
        
        super(CNO, self).__init__()

        ###################### Define the parameters & specifications #################################################

        assert n_spatial_dims == 2, "Only 2D networks are supported"
        # Number od (D) & (U) Blocks
        self.N_layers = int(N_layers)
        
        # Input is lifted to the half on channel_multiplier dimension
        self.lift_dim = channel_multiplier//2         
        self.out_dim  = out_dim
        
        #Should we add invariant layers in the decoder?
        self.add_inv = add_inv
        
        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier        
        
        # Is the filter radial? We always use NOT radial
        if radial ==0:
            self.radial = False
        else:
            self.radial = True
        
        ###################### Define evolution of the number features ################################################

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)
        
        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets
        
        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])
        
        ###################### Define evolution of sampling rates #####################################################
        
        if not expand_input:
            latent_size = in_size # No change in in_size
        else:
            down_exponent = 2 ** N_layers
            latent_size = in_size - (in_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        #Are inputs and outputs of the same size? If not, how should the size of the decoder evolve?
        if out_size == 1:
            latent_size_out = latent_size
        else:
            if not expand_input:
                latent_size_out = out_size # No change in in_size
            else:
                down_exponent = 2 ** N_layers
                latent_size_out = out_size - (out_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))
        
        
        
        ###################### Define Projection & Lift ##############################################################
    
        self.lift = LiftProjectBlock(in_channels  = in_dim,
                                     out_channels = self.encoder_features[0],
                                     in_size      = in_size,
                                     out_size     = self.encoder_sizes[0],
                                     latent_dim   = latent_lift_proj_dim,
                                     cutoff_den   = cutoff_den,
                                     conv_kernel  = conv_kernel,
                                     filter_size  = filter_size,
                                     lrelu_upsampling  = lrelu_upsampling,
                                     half_width_mult   = half_width_mult,
                                     radial            = radial,
                                     batch_norm        = False,
                                     activation = activation)
        _out_size = out_size
        if out_size == 1:
            _out_size = in_size
            
        self.project = LiftProjectBlock(in_channels  = self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels = out_dim,
                                        in_size      = self.decoder_sizes[-1],
                                        out_size     = _out_size,
                                        latent_dim   = latent_lift_proj_dim,
                                        cutoff_den   = cutoff_den,
                                        conv_kernel  = conv_kernel,
                                        filter_size  = filter_size,
                                        lrelu_upsampling  = lrelu_upsampling,
                                        half_width_mult   = half_width_mult,
                                        radial            = radial,
                                        batch_norm        = False,
                                        activation = activation) 

        ###################### Define U & D blocks ###################################################################

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers)])
        
        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers + 1)])
        
        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers)])
        
        
        self.decoder_inv    = nn.ModuleList([(CNOBlock(in_channels  =  self.inv_features[i],
                                                        out_channels = self.inv_features[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers + 1)])
        
        
        ####################### Define ResNets Blocks ################################################################

        # Here, we define ResNet Blocks. 
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks 
        
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder 

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(ResidualBlock(channels = self.encoder_features[l],
                                                   size     = self.encoder_sizes[l],
                                                   cutoff_den = cutoff_den,
                                                   conv_kernel = conv_kernel,
                                                   filter_size = filter_size,
                                                   lrelu_upsampling = lrelu_upsampling,
                                                   half_width_mult  = half_width_mult,
                                                   radial = radial,
                                                   batch_norm = batch_norm,
                                                   activation = activation))
        for i in range(self.N_res_neck):
            self.res_nets.append(ResidualBlock(channels = self.encoder_features[self.N_layers],
                                               size     = self.encoder_sizes[self.N_layers],
                                               cutoff_den = cutoff_den,
                                               conv_kernel = conv_kernel,
                                               filter_size = filter_size,
                                               lrelu_upsampling = lrelu_upsampling,
                                               half_width_mult  = half_width_mult,
                                               radial = radial,
                                               batch_norm = batch_norm,
                                               activation = activation))
        
        self.res_nets = torch.nn.Sequential(*self.res_nets)    


    def forward(self, x):
        x = x['x']
        x = x.permute(0, 4, 2, 3, 1).squeeze(-1)
        #Execute Lift ---------------------------------------------------------
        x = self.lift(x)
        skip = []
        
        # Execute Encoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            #Apply ResNet & save the result
            y = x
            for j in range(self.N_res):
                y = self.res_nets[i*self.N_res + j](y)
            skip.append(y)
            
            # Apply (D) block
            x = self.encoder[i](x)   
        
        #----------------------------------------------------------------------
        
        # Apply the deepest ResNet (bottle neck)
        for j in range(self.N_res_neck):
            x = self.res_nets[-j-1](x)

        # Execute Decoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])),1)
            
            if self.add_inv:
                x = self.decoder_inv[i](x)
            # Apply (U) block
            x = self.decoder[i](x)
        # Cat & Execute Projetion ---------------------------------------------
        
        x = torch.cat((x, self.ED_expansion[0](skip[0])),1)
        x = self.project(x)
        
        del skip
        del y
        
        x = x.unsqueeze(-1).permute(0, 4, 2, 3, 1)
        return x


    def get_n_params(self):
        pp = 0
        
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams