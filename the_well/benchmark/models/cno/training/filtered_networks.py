# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Copyright (c) 2021, NVIDIA Corporation & affiliates. All rights reserved.

NVIDIA Source Code License for StyleGAN3

=======================================================================

1. Definitions

"Licensor" means any person or entity that distributes its Work.

"Software" means the original work of authorship made available under
this License.

"Work" means the Software and any additions to or derivative works of
the Software that are made available under this License.

The terms "reproduce," "reproduction," "derivative works," and
"distribution" have the meaning as provided under U.S. copyright law;
provided, however, that for the purposes of this License, derivative
works shall not include works that remain separable from, or merely
link (or bind by name) to the interfaces of, the Work.

Works, including the Software, are "made available" under this License
by including in or with the Work either (a) a copyright notice
referencing the applicability of this License to the Work, or (b) a
copy of this License.

2. License Grants

    2.1 Copyright Grant. Subject to the terms and conditions of this
    License, each Licensor grants to you a perpetual, worldwide,
    non-exclusive, royalty-free, copyright license to reproduce,
    prepare derivative works of, publicly display, publicly perform,
    sublicense and distribute its Work and any resulting derivative
    works in any form.

3. Limitations

    3.1 Redistribution. You may reproduce or distribute the Work only
    if (a) you do so under this License, (b) you include a complete
    copy of this License with your distribution, and (c) you retain
    without modification any copyright, patent, trademark, or
    attribution notices that are present in the Work.

    3.2 Derivative Works. You may specify that additional or different
    terms apply to the use, reproduction, and distribution of your
    derivative works of the Work ("Your Terms") only if (a) Your Terms
    provide that the use limitation in Section 3.3 applies to your
    derivative works, and (b) you identify the specific derivative
    works that are subject to Your Terms. Notwithstanding Your Terms,
    this License (including the redistribution requirements in Section
    3.1) will continue to apply to the Work itself.

    3.3 Use Limitation. The Work and any derivative works thereof only
    may be used or intended for use non-commercially. Notwithstanding
    the foregoing, NVIDIA and its affiliates may use the Work and any
    derivative works commercially. As used herein, "non-commercially"
    means for research or evaluation purposes only.

    3.4 Patent Claims. If you bring or threaten to bring a patent claim
    against any Licensor (including any claim, cross-claim or
    counterclaim in a lawsuit) to enforce any patents that you allege
    are infringed by any Work, then your rights under this License from
    such Licensor (including the grant in Section 2.1) will terminate
    immediately.

    3.5 Trademarks. This License does not grant any rights to use any
    Licensor’s or its affiliates’ names, logos, or trademarks, except
    as necessary to reproduce the notices described in this License.

    3.6 Termination. If you violate any term of this License, then your
    rights under this License (including the grant in Section 2.1) will
    terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
THIS LICENSE.

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
(INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGES.

Generator architecture from the paper
"Alias-Free Generative Adversarial Networks".

=======================================================================
"""
#The code is borrowed for Research purposes only


#------------------------------------------------------------------------------

import numpy as np
import scipy.signal
import scipy.optimize
import torch
from ..torch_utils import misc
from ..torch_utils import persistence
from ..torch_utils.ops import conv2d_gradfix
from ..torch_utils.ops import filtered_lrelu
from ..torch_utils.ops import bias_act

import torch.nn.utils.parametrize as parametrize
import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
# CNO: RADIAL FILTERS: BETA PHASE, WE STILL DO NOT USE THIS FEATURE

class RadialConv2d(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Input spatial size: int or [width, height]
                 kernel_size,
                 stride,
                 padding):
        super(RadialConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = torch.nn.Parameter(torch.zeros(out_channels, in_channels, self.kernel_size, self.kernel_size)) 
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        
        # Random Seed for weight initialization
        self.retrain = 32
        # Xavier weight initialization
        self.init_xavier()
        
        #print(self.weight)
        
    def forward(self, x):
        
        return F.conv2d(x, torch.nn.Parameter((self.weight + torch.rot90(self.weight, 1, [-1,-2]) + torch.rot90(self.weight, 2, [-1,-2]) + 
                                              torch.rot90(self.weight, 3, [-1,-2])/4.0)), 
                                              torch.nn.Parameter(self.bias), 1, self.kernel_size - 1)
    
    def init_xavier(self):
        torch.manual_seed(self.retrain)
        def init_weights(m):
            if  m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_normal_(m.weight, gain=g)
                #torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)
        self.apply(init_weights)
        
#----------------------------------------------------------------------------

#SynthesisLayer does the following:
    
#   1. APPLY 2D CONVOLUTION
#   2. APPLY MODIFIED ACTIVATION LAYER

# @persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        #w_dim,                          # Intermediate latent (W) dimensionality.
        #is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        #use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,     # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
    ):
        super().__init__()
        

        self.is_critically_sampled = is_critically_sampled
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * lrelu_upsampling
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = conv_kernel
        
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.#-------------------------------------------
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))
        #----------------------------------------------------------------------

        # Design downsampling filter.#-----------------------------------------
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))
        #----------------------------------------------------------------------
            
        # Compute padding -----------------------------------------------------
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]
        #----------------------------------------------------------------------

        
        # CNO: WE DO NOT USE RADIAL FILTER
        if not use_radial_filters:
            self.convolution = torch.nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels, 
                                               kernel_size=self.conv_kernel, stride = 1, 
                                               padding = (conv_kernel-1, conv_kernel-1))
        
        # CNO: RADIAL FILTERS: BETA PHASE, WE STILL DO NOT USE THIS FEATURE
        else: 
            self.convolution = RadialConv2d(self.in_channels, self.out_channels, self.conv_kernel, 1, conv_kernel-1)

        #----------------------------------------------------------------------

    def forward(self, x, noise_mode='random', force_fp32=False, update_emas=False):
       
        dtype = torch.float32
        
        x = self.convolution(x.to(dtype))

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = np.sqrt(2)
        slope =  0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=None)


        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# @persistence.persistent_class
class LReLu(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input  transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        
        is_critically_sampled = False,  # Does this layer use critical sampling?  #NOT IMPORTANT FOR CNO.
        use_radial_filters    = False,  # Use radially symmetric downsampling filter?
    ):
        super().__init__()
        
        
        self.is_critically_sampled = is_critically_sampled

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) *lrelu_upsampling
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1  else 1
        
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

            
        # Compute padding. ------------------------------------------------------------------------------
        
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size * self.up_factor) # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
                
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]
            
        #------------------------------------------------------------------------------------------------

    def forward(self, x, noise_mode='random', force_fp32=False, update_emas=False):
       
        dtype = torch.float32

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = np.sqrt(2)
        slope = 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=None)


        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

import torch.nn as nn

class LReLu_regular(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
    ):
        super().__init__()
        
        
        self.activation = nn.LeakyReLU() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate

                    
        #------------------------------------------------------------------------------------------------

    def forward(self, x):
        
        if self.in_sampling_rate == 2*self.out_sampling_rate:
            return nn.AvgPool2d(2, stride=2, padding=0)(self.activation(x))
        elif self.in_sampling_rate == 4*self.out_sampling_rate:
            return nn.AvgPool2d(4, stride=4, padding=1)(self.activation(x))
        else:
            return nn.functional.interpolate(self.activation(x), size=self.out_size)

#------------------------------------------------------------------------------------------------

# This function is slower than the cno_lrelu, but it is easier to set it up

class LReLu_torch(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
    ):
        super().__init__()
        
        
        self.activation = nn.LeakyReLU() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert in_channels == out_channels
        
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate

        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        

        #------------------------------------------------------------------------------------------------

    def forward(self, x):
        
        x = nn.functional.interpolate(x, size = 2*self.in_size,mode='bicubic', antialias = True)
        x = self.activation(x)
        x = nn.functional.interpolate(x, size = self.in_size,mode='bicubic', antialias = True)
        x = nn.functional.interpolate(x, size = self.out_size,mode='bicubic', antialias = True)

        x = x.permute(0,2,3,1)
        x = torch.add(x, torch.broadcast_to(self.bias, x.shape))
        x = x.permute(0,3,1,2)
        return x