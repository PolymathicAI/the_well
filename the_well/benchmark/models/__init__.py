from .afno import AFNO
from .avit import AViT
from .dilated_resnet import DilatedResNet
from .fno import FNO
from .refno import ReFNO
from .tfno import TFNO
from .unet_classic import UNetClassic
from .unet_convnext import UNetConvNext

__all__ = [
    "FNO",
    "TFNO",
    "UNetClassic",
    "UNetConvNext",
    "DilatedResNet",
    "ReFNO",
    "AViT",
    "AFNO",
]
