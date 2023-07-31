

from .sr1 import *
from .AE import *
MODELS = {
            "ConAE": ConvAutoencoder,
            "Coarse_sr5": Coarse_sr5,
            "Coarse_sr4": Coarse_sr4,
            "Coarse_sr7": Coarse_sr7,
            "Coarse_sr8": Coarse_sr8,
            "sr6": Coarse_sr6,
            "sr61": Coarse_sr61,
            }