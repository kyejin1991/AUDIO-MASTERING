from .audio_modules import PROCESSORS
from .equalizer import process_equalizer_advanced as EQModule
from .compressor import process_compressor_advanced as CompressorModule
from .maximizer import process_maximizer_advanced as MaximizerModule

__all__ = [
    "PROCESSORS",
    "EQModule",
    "CompressorModule",
    "MaximizerModule",
]
