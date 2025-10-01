"""
SpectraLM — Physics-informed IR spectrum → SMILES translation.

Package layout:
    spectralm.models    — SpectralEncoder, SpectralTransformer, SpectraLM
    spectralm.physics   — BeerLambertConstraint, GroupFrequencyChecker
    spectralm.data      — SpectralAugmentor, IRSpectraDataset
"""

from spectralm.models import SpectraLM, SpectraLMConfig

__version__ = "0.3.0"
__all__ = ["SpectraLM", "SpectraLMConfig"]