"""
Core SMC modules: spectral analysis, manifold building, and weighted control.
"""

from .spectral_analyzer import SpectralAnalyzer
from .manifold_builder import SpectralManifold, ManifoldBuilder
from .weighted_control import WeightedControlVector, WeightedControlVectorComputer

__all__ = [
    "SpectralAnalyzer",
    "SpectralManifold",
    "ManifoldBuilder",
    "WeightedControlVector",
    "WeightedControlVectorComputer",
]

