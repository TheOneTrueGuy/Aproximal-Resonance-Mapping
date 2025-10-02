"""
Spectral Manifold Control (SMC) - Compositional Language Model Steering

This package implements SMC: a system for steering language models using
weighted composition of spectral signatures from a pre-built manifold.

Key Innovation: Build a reusable manifold once, then compose control vectors
instantly via weighted index mixing - like a "style mixing board."

Example:
    >>> from smc.core import SpectralAnalyzer, WeightedControlVectorComputer
    >>> 
    >>> # Build manifold (one-time cost)
    >>> analyzer = SpectralAnalyzer(n_modes=8)
    >>> signatures = [analyzer.compute_signature(act) for act in activations]
    >>> 
    >>> # Mix with weights (instant)
    >>> weights = {'formal': 0.7, 'friendly': 0.5, 'casual': -0.2}
    >>> cv = computer.compute_from_labels(weights, labels, corpus, layer=3)
    >>> 
    >>> # Generate
    >>> output = generate_with_control(prompt, cv)
"""

__version__ = "0.1.0"

from .core.spectral_analyzer import SpectralAnalyzer
from .core.weighted_control import WeightedControlVector, WeightedControlVectorComputer

__all__ = [
    "SpectralAnalyzer",
    "WeightedControlVector",
    "WeightedControlVectorComputer",
]

