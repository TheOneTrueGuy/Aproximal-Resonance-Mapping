"""
Configuration management for ARM hyperparameters and model settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class ARMConfig:
    """Configuration for Aproximal Resonance Mapping."""

    # Model settings
    model_name: str = "distilgpt2"
    device: Optional[torch.device] = None

    # ARM hyperparameters
    n_seeds: int = 200
    probes_per_seed: int = 16
    steps_per_probe: int = 9
    eps: float = 0.03  # perturbation magnitude (relative to hidden vector norm)
    layer_to_probe: int = 6  # transformer layer to inject perturbations
    neighbor_pca_samples: int = 128

    # Resonance analysis settings
    n_modes: int = 8  # number of modes to keep in resonance signature
    feature_type: str = "hidden_pooled"  # "hidden_pooled" or "logits_last"

    # Topology settings
    max_homology_dim: int = 1
    topology_neighbors: int = 10

    # Random seed for reproducibility
    random_seed: Optional[int] = 42

    def __post_init__(self):
        """Set device if not provided."""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ARMConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'device': str(self.device) if self.device else None,
            'n_seeds': self.n_seeds,
            'probes_per_seed': self.probes_per_seed,
            'steps_per_probe': self.steps_per_probe,
            'eps': self.eps,
            'layer_to_probe': self.layer_to_probe,
            'neighbor_pca_samples': self.neighbor_pca_samples,
            'n_modes': self.n_modes,
            'feature_type': self.feature_type,
            'max_homology_dim': self.max_homology_dim,
            'topology_neighbors': self.topology_neighbors,
            'random_seed': self.random_seed,
        }


@dataclass
class ModelConfig:
    """Configuration for transformer model interface."""

    model_name: str = "distilgpt2"
    device: Optional[torch.device] = None
    output_hidden_states: bool = True
    torch_dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        """Set device if not provided."""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
