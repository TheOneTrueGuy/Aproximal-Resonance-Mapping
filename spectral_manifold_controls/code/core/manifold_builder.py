"""
Manifold builder for SMC.

Builds spectral manifolds from text corpora by analyzing activation patterns.
Integrates with ARM's model interface for transformer access.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import numpy.typing as npt

# Import from legacy ARM (infrastructure only)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from arm_library.interfaces.model_interface import TransformerModelInterface
from arm_library.utils.config import ModelConfig

from .spectral_analyzer import SpectralAnalyzer


class SpectralManifold:
    """
    Represents a pre-built spectral manifold.
    
    The manifold contains spectral signatures for a corpus of text examples,
    along with semantic labels that map to groups of indices.
    
    Think of this as a "palette" or "library" of behavioral patterns that
    can be mixed and matched via weighted composition.
    """
    
    def __init__(
        self,
        corpus: List[str],
        signatures: List[Dict[str, Any]],
        descriptors: npt.NDArray[np.float32],
        model_name: str,
        layer: int,
        index_labels: Optional[Dict[str, List[int]]] = None
    ):
        """
        Initialize manifold.
        
        Args:
            corpus: List of text examples
            signatures: List of spectral signatures (one per example)
            descriptors: Descriptor vectors for each example
            model_name: Name of model used to build manifold
            layer: Layer that was probed
            index_labels: Optional semantic labels mapping to index groups
        """
        self.corpus = corpus
        self.signatures = signatures
        self.descriptors = descriptors
        self.model_name = model_name
        self.layer = layer
        self.index_labels = index_labels or {}
        
    def add_labels(self, labels: Dict[str, List[int]]) -> None:
        """
        Add or update semantic labels.
        
        Args:
            labels: Dictionary mapping label names to index lists
        """
        self.index_labels.update(labels)
    
    def get_label_indices(self, label: str) -> List[int]:
        """Get indices for a semantic label."""
        if label not in self.index_labels:
            raise ValueError(f"Unknown label: {label}. Available: {list(self.index_labels.keys())}")
        return self.index_labels[label]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifold to dictionary."""
        return {
            'corpus': self.corpus,
            'signatures': self.signatures,
            'descriptors': self.descriptors.tolist(),
            'model_name': self.model_name,
            'layer': self.layer,
            'index_labels': self.index_labels,
        }


class ManifoldBuilder:
    """
    Builds spectral manifolds from text corpora.
    
    This is the "expensive" one-time operation that analyzes a corpus
    and creates a reusable manifold for instant weighted composition.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        n_modes: int = 8
    ):
        """
        Initialize manifold builder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            n_modes: Number of spectral modes to extract
        """
        # Initialize model interface
        model_config = ModelConfig(
            model_name=model_name,
            device=device,
            output_hidden_states=True
        )
        self.model_interface = TransformerModelInterface(model_config)
        
        # Initialize spectral analyzer
        self.spectral_analyzer = SpectralAnalyzer(n_modes=n_modes)
        
        self.model_name = model_name
        self.device = device
        self.n_modes = n_modes
    
    def build_manifold(
        self,
        corpus: List[str],
        layer: int = 3,
        probes_per_seed: int = 4,
        steps_per_probe: int = 3,
        eps: float = 0.03,
        progress_callback: Optional[callable] = None
    ) -> SpectralManifold:
        """
        Build spectral manifold from text corpus.
        
        For each text in the corpus:
        1. Generate probe perturbations around its activation
        2. Collect activation matrix
        3. Compute spectral signature via SVD
        4. Extract descriptor vector
        
        Args:
            corpus: List of text examples
            layer: Transformer layer to probe
            probes_per_seed: Number of directional probes per text
            steps_per_probe: Number of steps along each probe
            eps: Perturbation magnitude
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            SpectralManifold ready for weighted composition
        """
        signatures = []
        descriptors = []
        
        for i, text in enumerate(corpus):
            if progress_callback:
                progress_callback(i, len(corpus), f"Analyzing: {text[:50]}...")
            
            # Collect activation matrix via probing
            activation_matrix = self._collect_activation_matrix(
                text, layer, probes_per_seed, steps_per_probe, eps
            )
            
            # Compute spectral signature
            signature = self.spectral_analyzer.compute_signature(activation_matrix)
            signatures.append(signature)
            
            # Extract descriptor
            descriptor = self.spectral_analyzer.compute_descriptor(signature)
            descriptors.append(descriptor)
        
        descriptors_array = np.array(descriptors)
        
        return SpectralManifold(
            corpus=corpus,
            signatures=signatures,
            descriptors=descriptors_array,
            model_name=self.model_name,
            layer=layer
        )
    
    def _collect_activation_matrix(
        self,
        text: str,
        layer: int,
        k: int,
        steps: int,
        eps: float
    ) -> npt.NDArray[np.float32]:
        """
        Collect activation matrix by probing around a text's activation.
        
        Args:
            text: Text to analyze
            layer: Layer to probe
            k: Number of probes
            steps: Steps per probe
            eps: Perturbation magnitude
        
        Returns:
            Activation matrix, shape (k*steps, n_features)
        """
        # Get base hidden state
        hidden_base = self.model_interface.get_hidden_at_layer(text, layer)
        
        # Convert to numpy
        if isinstance(hidden_base, torch.Tensor):
            hidden_base_np = hidden_base.detach().cpu().numpy()
        else:
            hidden_base_np = hidden_base
        
        # Generate random probe directions
        seq_len, hidden_dim = hidden_base_np.shape
        probe_directions = []
        for _ in range(k):
            # Random direction in hidden space
            direction = np.random.randn(seq_len, hidden_dim).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            probe_directions.append(direction)
        
        # Collect activations along probes
        rows = []
        for direction in probe_directions:
            for step in range(steps):
                # Perturb along direction
                alpha = (step + 1) * eps
                hidden_pert = hidden_base_np + alpha * direction
                
                # Convert to tensor
                h_t = torch.tensor(
                    hidden_pert[None, :, :],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Forward from this layer
                logits, final_h, _ = self.model_interface.forward_from_layer(
                    h_t, start_layer=layer
                )
                
                # Extract feature (mean-pooled hidden state)
                feat = final_h.squeeze(0).mean(dim=0).detach().cpu().numpy()
                rows.append(feat)
        
        return np.stack(rows, axis=0).astype(np.float32)

