"""
Weighted control vector composition for SMC.

This module implements the core innovation of Spectral Manifold Control:
weighted composition of control vectors from manifold indices.

Unlike traditional RepE which requires recomputation for each variant,
SMC builds a manifold once and then mixes indices with individual weights
to create control vectors instantly.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import numpy.typing as npt


class WeightedControlVector:
    """
    Represents a control vector with weighted composition from manifold indices.
    
    This is the "mixing board" concept - each index has an individual weight
    that can be positive (steer toward) or negative (steer away from).
    """

    def __init__(self, direction: torch.Tensor, layer: int, weights: Dict[int, float] = None):
        """
        Initialize weighted control vector.

        Args:
            direction: The steering direction vector (shape: [hidden_size])
            layer: The transformer layer to apply steering at
            weights: Dictionary of {index: weight} used to create this vector
        """
        self.direction = direction
        self.layer = layer
        self.weights = weights or {}
        self.strength = 1.0  # Master strength multiplier

    def apply(self, hidden_states: torch.Tensor, strength: Optional[float] = None) -> torch.Tensor:
        """
        Apply the control vector to hidden states.

        Args:
            hidden_states: Hidden states tensor (shape: [batch_size, seq_len, hidden_size])
            strength: Override strength (default: use self.strength)

        Returns:
            Modified hidden states with control vector applied
        """
        effective_strength = strength if strength is not None else self.strength
        
        # Add the control vector to all positions in the sequence
        steering_vector = self.direction.unsqueeze(0).unsqueeze(0) * effective_strength
        return hidden_states + steering_vector

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary with direction, layer, and weights
        """
        return {
            'direction': self.direction.cpu().numpy().tolist(),
            'layer': self.layer,
            'weights': self.weights,
            'strength': self.strength,
        }


class WeightedControlVectorComputer:
    """
    Computes control vectors as weighted combinations of manifold indices.
    
    This is the core of SMC's compositional power. Instead of binary
    positive/negative selection, each index gets an individual weight.
    
    Think of it like an audio mixing board:
    - Each index is a "track"
    - Each weight is the track's "volume"
    - Positive weights = include
    - Negative weights = avoid
    - Zero weights = mute
    """

    def __init__(self, model_interface: Any):
        """
        Initialize the weighted control vector computer.

        Args:
            model_interface: Interface to the transformer model
        """
        self.model_interface = model_interface

    def compute_from_weights(
        self,
        index_weights: Dict[int, float],
        corpus: List[str],
        layer: int,
        normalize: bool = True
    ) -> WeightedControlVector:
        """
        Create control vector from weighted index combination.
        
        This is the "mixing board" in action. Each index gets an individual
        weight, and we compute the weighted sum of their activations.

        Args:
            index_weights: Dictionary {index: weight} where:
                - Positive weights steer TOWARD that index
                - Negative weights steer AWAY from that index
                - Magnitude controls strength of influence
            corpus: List of text examples (manifold corpus)
            layer: Transformer layer to extract activations from
            normalize: Whether to normalize the final direction vector

        Returns:
            WeightedControlVector ready for steering

        Example:
            >>> weights = {
            ...     0: 0.8,    # 80% formal
            ...     12: 0.5,   # 50% friendly
            ...     5: -0.2,   # Avoid casual
            ... }
            >>> cv = compute_from_weights(weights, corpus, layer=3)
        """
        if not index_weights:
            raise ValueError("index_weights cannot be empty")

        # Collect weighted activations
        weighted_activations = []
        
        for idx, weight in index_weights.items():
            if idx < 0 or idx >= len(corpus):
                raise ValueError(f"Index {idx} out of range for corpus of size {len(corpus)}")
            
            # Get activation for this index
            prompt = corpus[idx]
            hidden = self.model_interface.get_hidden_at_layer(prompt, layer)
            
            # Pool over sequence dimension (mean across tokens)
            activation = hidden.mean(dim=0)  # Shape: [hidden_size]
            
            # Apply weight
            weighted_activations.append(activation * weight)
        
        # Sum all weighted activations
        direction = torch.stack(weighted_activations).sum(dim=0)
        
        # Normalize if requested
        if normalize:
            direction = direction / (torch.norm(direction) + 1e-8)
        
        return WeightedControlVector(direction, layer, weights=index_weights)

    def compute_from_labels(
        self,
        label_weights: Dict[str, float],
        index_labels: Dict[str, List[int]],
        corpus: List[str],
        layer: int,
        normalize: bool = True
    ) -> WeightedControlVector:
        """
        Create control vector from semantic label weights.
        
        This is the user-friendly interface - instead of raw indices,
        users specify semantic labels (e.g., "formal", "friendly").
        
        Args:
            label_weights: Dictionary {label: weight}
            index_labels: Dictionary {label: [indices]}
            corpus: List of text examples
            layer: Transformer layer
            normalize: Whether to normalize
        
        Returns:
            WeightedControlVector
        
        Example:
            >>> label_weights = {'formal': 0.7, 'friendly': 0.5, 'casual': -0.2}
            >>> index_labels = {'formal': [0,1,2], 'friendly': [10,11], 'casual': [20,21]}
            >>> cv = compute_from_labels(label_weights, index_labels, corpus, layer=3)
        """
        # Convert label weights to index weights
        index_weights = {}
        
        for label, weight in label_weights.items():
            if label not in index_labels:
                raise ValueError(f"Unknown label: {label}. Available: {list(index_labels.keys())}")
            
            indices = index_labels[label]
            if not indices:
                continue
            
            # Distribute weight evenly across indices in this label
            weight_per_index = weight / len(indices)
            for idx in indices:
                # If multiple labels map to same index, average the weights
                if idx in index_weights:
                    index_weights[idx] = (index_weights[idx] + weight_per_index) / 2
                else:
                    index_weights[idx] = weight_per_index
        
        if not index_weights:
            raise ValueError("No valid indices found for given labels")
        
        return self.compute_from_weights(index_weights, corpus, layer, normalize)

    def interpolate_vectors(
        self,
        cv1: WeightedControlVector,
        cv2: WeightedControlVector,
        alpha: float
    ) -> WeightedControlVector:
        """
        Interpolate between two control vectors.
        
        Useful for smooth transitions or exploring the space between
        two "recipes".
        
        Args:
            cv1: First control vector
            cv2: Second control vector
            alpha: Interpolation factor (0 = cv1, 1 = cv2)
        
        Returns:
            Interpolated control vector
        """
        if cv1.layer != cv2.layer:
            raise ValueError(f"Cannot interpolate vectors from different layers: {cv1.layer} vs {cv2.layer}")
        
        # Linear interpolation
        direction = (1 - alpha) * cv1.direction + alpha * cv2.direction
        
        # Interpolate weights (for tracking)
        weights = {}
        all_indices = set(cv1.weights.keys()) | set(cv2.weights.keys())
        for idx in all_indices:
            w1 = cv1.weights.get(idx, 0)
            w2 = cv2.weights.get(idx, 0)
            weights[idx] = (1 - alpha) * w1 + alpha * w2
        
        return WeightedControlVector(direction, cv1.layer, weights=weights)

