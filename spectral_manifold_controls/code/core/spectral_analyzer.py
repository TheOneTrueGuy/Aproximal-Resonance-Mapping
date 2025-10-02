"""
Spectral analysis for SMC - SVD decomposition of activation matrices.

This module performs Singular Value Decomposition on activation matrices to
extract spectral features. The term "spectral" refers to the decomposition
into singular values and vectors, not physical resonance.

Note: Originally called "resonance" in ARM, but renamed for mathematical accuracy
after empirical testing showed no oscillatory behavior in transformers.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import numpy.typing as npt


class SpectralAnalyzer:
    """
    Analyzes spectral patterns in activation matrices using SVD.
    
    Computes spectral signatures from activation matrices via Singular Value
    Decomposition. Extracts features including singular values, entropy, and
    participation ratio to characterize the distribution of activation modes.
    """

    def __init__(self, n_modes: int = 8):
        """
        Initialize spectral analyzer.
        
        Args:
            n_modes: Number of spectral modes (singular values) to keep
        """
        self.n_modes = n_modes

    def compute_signature(
        self,
        activation_matrix: npt.NDArray[np.float32],
        n_modes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute spectral signature from activation matrix using SVD.
        
        Performs Singular Value Decomposition on the centered activation matrix
        and extracts spectral features including singular values, entropy, and
        participation ratio.

        Args:
            activation_matrix: Activation matrix, shape (n_samples, n_features)
            n_modes: Number of modes to keep (default: self.n_modes)

        Returns:
            Dictionary containing spectral metrics:
            - singular_values: Top k singular values from SVD
            - s_norm: Normalized singular values (probability distribution)
            - entropy: Shannon entropy of singular value distribution
            - participation_ratio: Inverse participation ratio (spectral uniformity)
            - participation_ratio_normalized: PR normalized to [0, 1]
            - top_singular_vectors: Right singular vectors (principal directions)
            - explained_variance_ratio: Variance explained by each mode
        """
        n_modes = n_modes or self.n_modes

        # Center the activation matrix
        A0 = activation_matrix - activation_matrix.mean(axis=0, keepdims=True)

        # Compute SVD
        U, s, Vt = np.linalg.svd(A0, full_matrices=False)

        # Ensure no zero singular values
        s = np.maximum(s, 1e-12)

        # Normalized singular values
        s_norm = s / s.sum()

        # Shannon entropy of singular value distribution
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))

        # Participation ratio (measure of mode concentration)
        pr = (s**2).sum()**2 / (np.sum(s**4) + 1e-12)

        # Participation ratio normalized to [0, 1] range
        # For uniform distribution: PR = n_modes, for single mode: PR = 1
        n_total_modes = len(s)
        pr_normalized = (pr - 1) / (n_total_modes - 1) if n_total_modes > 1 else 1.0

        return {
            "singular_values": s[:n_modes].astype(np.float32),
            "s_norm": s_norm[:n_modes].astype(np.float32),
            "entropy": float(entropy),
            "participation_ratio": float(pr),
            "participation_ratio_normalized": float(pr_normalized),
            "top_singular_vectors": Vt[:n_modes].astype(np.float32),  # shape (n_modes, n_features)
            "explained_variance_ratio": (s[:n_modes]**2 / (s**2).sum()).astype(np.float32),
        }

    def batch_signatures(
        self,
        activation_matrices: List[npt.NDArray[np.float32]],
        n_modes: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute spectral signatures for multiple activation matrices.

        Args:
            activation_matrices: List of activation matrices
            n_modes: Number of modes to keep

        Returns:
            List of spectral signatures
        """
        return [self.compute_signature(A, n_modes) for A in activation_matrices]

    def compare_signatures(
        self,
        sig1: Dict[str, Any],
        sig2: Dict[str, Any],
        metric: str = "cosine"
    ) -> float:
        """
        Compare two spectral signatures.

        Args:
            sig1: First spectral signature
            sig2: Second spectral signature
            metric: Comparison metric ("cosine", "euclidean", "entropy_diff")

        Returns:
            Similarity/distance score
        """
        if metric == "cosine":
            # Cosine similarity between top singular vectors
            v1 = sig1["top_singular_vectors"].flatten()
            v2 = sig2["top_singular_vectors"].flatten()
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            return dot_product / (norm1 * norm2 + 1e-12)

        elif metric == "euclidean":
            # Euclidean distance between normalized singular values
            return np.linalg.norm(sig1["s_norm"] - sig2["s_norm"])

        elif metric == "entropy_diff":
            # Absolute difference in entropy
            return abs(sig1["entropy"] - sig2["entropy"])

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def detect_significant_modes(
        self,
        activation_matrix: npt.NDArray[np.float32],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect significant spectral modes based on explained variance.

        Args:
            activation_matrix: Activation matrix
            threshold: Minimum explained variance ratio for significant modes

        Returns:
            Dictionary with mode detection results
        """
        sig = self.compute_signature(activation_matrix)

        explained_var = sig["explained_variance_ratio"]
        significant_modes = np.where(explained_var >= threshold)[0]

        return {
            "n_significant_modes": len(significant_modes),
            "significant_mode_indices": significant_modes.tolist(),
            "significant_explained_var": explained_var[significant_modes].tolist(),
            "cumulative_explained_var": np.cumsum(explained_var).tolist(),
        }
    
    def compute_descriptor(self, signature: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """
        Compute descriptor vector from spectral signature.
        
        Combines spectral features into a single descriptor vector suitable
        for downstream tasks like weighted composition.
        
        Args:
            signature: Spectral signature from compute_signature()
        
        Returns:
            Descriptor vector combining s_norm, entropy, and participation ratio
        """
        features = np.concatenate([
            signature['s_norm'],
            [signature['entropy'], signature['participation_ratio_normalized']]
        ])
        return features.astype(np.float32)

