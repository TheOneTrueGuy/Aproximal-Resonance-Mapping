"""
Resonance analysis for ARM - spectral decomposition of activation matrices.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import numpy.typing as npt
from sklearn.decomposition import TruncatedSVD

from ..utils.config import ARMConfig


class ResonanceAnalyzer:
    """Analyzes resonance patterns in activation matrices using spectral methods."""

    def __init__(self, config: ARMConfig):
        self.config = config

    def resonance_signature(
        self,
        activation_matrix: npt.NDArray[np.float32],
        n_modes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute resonance signature from activation matrix using SVD.

        Args:
            activation_matrix: Activation matrix, shape (n_samples, n_features)
            n_modes: Number of modes to keep (default: config.n_modes)

        Returns:
            Dictionary containing resonance metrics
        """
        n_modes = n_modes or self.config.n_modes

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

    def batch_resonance_signatures(
        self,
        activation_matrices: List[npt.NDArray[np.float32]],
        n_modes: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute resonance signatures for multiple activation matrices.

        Args:
            activation_matrices: List of activation matrices
            n_modes: Number of modes to keep

        Returns:
            List of resonance signatures
        """
        return [self.resonance_signature(A, n_modes) for A in activation_matrices]

    def compare_resonance_signatures(
        self,
        sig1: Dict[str, Any],
        sig2: Dict[str, Any],
        metric: str = "cosine"
    ) -> float:
        """
        Compare two resonance signatures.

        Args:
            sig1: First resonance signature
            sig2: Second resonance signature
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

    def detect_resonance_modes(
        self,
        activation_matrix: npt.NDArray[np.float32],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect significant resonance modes based on explained variance.

        Args:
            activation_matrix: Activation matrix
            threshold: Minimum explained variance ratio for significant modes

        Returns:
            Dictionary with mode detection results
        """
        sig = self.resonance_signature(activation_matrix)

        explained_var = sig["explained_variance_ratio"]
        significant_modes = np.where(explained_var >= threshold)[0]

        return {
            "n_significant_modes": len(significant_modes),
            "significant_mode_indices": significant_modes.tolist(),
            "significant_explained_var": explained_var[significant_modes].tolist(),
            "cumulative_explained_var": np.cumsum(explained_var).tolist(),
        }
