"""
Probe generation for ARM - creates directional perturbations around seed points.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import numpy.typing as npt

from ..utils.config import ARMConfig


class ProbeGenerator:
    """Generates directional probes for ARM analysis."""

    def __init__(self, config: ARMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def sample_probes_for_hidden(
        self,
        hidden_vec: npt.NDArray[np.float32],
        k: Optional[int] = None,
        eps: Optional[float] = None
    ) -> npt.NDArray[np.float32]:
        """
        Sample directional probes around a hidden state vector.

        Args:
            hidden_vec: Hidden state array, shape (seq_len, d_model)
            k: Number of probes to generate (default: config.probes_per_seed)
            eps: Perturbation magnitude (default: config.eps)

        Returns:
            Probe deltas, shape (k, d_model)
        """
        k = k or self.config.probes_per_seed
        eps = eps or self.config.eps

        # Pool hidden state to single vector for direction construction
        pooled = hidden_vec.mean(axis=0)  # (d_model,)
        d = pooled.shape[0]

        # Sample isotropic Gaussian directions
        dirs = self.rng.normal(size=(k, d))
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

        # Scale by perturbation magnitude relative to hidden vector norm
        hidden_norm = np.linalg.norm(pooled) + 1e-12
        scale = eps * hidden_norm
        dirs = dirs * scale

        return dirs.astype(np.float32)

    def expand_delta_to_sequence(
        self,
        delta_vec: npt.NDArray[np.float32],
        seq_len: int
    ) -> npt.NDArray[np.float32]:
        """
        Expand a pooled delta vector to apply across all sequence positions.

        Args:
            delta_vec: Delta vector, shape (d_model,)
            seq_len: Sequence length

        Returns:
            Expanded deltas, shape (seq_len, d_model)
        """
        return np.tile(delta_vec[None, :], (seq_len, 1)).astype(np.float32)

    def build_probe_path(
        self,
        hidden_base: npt.NDArray[np.float32],
        dir_vec: npt.NDArray[np.float32],
        steps: Optional[int] = None,
        tau: float = 1.0
    ) -> Tuple[List[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
        """
        Build a probe path along a direction.

        Args:
            hidden_base: Base hidden state, shape (seq_len, d_model)
            dir_vec: Direction vector, shape (d_model,)
            steps: Number of steps (default: config.steps_per_probe)
            tau: Scaling factor for step range [-tau, tau]

        Returns:
            Tuple of (path_hidden_states, step_values)
        """
        steps = steps or self.config.steps_per_probe
        seq_len = hidden_base.shape[0]

        # Expand direction across sequence positions
        dir_seq = self.expand_delta_to_sequence(dir_vec, seq_len)

        # Generate step values
        ts = np.linspace(-tau, tau, steps, dtype=np.float32)

        # Build perturbed hidden states
        path = [hidden_base + (t * dir_seq) for t in ts]

        return path, ts

    def generate_probe_batch(
        self,
        hidden_base: npt.NDArray[np.float32],
        k: Optional[int] = None,
        steps: Optional[int] = None,
        eps: Optional[float] = None,
        tau: float = 1.0
    ) -> Tuple[List[List[npt.NDArray[np.float32]]], npt.NDArray[np.float32]]:
        """
        Generate a batch of probe paths around a base hidden state.

        Args:
            hidden_base: Base hidden state, shape (seq_len, d_model)
            k: Number of probes (default: config.probes_per_seed)
            steps: Steps per probe (default: config.steps_per_probe)
            eps: Perturbation magnitude (default: config.eps)
            tau: Step scaling factor

        Returns:
            Tuple of (probe_paths, probe_directions)
                - probe_paths: List of paths, each path is list of hidden states
                - probe_directions: Directions used, shape (k, d_model)
        """
        k = k or self.config.probes_per_seed
        steps = steps or self.config.steps_per_probe
        eps = eps or self.config.eps

        # Generate probe directions
        probe_directions = self.sample_probes_for_hidden(hidden_base, k=k, eps=eps)

        # Build paths for each direction
        probe_paths = []
        for j in range(k):
            path, _ = self.build_probe_path(hidden_base, probe_directions[j], steps=steps, tau=tau)
            probe_paths.append(path)

        return probe_paths, probe_directions
