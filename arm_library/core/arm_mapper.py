"""
Main ARM (Aproximal Resonance Mapping) orchestrator class.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import numpy.typing as npt

from .probe_generator import ProbeGenerator
from .resonance_analyzer import ResonanceAnalyzer
from .topology_mapper import TopologyMapper
from .steering import ARMControlVectorComputer, ARMControlledGenerator, ARMControlVector
from ..interfaces.model_interface import TransformerModelInterface
from ..utils.config import ARMConfig, ModelConfig


class ARMMapper:
    """Main class for Aproximal Resonance Mapping of transformer latent manifolds."""

    def __init__(self, config: ARMConfig):
        self.config = config

        # Initialize components
        self.model_interface = TransformerModelInterface(ModelConfig(
            model_name=config.model_name,
            device=config.device,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit
        ))

        self.probe_generator = ProbeGenerator(config)
        self.resonance_analyzer = ResonanceAnalyzer(config)
        self.topology_mapper = TopologyMapper(config)

    def collect_activation_matrix(
        self,
        prompt: str,
        k: Optional[int] = None,
        steps: Optional[int] = None,
        eps: Optional[float] = None
    ) -> npt.NDArray[np.float32]:
        """
        Collect activation matrix for a seed prompt by probing its neighborhood.

        Args:
            prompt: Seed prompt
            k: Number of probes
            steps: Steps per probe
            eps: Perturbation magnitude

        Returns:
            Activation matrix, shape (k*steps, n_features)
        """
        k = k or self.config.probes_per_seed
        steps = steps or self.config.steps_per_probe
        eps = eps or self.config.eps

        # Get base hidden state
        hidden_base = self.model_interface.get_hidden_at_layer(prompt, self.config.layer_to_probe)
        # Support both torch.Tensor and numpy.ndarray from model interface/mocks
        if isinstance(hidden_base, torch.Tensor):
            hidden_base_np = hidden_base.detach().cpu().numpy()
        elif isinstance(hidden_base, np.ndarray):
            hidden_base_np = hidden_base
        else:
            raise TypeError(f"Unsupported hidden_base type: {type(hidden_base)}")

        # Generate probe batch
        probe_paths, _ = self.probe_generator.generate_probe_batch(
            hidden_base_np, k=k, steps=steps, eps=eps
        )

        # Collect activations for each probe point
        rows = []
        for path in probe_paths:
            for hidden_pert in path:
                # Convert to tensor with batch dimension
                h_t = torch.tensor(hidden_pert[None, :, :], dtype=torch.float32, device=self.config.device)

                # Forward from probe layer
                logits, final_h, _ = self.model_interface.forward_from_layer(
                    h_t, start_layer=self.config.layer_to_probe
                )

                # Extract features based on config
                if self.config.feature_type == "hidden_pooled":
                    feat = final_h.squeeze(0).mean(dim=0).detach().cpu().numpy()
                elif self.config.feature_type == "logits_last":
                    feat = logits[0, -1, :].detach().cpu().numpy()
                else:
                    raise ValueError(f"Unknown feature_type: {self.config.feature_type}")

                rows.append(feat)

        return np.stack(rows, axis=0).astype(np.float32)

    def analyze_seed_point(self, prompt: str) -> Dict[str, Any]:
        """
        Perform complete ARM analysis for a single seed point.

        Args:
            prompt: Seed prompt

        Returns:
            Complete analysis results
        """
        # Collect activation matrix
        A = self.collect_activation_matrix(prompt)

        # Resonance analysis
        resonance_sig = self.resonance_analyzer.resonance_signature(A)

        # Topology analysis
        persistence_data = self.topology_mapper.local_persistence(A)

        # Combined descriptor
        descriptor = self.topology_mapper.compute_topological_descriptor(
            resonance_sig, persistence_data
        )

        return {
            'prompt': prompt,
            'activation_matrix': A,
            'resonance_signature': resonance_sig,
            'persistence_data': persistence_data,
            'descriptor': descriptor,
        }

    def map_latent_manifold(
        self,
        seed_prompts: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Map the latent manifold by analyzing multiple seed points.

        Args:
            seed_prompts: List of seed prompts
            progress_callback: Optional callback for progress updates

        Returns:
            Complete manifold mapping results
        """
        seed_analyses = []

        for i, prompt in enumerate(seed_prompts):
            if progress_callback:
                progress_callback(i, len(seed_prompts), f"Analyzing seed {i+1}/{len(seed_prompts)}")

            analysis = self.analyze_seed_point(prompt)
            seed_analyses.append(analysis)

        # Extract resonance signatures for global analysis
        resonance_sigs = [analysis['resonance_signature'] for analysis in seed_analyses]

        # Build resonance graph
        graph_data = self.topology_mapper.build_resonance_graph(resonance_sigs)

        # Detect attractor basins
        clustering_data = self.topology_mapper.detect_attractor_basins(resonance_sigs, graph_data)

        # Extract descriptors
        descriptors = np.array([analysis['descriptor'] for analysis in seed_analyses])

        # Store results for steering methods
        results = {
            'seed_analyses': seed_analyses,
            'graph_data': graph_data,
            'clustering_data': clustering_data,
            'descriptors': descriptors,
            'n_seeds': len(seed_prompts),
            'prompts': seed_prompts,  # Store prompts for steering
            'layer_to_probe': self.config.layer_to_probe,  # Store layer info
        }
        self._last_results = results

        return results

    def find_similar_seeds(
        self,
        target_signature: Dict[str, Any],
        seed_analyses: List[Dict[str, Any]],
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """
        Find seeds with similar resonance patterns to a target signature.

        Args:
            target_signature: Target resonance signature
            seed_analyses: List of seed analyses
            top_k: Number of most similar seeds to return
            metric: Similarity metric

        Returns:
            List of (seed_index, similarity_score) tuples
        """
        similarities = []

        for i, analysis in enumerate(seed_analyses):
            sig = analysis['resonance_signature']
            similarity = self.resonance_analyzer.compare_resonance_signatures(
                target_signature, sig, metric=metric
            )
            similarities.append((i, similarity))

        # Sort by similarity (higher is better for cosine, lower for distances)
        if metric == "cosine":
            similarities.sort(key=lambda x: x[1], reverse=True)
        else:
            similarities.sort(key=lambda x: x[1])

        return similarities[:top_k]

    def create_control_vector_computer(self) -> ARMControlVectorComputer:
        """Create a control vector computer for steering."""
        return ARMControlVectorComputer(self.model_interface)

    def create_controlled_generator(self) -> ARMControlledGenerator:
        """Create a controlled generator for steered text generation."""
        return ARMControlledGenerator(self.model_interface)

    def compute_steering_vector_from_manifold(
        self,
        positive_region_indices: List[int],
        negative_region_indices: List[int],
        layer: Optional[int] = None
    ) -> ARMControlVector:
        """
        Compute a steering vector from positive and negative regions in the manifold.

        Args:
            positive_region_indices: Indices of seeds representing positive behavior
            negative_region_indices: Indices of seeds representing negative behavior
            layer: Layer to compute steering for (defaults to config.layer_to_probe)

        Returns:
            Control vector for steering
        """
        if not hasattr(self, '_last_results') or self._last_results is None:
            raise ValueError("Must run map_latent_manifold() first to have manifold data")

        if layer is None:
            layer = self.config.layer_to_probe

        computer = self.create_control_vector_computer()

        # Get positive and negative prompts from the manifold data
        manifold_data = self._last_results
        all_prompts = manifold_data.get('prompts', [])

        positive_prompts = [all_prompts[i] for i in positive_region_indices if i < len(all_prompts)]
        negative_prompts = [all_prompts[i] for i in negative_region_indices if i < len(all_prompts)]

        if not positive_prompts or not negative_prompts:
            raise ValueError("Need at least one positive and one negative example")

        return computer.compute_control_vector(positive_prompts, negative_prompts, layer)

    def steer_generation_toward_signature(
        self,
        prompt: str,
        target_signature: np.ndarray,
        max_length: int = 50,
        temperature: float = 0.8,
        steering_strength: float = 1.0,
        mode_indices: Optional[List[int]] = None,
        mode_weights: Optional[List[float]] = None,
    ) -> str:
        """
        Generate text steered toward a target resonance signature.

        Args:
            prompt: Starting prompt
            target_signature: Target resonance signature
            max_length: Maximum tokens to generate
            temperature: Generation temperature
            steering_strength: How strongly to apply steering (0 = no steering)

        Returns:
            Generated text
        """
        if not hasattr(self, '_last_results') or self._last_results is None:
            raise ValueError("Must run map_latent_manifold() first to have manifold data")

        generator = self.create_controlled_generator()
        return generator.generate_with_manifold_steering(
            prompt=prompt,
            target_signature=target_signature,
            manifold_data=self._last_results,
            max_length=max_length,
            temperature=temperature,
            steering_strength=steering_strength,
            mode_indices=mode_indices,
            mode_weights=mode_weights,
        )
