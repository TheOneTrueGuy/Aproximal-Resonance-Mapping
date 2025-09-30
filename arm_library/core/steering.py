#!/usr/bin/env python3
"""
ARM Steering Module

Implements control vector-based steering for ARM, inspired by Representation Engineering (RepE).
Provides functionality to compute control vectors from positive/negative examples and apply
them during text generation.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from ..interfaces.model_interface import TransformerModelInterface


class ARMControlVector:
    """Represents a control vector for steering model behavior."""

    def __init__(self, direction: torch.Tensor, layer: int, coefficient: float = 1.0):
        """
        Initialize a control vector.

        Args:
            direction: The steering direction vector (shape: [hidden_size])
            layer: The transformer layer to apply steering at
            coefficient: Scaling coefficient for the control vector
        """
        self.direction = direction
        self.layer = layer
        self.coefficient = coefficient

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply the control vector to hidden states.

        Args:
            hidden_states: Hidden states tensor (shape: [batch_size, seq_len, hidden_size])

        Returns:
            Modified hidden states with control vector applied
        """
        # Add the control vector to all positions in the sequence
        steering_vector = self.direction.unsqueeze(0).unsqueeze(0) * self.coefficient
        return hidden_states + steering_vector


class ARMControlVectorComputer:
    """Computes control vectors from positive and negative examples."""

    def __init__(self, model_interface: TransformerModelInterface):
        """
        Initialize the control vector computer.

        Args:
            model_interface: Interface to the transformer model
        """
        self.model_interface = model_interface

    def compute_control_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer: int,
        normalize: bool = True
    ) -> ARMControlVector:
        """
        Compute a control vector from positive and negative examples.

        This follows the RepE approach: control_vector = mean(pos_activations) - mean(neg_activations)

        Args:
            positive_prompts: Prompts that exemplify the desired behavior
            negative_prompts: Prompts that exemplify the undesired behavior
            layer: Transformer layer to extract activations from
            normalize: Whether to normalize the control vector

        Returns:
            ARMControlVector for steering
        """
        # Get activations for positive prompts
        pos_activations = []
        for prompt in positive_prompts:
            hidden = self.model_interface.get_hidden_at_layer(prompt, layer)
            # Use the mean activation across the sequence (excluding padding if any)
            pos_activations.append(hidden.mean(dim=0))

        # Get activations for negative prompts
        neg_activations = []
        for prompt in negative_prompts:
            hidden = self.model_interface.get_hidden_at_layer(prompt, layer)
            # Use the mean activation across the sequence (excluding padding if any)
            neg_activations.append(hidden.mean(dim=0))

        # Compute means
        pos_mean = torch.stack(pos_activations).mean(dim=0)
        neg_mean = torch.stack(neg_activations).mean(dim=0)

        # Compute control vector: positive - negative
        control_direction = pos_mean - neg_mean

        # Normalize if requested
        if normalize:
            control_direction = control_direction / (torch.norm(control_direction) + 1e-8)

        return ARMControlVector(control_direction, layer)

    def compute_manifold_control_vector(
        self,
        target_signature: np.ndarray,
        layer: int,
        manifold_data: Dict[str, Any],
        normalize: bool = True
    ) -> ARMControlVector:
        """
        Compute a control vector that steers toward a specific resonance signature
        within a discovered manifold.

        Args:
            target_signature: Target resonance signature to steer toward
            layer: Transformer layer
            manifold_data: Results from ARM manifold analysis
            normalize: Whether to normalize the control vector

        Returns:
            ARMControlVector for manifold-based steering
        """
        # Find the seed with the closest resonance signature to the target
        seed_analyses = manifold_data['seed_analyses']
        best_seed_idx = None
        best_distance = float('inf')

        for i, analysis in enumerate(seed_analyses):
            signature = analysis['resonance_signature']['s_norm']
            # Compare signatures (truncate to same length if needed)
            min_len = min(len(signature), len(target_signature))
            distance = np.linalg.norm(signature[:min_len] - target_signature[:min_len])
            if distance < best_distance:
                best_distance = distance
                best_seed_idx = i

        if best_seed_idx is None:
            raise ValueError("No suitable seed found for manifold steering")

        # Derive a hidden-sized direction from resonance data
        # Prefer the first right singular vector (top principal direction)
        best_sig = seed_analyses[best_seed_idx]['resonance_signature']
        top_singular_vectors = best_sig.get('top_singular_vectors', None)
        if top_singular_vectors is None:
            raise ValueError("Resonance signature missing 'top_singular_vectors'. Re-run analysis with n_modes > 0.")

        # Validate dimensionality against model hidden size
        model_dims = self.model_interface.get_model_dimensions()
        hidden_size = model_dims.get('d_model')
        if top_singular_vectors.shape[1] != hidden_size:
            raise ValueError(
                "Manifold steering requires analysis with feature_type='hidden_pooled' so singular vectors match hidden size. "
                f"Got singular vector size {top_singular_vectors.shape[1]} vs hidden size {hidden_size}."
            )

        control_direction = torch.tensor(top_singular_vectors[0], dtype=torch.float32)
        if normalize:
            control_direction = control_direction / (torch.norm(control_direction) + 1e-8)

        return ARMControlVector(control_direction, layer)


class ARMControlledGenerator:
    """Generates text with ARM control vector steering."""

    def __init__(self, model_interface: TransformerModelInterface):
        """
        Initialize the controlled generator.

        Args:
            model_interface: Interface to the transformer model
        """
        self.model_interface = model_interface
        self.active_controls = {}  # layer -> ARMControlVector
        self._hook_handles = []

    def _validate_control_dimension(self, control_vector: ARMControlVector):
        """Ensure control vector matches model hidden size."""
        model_dims = self.model_interface.get_model_dimensions()
        hidden_size = model_dims.get('d_model')
        vec = control_vector.direction
        if vec.dim() != 1 or vec.shape[0] != hidden_size:
            raise ValueError(
                f"Control vector dimension {tuple(vec.shape)} incompatible with model hidden size {hidden_size}."
            )

    def _register_layer_hooks(self) -> bool:
        """Register forward-pre hooks on targeted transformer blocks to inject controls."""
        # Clear any previous hooks
        self._remove_layer_hooks()

        try:
            blocks = self.model_interface.model.transformer.h
        except Exception:
            # Model structure not supported for hooking
            return False

        for layer_idx, cv in list(self.active_controls.items()):
            # Validate indices and dimensions
            if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= len(blocks):
                continue
            self._validate_control_dimension(cv)

            def make_pre_hook(idx: int):
                def pre_hook(module, inputs):
                    # inputs is a tuple: (hidden_states, ...)
                    if idx not in self.active_controls:
                        return None
                    try:
                        hidden_states = inputs[0]
                        cv_local = self.active_controls[idx]
                        steer = (cv_local.direction.to(hidden_states.device) * cv_local.coefficient)
                        steer = steer.unsqueeze(0).unsqueeze(0)
                        modified = hidden_states + steer
                        return (modified,) + inputs[1:]
                    except Exception:
                        # If anything goes wrong, skip modification for this call
                        return None
                return pre_hook

            handle = blocks[layer_idx].register_forward_pre_hook(make_pre_hook(layer_idx))
            self._hook_handles.append(handle)

        return len(self._hook_handles) > 0

    def _remove_layer_hooks(self):
        """Remove all registered hooks."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def set_control(self, control_vector: ARMControlVector):
        """
        Set a control vector for steering.

        Args:
            control_vector: The control vector to apply
        """
        # Validate before setting
        self._validate_control_dimension(control_vector)
        self.active_controls[control_vector.layer] = control_vector

    def clear_controls(self):
        """Clear all active control vectors."""
        self.active_controls.clear()

    def generate_with_steering(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Generate text with active control vectors applied.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        # Encode the prompt
        input_ids, attention_mask = self.model_interface.encode_prompt(prompt)

        outputs = None
        hooks_active = False
        try:
            if self.active_controls:
                hooks_active = self._register_layer_hooks()

            # Run generation (hooks will apply at each step if active)
            outputs = self.model_interface.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.model_interface.tokenizer.pad_token_id,
                eos_token_id=self.model_interface.tokenizer.eos_token_id,
                **generation_kwargs
            )
        finally:
            if hooks_active:
                self._remove_layer_hooks()

        # Decode the generated text
        generated_text = self.model_interface.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return generated_text

    def generate_with_manifold_steering(
        self,
        prompt: str,
        target_signature: np.ndarray,
        manifold_data: Dict[str, Any],
        max_length: int = 50,
        temperature: float = 1.0,
        steering_strength: float = 1.0,
        **generation_kwargs
    ) -> str:
        """
        Generate text with manifold-based steering toward a target resonance signature.

        Args:
            prompt: Input prompt
            target_signature: Target resonance signature
            manifold_data: ARM manifold analysis results
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            steering_strength: Strength of the steering (0 = no steering)
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated text with manifold steering applied
        """
        if steering_strength == 0:
            # No steering requested
            return self.generate_with_steering(prompt, max_length, temperature, **generation_kwargs)

        # Compute control vector for manifold steering
        computer = ARMControlVectorComputer(self.model_interface)
        layer = manifold_data.get('layer_to_probe', 6)  # Default to layer 6
        control_vector = computer.compute_manifold_control_vector(
            target_signature, layer, manifold_data
        )
        control_vector.coefficient = steering_strength

        # Apply the control and generate
        self.set_control(control_vector)
        try:
            result = self.generate_with_steering(prompt, max_length, temperature, **generation_kwargs)
        finally:
            self.clear_controls()

        return result

