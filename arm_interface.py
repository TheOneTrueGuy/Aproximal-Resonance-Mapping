#!/usr/bin/env python3
"""
Interactive ARM (Aproximal Resonance Mapping) Interface using Gradio.

This provides a complete web interface for ARM operations including:
- Model selection and configuration
- Prompt input and analysis
- Parameter tuning (probes, layers, eps, etc.)
- Results visualization
- Text generation with ARM steering
"""

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
import json
import pickle
import base64
import io
import os
import requests

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig
from arm_library.interfaces.model_interface import TransformerModelInterface, ModelConfig

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create output directory for plots
OUTPUT_DIR = "arm_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ARMInterface:
    """Interactive interface for ARM operations."""

    def __init__(self):
        self.current_mapper = None
        self.current_results = None
        self.current_config = None
        self.current_prompts = None
        self.model_directory = None  # User's local model directory
        self.available_models = [
            "distilgpt2",  # Small, fast
            "gpt2",        # Medium size
            "gpt2-medium", # Larger
        ]
        self.local_models = []  # Models found in local directory
        self.hf_compatible_models = []  # HuggingFace models compatible with system
        self.is_processing = False  # Prevent concurrent operations
        
        # Curated list of known transformer models with size estimates (in GB)
        self.known_models = {
            # GPT-2 family
            "distilgpt2": {"size_gb": 0.24, "params": "82M", "type": "GPT-2"},
            "gpt2": {"size_gb": 0.5, "params": "124M", "type": "GPT-2"},
            "gpt2-medium": {"size_gb": 1.5, "params": "355M", "type": "GPT-2"},
            "gpt2-large": {"size_gb": 3.0, "params": "774M", "type": "GPT-2"},
            "gpt2-xl": {"size_gb": 6.0, "params": "1.5B", "type": "GPT-2"},
            
            # GPT-Neo family
            "EleutherAI/gpt-neo-125M": {"size_gb": 0.5, "params": "125M", "type": "GPT-Neo"},
            "EleutherAI/gpt-neo-1.3B": {"size_gb": 5.0, "params": "1.3B", "type": "GPT-Neo"},
            "EleutherAI/gpt-neo-2.7B": {"size_gb": 10.0, "params": "2.7B", "type": "GPT-Neo"},
            
            # GPT-J
            "EleutherAI/gpt-j-6B": {"size_gb": 24.0, "params": "6B", "type": "GPT-J"},
            
            # BERT family (smaller)
            "bert-base-uncased": {"size_gb": 0.4, "params": "110M", "type": "BERT"},
            "bert-large-uncased": {"size_gb": 1.3, "params": "340M", "type": "BERT"},
            "distilbert-base-uncased": {"size_gb": 0.25, "params": "66M", "type": "DistilBERT"},
            
            # Other causal LMs
            "facebook/opt-125m": {"size_gb": 0.5, "params": "125M", "type": "OPT"},
            "facebook/opt-350m": {"size_gb": 1.3, "params": "350M", "type": "OPT"},
            "facebook/opt-1.3b": {"size_gb": 5.0, "params": "1.3B", "type": "OPT"},
            "facebook/opt-2.7b": {"size_gb": 10.0, "params": "2.7B", "type": "OPT"},
            
            # Pythia models (good for research)
            "EleutherAI/pythia-70m": {"size_gb": 0.3, "params": "70M", "type": "Pythia"},
            "EleutherAI/pythia-160m": {"size_gb": 0.6, "params": "160M", "type": "Pythia"},
            "EleutherAI/pythia-410m": {"size_gb": 1.6, "params": "410M", "type": "Pythia"},
            "EleutherAI/pythia-1b": {"size_gb": 4.0, "params": "1B", "type": "Pythia"},
            "EleutherAI/pythia-1.4b": {"size_gb": 5.6, "params": "1.4B", "type": "Pythia"},
            "EleutherAI/pythia-2.8b": {"size_gb": 11.0, "params": "2.8B", "type": "Pythia"},
        }

    def filter_compatible_models(
        self,
        ram_gb: float,
        vram_gb: float,
        use_gpu: bool,
        quantization_mode: str = "none"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter HuggingFace models based on system resources.
        
        Args:
            ram_gb: Available system RAM in GB
            vram_gb: Available GPU VRAM in GB
            use_gpu: Whether to use GPU for inference
            
        Returns:
            Status message and dropdown update dict
        """
        try:
            # Calculate available memory with safety margin (use 80% of available)
            safe_ram = ram_gb * 0.8
            safe_vram = vram_gb * 0.8 if use_gpu else 0
            
            # Apply quantization reduction factor
            if quantization_mode == "8bit":
                quant_factor = 0.25  # 8-bit uses ~25% of original size (4x reduction)
                quant_label = " [8-bit]"
            elif quantization_mode == "4bit":
                quant_factor = 0.125  # 4-bit uses ~12.5% of original size (8x reduction)
                quant_label = " [4-bit]"
            else:
                quant_factor = 1.0  # No quantization
                quant_label = ""
            
            # Model requires: model_size + activation_memory + overhead
            # Rule of thumb: model needs ~1.5x its size for inference
            # Add extra overhead for ARM analysis (probes, activations)
            memory_multiplier = 2.0  # Conservative for ARM operations
            
            compatible_models = []
            incompatible_count = 0
            
            for model_id, info in self.known_models.items():
                model_size = info["size_gb"] * quant_factor  # Apply quantization
                required_memory = model_size * memory_multiplier
                
                # Check if it fits
                if use_gpu:
                    # Try GPU first, fallback to RAM
                    fits_gpu = required_memory <= safe_vram
                    fits_ram = required_memory <= safe_ram
                    is_compatible = fits_gpu or fits_ram
                    
                    if is_compatible:
                        location = "🎮 GPU" if fits_gpu else "💾 CPU"
                        display_name = f"{model_id} [{info['params']}, {model_size:.1f}GB{quant_label}] {location}"
                        compatible_models.append((model_id, display_name, model_size))
                    else:
                        incompatible_count += 1
                else:
                    # CPU only
                    if required_memory <= safe_ram:
                        display_name = f"{model_id} [{info['params']}, {model_size:.1f}GB{quant_label}] 💾 CPU"
                        compatible_models.append((model_id, display_name, model_size))
                    else:
                        incompatible_count += 1
            
            # Sort by size (smallest first)
            compatible_models.sort(key=lambda x: x[2])
            
            # Update dropdown choices
            all_models = self.available_models.copy()
            
            if compatible_models:
                all_models.append("--- Compatible HuggingFace Models ---")
                for model_id, display_name, _ in compatible_models:
                    all_models.append(display_name)
                    # Store for later retrieval
                    if model_id not in self.hf_compatible_models:
                        self.hf_compatible_models.append(model_id)
            
            # Add local models if any
            if self.local_models:
                all_models.append("--- Local Models ---")
                for model_path in self.local_models:
                    relative_path = os.path.relpath(model_path, self.model_directory)
                    display_name = relative_path.replace(os.sep, '/')
                    all_models.append(f"📁 {display_name}")
            
            # Create status message
            status = f"✅ Found {len(compatible_models)} compatible model(s)\n"
            if quantization_mode != "none":
                status += f"   Quantization: {quantization_mode} ({int(1/quant_factor)}x memory reduction)\n"
            status += f"   RAM: {ram_gb}GB (using {safe_ram:.1f}GB safely)\n"
            if use_gpu:
                status += f"   VRAM: {vram_gb}GB (using {safe_vram:.1f}GB safely)\n"
            status += f"   Excluded {incompatible_count} models (too large)\n\n"
            
            if compatible_models:
                status += "Smallest to largest:\n"
                for model_id, display_name, size in compatible_models[:5]:
                    status += f"  • {model_id} ({size:.1f}GB)\n"
                if len(compatible_models) > 5:
                    status += f"  ... and {len(compatible_models) - 5} more\n"
            else:
                status = "❌ No compatible models found with current settings.\n"
                status += "Try increasing RAM limit or disabling GPU mode."
            
            return status, gr.update(choices=all_models, value=all_models[0] if all_models else None)
            
        except Exception as e:
            return f"❌ Error filtering models: {str(e)}", gr.update(choices=self.available_models)

    def scan_model_directory(self, directory_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Recursively scan a directory for valid transformer models.
        
        Args:
            directory_path: Path to directory containing model folders
            
        Returns:
            Status message and dropdown update dict
        """
        if not directory_path:
            return "⚠️ No directory selected", gr.update(choices=self.available_models)
        
        try:
            directory_path = directory_path.strip()
            
            if not os.path.isdir(directory_path):
                return f"❌ Not a valid directory: {directory_path}", gr.update(choices=self.available_models)
            
            self.model_directory = directory_path
            self.local_models = []
            
            # Recursively walk through all subdirectories
            for root, dirs, files in os.walk(directory_path):
                # Check if current directory contains a valid model
                # (must have config.json at minimum)
                if "config.json" in files:
                    self.local_models.append(root)
                    # Don't descend into this directory's subdirectories
                    # since we found a model here
                    dirs.clear()
            
            # Sort models by path for easier navigation
            self.local_models.sort()
            
            # Combine HuggingFace models and local models
            all_models = self.available_models.copy()
            if self.local_models:
                all_models.append("--- Local Models ---")
                # Show relative path from base directory for clarity
                for model_path in self.local_models:
                    relative_path = os.path.relpath(model_path, directory_path)
                    # Use forward slashes for consistency across platforms
                    display_name = relative_path.replace(os.sep, '/')
                    all_models.append(f"📁 {display_name}")
            
            status = f"✅ Found {len(self.local_models)} model(s) in: {directory_path}"
            if len(self.local_models) == 0:
                status += "\n\n⚠️ No valid models found. Models must contain 'config.json'.\n"
                status += "Searched recursively through all subdirectories."
            else:
                status += f"\n\nModels found:\n" + "\n".join([f"  • {os.path.relpath(m, directory_path)}" for m in self.local_models[:10]])
                if len(self.local_models) > 10:
                    status += f"\n  ... and {len(self.local_models) - 10} more"
            
            return status, gr.update(choices=all_models, value=all_models[0])
            
        except Exception as e:
            return f"❌ Error scanning directory: {str(e)}", gr.update(choices=self.available_models)
    
    def get_model_path(self, model_selection: str) -> str:
        """
        Convert UI model selection to actual model path/name.
        
        Args:
            model_selection: Selected value from dropdown
            
        Returns:
            Full path for local models, or HuggingFace ID for online models
        """
        # Skip separators
        if "--- " in model_selection and " ---" in model_selection:
            return self.available_models[0]  # Default to first HF model
        
        # Handle HuggingFace models with metadata (e.g., "gpt2 [124M, 0.5GB] 💾 CPU")
        if "[" in model_selection and "]" in model_selection:
            # Extract model ID (everything before the bracket)
            model_id = model_selection.split("[")[0].strip()
            return model_id
        
        # Handle local model (prefixed with 📁)
        if model_selection.startswith("📁 "):
            relative_path = model_selection[2:]  # Remove emoji prefix
            # Convert forward slashes back to OS-specific separators
            relative_path = relative_path.replace('/', os.sep)
            
            # Find full path by matching relative path
            for local_path in self.local_models:
                if self.model_directory:
                    model_relative = os.path.relpath(local_path, self.model_directory)
                    if model_relative == relative_path:
                        return local_path
            
            # Fallback: try to construct path directly
            if self.model_directory:
                constructed_path = os.path.join(self.model_directory, relative_path)
                if os.path.exists(constructed_path):
                    return constructed_path
            
            return model_selection  # Last resort fallback
        
        # Plain HuggingFace model ID or other
        return model_selection

    def create_config_from_inputs(
        self,
        model_name: str,
        n_seeds: int,
        probes_per_seed: int,
        steps_per_probe: int,
        eps: float,
        layer_to_probe: int,
        n_modes: int,
        temperature: float,
        max_tokens: int,
        quantization_mode: str = "none"
    ) -> ARMConfig:
        """Create ARM config from interface inputs."""
        return ARMConfig(
            model_name=model_name,
            n_seeds=n_seeds,
            probes_per_seed=probes_per_seed,
            steps_per_probe=steps_per_probe,
            eps=eps,
            layer_to_probe=layer_to_probe,
            n_modes=n_modes,
            load_in_8bit=(quantization_mode == "8bit"),
            load_in_4bit=(quantization_mode == "4bit"),
            random_seed=42,  # For reproducibility
        )

    def analyze_prompts(
        self,
        model_name: str,
        prompts_text: str,
        n_seeds: int,
        probes_per_seed: int,
        steps_per_probe: int,
        eps: float,
        layer_to_probe: int,
        n_modes: int,
        quantization_mode: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, str, str, str]:
        """Run ARM analysis on input prompts."""

        # Prevent concurrent operations
        if self.is_processing:
            return "⚠️ Analysis already in progress. Please wait for it to complete.", "", "", "", ""
        
        self.is_processing = True
        
        try:
            progress(0.1, desc="⚙️ Initializing ARM configuration...")

            # Parse prompts
            prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
            if not prompts:
                return "❌ Error: No prompts provided", "", "", "", ""

            # Convert model selection to actual path/name
            actual_model_path = self.get_model_path(model_name)

            # Create configuration
            config = self.create_config_from_inputs(
                actual_model_path, n_seeds, probes_per_seed, steps_per_probe,
                eps, layer_to_probe, n_modes, 1.0, 50, quantization_mode
            )

            progress(0.2, desc=f"📥 Loading model: {model_name}")
            progress(0.25, desc="This may take 1-5 minutes on first use (downloading model)...")

            # Initialize ARM
            self.current_mapper = ARMMapper(config)
            self.current_config = config  # Store config for saving
            self.current_prompts = prompts  # Store prompts for saving

            progress(0.4, desc=f"🔬 Running ARM analysis on {len(prompts)} prompt(s)...")
            progress(0.5, desc="Generating directional probes...")

            # Run analysis
            results = self.current_mapper.map_latent_manifold(prompts)
            self.current_results = results

            progress(0.8, desc="📊 Analyzing resonance patterns...")
            progress(0.85, desc="🗺️ Computing topological features...")
            progress(0.9, desc="🎨 Generating visualizations...")

            # Create summary
            summary = self.create_analysis_summary(results, prompts, config)

            # Create plots
            resonance_plot = self.create_resonance_plot(results)
            topology_plot = self.create_topology_plot(results)
            descriptor_plot = self.create_descriptor_plot(results)

            progress(1.0, desc="✅ Analysis complete!")

            return (
                "✅ Analysis completed successfully!",
                summary,
                resonance_plot,
                topology_plot,
                descriptor_plot
            )

        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            return error_msg, "", "", "", ""
        
        finally:
            # Always reset processing flag
            self.is_processing = False

    def create_analysis_summary(self, results: Dict[str, Any],
                              prompts: List[str], config: ARMConfig) -> str:
        """Create a detailed summary of the ARM analysis."""

        summary_lines = [
            f"# ARM Analysis Summary",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Configuration",
            f"- **Model:** {config.model_name}",
            f"- **Seeds:** {config.n_seeds}",
            f"- **Probes per seed:** {config.probes_per_seed}",
            f"- **Steps per probe:** {config.steps_per_probe}",
            f"- **Epsilon:** {config.eps}",
            f"- **Layer:** {config.layer_to_probe}",
            f"- **Modes:** {config.n_modes}",
            "",
            f"## Input Prompts ({len(prompts)})",
            f"_(Index in brackets for steering)_",
        ]

        for i, prompt in enumerate(prompts):
            summary_lines.append(f"{i+1}. **[Index {i}]** `{prompt}`")

        summary_lines.extend([
            "",
            f"## Results",
            f"- **Descriptors shape:** {results['descriptors'].shape}",
            f"- **Seeds analyzed:** {len(results['seed_analyses'])}",
            "",
            f"### Resonance Signatures"
        ])

        # Show resonance stats for each seed
        for i, analysis in enumerate(results['seed_analyses']):
            sig = analysis['resonance_signature']
            summary_lines.extend([
                f"**Seed {i+1}** (`{prompts[i]}`):",
                f"  - Entropy: {sig['entropy']:.4f}",
                f"  - Participation Ratio: {sig['participation_ratio']:.4f}",
                f"  - Top singular values: {sig['singular_values'][:3]}",
                ""
            ])

        # Show topology information
        if 'clustering_data' in results:
            clusters = results['clustering_data']
            summary_lines.extend([
                f"### Topology Analysis",
                f"- **Clusters detected:** {clusters['n_clusters']}",
                f"- **Cluster sizes:** {clusters['cluster_sizes']}",
                ""
            ])

        return "\n".join(summary_lines)

    def create_resonance_plot(self, results: Dict[str, Any]) -> str:
        """Create resonance signature visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('ARM Resonance Analysis', fontsize=16)

        analyses = results['seed_analyses']

        # Entropy plot
        entropies = [a['resonance_signature']['entropy'] for a in analyses]
        axes[0, 0].bar(range(len(entropies)), entropies, color='skyblue')
        axes[0, 0].set_title('Resonance Entropy by Seed')
        axes[0, 0].set_xlabel('Seed Index')
        axes[0, 0].set_ylabel('Entropy')

        # Participation ratio plot
        ratios = [a['resonance_signature']['participation_ratio_normalized'] for a in analyses]
        axes[0, 1].bar(range(len(ratios)), ratios, color='lightgreen')
        axes[0, 1].set_title('Participation Ratio by Seed')
        axes[0, 1].set_xlabel('Seed Index')
        axes[0, 1].set_ylabel('Participation Ratio')

        # Singular values comparison
        n_modes = min(5, len(analyses[0]['resonance_signature']['singular_values']))
        singular_data = []
        for i, analysis in enumerate(analyses):
            sig = analysis['resonance_signature']
            singular_data.append(sig['singular_values'][:n_modes])

        singular_df = pd.DataFrame(singular_data).T
        singular_df.plot(kind='line', ax=axes[1, 0], marker='o')
        axes[1, 0].set_title(f'Top {n_modes} Singular Values')
        axes[1, 0].set_xlabel('Mode Index')
        axes[1, 0].set_ylabel('Singular Value')
        axes[1, 0].legend([f'Seed {i+1}' for i in range(len(analyses))])

        # Descriptor scatter plot (first 2 dimensions)
        descriptors = results['descriptors']
        if descriptors.shape[1] >= 2:
            axes[1, 1].scatter(descriptors[:, 0], descriptors[:, 1], alpha=0.7, s=50)
            axes[1, 1].set_title('Descriptor Space (First 2 Dimensions)')
            axes[1, 1].set_xlabel('Dimension 1')
            axes[1, 1].set_ylabel('Dimension 2')

            # Add seed labels
            for i, (x, y) in enumerate(zip(descriptors[:, 0], descriptors[:, 1])):
                axes[1, 1].annotate(f'Seed {i+1}', (x, y), xytext=(5, 5),
                                   textcoords='offset points')

        plt.tight_layout()

        # Save plot to output directory with unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plot_path = os.path.join(OUTPUT_DIR, f"resonance_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def create_topology_plot(self, results: Dict[str, Any]) -> str:
        """Create topology visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('ARM Topology Analysis', fontsize=16)

        # Spectral embedding plot
        if 'graph_data' in results and results['graph_data']['spectral_embedding'].shape[1] >= 2:
            embedding = results['graph_data']['spectral_embedding']
            axes[0].scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50, c='purple')
            axes[0].set_title('Spectral Embedding (First 2 Dimensions)')
            axes[0].set_xlabel('Dimension 1')
            axes[0].set_ylabel('Dimension 2')

            # Add seed labels
            for i, (x, y) in enumerate(zip(embedding[:, 0], embedding[:, 1])):
                axes[0].annotate(f'Seed {i+1}', (x, y), xytext=(5, 5),
                               textcoords='offset points')

        # Clustering results
        if 'clustering_data' in results:
            clusters = results['clustering_data']
            cluster_labels = clusters['cluster_labels']

            # Color by cluster
            colors = plt.cm.rainbow(np.linspace(0, 1, clusters['n_clusters']))
            for i, label in enumerate(cluster_labels):
                axes[1].scatter(i, label, c=[colors[label]], s=100, alpha=0.7)
                axes[1].annotate(f'Seed {i+1}', (i, label), xytext=(0, 10),
                               textcoords='offset points', ha='center')

            axes[1].set_title('Cluster Assignment by Seed')
            axes[1].set_xlabel('Seed Index')
            axes[1].set_ylabel('Cluster ID')
            axes[1].set_yticks(range(clusters['n_clusters']))
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot to output directory with unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plot_path = os.path.join(OUTPUT_DIR, f"topology_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def create_descriptor_plot(self, results: Dict[str, Any]) -> str:
        """Create descriptor space visualization."""
        descriptors = results['descriptors']

        if descriptors.shape[1] < 2:
            # Create a simple bar plot for single dimension
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(descriptors)), descriptors[:, 0])
            ax.set_title('Descriptor Values (Single Dimension)')
            ax.set_xlabel('Seed Index')
            ax.set_ylabel('Descriptor Value')
        else:
            # Create a heatmap for multi-dimensional descriptors
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(descriptors.T, aspect='auto', cmap='viridis')
            ax.set_title('Descriptor Space Heatmap')
            ax.set_xlabel('Seed Index')
            ax.set_ylabel('Descriptor Dimension')

            # Add colorbar
            plt.colorbar(im, ax=ax)

        plt.tight_layout()

        # Save plot to output directory with unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plot_path = os.path.join(OUTPUT_DIR, f"descriptor_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def generate_steered_text(
        self,
        target_prompt: str,
        temperature: float,
        max_tokens: int,
        steering_mode: str,
        positive_indices: str = "",
        negative_indices: str = "",
        target_signature_indices: str = "0",
        steering_strength: float = 1.0
    ) -> str:
        """Generate text with optional ARM steering."""

        if not self.current_mapper or not self.current_results:
            return "❌ Error: Please run ARM analysis first before generating steered text."

        try:
            if steering_mode == "none":
                # Generate baseline text without steering
                generator = self.current_mapper.create_controlled_generator()
                generated_text = generator.generate_with_steering(
                    target_prompt, max_tokens, temperature, do_sample=True
                )
                return f"📝 Baseline generation:\n\n{generated_text}"

            elif steering_mode == "control_vector":
                # Parse positive and negative indices
                try:
                    pos_indices = [int(x.strip()) for x in positive_indices.split(',') if x.strip()]
                    neg_indices = [int(x.strip()) for x in negative_indices.split(',') if x.strip()]
                except ValueError:
                    return "❌ Error: Positive and negative indices must be comma-separated integers (e.g., '0,1,2')"

                if not pos_indices or not neg_indices:
                    return "❌ Error: Need at least one positive and one negative example index"

                # Compute control vector
                control_vector = self.current_mapper.compute_steering_vector_from_manifold(
                    pos_indices, neg_indices
                )
                control_vector.coefficient = steering_strength

                # Generate with steering
                generator = self.current_mapper.create_controlled_generator()
                generator.set_control(control_vector)
                try:
                    generated_text = generator.generate_with_steering(
                        target_prompt, max_tokens, temperature, do_sample=True
                    )
                finally:
                    generator.clear_controls()

                return f"🎯 Control vector steering (strength: {steering_strength}):\n\nPositive examples: {pos_indices}\nNegative examples: {neg_indices}\n\n{generated_text}"

            elif steering_mode == "manifold_signature":
                # Steer toward blended resonance signature(s)
                import numpy as np
                
                # Parse target indices
                try:
                    indices = [int(x.strip()) for x in target_signature_indices.split(',') if x.strip()]
                except ValueError:
                    return "❌ Error: Target signature indices must be comma-separated integers (e.g., '0,1,2')"
                
                if not indices:
                    return "❌ Error: Need at least one target signature index"
                
                # Validate indices
                max_index = len(self.current_results['seed_analyses']) - 1
                for idx in indices:
                    if idx < 0 or idx > max_index:
                        return f"❌ Error: Index {idx} is out of range (0-{max_index})"
                
                # Collect and blend signatures
                signatures = []
                prompt_texts = []
                for idx in indices:
                    target_analysis = self.current_results['seed_analyses'][idx]
                    signatures.append(target_analysis['resonance_signature']['s_norm'])
                    prompt_texts.append(self.current_results['prompts'][idx])
                
                # Average signatures if multiple
                if len(signatures) > 1:
                    blended_signature = np.mean(signatures, axis=0)
                    signature_desc = f"Blended from {len(indices)} signatures:\n" + "\n".join([f"  [{i}] {t}" for i, t in zip(indices, prompt_texts)])
                else:
                    blended_signature = signatures[0]
                    signature_desc = f"Single signature from: [{indices[0]}] {prompt_texts[0]}"

                generated_text = self.current_mapper.steer_generation_toward_signature(
                    prompt=target_prompt,
                    target_signature=blended_signature,
                    max_length=max_tokens,
                    temperature=temperature,
                    steering_strength=steering_strength
                )

                return f"🌀 Manifold signature steering (strength: {steering_strength}):\n\n{signature_desc}\n\n{generated_text}"

            else:
                return f"❌ Error: Unknown steering mode '{steering_mode}'"

        except Exception as e:
            return f"❌ Text generation failed: {str(e)}"

    def save_results(self, format_type: str = "json") -> Tuple[str, str]:
        """Save current ARM results to file."""

        if not self.current_results or not self.current_config or not self.current_prompts:
            return "❌ No results to save. Please run an analysis first.", ""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arm_results_{timestamp}"

            # Prepare data for saving
            save_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "format_version": "1.0",
                    "arm_version": "1.0"
                },
                "configuration": self.current_config.to_dict(),
                "prompts": self.current_prompts,
                "results": self.convert_results_for_saving(self.current_results)
            }

            if format_type == "json":
                # Convert numpy arrays to lists for JSON
                json_str = json.dumps(save_data, indent=2, default=str)

                # Create download link
                b64_data = base64.b64encode(json_str.encode()).decode()
                href = f"data:text/json;base64,{b64_data}"

                filename += ".json"
                download_link = f'<a href="{href}" download="{filename}" style="color: #4CAF50; text-decoration: none; font-weight: bold;">📥 Click to Download {filename}</a>'
                return f"✅ Results saved as {filename}", download_link

            elif format_type == "pickle":
                # Save as pickle (preserves numpy arrays)
                pickle_data = pickle.dumps(save_data)

                b64_data = base64.b64encode(pickle_data).decode()
                href = f"data:application/octet-stream;base64,{b64_data}"

                filename += ".pkl"
                download_link = f'<a href="{href}" download="{filename}" style="color: #2196F3; text-decoration: none; font-weight: bold;">📥 Click to Download {filename}</a>'
                return f"✅ Results saved as {filename}", download_link

            else:
                return "❌ Unsupported format", ""

        except Exception as e:
            return f"❌ Save failed: {str(e)}", ""

    def load_results(self, file_obj) -> Tuple[str, str, str, str, str]:
        """Load ARM results from file."""

        if file_obj is None:
            return "❌ No file selected", "", "", "", ""

        try:
            # Determine file type from name
            filename = getattr(file_obj, 'name', 'unknown')
            if filename.endswith('.json'):
                data = json.loads(file_obj.read().decode('utf-8'))
            elif filename.endswith('.pkl') or filename.endswith('.pickle'):
                data = pickle.loads(file_obj.read())
            else:
                return "❌ Unsupported file format. Use .json or .pkl files.", "", "", "", ""

            # Validate data structure
            if not all(key in data for key in ['configuration', 'prompts', 'results']):
                return "❌ Invalid file format - missing required data", "", "", "", ""

            # Load configuration
            config_dict = data['configuration']
            self.current_config = ARMConfig.from_dict(config_dict)

            # Load prompts
            self.current_prompts = data['prompts']

            # Load results
            self.current_results = self.convert_results_from_saved(data['results'])
            
            # Restore prompts into results dict (required for text generation)
            self.current_results['prompts'] = self.current_prompts

            # Create summary
            summary = self.create_analysis_summary(
                self.current_results,
                self.current_prompts,
                self.current_config
            )

            # Create plots
            resonance_plot = self.create_resonance_plot(self.current_results)
            topology_plot = self.create_topology_plot(self.current_results)
            descriptor_plot = self.create_descriptor_plot(self.current_results)

            # Re-initialize mapper for text generation
            mapper_status = ""
            try:
                self.current_mapper = ARMMapper(self.current_config)
                mapper_status = f"\n\n✅ Model '{self.current_config.model_name}' loaded successfully for text generation."
            except Exception as e:
                # Mapper initialization might fail if model isn't available
                self.current_mapper = None
                mapper_status = f"\n\n⚠️ Warning: Could not load model '{self.current_config.model_name}' ({str(e)}). Visualizations available, but text generation disabled."

            return (
                f"✅ Results loaded successfully!{mapper_status}",
                summary,
                resonance_plot,
                topology_plot,
                descriptor_plot
            )

        except Exception as e:
            return f"❌ Load failed: {str(e)}", "", "", "", ""

    def load_prompt_file(self, file_input) -> str:
        """Load prompts from a text file."""
        if file_input is None:
            return "Please select a .txt file first using the file upload component above."

        try:
            # Handle different types of file input from Gradio
            content = None

            if hasattr(file_input, 'read'):
                # File-like object
                content = file_input.read().decode('utf-8')
            elif hasattr(file_input, 'name') and isinstance(file_input.name, str):
                # File path string - open and read the file
                with open(file_input.name, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif isinstance(file_input, str):
                # Direct file path string
                with open(file_input, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif isinstance(file_input, list) and len(file_input) > 0:
                # Sometimes Gradio returns a list
                first_item = file_input[0]
                if hasattr(first_item, 'name'):
                    with open(first_item.name, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif isinstance(first_item, str):
                    with open(first_item, 'r', encoding='utf-8') as f:
                        content = f.read()
            else:
                return f"❌ Unsupported file input type: {type(file_input)}. Please select a valid .txt file."

            if content is None:
                return "❌ Could not read file content. Please check the file and try again."

            # Parse prompts: skip comments (lines starting with #) and empty lines
            prompts = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)

            if not prompts:
                return "No valid prompts found in file. Make sure your .txt file contains prompts (one per line) that don't start with #."

            # Join prompts for the textbox
            prompt_text = '\n'.join(prompts)
            return f"✅ Successfully loaded {len(prompts)} prompts from file!\n\n{prompt_text}"

        except Exception as e:
            return f"❌ Failed to load prompt file: {str(e)}. Please check that the file is a valid .txt file."

    def convert_results_for_saving(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        save_results = {}

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                save_results[key] = value.tolist()
            elif isinstance(value, dict):
                save_results[key] = self.convert_results_for_saving(value)
            elif isinstance(value, list):
                save_results[key] = [
                    self.convert_results_for_saving(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                save_results[key] = value

        return save_results

    def convert_results_from_saved(self, saved_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert saved results back to working format."""
        results = {}

        for key, value in saved_results.items():
            if key == 'descriptors' and isinstance(value, list):
                results[key] = np.array(value)
            elif key == 'graph_data' and isinstance(value, dict):
                graph_data = value.copy()
                if 'feature_vectors' in graph_data:
                    graph_data['feature_vectors'] = np.array(graph_data['feature_vectors'])
                if 'spectral_embedding' in graph_data:
                    graph_data['spectral_embedding'] = np.array(graph_data['spectral_embedding'])
                if 'adjacency_matrix' in graph_data:
                    graph_data['adjacency_matrix'] = np.array(graph_data['adjacency_matrix'])
                results[key] = graph_data
            elif isinstance(value, dict):
                results[key] = self.convert_results_from_saved(value)
            else:
                results[key] = value

        return results


def create_gradio_interface():
    """Create the Gradio web interface."""

    arm_interface = ARMInterface()

    with gr.Blocks(title="ARM: Aproximal Resonance Mapping", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🔬 ARM: Aproximal Resonance Mapping

        **Explore and control AI latent manifolds through topological analysis**

        This interface allows you to:
        - Analyze how different prompts behave in AI latent space
        - Visualize resonance patterns and topological structures
        - Understand AI behavior through mathematical topology
        - Generate steered text (coming soon)
        """)

        with gr.Tab("ARM Analysis"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model Configuration")

                    model_dropdown = gr.Dropdown(
                        choices=arm_interface.available_models,
                        value="distilgpt2",
                        label="Model",
                        info="Choose the transformer model to analyze"
                    )

                    # HuggingFace model browser with compatibility check
                    with gr.Accordion("🤗 Browse Compatible HuggingFace Models", open=False):
                        gr.Markdown("""
                        **Find Models Compatible with Your System**
                        
                        Filter curated HuggingFace models based on your available RAM and VRAM.
                        Models are categorized by size and compatibility.
                        """)
                        
                        with gr.Row():
                            ram_input = gr.Slider(
                                minimum=1, maximum=128, value=32, step=1,
                                label="Available RAM (GB)",
                                info="Total system memory available"
                            )
                            vram_input = gr.Slider(
                                minimum=0, maximum=80, value=4, step=1,
                                label="Available VRAM (GB)",
                                info="GPU memory (0 if CPU-only)"
                            )
                        
                        use_gpu_checkbox = gr.Checkbox(
                            value=True,
                            label="Use GPU if available",
                            info="Try GPU first, fallback to CPU if needed"
                        )
                        
                        filter_models_btn = gr.Button(
                            "🔍 Find Compatible Models",
                            variant="primary",
                            size="sm"
                        )
                        
                        hf_filter_status = gr.Textbox(
                            label="Compatibility Results",
                            interactive=False,
                            lines=8
                        )

                    # Local model directory browser
                    with gr.Accordion("📁 Local Model Directory", open=False):
                        gr.Markdown("""
                        **Manage Local Models**
                        
                        Point to your models directory - the scanner will **recursively search** all subdirectories.
                        Models can be nested at any depth (e.g., `models/huggingface/my-model/`).
                        Each model needs a `config.json` file to be recognized.
                        """)
                        
                        model_dir_input = gr.Textbox(
                            label="Model Directory Path",
                            placeholder="e.g., C:\\Users\\YourName\\models or /home/user/models",
                            info="Enter or paste the path to your models directory"
                        )
                        
                        with gr.Row():
                            scan_dir_btn = gr.Button(
                                "🔍 Scan Directory",
                                variant="secondary",
                                size="sm"
                            )
                        
                        model_dir_status = gr.Textbox(
                            label="Scan Status",
                            interactive=False,
                            lines=2
                        )

                    with gr.Row():
                        n_seeds = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Number of Seeds",
                            info="How many prompts to analyze"
                        )
                        probes_per_seed = gr.Slider(
                            minimum=1, maximum=8, value=2, step=1,
                            label="Probes per Seed",
                            info="Directional probes around each prompt"
                        )

                    with gr.Row():
                        steps_per_probe = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Steps per Probe",
                            info="Resolution along each probe direction"
                        )
                        eps = gr.Slider(
                            minimum=0.001, maximum=0.5, value=0.03, step=0.01,
                            label="Epsilon (ε)",
                            info="Perturbation magnitude"
                        )

                    with gr.Row():
                        layer_to_probe = gr.Slider(
                            minimum=0, maximum=11, value=2, step=1,
                            label="Layer to Probe",
                            info="Transformer layer for analysis (0-11 for GPT-2)"
                        )
                        n_modes = gr.Slider(
                            minimum=1, maximum=8, value=3, step=1,
                            label="Resonance Modes",
                            info="Number of spectral modes to analyze"
                        )
                    
                    # Quantization settings
                    quantization_mode = gr.Radio(
                        choices=["none", "8bit", "4bit"],
                        value="none",
                        label="🔧 Quantization (Memory Saver)",
                        info="8-bit=4x less RAM, 4-bit=8x less RAM (needs: pip install bitsandbytes)"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Input Prompts")

                    prompts_input = gr.Textbox(
                        label="Prompts (one per line)",
                        lines=6,
                        placeholder="Enter prompts to analyze, one per line...\n\nExample:\nThe cat sat on the mat\nOnce upon a time\nIn the beginning",
                        value="The cat sat on the mat\nOnce upon a time\nIn the beginning"
                    )

                    # File upload for prompt files (auto-loads when selected)
                    prompt_file_input = gr.File(
                        label="Upload prompt file (.txt) - prompts auto-load when selected",
                        file_types=[".txt"],
                        file_count="single"
                    )

                    analyze_btn = gr.Button(
                        "🔬 Run ARM Analysis",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Status output directly under button for visibility
                    status_output = gr.Textbox(
                        label="🔄 Analysis Status",
                        interactive=False,
                        lines=3,
                        placeholder="Click 'Run ARM Analysis' to start. Progress will show here...",
                        show_label=True
                    )
                    
                    gr.Markdown("""
                    ⏱️ **Processing Time**: 30 seconds to several minutes depending on model size.
                    **Watch the status box above** - progress updates will appear there!
                    """, elem_classes=["warning-text"])

            with gr.Row():
                summary_output = gr.Markdown(
                    label="Analysis Summary",
                    height=300
                )

            with gr.Row():
                with gr.Column():
                    resonance_plot = gr.Image(label="Resonance Analysis")
                with gr.Column():
                    topology_plot = gr.Image(label="Topology Analysis")

            with gr.Row():
                descriptor_plot = gr.Image(label="Descriptor Space")

        with gr.Tab("Save/Load Results"):

            gr.Markdown("""
            ### Save & Load ARM Results

            Save your analysis results for sharing, reproducibility, and future reference.
            Load previously saved results to continue analysis or comparison.
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Save Current Results")

                    save_format = gr.Radio(
                        choices=["json", "pickle"],
                        value="json",
                        label="Save Format",
                        info="JSON: Human-readable, Pickle: Preserves all data types"
                    )

                    save_btn = gr.Button(
                        "💾 Save Results",
                        variant="secondary"
                    )

                    save_status = gr.Textbox(
                        label="Save Status",
                        interactive=False,
                        lines=2
                    )

                    save_download = gr.Markdown(
                        label="Download Link",
                        value="",
                        visible=True
                    )

                with gr.Column():
                    gr.Markdown("#### Load Previous Results")

                    load_file = gr.File(
                        label="Select ARM Results File",
                        file_types=[".json", ".pkl", ".pickle"],
                        file_count="single"
                    )

                    load_btn = gr.Button(
                        "📂 Load Results",
                        variant="secondary"
                    )

                    load_status = gr.Textbox(
                        label="Load Status",
                        interactive=False,
                        lines=2
                    )

            with gr.Row():
                gr.Markdown("""
                ### File Format Details

                **JSON Format (.json):**
                - Human-readable text format
                - Compatible with any programming language
                - Slightly larger file size
                - Good for sharing and version control

                **Pickle Format (.pkl):**
                - Python-specific binary format
                - Preserves exact numpy arrays and data types
                - Smaller file size
                - Faster loading
                - Only works with Python

                ### Usage Tips

                1. **Save after analysis** to preserve your results
                2. **Load saved results** to continue work without re-running analysis
                3. **Share JSON files** for collaboration
                4. **Use pickle for speed** when working locally
                5. **Include original prompts** when saving for context
                """)

        with gr.Tab("Text Generation"):

            gr.Markdown("""
            ### Generate Text with ARM Steering

            Generate text that follows behavioral patterns discovered by ARM analysis.
            Choose from different steering modes to control the generated output.
            """)

            with gr.Row():
                with gr.Column():
                    target_prompt = gr.Textbox(
                        label="Target Prompt",
                        placeholder="Enter a prompt to continue...",
                        value="The Jabberwock, with eyes of flame,"
                    )

                    steering_mode = gr.Radio(
                        choices=["none", "control_vector", "manifold_signature"],
                        value="none",
                        label="Steering Mode",
                        info="Type of steering to apply"
                    )

                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                            label="Temperature",
                            info="Controls randomness (higher = more random)"
                        )

                        max_tokens = gr.Slider(
                            minimum=10, maximum=200, value=50, step=10,
                            label="Max Tokens",
                            info="Maximum tokens to generate"
                        )

                    steering_strength = gr.Slider(
                        minimum=0.0, maximum=3.0, value=1.0, step=0.1,
                        label="Steering Strength",
                        info="How strongly to apply steering (0 = no steering)"
                    )

                    # Control vector steering options
                    with gr.Group(visible=False) as control_vector_group:
                        gr.Markdown("**Control Vector Steering**")
                        gr.Markdown("Specify seed indices (0-based) for positive and negative examples:")

                        positive_indices = gr.Textbox(
                            label="Positive Example Indices",
                            placeholder="e.g., 0,1,2",
                            info="Comma-separated indices of seeds to use as positive examples"
                        )

                        negative_indices = gr.Textbox(
                            label="Negative Example Indices",
                            placeholder="e.g., 3,4",
                            info="Comma-separated indices of seeds to use as negative examples"
                        )

                    # Manifold signature steering options
                    with gr.Group(visible=False) as manifold_group:
                        gr.Markdown("**Manifold Signature Steering**")
                        gr.Markdown("Steer toward resonance signature(s). Multiple indices will be blended:")

                        target_signature_indices = gr.Textbox(
                            label="Target Signature Indices",
                            placeholder="e.g., 0,1,2",
                            value="0",
                            info="Comma-separated indices of seeds whose signatures to blend and target"
                        )

                with gr.Column():
                    generate_btn = gr.Button(
                        "🎭 Generate Text",
                        variant="secondary",
                        size="lg"
                    )

                    generated_output = gr.Textbox(
                        label="Generated Text",
                        interactive=False,
                        lines=15,
                        placeholder="Generated text will appear here..."
                    )

                    gr.Markdown("""
                    ### Steering Modes

                    **None**: Standard generation without steering

                    **Control Vector**: Like RepE - steer away from negative examples toward positive examples
                    - Use comma-separated indices (e.g., "0,1,2" for positive, "3,4" for negative)
                    - Indices correspond to seed prompts (see Results tab for index mapping)

                    **Manifold Signature**: Steer toward resonance pattern(s) of analyzed prompt(s)
                    - Use single index (e.g., "0") or multiple comma-separated (e.g., "0,1,2")
                    - Multiple signatures are blended (averaged) together for combined steering
                    - Check Results tab to see which index corresponds to which prompt
                    """)

        with gr.Tab("Help & Documentation"):

            gr.Markdown("""
            ## ARM Parameter Guide

            ### Core Parameters
            - **Number of Seeds**: How many different prompts to analyze
            - **Probes per Seed**: How many random directions to explore around each prompt
            - **Steps per Probe**: Resolution along each probe direction (higher = more detailed)
            - **Epsilon (ε)**: How far to perturb from the original prompt (0.01-0.1 typical)

            ### Model Parameters
            - **Layer to Probe**: Which transformer layer to analyze (earlier layers = more syntax, later = more semantics)
            - **Resonance Modes**: How many spectral components to analyze

            ### Understanding Results
            - **Resonance Entropy**: How "complex" the activation patterns are (higher = more diverse)
            - **Participation Ratio**: How concentrated the spectral energy is (higher = more uniform spread)
            - **Topology Clusters**: Groups of prompts with similar latent space behavior
            - **Descriptor Space**: Mathematical representation of each prompt's behavioral signature

            ### Tips for Good Results
            1. Start with 3-5 related prompts to see meaningful patterns
            2. Try different epsilon values (0.01, 0.03, 0.1) to see sensitivity
            3. Experiment with different layers to understand hierarchical behavior
            4. Use prompts from the same domain for more coherent topology
            5. **Wait for progress bars** - Don't click buttons multiple times!
            6. **Watch the Status box** for feedback during long operations
            7. Model downloads happen automatically on first use (can take 1-5 minutes)

            ### Quantization (Memory Optimization) - NEW!
            
            **What is Quantization?**
            Reduces model memory usage by using lower precision numbers:
            - **8-bit**: 4x memory reduction (recommended)
            - **4-bit**: 8x memory reduction (more aggressive)
            - **none**: Full precision (no reduction)
            
            **How to Use:**
            1. Select quantization mode in "🔧 Quantization" radio buttons
            2. Install bitsandbytes: `pip install bitsandbytes`
            3. Run analysis normally - model loads with reduced memory
            
            **Benefits:**
            - Run larger models on your hardware
            - Faster loading times
            - Same ARM analysis quality
            - No code changes needed
            
            **Example:**
            - GPT-2 Medium (1.5GB) → 0.4GB with 8-bit
            - Can fit GPT-J-6B (24GB) → 6GB with 4-bit!
            
            ### Finding Compatible Models
            
            **HuggingFace Model Browser:**
            1. Click "🤗 Browse Compatible HuggingFace Models" to expand
            2. Set your system specs:
               - **RAM**: Your system memory (default: 32GB)
               - **VRAM**: Your GPU memory (default: 4GB, or 0 for CPU-only)
               - **Use GPU**: Check if you want to use GPU when possible
               - **Quantization**: Select quantization mode to see MORE compatible models!
            3. Click "🔍 Find Compatible Models"
            4. Compatible models will be added to the dropdown with:
               - Model name and organization
               - Parameter count (e.g., 124M, 1.3B)
               - Size in GB (adjusted for quantization if selected)
               - Quantization mode [8-bit] or [4-bit] if enabled
               - Where it will run: 🎮 GPU or 💾 CPU
            
            **How It Works:**
            - Estimates memory requirements (model size × 2 for ARM operations)
            - Applies quantization reduction (4x for 8-bit, 8x for 4-bit)
            - Filters models that fit in your available memory
            - Prioritizes GPU if available, falls back to CPU
            - Uses 80% of available memory for safety
            - Shows smallest models first
            
            **Example Output (with 8-bit quantization):**
            ```
            distilgpt2 [82M, 0.1GB [8-bit]] 🎮 GPU
            gpt2 [124M, 0.1GB [8-bit]] 🎮 GPU
            gpt2-medium [355M, 0.4GB [8-bit]] 🎮 GPU
            gpt2-large [774M, 0.8GB [8-bit]] 💾 CPU
            EleutherAI/pythia-1.4b [1.4B, 1.4GB [8-bit]] 💾 CPU
            ```
            
            ### Using Local Models
            
            **Local Model Directory Feature:**
            1. Click "📁 Local Model Directory" to expand the section
            2. Enter or paste the path to your models directory
            3. Click "🔍 Scan Directory" to search for models **recursively**
            4. Local models will appear in the Model dropdown with a 📁 icon and their path
            
            **How Scanning Works:**
            - **Recursive search**: Searches through ALL subdirectories automatically
            - Finds models at any depth in your directory structure
            - Each model identified by presence of `config.json`
            - Displays relative path from base directory (e.g., `📁 huggingface/my-model`)
            
            **Requirements for Local Models:**
            - Must contain at least a `config.json` file
            - Should also have model weights (`pytorch_model.bin` or `.safetensors`)
            - Compatible with HuggingFace transformers format
            
            **Example Directory Structure:**
            ```
            C:\\Users\\YourName\\models\\
            ├── huggingface\\
            │   ├── my-finetuned-gpt2\\
            │   │   ├── config.json
            │   │   ├── pytorch_model.bin
            │   │   └── tokenizer files...
            │   └── another-model\\
            │       ├── config.json
            │       └── model.safetensors
            ├── custom\\
            │   └── experimental-v3\\
            │       ├── config.json
            │       └── pytorch_model.bin
            ```
            All models will be found regardless of nesting!

            ### Current Limitations
            - GPU acceleration requires CUDA installation
            - Very large models (>1B parameters) may be slow
            - Topology analysis works best with 3+ prompts
            """)

        # Event handlers
        # Auto-load prompts when file is selected
        prompt_file_input.change(
            fn=arm_interface.load_prompt_file,
            inputs=[prompt_file_input],
            outputs=[prompts_input]
        )

        # Filter HuggingFace models by compatibility
        filter_models_btn.click(
            fn=arm_interface.filter_compatible_models,
            inputs=[ram_input, vram_input, use_gpu_checkbox, quantization_mode],
            outputs=[hf_filter_status, model_dropdown]
        )

        # Scan local model directory
        scan_dir_btn.click(
            fn=arm_interface.scan_model_directory,
            inputs=[model_dir_input],
            outputs=[model_dir_status, model_dropdown]
        )

        analyze_btn.click(
            fn=arm_interface.analyze_prompts,
            inputs=[
                model_dropdown, prompts_input, n_seeds, probes_per_seed,
                steps_per_probe, eps, layer_to_probe, n_modes, quantization_mode
            ],
            outputs=[status_output, summary_output, resonance_plot, topology_plot, descriptor_plot]
        )

        # Show/hide steering options based on mode
        def update_steering_visibility(mode):
            if mode == "control_vector":
                return [gr.Group(visible=True), gr.Group(visible=False)]
            elif mode == "manifold_signature":
                return [gr.Group(visible=False), gr.Group(visible=True)]
            else:
                return [gr.Group(visible=False), gr.Group(visible=False)]

        steering_mode.change(
            fn=update_steering_visibility,
            inputs=[steering_mode],
            outputs=[control_vector_group, manifold_group]
        )

        generate_btn.click(
            fn=arm_interface.generate_steered_text,
            inputs=[
                target_prompt, temperature, max_tokens, steering_mode,
                positive_indices, negative_indices, target_signature_indices, steering_strength
            ],
            outputs=[generated_output]
        )

        # Save/Load event handlers
        save_btn.click(
            fn=arm_interface.save_results,
            inputs=[save_format],
            outputs=[save_status, save_download]
        )

        load_btn.click(
            fn=arm_interface.load_results,
            inputs=[load_file],
            outputs=[load_status, summary_output, resonance_plot, topology_plot, descriptor_plot]
        )

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True
    )
