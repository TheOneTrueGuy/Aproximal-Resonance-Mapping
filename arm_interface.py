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
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
import json
import pickle
import base64
import io

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig
from arm_library.interfaces.model_interface import TransformerModelInterface, ModelConfig

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class ARMInterface:
    """Interactive interface for ARM operations."""

    def __init__(self):
        self.current_mapper = None
        self.current_results = None
        self.current_config = None
        self.current_prompts = None
        self.available_models = [
            "distilgpt2",  # Small, fast
            "gpt2",        # Medium size
            "gpt2-medium", # Larger
        ]

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
        max_tokens: int
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
        progress=gr.Progress()
    ) -> Tuple[str, str, str, str, str]:
        """Run ARM analysis on input prompts."""

        progress(0.1, "Initializing ARM configuration...")

        # Parse prompts
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        if not prompts:
            return "❌ Error: No prompts provided", "", "", "", ""

        # Create configuration
        config = self.create_config_from_inputs(
            model_name, n_seeds, probes_per_seed, steps_per_probe,
            eps, layer_to_probe, n_modes, 1.0, 50
        )

        progress(0.3, f"Loading model {model_name}...")

        try:
            # Initialize ARM
            self.current_mapper = ARMMapper(config)

            progress(0.6, f"Analyzing {len(prompts)} prompts...")

            # Run analysis
            results = self.current_mapper.map_latent_manifold(prompts)
            self.current_results = results

            progress(0.9, "Generating visualizations...")

            # Create summary
            summary = self.create_analysis_summary(results, prompts, config)

            # Create plots
            resonance_plot = self.create_resonance_plot(results)
            topology_plot = self.create_topology_plot(results)
            descriptor_plot = self.create_descriptor_plot(results)

            progress(1.0, "Analysis complete!")

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
        ]

        for i, prompt in enumerate(prompts, 1):
            summary_lines.append(f"{i}. `{prompt}`")

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

        # Save plot
        plot_path = "resonance_analysis.png"
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

        # Save plot
        plot_path = "topology_analysis.png"
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

        # Save plot
        plot_path = "descriptor_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def generate_steered_text(
        self,
        target_prompt: str,
        temperature: float,
        max_tokens: int,
        use_arm_steering: bool
    ) -> str:
        """Generate text with optional ARM steering."""

        if not self.current_mapper or not self.current_results:
            return "❌ Error: Please run ARM analysis first before generating steered text."

        try:
            # Get the model interface from the mapper
            model_interface = self.current_mapper.model_interface

            if use_arm_steering:
                # This would require implementing actual steering logic
                # For now, just generate normally but note that steering isn't implemented yet
                result = f"⚠️ ARM steering not yet implemented in this interface.\n\nGenerating baseline text instead:\n\n"
            else:
                result = "Generating baseline text:\n\n"

            # Generate baseline text
            inputs = model_interface.encode_prompt(target_prompt)
            outputs = model_interface.model.generate(
                inputs[0],
                max_length=len(inputs[0][0]) + max_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                pad_token_id=model_interface.tokenizer.pad_token_id,
                eos_token_id=model_interface.tokenizer.eos_token_id,
            )

            generated_text = model_interface.tokenizer.decode(
                outputs[0][len(inputs[0][0]):],
                skip_special_tokens=True
            ).strip()

            result += generated_text

            return result

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
            try:
                self.current_mapper = ARMMapper(self.current_config)
            except Exception:
                # Mapper initialization might fail if model isn't available
                pass

            return (
                "✅ Results loaded successfully!",
                summary,
                resonance_plot,
                topology_plot,
                descriptor_plot
            )

        except Exception as e:
            return f"❌ Load failed: {str(e)}", "", "", "", ""

    def load_prompt_file(self, file_obj) -> str:
        """Load prompts from a text file."""
        if file_obj is None:
            return "Enter prompts manually or upload a prompt file"

        try:
            content = file_obj.read().decode('utf-8')

            # Parse prompts: skip comments (lines starting with #) and empty lines
            prompts = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)

            if not prompts:
                return "No valid prompts found in file. Prompts should be one per line, not starting with #"

            # Join prompts for the textbox
            prompt_text = '\n'.join(prompts)

            return prompt_text

        except Exception as e:
            return f"❌ Failed to load prompt file: {str(e)}"

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

                with gr.Column(scale=2):
                    gr.Markdown("### Input Prompts")

                    prompts_input = gr.Textbox(
                        label="Prompts (one per line)",
                        lines=6,
                        placeholder="Enter prompts to analyze, one per line...\n\nExample:\nThe cat sat on the mat\nOnce upon a time\nIn the beginning",
                        value="The cat sat on the mat\nOnce upon a time\nIn the beginning"
                    )

                    # File upload for prompt files
                    with gr.Row():
                        prompt_file_input = gr.File(
                            label="Or upload prompt file (.txt)",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        load_prompts_btn = gr.Button(
                            "📂 Load Prompts from File",
                            variant="secondary",
                            size="sm"
                        )

                    analyze_btn = gr.Button(
                        "🔬 Run ARM Analysis",
                        variant="primary",
                        size="lg"
                    )

            with gr.Row():
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )

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

            This feature allows you to generate text that follows the behavioral patterns
            discovered by ARM analysis. (Steering implementation coming soon)
            """)

            with gr.Row():
                with gr.Column():
                    target_prompt = gr.Textbox(
                        label="Target Prompt",
                        placeholder="Enter a prompt to continue...",
                        value="The Jabberwock, with eyes of flame,"
                    )

                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="Temperature",
                        info="Controls randomness (higher = more random)"
                    )

                    max_tokens = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10,
                        label="Max Tokens",
                        info="Maximum tokens to generate (0 = until stop token)"
                    )

                    use_steering = gr.Checkbox(
                        label="Use ARM Steering",
                        value=False,
                        info="Apply ARM-discovered patterns to guide generation (not yet implemented)"
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
                        lines=12,
                        placeholder="Generated text will appear here..."
                    )

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

            ### Current Limitations
            - Text generation steering not yet implemented
            - GPU acceleration requires CUDA installation
            - Very large models (>1B parameters) may be slow
            - Topology analysis works best with 3+ prompts
            """)

        # Event handlers
        load_prompts_btn.click(
            fn=arm_interface.load_prompt_file,
            inputs=[prompt_file_input],
            outputs=[prompts_input]
        )

        analyze_btn.click(
            fn=arm_interface.analyze_prompts,
            inputs=[
                model_dropdown, prompts_input, n_seeds, probes_per_seed,
                steps_per_probe, eps, layer_to_probe, n_modes
            ],
            outputs=[status_output, summary_output, resonance_plot, topology_plot, descriptor_plot]
        )

        generate_btn.click(
            fn=arm_interface.generate_steered_text,
            inputs=[target_prompt, temperature, max_tokens, use_steering],
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
