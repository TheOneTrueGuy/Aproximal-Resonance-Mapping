#!/usr/bin/env python3
"""
Refactored ARM Transformer implementation using the new modular library.

This script demonstrates the original ARM functionality using the new
modular architecture for better maintainability and testing.
"""

import numpy as np
from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


def main():
    """Main function demonstrating the refactored ARM implementation."""

    print("🔬 ARM Transformer Analysis (Refactored)")
    print("=" * 50)

    # Use the same configuration as the original script
    config = ARMConfig(
        model_name="distilgpt2",   # small, efficient; switch to "gpt2" if you prefer
        n_seeds=10,                # Reduced from 200 for demonstration
        probes_per_seed=16,
        steps_per_probe=9,
        eps=0.03,                  # perturbation magnitude (relative to hidden vector norm)
        layer_to_probe=6,          # index of transformer block to inject perturbations (0-based)
        n_modes=8,                 # number of modes for resonance analysis
        random_seed=42,            # for reproducible results
    )

    print(f"Configuration: {config.n_seeds} seeds, {config.probes_per_seed} probes, {config.steps_per_probe} steps")
    print(f"Model: {config.model_name}, Layer: {config.layer_to_probe}, Eps: {config.eps}")

    # Initialize ARM mapper
    print("\nInitializing ARM mapper...")
    arm_mapper = ARMMapper(config)

    # Define seed prompts (can be expanded to 200+ for full analysis)
    seed_prompts = [
        "The cat sat on the mat",
        "Machine learning is fascinating",
        "The weather is beautiful",
        "I love programming",
        "Natural language processing",
        "Artificial intelligence advances",
        "The future of technology",
        "Deep learning models",
        "Computer vision systems",
        "Large language models",
    ]

    print(f"\nAnalyzing {len(seed_prompts)} seed prompts...")

    # Progress callback
    def progress_callback(current, total, message):
        print(f"[{current}/{total}] {message}")

    # Map the latent manifold
    manifold_result = arm_mapper.map_latent_manifold(
        seed_prompts,
        progress_callback=progress_callback
    )

    # Extract results
    descriptors = manifold_result['descriptors']
    seed_analyses = manifold_result['seed_analyses']

    print("
Results:")
    print(f"  Descriptor matrix shape: {descriptors.shape}")
    print(f"  Seeds analyzed: {len(seed_analyses)}")

    # Show some resonance statistics
    print("
Resonance Analysis Summary:")
    entropies = [analysis['resonance_signature']['entropy'] for analysis in seed_analyses]
    participation_ratios = [analysis['resonance_signature']['participation_ratio']
                           for analysis in seed_analyses]

    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    # Demonstrate seed similarity
    print("
Seed Similarity Analysis:")
    if len(seed_analyses) >= 2:
        target_sig = seed_analyses[0]['resonance_signature']
        similar_seeds = arm_mapper.find_similar_seeds(target_sig, seed_analyses, top_k=3)

        print(f"Seeds most similar to: '{seed_analyses[0]['prompt']}'")
        for rank, (seed_idx, similarity) in enumerate(similar_seeds, 1):
            seed_prompt = seed_analyses[seed_idx]['prompt']
            print(".3f")

    # Show manifold topology information
    if 'clustering_data' in manifold_result:
        clusters = manifold_result['clustering_data']
        print("
Topological Clustering:")
        print(f"  Number of clusters: {clusters['n_clusters']}")
        print(f"  Cluster sizes: {clusters['cluster_sizes']}")

    print("
✓ Analysis completed successfully!")
    print("\nThe modular ARM library provides:")
    print("- Clean separation of concerns")
    print("- Comprehensive unit testing")
    print("- Easy configuration management")
    print("- Extensible architecture for new models/features")


if __name__ == "__main__":
    main()
