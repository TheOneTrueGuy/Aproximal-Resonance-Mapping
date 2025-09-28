#!/usr/bin/env python3
"""
Example usage of the modular ARM (Aproximal Resonance Mapping) library.

This script demonstrates how to use the new modular ARM library for
analyzing transformer latent manifolds.
"""

import numpy as np
from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


def progress_callback(current, total, message):
    """Progress callback for long-running operations."""
    print(f"[{current}/{total}] {message}")


def main():
    """Main example demonstrating ARM usage."""

    print("🔬 ARM (Aproximal Resonance Mapping) Library Example")
    print("=" * 60)

    # 1. Configure ARM parameters
    print("\n1. Configuring ARM parameters...")
    config = ARMConfig(
        model_name="distilgpt2",  # Small, efficient model for examples
        n_seeds=5,                # Number of seed points to analyze
        probes_per_seed=8,        # Directional probes per seed
        steps_per_probe=5,        # Steps along each probe direction
        eps=0.03,                 # Perturbation magnitude
        layer_to_probe=3,         # Transformer layer to analyze
        n_modes=6,                # Number of resonance modes to track
        random_seed=42,           # For reproducible results
    )

    print(f"Model: {config.model_name}")
    print(f"Seeds: {config.n_seeds}, Probes: {config.probes_per_seed}, Steps: {config.steps_per_probe}")
    print(f"Layer: {config.layer_to_probe}, Modes: {config.n_modes}")

    # 2. Initialize ARM mapper
    print("\n2. Initializing ARM mapper...")
    try:
        arm_mapper = ARMMapper(config)
        print("✓ ARM mapper initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize ARM mapper: {e}")
        return

    # 3. Define seed prompts for manifold exploration
    seed_prompts = [
        "The cat sat on the mat",
        "Machine learning is powerful",
        "The weather is beautiful today",
        "I enjoy programming",
        "Natural language processing advances rapidly",
    ]

    print(f"\n3. Analyzing {len(seed_prompts)} seed prompts...")

    # 4. Map the latent manifold
    try:
        manifold_result = arm_mapper.map_latent_manifold(
            seed_prompts,
            progress_callback=progress_callback
        )
        print("✓ Manifold mapping completed successfully")
    except Exception as e:
        print(f"✗ Failed to map manifold: {e}")
        return

    # 5. Analyze results
    print("\n4. Analyzing results...")

    descriptors = manifold_result['descriptors']
    seed_analyses = manifold_result['seed_analyses']

    print(f"Descriptor matrix shape: {descriptors.shape}")
    print(f"Number of seed analyses: {len(seed_analyses)}")

    # 6. Show resonance signatures for each seed
    print("\n5. Resonance signatures:")
    for i, analysis in enumerate(seed_analyses):
        sig = analysis['resonance_signature']
        print(f"  Seed {i+1} ({analysis['prompt'][:30]}...):")
        print(f"    Entropy: {sig['entropy']:.3f}")
        print(f"    Participation ratio: {sig['participation_ratio']:.3f}")
        print(f"    Top 3 singular values: {sig['singular_values'][:3]}")

    # 7. Demonstrate seed similarity search
    print("\n6. Finding similar seeds...")
    target_prompt = "The cat sat on the mat"
    target_analysis = next(a for a in seed_analyses if a['prompt'] == target_prompt)
    target_signature = target_analysis['resonance_signature']

    similar_seeds = arm_mapper.find_similar_seeds(
        target_signature,
        seed_analyses,
        top_k=3
    )

    print(f"Seeds most similar to: '{target_prompt}'")
    for rank, (seed_idx, similarity) in enumerate(similar_seeds, 1):
        seed_prompt = seed_analyses[seed_idx]['prompt']
        print(".3f")

    # 8. Show manifold statistics
    print("\n7. Manifold statistics:")
    print(f"  Seeds analyzed: {manifold_result['n_seeds']}")
    print(f"  Graph nodes: {manifold_result['graph_data']['n_nodes']}")

    if 'clustering_data' in manifold_result:
        clusters = manifold_result['clustering_data']
        print(f"  Detected clusters: {clusters['n_clusters']}")
        print(f"  Cluster sizes: {clusters['cluster_sizes']}")

    print("\n✓ ARM analysis completed successfully!")
    print("\nNext steps:")
    print("- Try different seed prompts to explore different regions")
    print("- Adjust eps, probes_per_seed, or layer_to_probe for different analysis depths")
    print("- Use the resonance signatures for control or classification tasks")


if __name__ == "__main__":
    main()
