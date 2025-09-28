#!/usr/bin/env python3
"""
Demo script showing how to save and load ARM results programmatically.

This demonstrates the save/load functionality for research reproducibility.
"""

import json
import pickle
from datetime import datetime

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


def demo_save_load():
    """Demonstrate saving and loading ARM results."""

    print("💾 ARM Save/Load Demo")
    print("=" * 50)

    # Create a simple configuration
    config = ARMConfig(
        model_name="distilgpt2",
        n_seeds=2,
        probes_per_seed=2,
        steps_per_probe=2,
        eps=0.03,
        layer_to_probe=1,
        n_modes=3,
        random_seed=42,
    )

    # Sample prompts
    prompts = ["Hello world", "How are you"]

    print("Running ARM analysis...")
    try:
        mapper = ARMMapper(config)
        results = mapper.map_latent_manifold(prompts)

        print("✅ Analysis complete")
        print(f"Results keys: {list(results.keys())}")

        # Prepare data for saving
        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "format_version": "1.0",
                "arm_version": "1.0"
            },
            "configuration": config.to_dict(),
            "prompts": prompts,
            "results": convert_results_for_saving(results)
        }

        # Save as JSON
        json_filename = "demo_results.json"
        with open(json_filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"💾 Saved as JSON: {json_filename}")

        # Save as Pickle
        pickle_filename = "demo_results.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"💾 Saved as Pickle: {pickle_filename}")

        # Load from JSON
        print("\n📂 Loading from JSON...")
        with open(json_filename, 'r') as f:
            loaded_json = json.load(f)

        loaded_config = ARMConfig.from_dict(loaded_json['configuration'])
        loaded_prompts = loaded_json['prompts']
        loaded_results = convert_results_from_saved(loaded_json['results'])

        print("✅ JSON load successful")
        print(f"  Config model: {loaded_config.model_name}")
        print(f"  Prompts: {loaded_prompts}")
        print(f"  Results keys: {list(loaded_results.keys())}")

        # Load from Pickle
        print("\n📂 Loading from Pickle...")
        with open(pickle_filename, 'rb') as f:
            loaded_pickle = pickle.load(f)

        print("✅ Pickle load successful")
        print(f"  Same config: {loaded_pickle['configuration']['model_name'] == loaded_json['configuration']['model_name']}")
        print(f"  Same prompts: {loaded_pickle['prompts'] == loaded_json['prompts']}")

        # Compare file sizes
        import os
        json_size = os.path.getsize(json_filename)
        pickle_size = os.path.getsize(pickle_filename)

        print("\n📊 File Size Comparison:")
        print(f"  JSON: {json_size} bytes ({json_size/1024:.1f} KB)")
        print(f"  Pickle: {pickle_size} bytes ({pickle_size/1024:.1f} KB)")
        print(f"  Ratio: {pickle_size/json_size:.2f}x smaller")

        print("\n🎉 Demo completed successfully!")
        print("💡 Use these files to test the web interface load functionality")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def convert_results_for_saving(results):
    """Convert results to JSON-serializable format."""
    import numpy as np

    save_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            save_results[key] = value.tolist()
        elif isinstance(value, dict):
            save_results[key] = convert_results_for_saving(value)
        elif isinstance(value, list):
            save_results[key] = [
                convert_results_for_saving(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            save_results[key] = value
    return save_results


def convert_results_from_saved(saved_results):
    """Convert saved results back to working format."""
    import numpy as np

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
            results[key] = convert_results_from_saved(value)
        else:
            results[key] = value
    return results


if __name__ == "__main__":
    demo_save_load()
