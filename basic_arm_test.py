#!/usr/bin/env python3
"""
Basic ARM functionality test for your hardware setup.

This script tests the minimal ARM pipeline to ensure it works on your
Quadro K2200 with 4GB VRAM and 32GB RAM.
"""

import torch
import sys
from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


def test_hardware_setup():
    """Test basic hardware and library setup."""
    print("🔧 Testing Hardware Setup")
    print("-" * 30)

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Running on CPU")

    # Check system memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1024**3:.1f} GB")
    print("✅ Hardware check complete")


def test_basic_arm():
    """Test basic ARM functionality with minimal configuration."""
    print("\n🔬 Testing Basic ARM Functionality")
    print("-" * 40)

    try:
    # Minimal configuration for testing
    config = ARMConfig(
        model_name="distilgpt2",  # Small model
        n_seeds=3,                # Need at least 3 seeds for topology
        probes_per_seed=2,        # Minimal probes
        steps_per_probe=2,        # Minimal steps
        eps=0.01,                 # Small perturbation
        layer_to_probe=1,         # Early layer
        n_modes=2,                # Few modes
        topology_neighbors=2,     # Match number of seeds
        random_seed=42,
    )

        print("Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  Seeds: {config.n_seeds}, Probes: {config.probes_per_seed}, Steps: {config.steps_per_probe}")
        print(f"  Device: {config.device}")

    # Test prompts
    test_prompts = ["The cat sat on", "Once upon a time", "In the beginning"]
    print(f"  Test prompts: {test_prompts}")

        # Initialize ARM
        print("Initializing ARM mapper...")
        arm_mapper = ARMMapper(config)
        print("✅ ARM mapper initialized")

        # Run analysis
        print("Running ARM analysis...")
        result = arm_mapper.map_latent_manifold(test_prompts)
        print("✅ ARM analysis completed")

        # Check results
        descriptors = result['descriptors']
        analyses = result['seed_analyses']

        print("Results:")
        print(f"  Descriptors shape: {descriptors.shape}")
        print(f"  Number of analyses: {len(analyses)}")

        if len(analyses) > 0:
            sig = analyses[0]['resonance_signature']
            print("  Resonance signature sample:")
            print(f"    Entropy: {sig['entropy']:.4f}")
            print(f"    Participation ratio: {sig['participation_ratio']:.4f}")
            print(f"    Singular values: {sig['singular_values'][:3]}")

        return True

    except Exception as e:
        print(f"❌ ARM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test memory usage during ARM operations."""
    print("\n📊 Testing Memory Usage")
    print("-" * 25)

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024**2  # MB
        print(f"Initial memory: {initial_memory:.1f} MB")
        # Run a small ARM test
        config = ARMConfig(
            model_name="distilgpt2",
            n_seeds=3,
            probes_per_seed=2,
            steps_per_probe=2,
            layer_to_probe=1,
            n_modes=2,
            topology_neighbors=2,
        )

        arm_mapper = ARMMapper(config)
        result = arm_mapper.map_latent_manifold(["Test one", "Test two", "Test three"])

        final_memory = process.memory_info().rss / 1024**2  # MB
        memory_used = final_memory - initial_memory

        print(f"Memory used: {memory_used:.1f} MB")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"GPU memory used: {gpu_memory:.1f} MB")
        print("✅ Memory test completed")

    except ImportError:
        print("⚠️  psutil not available for memory testing")
    except Exception as e:
        print(f"❌ Memory test failed: {e}")


def main():
    """Run all basic tests."""
    print("🧪 ARM Basic Functionality Test Suite")
    print("=" * 50)
    print("Testing ARM on Quadro K2200 (4GB VRAM, 32GB RAM)")
    print()

    # Test hardware
    test_hardware_setup()

    # Test basic ARM
    arm_success = test_basic_arm()

    # Test memory
    test_memory_usage()

    # Summary
    print("\n" + "=" * 50)
    if arm_success:
        print("🎉 SUCCESS: ARM is working on your hardware!")
        print()
        print("Next steps:")
        print("1. Run hyperparameter_optimization.py for parameter tuning")
        print("2. Run style_preservation_test.py for style control testing")
        print("3. Gradually increase complexity (more seeds, probes, steps)")
        print("4. Experiment with different layers and eps values")
    else:
        print("❌ FAILURE: ARM is not working on your hardware")
        print()
        print("Troubleshooting:")
        print("1. Check PyTorch installation: pip install torch")
        print("2. Check transformers: pip install transformers")
        print("3. Try CPU-only: modify config to use device='cpu'")
        print("4. Reduce model size: use even smaller models if available")
        print("5. Check memory: close other applications")


if __name__ == "__main__":
    main()
