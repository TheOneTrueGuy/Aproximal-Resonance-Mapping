#!/usr/bin/env python3
"""
Test script for ARM steering functionality.

Tests the new steering capabilities with Jabberwocky manifold examples.
"""

import sys
import os
sys.path.append('.')

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig
from arm_library.core.steering import ARMControlVectorComputer

def test_steering_basic():
    """Test basic steering functionality."""
    print("🧪 Testing ARM Steering Functionality")
    print("=" * 50)

    # Load Jabberwocky prompts
    try:
        with open('test-data/jwok_prompts.txt', 'r') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("❌ Jabberwocky prompts file not found. Using sample prompts.")
        prompts = [
            "Twas brillig, and the slithy toves",
            "Did gyre and gimble in the wabe",
            "All mimsy were the borogoves",
            "And the mome raths outgrabe",
            "Beware the Jabberwock, my son",
            "The jaws that bite, the claws that catch"
        ]

    print(f"📚 Loaded {len(prompts)} prompts")
    print("Sample prompts:")
    for i, prompt in enumerate(prompts[:3]):
        print(f"  {i}: {prompt}")
    print("  ...")

    # Create ARM configuration
    config = ARMConfig(
        model_name="distilgpt2",
        n_seeds=min(len(prompts), 3),  # Use fewer seeds for faster testing
        probes_per_seed=2,
        steps_per_probe=2,
        eps=0.03,
        layer_to_probe=6,
        n_modes=3
    )

    print(f"\n⚙️ ARM Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Probes per seed: {config.probes_per_seed}")
    print(f"  Layer: {config.layer_to_probe}")

    try:
        # Initialize ARM
        print("\n🔬 Initializing ARM Mapper...")
        mapper = ARMMapper(config)

        # Run analysis on subset of prompts
        test_prompts = prompts[:config.n_seeds]
        print(f"📊 Running ARM analysis on {len(test_prompts)} prompts...")

        results = mapper.map_latent_manifold(test_prompts)

        print("✅ ARM analysis completed!")
        print(f"  Seeds analyzed: {results['n_seeds']}")
        print(f"  Descriptors shape: {results['descriptors'].shape}")

        # Test control vector computation
        print("\n🎯 Testing Control Vector Computation...")

        # Use first prompt as positive, second as negative
        computer = mapper.create_control_vector_computer()
        control_vector = computer.compute_control_vector(
            positive_prompts=[test_prompts[0]],
            negative_prompts=[test_prompts[1]],
            layer=config.layer_to_probe
        )

        print("✅ Control vector computed!")
        print(f"  Direction shape: {control_vector.direction.shape}")
        print(f"  Layer: {control_vector.layer}")
        print(f"  Coefficient: {control_vector.coefficient}")

        # Test steered generation
        print("\n🎭 Testing Steered Text Generation...")

        generator = mapper.create_controlled_generator()

        # Generate baseline text
        baseline_text = generator.generate_with_steering(
            "The Jabberwock", max_length=20, temperature=0.8
        )
        print("📝 Baseline generation:")
        print(f"  '{baseline_text}'")

        # Generate steered text
        generator.set_control(control_vector)
        try:
            steered_text = generator.generate_with_steering(
                "The Jabberwock", max_length=20, temperature=0.8
            )
            print("🎯 Steered generation:")
            print(f"  '{steered_text}'")
        finally:
            generator.clear_controls()

        # Test manifold steering
        print("\n🌀 Testing Manifold Signature Steering...")

        target_signature = results['seed_analyses'][0]['resonance_signature']['s_norm']
        steered_manifold_text = mapper.steer_generation_toward_signature(
            prompt="The Jabberwock",
            target_signature=target_signature,
            max_length=20,
            temperature=0.8,
            steering_strength=0.5
        )
        print("🌀 Manifold signature steering:")
        print(f"  '{steered_manifold_text}'")

        print("\n🎉 All steering tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_steering_from_manifold():
    """Test computing steering vectors directly from manifold regions."""
    print("\n🔄 Testing Steering Vector from Manifold Regions")

    try:
        config = ARMConfig(model_name="distilgpt2", n_seeds=3)
        mapper = ARMMapper(config)

        # Simple test prompts
        prompts = [
            "Hello world",
            "Goodbye world",
            "How are you"
        ]

        results = mapper.map_latent_manifold(prompts)

        # Compute steering vector using manifold indices
        control_vector = mapper.compute_steering_vector_from_manifold(
            positive_region_indices=[0],  # "Hello world" as positive
            negative_region_indices=[1]   # "Goodbye world" as negative
        )

        print("✅ Manifold-based control vector computed!")
        print(f"  Direction shape: {control_vector.direction.shape}")

        return True

    except Exception as e:
        print(f"❌ Manifold steering test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 ARM Steering Test Suite")
    print("=" * 50)

    success1 = test_steering_basic()
    success2 = test_steering_from_manifold()

    if success1 and success2:
        print("\n🎉 All tests passed! ARM steering is ready for use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
