#!/usr/bin/env python3
"""
Hyperparameter optimization for ARM using experimental testing.

This script systematically tests different ARM configurations to find optimal
settings for latent manifold mapping and control vector effectiveness.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple
import json
import time
from datetime import datetime
import os

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


class ARMExperiment:
    """Manages ARM hyperparameter experiments."""

    def __init__(self, test_data_path: str = "test-data/jwok.txt"):
        self.test_data_path = test_data_path
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_test_prompts(self, n_prompts: int = 5) -> List[str]:
        """Load test prompts from Jabberwocky text."""
        with open(self.test_data_path, 'r') as f:
            text = f.read()

        # Split into lines and filter out empty ones
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Take first n_prompts lines as seed prompts
        prompts = lines[:min(n_prompts, len(lines))]

        return prompts

    def evaluate_style_preservation(self, original_prompts: List[str],
                                  generated_completions: List[str]) -> float:
        """
        Evaluate how well the style is preserved in completions.
        This is a simple heuristic - in practice you'd want more sophisticated metrics.
        """
        # Simple style metrics for Jabberwocky (nonsense literature)
        style_indicators = [
            'slithy', 'toves', 'gyre', 'gimble', 'wabe', 'mimsy', 'borogoves',
            'mome', 'raths', 'outgrabe', 'Jabberwock', 'vorpal', 'snicker-snack',
            'galumphing', 'frabjous', 'Callooh', 'Callay'
        ]

        total_score = 0
        for completion in generated_completions:
            completion_lower = completion.lower()
            style_matches = sum(1 for word in style_indicators if word.lower() in completion_lower)
            # Normalize by completion length (penalty for bland, generic text)
            length_penalty = min(1.0, len(completion.split()) / 20)  # Prefer longer, more creative completions
            score = style_matches * length_penalty
            total_score += score

        return total_score / len(generated_completions) if generated_completions else 0

    def run_single_experiment(self, config: ARMConfig,
                            test_prompts: List[str]) -> Dict[str, Any]:
        """Run a single ARM experiment and evaluate results."""

        print(f"Running experiment with config: eps={config.eps}, probes={config.probes_per_seed}, steps={config.steps_per_probe}")

        start_time = time.time()

        try:
            # Initialize ARM mapper
            arm_mapper = ARMMapper(config)

            # Run manifold mapping
            manifold_result = arm_mapper.map_latent_manifold(test_prompts)

            # Extract results
            descriptors = manifold_result['descriptors']
            seed_analyses = manifold_result['seed_analyses']

            # Simple evaluation: check descriptor diversity
            descriptor_std = np.std(descriptors, axis=0).mean()
            descriptor_range = np.ptp(descriptors, axis=0).mean()

            # Check resonance signatures
            entropies = [a['resonance_signature']['entropy'] for a in seed_analyses]
            participation_ratios = [a['resonance_signature']['participation_ratio_normalized'] for a in seed_analyses]

            avg_entropy = np.mean(entropies)
            avg_participation = np.mean(participation_ratios)

            # Test style preservation (this would need actual generation, placeholder for now)
            style_score = 0.5  # Placeholder

            experiment_result = {
                'config': config.to_dict(),
                'success': True,
                'runtime_seconds': time.time() - start_time,
                'n_seeds_processed': len(seed_analyses),
                'descriptor_diversity': float(descriptor_std),
                'descriptor_range': float(descriptor_range),
                'avg_entropy': float(avg_entropy),
                'avg_participation_ratio': float(avg_participation),
                'style_preservation_score': style_score,
                'error': None
            }

        except Exception as e:
            print(f"Experiment failed: {e}")
            experiment_result = {
                'config': config.to_dict(),
                'success': False,
                'runtime_seconds': time.time() - start_time,
                'error': str(e)
            }

        return experiment_result

    def grid_search_optimization(self, param_ranges: Dict[str, List[Any]],
                               base_config: ARMConfig,
                               test_prompts: List[str]) -> List[Dict[str, Any]]:
        """Perform grid search over hyperparameter ranges."""

        print(f"Starting grid search with {len(param_ranges)} parameters to vary")
        print(f"Parameter ranges: {param_ranges}")

        results = []

        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        from itertools import product
        combinations = list(product(*param_values))

        print(f"Testing {len(combinations)} parameter combinations...")

        for i, combo in enumerate(combinations):
            print(f"\n--- Experiment {i+1}/{len(combinations)} ---")

            # Create config with this parameter combination
            config_dict = base_config.to_dict()
            for name, value in zip(param_names, combo):
                config_dict[name] = value

            config = ARMConfig.from_dict(config_dict)

            # Run experiment
            result = self.run_single_experiment(config, test_prompts)
            result['experiment_id'] = i
            result['param_combo'] = dict(zip(param_names, combo))

            results.append(result)

            # Save intermediate results
            self.save_results(results, f"intermediate_results_{len(results)}.json")

        return results

    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save experiment results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                serializable_result[key] = convert_numpy(value)
            serializable_results.append(serializable_result)

        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(results),
                'results': serializable_results
            }, f, indent=2)

        print(f"Saved {len(results)} results to {filepath}")

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results to find optimal configurations."""

        successful_results = [r for r in results if r.get('success', False)]

        if not successful_results:
            return {'error': 'No successful experiments to analyze'}

        # Find best configurations by different metrics
        analysis = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'best_configs': {}
        }

        # Sort by different metrics
        metrics = ['descriptor_diversity', 'descriptor_range', 'avg_entropy', 'avg_participation_ratio']

        for metric in metrics:
            sorted_results = sorted(successful_results,
                                  key=lambda x: x.get(metric, 0),
                                  reverse=True)
            if sorted_results:
                best = sorted_results[0]
                analysis['best_configs'][f'by_{metric}'] = {
                    'score': best[metric],
                    'config': best['config'],
                    'runtime': best['runtime_seconds']
                }

        # Statistical summary
        runtimes = [r['runtime_seconds'] for r in successful_results]
        analysis['runtime_stats'] = {
            'mean': np.mean(runtimes),
            'std': np.std(runtimes),
            'min': np.min(runtimes),
            'max': np.max(runtimes)
        }

        return analysis


def main():
    """Main experimental optimization loop."""

    print("🔬 ARM Hyperparameter Optimization Experiment")
    print("=" * 60)

    # Initialize experiment manager
    experiment = ARMExperiment()

    # Load test data
    test_prompts = experiment.load_test_prompts(n_prompts=3)
    print(f"Loaded {len(test_prompts)} test prompts from Jabberwocky:")
    for i, prompt in enumerate(test_prompts):
        print(f"  {i+1}: {prompt[:50]}...")

    # Base configuration (optimized for user's hardware)
    base_config = ARMConfig(
        model_name="distilgpt2",  # Small model for Quadro K2200
        n_seeds=3,                # Very small for quick iteration
        probes_per_seed=4,        # Reduced for speed
        steps_per_probe=3,        # Reduced for speed
        eps=0.03,                 # Default perturbation
        layer_to_probe=2,         # Earlier layer for experimentation
        n_modes=4,                # Fewer modes for speed
        random_seed=42,
    )

    # Define hyperparameter ranges to test
    # Start conservative given hardware constraints
    param_ranges = {
        'eps': [0.01, 0.03, 0.05, 0.1],           # Perturbation magnitudes
        'probes_per_seed': [2, 4, 8],              # Number of probes
        'steps_per_probe': [3, 5, 7],              # Steps per probe
        'layer_to_probe': [1, 2, 3, 4],            # Which transformer layer
    }

    print("
Hyperparameter ranges to test:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")

    # Run grid search
    print("
🚀 Starting hyperparameter optimization...")
    start_time = time.time()

    results = experiment.grid_search_optimization(param_ranges, base_config, test_prompts)

    total_time = time.time() - start_time
    print(".2f"
    # Analyze results
    analysis = experiment.analyze_results(results)

    # Save final results
    experiment.save_results(results, "final_optimization_results.json")

    # Print summary
    print("
📊 Optimization Results Summary:")
    print(f"Total experiments: {analysis['total_experiments']}")
    print(f"Successful: {analysis['successful_experiments']} ({analysis['success_rate']*100:.1f}%)")
    print(".2f")

    print("
🏆 Best Configurations:"    for metric, best in analysis['best_configs'].items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Score: {best['score']:.4f}")
        print(f"  Config: eps={best['config']['eps']}, probes={best['config']['probes_per_seed']}, steps={best['config']['steps_per_probe']}, layer={best['config']['layer_to_probe']}")
        print(f"  Runtime: {best['runtime']:.2f}s")

    print("
💡 Recommendations:"    if analysis['success_rate'] > 0.8:
        print("  ✓ High success rate - ARM is viable on your hardware")
    else:
        print("  ⚠ Low success rate - may need hardware upgrades or smaller models")

    best_diversity = analysis['best_configs'].get('by_descriptor_diversity', {})
    if best_diversity:
        print(f"  🎯 Best diversity config found: eps={best_diversity['config']['eps']}")
        print("  📈 Try this configuration for style preservation experiments")


if __name__ == "__main__":
    main()
