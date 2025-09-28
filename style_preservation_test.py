#!/usr/bin/env python3
"""
Test ARM's ability to preserve and control stylistic elements in text generation.

This script tests whether ARM can capture and steer the distinctive style of
Lewis Carroll's Jabberwocky (nonsense literature with invented words).
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
import re

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


class StyleEvaluator:
    """Evaluates stylistic elements in generated text."""

    def __init__(self):
        # Jabberwocky-specific style indicators
        self.nonsense_words = {
            'slithy', 'toves', 'gyre', 'gimble', 'wabe', 'mimsy', 'borogoves',
            'mome', 'raths', 'outgrabe', 'Jabberwock', 'vorpal', 'snicker-snack',
            'galumphing', 'frabjous', 'Callooh', 'Callay', 'uffish', 'tulgey',
            'burbled', 'whiffling', 'beamish', 'frumious', 'Bandersnatch',
            'Jubjub', 'Tumtum', 'manxome'
        }

        # Stylistic patterns
        self.patterns = {
            'capitalized_nonsense': re.compile(r'\b[A-Z][a-z]*\b'),  # Capitalized made-up words
            'compound_words': re.compile(r'\b\w+-\w+\b'),            # Hyphenated compounds
            'onomatopoeic': re.compile(r'\b\w*oo\w*\b', re.IGNORECASE), # Words with 'oo'
            'repetition': re.compile(r'\b(\w+)\s+\1\b'),            # Repeated words
        }

    def analyze_text_style(self, text: str) -> Dict[str, float]:
        """Analyze stylistic elements in text."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Count nonsense words
        nonsense_count = sum(1 for word in words if word in self.nonsense_words)

        # Count stylistic patterns
        pattern_scores = {}
        for pattern_name, pattern in self.patterns.items():
            matches = len(pattern.findall(text))
            pattern_scores[pattern_name] = matches

        # Calculate metrics
        total_words = len(words)
        if total_words == 0:
            return {'error': 'No words found'}

        metrics = {
            'nonsense_density': nonsense_count / total_words,
            'total_nonsense_words': nonsense_count,
            'total_words': total_words,
            'avg_word_length': np.mean([len(word) for word in words]),
            'unique_words_ratio': len(set(words)) / total_words,
        }

        # Add pattern metrics
        for pattern_name, count in pattern_scores.items():
            metrics[f'{pattern_name}_count'] = count
            metrics[f'{pattern_name}_density'] = count / total_words if total_words > 0 else 0

        return metrics

    def compare_texts(self, original: str, generated: str) -> Dict[str, Any]:
        """Compare original and generated text styles."""
        original_metrics = self.analyze_text_style(original)
        generated_metrics = self.analyze_text_style(generated)

        if 'error' in original_metrics or 'error' in generated_metrics:
            return {'error': 'Text analysis failed'}

        # Calculate style similarity
        similarity_scores = {}
        for key in original_metrics.keys():
            if key in generated_metrics and not key.startswith('total_'):
                orig_val = original_metrics[key]
                gen_val = generated_metrics[key]
                if orig_val + gen_val > 0:  # Avoid division by zero
                    similarity = 1 - abs(orig_val - gen_val) / (orig_val + gen_val)
                    similarity_scores[f'{key}_similarity'] = similarity

        return {
            'original_metrics': original_metrics,
            'generated_metrics': generated_metrics,
            'similarity_scores': similarity_scores,
            'overall_similarity': np.mean(list(similarity_scores.values())) if similarity_scores else 0
        }


class ARMStyleExperiment:
    """Test ARM's ability to capture and control stylistic elements."""

    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.style_evaluator = StyleEvaluator()

    def load_jabberwocky_prompts(self, n_prompts: int = 3) -> List[str]:
        """Load prompts from Jabberwocky text."""
        with open("test-data/jwok.txt", 'r') as f:
            text = f.read()

        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith("'")]

        # Take the first few meaningful lines
        prompts = []
        for line in lines[:n_prompts]:
            # Truncate to make good prompts
            if len(line.split()) > 10:
                words = line.split()[:8]  # First 8 words
                prompts.append(' '.join(words))
            else:
                prompts.append(line)

        return prompts

    def generate_baseline_completion(self, prompt: str, max_length: int = 50) -> str:
        """Generate completion without ARM control."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return completion.strip()

    def test_arm_style_capture(self, config: ARMConfig) -> Dict[str, Any]:
        """Test ARM's ability to capture Jabberwocky style."""

        print(f"Testing ARM configuration: eps={config.eps}, probes={config.probes_per_seed}, layer={config.layer_to_probe}")

        # Load test prompts
        prompts = self.load_jabberwocky_prompts(2)
        print(f"Using prompts: {[p[:30] + '...' for p in prompts]}")

        # Generate baseline completions (without ARM)
        print("Generating baseline completions...")
        baseline_completions = []
        for prompt in prompts:
            completion = self.generate_baseline_completion(prompt)
            baseline_completions.append(completion)
            print(f"Baseline for '{prompt[:20]}...': {completion[:50]}...")

        # Analyze baseline style
        baseline_metrics = []
        for completion in baseline_completions:
            metrics = self.style_evaluator.analyze_text_style(completion)
            baseline_metrics.append(metrics)

        # Calculate average baseline nonsense density
        baseline_nonsense_avg = np.mean([m.get('nonsense_density', 0) for m in baseline_metrics])

        print(".4f"
        # Run ARM analysis
        print("Running ARM analysis...")
        try:
            arm_mapper = ARMMapper(config)
            manifold_result = arm_mapper.map_latent_manifold(prompts)

            # Extract control information (this would be used for steering)
            descriptors = manifold_result['descriptors']
            print(f"ARM analysis successful. Generated {len(descriptors)} descriptors.")
            print(f"Descriptor shape: {descriptors.shape}")

            # For now, just evaluate if ARM runs successfully
            # In a full implementation, you'd use the descriptors to steer generation

            return {
                'success': True,
                'baseline_nonsense_density': baseline_nonsense_avg,
                'arm_descriptors_shape': descriptors.shape,
                'arm_entropy_avg': np.mean([a['resonance_signature']['entropy']
                                          for a in manifold_result['seed_analyses']]),
                'config': config.to_dict(),
                'baseline_completions': baseline_completions,
            }

        except Exception as e:
            print(f"ARM analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'baseline_nonsense_density': baseline_nonsense_avg,
                'config': config.to_dict(),
            }

    def run_parameter_sweep(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Run experiments across parameter ranges."""

        results = []

        # Generate parameter combinations
        from itertools import product
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))

        print(f"Running {len(combinations)} parameter combinations...")

        for i, combo in enumerate(combinations):
            print(f"\n--- Test {i+1}/{len(combinations)} ---")

            # Create configuration
            config_dict = {
                'model_name': 'distilgpt2',
                'n_seeds': 2,  # Very small for testing
                'steps_per_probe': 3,
                'n_modes': 3,
                'random_seed': 42,
            }

            for name, value in zip(param_names, combo):
                config_dict[name] = value

            config = ARMConfig.from_dict(config_dict)

            # Run test
            result = self.test_arm_style_capture(config)
            result['param_combo'] = dict(zip(param_names, combo))
            result['test_id'] = i

            results.append(result)

        return results


def main():
    """Run style preservation experiments."""

    print("🎭 ARM Style Preservation Experiment")
    print("=" * 50)
    print("Testing ARM's ability to capture Jabberwocky nonsense literature style")

    # Initialize experiment
    experiment = ARMStyleExperiment()

    # Test different configurations
    param_ranges = {
        'eps': [0.01, 0.05, 0.1],
        'probes_per_seed': [2, 4],
        'layer_to_probe': [1, 2, 3],
    }

    print(f"Testing {len(param_ranges['eps']) * len(param_ranges['probes_per_seed']) * len(param_ranges['layer_to_probe'])} configurations...")

    # Run parameter sweep
    results = experiment.run_parameter_sweep(param_ranges)

    # Analyze results
    successful_results = [r for r in results if r.get('success', False)]

    print("
📊 Results Summary:"    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")

    if successful_results:
        print("
✅ Successful configurations:"        for result in successful_results:
            config = result['config']
            entropy = result.get('arm_entropy_avg', 0)
            print(".4f"
        # Find best configuration
        best_result = max(successful_results,
                         key=lambda x: x.get('arm_entropy_avg', 0))

        print("
🏆 Best Configuration:"        print(f"  Parameters: eps={best_result['config']['eps']}, probes={best_result['config']['probes_per_seed']}, layer={best_result['config']['layer_to_probe']}")
        print(".4f"
    else:
        print("❌ All ARM tests failed - check configuration and hardware")

    print("
💡 Next Steps:"    print("1. If ARM runs successfully, implement actual generation steering")
    print("2. Compare steered generations vs. baselines for style preservation")
    print("3. Try different models or fine-tune on Jabberwocky text")
    print("4. Implement genetic algorithm for hyperparameter optimization")


if __name__ == "__main__":
    main()
