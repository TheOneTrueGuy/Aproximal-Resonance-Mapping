"""
Controlled Comparison: Generation Strategy Isolation

Tests SMC vs baselines with MATCHED generation strategies to isolate
the pure effect of steering vs prompting.

Configurations:
- 8 test samples (fast)
- 3 baselines (zero-shot, 2-shot, 5-shot)
- 3 SMC strengths (1.0, 1.5, 2.0) - peak ± neighbors
- 2 generation strategies each: beam search AND sampling

This isolates: Does steering help independent of generation strategy?
"""

import sys
from pathlib import Path
import json
import re
import numpy as np
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core import ManifoldBuilder, WeightedControlVectorComputer
from arm_library.core.steering import ARMControlledGenerator


def safe_json_adherence_score(text: str, required_keys: list) -> float:
    """Return 1.0 if valid JSON with required keys, else 0.0."""
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not candidates:
        return 0.0

    def try_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    for cand in candidates:
        obj = try_parse(cand)
        if obj is None:
            s = re.sub(r"(\b\w+\b)\s*:", r'"\1":', cand)
            s = s.replace("'", '"')
            s = s.replace('""', '"')
            obj = try_parse(s)
        if isinstance(obj, dict):
            if all(k in obj for k in required_keys):
                return 1.0
    return 0.0


def load_json_corpus(path="test-data/json_test.txt", max_examples=50):
    """Load JSON corpus."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        return [
            '{"name": "Alice", "age": 30, "city": "NYC"}',
            '{"name": "Bob", "age": 25, "city": "LA"}',
            '{"name": "Carol", "age": 35, "city": "Chicago"}',
        ] * 17
    
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:max_examples]


def test_baseline(model_interface, test_prompts, corpus, n_shots, required_keys, use_beam):
    """Test baseline with specified generation strategy."""
    strategy = "beam" if use_beam else "sample"
    print(f"\n  [{n_shots}-shot + {strategy}]")
    
    if n_shots > 0:
        rng = np.random.default_rng(42)
        examples = rng.choice(corpus, size=min(n_shots, len(corpus)), replace=False)
        examples_block = "\n".join(examples)
    
    generator = ARMControlledGenerator(model_interface)
    scores = []
    times = []
    
    for i, base_prompt in enumerate(test_prompts, 1):
        if n_shots > 0:
            prompt = f"Generate JSON similar to these examples (JSON only):\n{examples_block}\n\n{base_prompt}"
        else:
            prompt = base_prompt
        
        start_time = time.time()
        if use_beam:
            output = generator.generate_with_steering(
                prompt=prompt,
                max_length=40,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        else:
            output = generator.generate_with_steering(
                prompt=prompt,
                max_length=40,
                do_sample=True,
                temperature=0.8,
            )
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        score = safe_json_adherence_score(output, required_keys)
        scores.append(score)
        
        output_safe = output[:50].encode('ascii', 'replace').decode('ascii')
        print(f"    [{i}] [{score}] {elapsed:.2f}s - {output_safe}...")
    
    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    std_score = np.std(scores)
    print(f"    Result: score={avg_score:.3f} (±{std_score:.3f}), time={avg_time:.2f}s/sample")
    return avg_score, scores, avg_time


def test_smc(builder, manifold, test_prompts, corpus, required_keys, strength, use_beam):
    """Test SMC with specified generation strategy."""
    strategy = "beam" if use_beam else "sample"
    print(f"\n  [SMC strength={strength} + {strategy}]")
    
    # Create control vector
    cv_start = time.time()
    computer = WeightedControlVectorComputer(builder.model_interface)
    index_weights = {i: 1.0 / len(corpus) for i in range(len(corpus))}
    cv = computer.compute_from_weights(index_weights, corpus, layer=3)
    cv.strength = strength
    cv_time = time.time() - cv_start
    
    # 2-shot prefix
    rng = np.random.default_rng(42)
    examples = rng.choice(corpus, size=2, replace=False)
    examples_block = "\n".join(examples)
    
    generator = ARMControlledGenerator(builder.model_interface)
    scores = []
    times = []
    
    for i, base_prompt in enumerate(test_prompts, 1):
        prompt = f"Generate JSON similar to these examples (JSON only):\n{examples_block}\n\n{base_prompt}"
        
        generator.set_control(cv)
        start_time = time.time()
        if use_beam:
            output = generator.generate_with_steering(
                prompt=prompt,
                max_length=40,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        else:
            output = generator.generate_with_steering(
                prompt=prompt,
                max_length=40,
                do_sample=True,
                temperature=0.8,
            )
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        score = safe_json_adherence_score(output, required_keys)
        scores.append(score)
        
        output_safe = output[:50].encode('ascii', 'replace').decode('ascii')
        print(f"    [{i}] [{score}] {elapsed:.2f}s - {output_safe}...")
    
    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    std_score = np.std(scores)
    print(f"    Result: score={avg_score:.3f} (±{std_score:.3f}), time={avg_time:.2f}s/sample, CV={cv_time:.3f}s")
    return avg_score, scores, avg_time, cv_time


def main():
    print("="*80)
    print("GENERATION STRATEGY ISOLATION TEST")
    print("Controlled comparison: Beam vs Sampling")
    print("="*80)
    
    # Setup
    print("\n[1/4] Setup...")
    corpus = load_json_corpus()
    print(f"  Corpus: {len(corpus)} examples")
    
    required_keys = ["name", "age", "city"]
    test_prompts = [
        "Task: Given name=Alice, age=30, city=Paris. Respond with exactly:",
        "Task: Given name=Bob, age=25, city=London. Respond with exactly:",
        "Task: Given name=Carol, age=35, city=Tokyo. Respond with exactly:",
        "Task: Given name=David, age=28, city=Berlin. Respond with exactly:",
        "Task: Given name=Eve, age=32, city=Madrid. Respond with exactly:",
        "Task: Given name=Frank, age=27, city=Rome. Respond with exactly:",
        "Task: Given name=Grace, age=29, city=Sydney. Respond with exactly:",
        "Task: Given name=Henry, age=31, city=Toronto. Respond with exactly:",
    ]
    print(f"  Test samples: {len(test_prompts)}")
    
    # Load model
    print("\n[2/4] Loading model...")
    start = time.time()
    builder = ManifoldBuilder(model_name="gpt2-medium", device="cpu", n_modes=4)
    print(f"  Model loaded: {time.time() - start:.1f}s")
    
    # Build manifold
    print("\n[3/4] Building manifold...")
    start = time.time()
    manifold = builder.build_manifold(corpus, layer=3, probes_per_seed=2, steps_per_probe=2, eps=0.03)
    build_time = time.time() - start
    print(f"  Manifold built: {build_time:.1f}s")
    
    # Run tests
    print("\n[4/4] Running controlled comparison...")
    print("="*80)
    
    results = {}
    
    # Baselines with both strategies
    print("\nBASELINES:")
    for n_shots in [0, 2, 5]:
        for use_beam in [True, False]:
            key = f"{n_shots}shot_{'beam' if use_beam else 'sample'}"
            score, scores, avg_time = test_baseline(
                builder.model_interface, test_prompts, corpus, n_shots, required_keys, use_beam
            )
            results[key] = (score, scores, avg_time)
    
    # SMC with both strategies
    print("\n\nSMC (peak ± neighbors):")
    for strength in [1.0, 1.5, 2.0]:
        for use_beam in [True, False]:
            key = f"smc_{strength}_{'beam' if use_beam else 'sample'}"
            score, scores, avg_time, cv_time = test_smc(
                builder, manifold, test_prompts, corpus, required_keys, strength, use_beam
            )
            results[key] = (score, scores, avg_time, cv_time)
    
    # Analysis
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    print("\nAll Results:")
    print("-"*80)
    print(f"{'Method':<25} {'Strategy':<10} {'Score':<15} {'Time/sample':<12}")
    print("-"*80)
    
    # Baselines
    for n_shots in [0, 2, 5]:
        for strategy in ['beam', 'sample']:
            key = f"{n_shots}shot_{strategy}"
            data = results[key]
            score, scores, avg_time = data[0], data[1], data[2]
            std = np.std(scores)
            print(f"{n_shots}-shot{'':<17} {strategy:<10} {score:.3f} (±{std:.3f})  {avg_time:.2f}s")
    
    print()
    
    # SMC
    for strength in [1.0, 1.5, 2.0]:
        for strategy in ['beam', 'sample']:
            key = f"smc_{strength}_{strategy}"
            data = results[key]
            score, scores, avg_time = data[0], data[1], data[2]
            std = np.std(scores)
            print(f"SMC (strength={strength}){'':<7} {strategy:<10} {score:.3f} (±{std:.3f})  {avg_time:.2f}s")
    
    # Key comparisons
    print("\n" + "="*80)
    print("KEY COMPARISONS")
    print("="*80)
    
    # Compare same generation strategy
    print("\n1. BEAM SEARCH (apples-to-apples):")
    print("-"*80)
    best_baseline_beam = max([results[f"{n}shot_beam"][0] for n in [0, 2, 5]])
    best_smc_beam = max([results[f"smc_{s}_beam"][0] for s in [1.0, 1.5, 2.0]])
    print(f"  Best baseline (beam):  {best_baseline_beam:.3f}")
    print(f"  Best SMC (beam):       {best_smc_beam:.3f}")
    print(f"  SMC advantage:         {best_smc_beam - best_baseline_beam:+.3f} ({(best_smc_beam/best_baseline_beam-1)*100:.1f}% improvement)")
    
    print("\n2. SAMPLING (apples-to-apples):")
    print("-"*80)
    best_baseline_sample = max([results[f"{n}shot_sample"][0] for n in [0, 2, 5]])
    best_smc_sample = max([results[f"smc_{s}_sample"][0] for s in [1.0, 1.5, 2.0]])
    print(f"  Best baseline (sample): {best_baseline_sample:.3f}")
    print(f"  Best SMC (sample):      {best_smc_sample:.3f}")
    print(f"  SMC advantage:          {best_smc_sample - best_baseline_sample:+.3f} ({(best_smc_sample/best_baseline_sample-1)*100:.1f}% improvement)")
    
    print("\n3. GENERATION STRATEGY EFFECT:")
    print("-"*80)
    # For 2-shot
    beam_2shot = results['2shot_beam'][0]
    sample_2shot = results['2shot_sample'][0]
    print(f"  2-shot + beam:    {beam_2shot:.3f}")
    print(f"  2-shot + sample:  {sample_2shot:.3f}")
    print(f"  Beam advantage:   {beam_2shot - sample_2shot:+.3f}")
    
    # For SMC 1.5
    beam_smc = results['smc_1.5_beam'][0]
    sample_smc = results['smc_1.5_sample'][0]
    print(f"\n  SMC 1.5 + beam:   {beam_smc:.3f}")
    print(f"  SMC 1.5 + sample: {sample_smc:.3f}")
    print(f"  Beam advantage:   {beam_smc - sample_smc:+.3f}")
    
    print("\n4. SPEED COMPARISON:")
    print("-"*80)
    time_2shot_beam = results['2shot_beam'][2]
    time_smc_beam = results['smc_1.5_beam'][2]
    print(f"  2-shot + beam:    {time_2shot_beam:.2f}s/sample")
    print(f"  SMC 1.5 + beam:   {time_smc_beam:.2f}s/sample")
    print(f"  SMC speedup:      {time_2shot_beam/time_smc_beam:.2f}x")
    
    time_2shot_sample = results['2shot_sample'][2]
    time_smc_sample = results['smc_1.5_sample'][2]
    print(f"\n  2-shot + sample:  {time_2shot_sample:.2f}s/sample")
    print(f"  SMC 1.5 + sample: {time_smc_sample:.2f}s/sample")
    print(f"  SMC speedup:      {time_2shot_sample/time_smc_sample:.2f}x")
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    print("\nDoes steering help (independent of generation strategy)?")
    if best_smc_beam > best_baseline_beam and best_smc_sample > best_baseline_sample:
        print("  [YES] SMC beats baselines with BOTH beam search AND sampling")
        print("  -> Steering effect is real, not an artifact of generation strategy")
    else:
        print("  [MIXED] Results vary by generation strategy")
    
    print("\nWhich generation strategy is better?")
    if beam_2shot > sample_2shot and beam_smc > sample_smc:
        print("  [BEAM] Beam search consistently better than sampling")
    elif beam_2shot < sample_2shot and beam_smc < sample_smc:
        print("  [SAMPLE] Sampling consistently better than beam search")
    else:
        print("  [MIXED] Depends on method")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

