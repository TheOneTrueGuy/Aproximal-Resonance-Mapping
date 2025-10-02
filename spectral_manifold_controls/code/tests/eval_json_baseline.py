"""
JSON Adherence Evaluation with Baselines

Critical test: Does SMC actually beat simpler approaches?

Baselines tested:
1. Zero-shot (no examples)
2. Few-shot (2-shot, 5-shot)
3. SMC with manifold steering

This is the MOST IMPORTANT eval - if SMC can't beat few-shot,
the whole value proposition is questionable.
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
    """
    Return 1.0 if text contains valid JSON with all required keys, else 0.0.
    
    Ported from arm_eval_harness.py (proven to work).
    """
    # Find minimal brace substrings
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
            # Light normalization: quote bare keys, unify quotes
            s = re.sub(r"(\b\w+\b)\s*:", r'"\1":', cand)
            s = s.replace("'", '"')
            s = s.replace('""', '"')
            obj = try_parse(s)
        if isinstance(obj, dict):
            if all(k in obj for k in required_keys):
                return 1.0
    return 0.0


def load_json_corpus(path="test-data/json_test.txt", max_examples=50):
    """Load JSON examples from file."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        # Fallback to hardcoded examples
        return [
            '{"name": "Alice", "age": 30, "city": "NYC"}',
            '{"name": "Bob", "age": 25, "city": "LA"}',
            '{"name": "Carol", "age": 35, "city": "Chicago"}',
        ] * 17  # Repeat to get ~50
    
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:max_examples]


def eval_baseline_zeroshot(model_interface, test_prompts, required_keys, verbose=True):
    """Baseline 1: Zero-shot (no examples)."""
    print("\n  Testing: Zero-shot baseline")
    
    generator = ARMControlledGenerator(model_interface)
    scores = []
    times = []
    
    for i, prompt in enumerate(test_prompts, 1):
        start_time = time.time()
        output = generator.generate_with_steering(
            prompt=prompt,
            max_length=40,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        score = safe_json_adherence_score(output, required_keys)
        scores.append(score)
        
        # Only print first 3 and last for long runs
        if verbose or i <= 3 or i == len(test_prompts):
            output_safe = output[:60].encode('ascii', 'replace').decode('ascii')
            print(f"    [{i}/{len(test_prompts)}] [{score}] {elapsed:.2f}s - {output_safe}...")
        elif i == 4:
            print(f"    ... ({len(test_prompts) - 4} more samples) ...")
    
    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    std_score = np.std(scores)
    print(f"    Zero-shot: score={avg_score:.3f} (±{std_score:.3f}), time={avg_time:.2f}s/sample")
    return avg_score, scores, avg_time


def eval_baseline_fewshot(model_interface, test_prompts, corpus, n_shots, required_keys, verbose=False):
    """Baseline 2: Few-shot prompting."""
    print(f"\n  Testing: {n_shots}-shot baseline")
    
    # Sample examples for few-shot
    rng = np.random.default_rng(42)
    examples = rng.choice(corpus, size=min(n_shots, len(corpus)), replace=False)
    examples_block = "\n".join(examples)
    
    generator = ARMControlledGenerator(model_interface)
    scores = []
    times = []
    
    for i, base_prompt in enumerate(test_prompts, 1):
        # Add few-shot examples
        prompt = (
            f"Generate JSON similar to these examples (JSON only):\n{examples_block}\n\n"
            f"{base_prompt}"
        )
        
        start_time = time.time()
        output = generator.generate_with_steering(
            prompt=prompt,
            max_length=40,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        score = safe_json_adherence_score(output, required_keys)
        scores.append(score)
        
        # Only print first 3 and last for long runs
        if verbose or i <= 3 or i == len(test_prompts):
            output_safe = output[:60].encode('ascii', 'replace').decode('ascii')
            print(f"    [{i}/{len(test_prompts)}] [{score}] {elapsed:.2f}s - {output_safe}...")
        elif i == 4:
            print(f"    ... ({len(test_prompts) - 4} more samples) ...")
    
    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    std_score = np.std(scores)
    print(f"    {n_shots}-shot: score={avg_score:.3f} (±{std_score:.3f}), time={avg_time:.2f}s/sample")
    return avg_score, scores, avg_time


def eval_smc_steering(builder, manifold, test_prompts, corpus, required_keys, strength=1.0, verbose=False):
    """SMC: Manifold-based steering."""
    
    # Create control vector from manifold (timed separately as one-time cost)
    cv_start = time.time()
    computer = WeightedControlVectorComputer(builder.model_interface)
    
    # All indices get equal weight (positive-only)
    index_weights = {i: 1.0 / len(corpus) for i in range(len(corpus))}
    cv = computer.compute_from_weights(index_weights, corpus, layer=3)
    cv.strength = strength
    cv_time = time.time() - cv_start
    
    # Few-shot prefix (same as few-shot baseline for fair comparison)
    rng = np.random.default_rng(42)
    examples = rng.choice(corpus, size=min(2, len(corpus)), replace=False)
    examples_block = "\n".join(examples)
    
    generator = ARMControlledGenerator(builder.model_interface)
    scores = []
    times = []
    
    for i, base_prompt in enumerate(test_prompts, 1):
        prompt = (
            f"Generate JSON similar to these examples (JSON only):\n{examples_block}\n\n"
            f"{base_prompt}"
        )
        
        # Generate with control vector applied
        generator.set_control(cv)
        start_time = time.time()
        output = generator.generate_with_steering(
            prompt=prompt,
            max_length=40,
            temperature=0.8,
            do_sample=True,
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        score = safe_json_adherence_score(output, required_keys)
        scores.append(score)
        
        # Only print first 3 and last for long runs
        if verbose or i <= 3 or i == len(test_prompts):
            output_safe = output[:60].encode('ascii', 'replace').decode('ascii')
            print(f"      [{i}/{len(test_prompts)}] [{score}] {elapsed:.2f}s - {output_safe}...")
        elif i == 4:
            print(f"      ... ({len(test_prompts) - 4} more samples) ...")
    
    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    std_score = np.std(scores)
    print(f"    Strength {strength}: score={avg_score:.3f} (±{std_score:.3f}), time={avg_time:.2f}s/sample, CV build={cv_time:.3f}s")
    return avg_score, scores, avg_time, cv_time


def main():
    print("="*80)
    print("JSON ADHERENCE: SMC vs BASELINES")
    print("Critical Test: Does SMC beat simpler approaches?")
    print("="*80)
    
    # Timing breakdown
    timing = {}
    
    # Setup
    print("\n[1/5] Loading JSON corpus...")
    corpus_start = time.time()
    corpus = load_json_corpus()
    timing['corpus_load'] = time.time() - corpus_start
    print(f"  Loaded {len(corpus)} JSON examples ({timing['corpus_load']:.3f}s)")
    
    required_keys = ["name", "age", "city"]
    
    # Generate 50 diverse test prompts
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
             "Kate", "Leo", "Mary", "Nathan", "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tina",
             "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zack", "Anna", "Ben", "Claire", "Dan",
             "Emma", "Felix", "Gina", "Hugo", "Ivy", "James", "Kara", "Liam", "Mia", "Noah",
             "Ola", "Pete", "Rosa", "Seth", "Tara", "Uri", "Vera", "Will", "Xia", "Yuki"]
    ages = list(range(20, 70))
    cities = ["Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Sydney", "Toronto", "Mumbai", "Seoul",
              "Cairo", "Moscow", "Lima", "Oslo", "Dublin", "Vienna", "Prague", "Lisbon", "Athens", "Warsaw",
              "Bangkok", "Singapore", "Amsterdam", "Brussels", "Copenhagen", "Helsinki", "Stockholm", "Zurich",
              "Geneva", "Barcelona", "Milan", "Venice", "Florence", "Naples", "Kyoto", "Osaka", "Beijing",
              "Shanghai", "Hong Kong", "Taipei", "Jakarta", "Manila", "Hanoi", "Kuala Lumpur", "Perth",
              "Melbourne", "Auckland", "Wellington", "Vancouver", "Montreal"]
    
    test_prompts = []
    for i in range(50):
        name = names[i]
        age = ages[i]
        city = cities[i]
        test_prompts.append(f"Task: Given name={name}, age={age}, city={city}. Respond with exactly:")
    
    print(f"  Test prompts: {len(test_prompts)}")
    print(f"  Required keys: {required_keys}")
    
    # Load model (separate timing)
    print("\n[2/5] Loading model...")
    model_start = time.time()
    builder = ManifoldBuilder(
        model_name="gpt2-medium",
        device="cpu",
        n_modes=4
    )
    timing['model_load'] = time.time() - model_start
    print(f"  Model loaded: {timing['model_load']:.1f}s")
    print(f"  (Cached models load faster; first run may download)")
    
    # Build SMC manifold
    print("\n[3/5] Building SMC manifold...")
    manifold_start = time.time()
    
    manifold = builder.build_manifold(
        corpus=corpus,
        layer=3,
        probes_per_seed=2,
        steps_per_probe=2,
        eps=0.03
    )
    
    timing['manifold_build'] = time.time() - manifold_start
    build_time = timing['manifold_build']  # For backward compatibility
    print(f"  Manifold built: {timing['manifold_build']:.1f}s")
    
    # Run evaluations
    print("\n[4/5] Running evaluations...")
    eval_start = time.time()
    results = {}  # {name: (score, scores_list, avg_time, [cv_time])}
    
    # Baseline 1: Zero-shot
    score, scores, avg_time = eval_baseline_zeroshot(
        builder.model_interface, test_prompts, required_keys
    )
    results['zeroshot'] = (score, scores, avg_time)
    
    # Baseline 2: Few-shot (2-shot)
    score, scores, avg_time = eval_baseline_fewshot(
        builder.model_interface, test_prompts, corpus, 2, required_keys
    )
    results['fewshot_2'] = (score, scores, avg_time)
    
    # Baseline 3: Few-shot (5-shot)
    score, scores, avg_time = eval_baseline_fewshot(
        builder.model_interface, test_prompts, corpus, 5, required_keys
    )
    results['fewshot_5'] = (score, scores, avg_time)
    
    # SMC: Different strengths (sweep to find optimum)
    strength_range = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    print(f"\n  Testing SMC strengths: {strength_range}")
    print(f"  (This will test {len(strength_range)} x {len(test_prompts)} = {len(strength_range) * len(test_prompts)} generations)")
    print(f"  Estimated time: ~{len(strength_range) * len(test_prompts) * 5.5 / 60:.1f} minutes")
    
    for i, strength in enumerate(strength_range, 1):
        print(f"\n  [{i}/{len(strength_range)}] Strength {strength}...")
        score, scores, avg_time, cv_time = eval_smc_steering(
            builder, manifold, test_prompts, corpus, required_keys, strength
        )
        results[f'smc_{strength}'] = (score, scores, avg_time, cv_time)
    
    timing['total_eval'] = time.time() - eval_start
    
    # Analysis
    print("\n[5/5] RESULTS ANALYSIS")
    print("="*80)
    
    print("\nScores and timing by approach:")
    print("-"*80)
    print(f"{'Approach':<20} {'Score':<10} {'Time/sample':<15}")
    print("-"*80)
    
    for name, data in results.items():
        score = data[0]
        avg_time = data[2]
        std_score = np.std(data[1]) if len(data[1]) > 1 else 0.0
        print(f"{name:<20} {score:.3f} (±{std_score:.3f})  {avg_time:.2f}s")
    
    # SMC Dose-Response Curve
    print("\nSMC Dose-Response Curve (Strength vs Score):")
    print("-"*80)
    smc_results = [(k.replace('smc_', ''), v[0], np.std(v[1])) for k, v in results.items() if k.startswith('smc_')]
    smc_results.sort(key=lambda x: float(x[0]))
    
    for strength, score, std in smc_results:
        bar_length = int(score * 50)  # 50 chars = 1.0 score
        bar = '#' * bar_length
        print(f"  {strength:>4}  {score:.3f} (±{std:.3f})  {bar}")
    
    best_strength = max(smc_results, key=lambda x: x[1])[0]
    best_score = max(smc_results, key=lambda x: x[1])[1]
    print(f"\nOptimal strength: {best_strength} (score: {best_score:.3f})")
    
    # Critical comparisons
    print("\n" + "="*80)
    print("CRITICAL ANALYSIS")
    print("="*80)
    
    zeroshot_score = results['zeroshot'][0]
    fewshot_2_score = results['fewshot_2'][0]
    fewshot_5_score = results['fewshot_5'][0]
    smc_best_score = max([results[k][0] for k in results if k.startswith('smc_')])
    smc_best_name = max([k for k in results if k.startswith('smc_')], key=lambda k: results[k][0])
    
    print(f"\nBaseline scores:")
    print(f"  Zero-shot:    {zeroshot_score:.2f}")
    print(f"  2-shot:       {fewshot_2_score:.2f}")
    print(f"  5-shot:       {fewshot_5_score:.2f}")
    print(f"\nSMC best:       {smc_best_score:.2f} ({smc_best_name})")
    
    # Timing comparison
    print(f"\nTiming breakdown:")
    print(f"  Manifold build:    {build_time:.1f}s (one-time)")
    print(f"  Zero-shot gen:     {results['zeroshot'][2]:.2f}s/sample")
    print(f"  5-shot gen:        {results['fewshot_5'][2]:.2f}s/sample")
    print(f"  SMC gen:           {results[smc_best_name][2]:.2f}s/sample")
    print(f"  SMC CV creation:   {results[smc_best_name][3]:.3f}s (one-time)")
    print(f"\nFor {len(test_prompts)} samples:")
    total_baseline = results['fewshot_5'][2] * len(test_prompts)
    total_smc = build_time + results[smc_best_name][3] + results[smc_best_name][2] * len(test_prompts)
    print(f"  5-shot total:      {total_baseline:.1f}s")
    print(f"  SMC total:         {total_smc:.1f}s")
    print(f"  SMC overhead:      {total_smc - total_baseline:+.1f}s")
    
    print("\n" + "-"*80)
    print("VERDICT")
    print("-"*80)
    
    # Question 1: Does SMC beat zero-shot?
    if smc_best_score > zeroshot_score + 0.1:
        print(f"[PASS] SMC beats zero-shot: {smc_best_score:.2f} > {zeroshot_score:.2f}")
    else:
        print(f"[FAIL] SMC doesn't beat zero-shot: {smc_best_score:.2f} vs {zeroshot_score:.2f}")
    
    # Question 2: Does SMC beat few-shot? (CRITICAL)
    if smc_best_score > fewshot_5_score + 0.05:
        print(f"[PASS] SMC beats 5-shot: {smc_best_score:.2f} > {fewshot_5_score:.2f}")
        print("  -> SMC adds value beyond simple prompting!")
    elif smc_best_score >= fewshot_5_score - 0.05:
        print(f"[NEUTRAL] SMC matches 5-shot: {smc_best_score:.2f} ≈ {fewshot_5_score:.2f}")
        print("  -> SMC doesn't hurt, but doesn't clearly beat prompting")
    else:
        print(f"[FAIL] SMC worse than 5-shot: {smc_best_score:.2f} < {fewshot_5_score:.2f}")
        print("  -> WARNING: Simple prompting beats SMC!")
    
    # Question 3: Is the build cost worth it?
    print(f"\n[COST] SMC overhead for {len(test_prompts)} samples: {total_smc - total_baseline:+.1f}s")
    if total_smc < total_baseline:
        print(f"       SMC is FASTER overall!")
    elif total_smc < total_baseline * 2:
        print(f"       SMC overhead is reasonable (< 2x)")
    else:
        print(f"       SMC is expensive for small batches")
    
    # Calculate break-even
    if results[smc_best_name][2] < results['fewshot_5'][2]:
        print(f"       SMC generation is faster per-sample")
        print(f"       Build cost amortizes over any batch size")
    else:
        gen_overhead = results[smc_best_name][2] - results['fewshot_5'][2]
        setup_cost = build_time + results[smc_best_name][3]
        if gen_overhead > 0:
            breakeven = setup_cost / gen_overhead
            print(f"       Break-even: ~{breakeven:.0f} samples")
        else:
            print(f"       SMC is always faster per-sample")
    
    print("\n" + "="*80)
    print("HONEST ASSESSMENT")
    print("="*80)
    
    if smc_best_score < fewshot_2_score:
        print("\n⚠️  WARNING: SMC doesn't beat even 2-shot prompting.")
        print("   This suggests SMC may not add value for JSON task.")
        print("   Consider:")
        print("   - Testing on other tasks (style, sentiment)")
        print("   - Checking if manifold quality is poor")
        print("   - Reviewing steering implementation")
    elif smc_best_score < fewshot_5_score:
        print("\n⚠️  CAUTION: SMC doesn't beat 5-shot prompting.")
        print("   For JSON generation, few-shot might be simpler/better.")
        print("   SMC value may be in:")
        print("   - Compositional mixing (not tested here)")
        print("   - Other tasks where prompting is weaker")
        print("   - Recipe reusability across contexts")
    else:
        print("\n[SUCCESS] SMC beats strong baselines!")
        print("  This validates the approach for JSON task.")
        print("  Next: Test on other domains to confirm generalization")
    
    print("\n" + "="*80)
    print("TIMING BREAKDOWN (Where does the time go?)")
    print("="*80)
    
    print("\nSetup costs (one-time):")
    print(f"  Corpus load:       {timing['corpus_load']:.3f}s")
    print(f"  Model load:        {timing['model_load']:.1f}s")
    print(f"  Manifold build:    {timing['manifold_build']:.1f}s")
    print(f"  Total setup:       {timing['corpus_load'] + timing['model_load'] + timing['manifold_build']:.1f}s")
    
    print("\nEvaluation costs:")
    print(f"  All tests:         {timing['total_eval']:.1f}s")
    
    print("\nPer-sample generation time:")
    for name, data in results.items():
        avg_time = data[2]
        print(f"  {name:<18} {avg_time:.2f}s/sample")
    
    print("\nCostliest operations:")
    costs = [
        ("Model loading", timing['model_load']),
        ("Manifold building", timing['manifold_build']),
        ("Running all evals", timing['total_eval']),
    ]
    costs.sort(key=lambda x: x[1], reverse=True)
    for i, (name, cost) in enumerate(costs, 1):
        pct = 100 * cost / sum(c[1] for c in costs)
        print(f"  {i}. {name:<25} {cost:>6.1f}s ({pct:>5.1f}%)")
    
    total_runtime = timing['corpus_load'] + timing['model_load'] + timing['manifold_build'] + timing['total_eval']
    print(f"\nTotal runtime: {total_runtime:.1f}s")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if timing['model_load'] > 10:
        print("\n[NOTE] Model loading took {:.1f}s.".format(timing['model_load']))
        print("       Models should be cached after first run.")
        print("       If this is slow every time, check your HF_HOME cache.")
    
    if timing['manifold_build'] > timing['total_eval']:
        print("\n[NOTE] Manifold building is the bottleneck ({:.1f}s).".format(timing['manifold_build']))
        print("       This is a one-time cost - manifolds can be saved/reused.")
        print("       For {:.0f} samples, build cost = {:.1f}s per sample".format(
            len(corpus), timing['manifold_build'] / len(corpus)
        ))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

