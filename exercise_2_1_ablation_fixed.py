#!/usr/bin/env python3
"""
Exercise 2.1: Ablation Study - Using Working Harness Infrastructure
Modified from examples/arm_eval_harness.py to test topology contribution
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig

OUTPUT_DIR = "arm_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_json_adherence_score(text: str, required_keys: List[str]) -> float:
    """Return 1.0 if text contains a valid JSON object with all required keys, else 0.0."""
    import json
    import re

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


def compute_descriptor_spectral_only(resonance_signature: Dict[str, Any]) -> np.ndarray:
    """Version A: Spectral features only."""
    features = np.concatenate([
        resonance_signature['s_norm'],
        [resonance_signature['entropy'], resonance_signature['participation_ratio_normalized']]
    ])
    return features.astype(np.float32)


def compute_descriptor_topology_only(persistence_data: Dict[str, Any], max_homology_dim: int = 1) -> np.ndarray:
    """Version B: Topology features only."""
    topo_features = []
    for dim in range(max_homology_dim + 1):
        key = f'h{dim}_features'
        if key in persistence_data['persistence_features']:
            features = persistence_data['persistence_features'][key]
            topo_features.extend([
                features['max_persistence'],
                features['mean_persistence'],
                features['n_features']
            ])
        else:
            topo_features.extend([0.0, 0.0, 0.0])
    return np.array(topo_features, dtype=np.float32)


def compute_descriptor_combined(resonance_signature: Dict[str, Any], 
                                persistence_data: Dict[str, Any],
                                max_homology_dim: int = 1) -> np.ndarray:
    """Version C: Combined (current implementation)."""
    spectral = compute_descriptor_spectral_only(resonance_signature)
    topology = compute_descriptor_topology_only(persistence_data, max_homology_dim)
    return np.concatenate([spectral, topology])


def eval_json_with_descriptor_type(arm: ARMMapper, descriptor_type: str, 
                                   strengths: List[float]) -> Dict[str, Any]:
    """
    Evaluate JSON adherence using specified descriptor type.
    Uses exact same approach as working harness, but swaps descriptor computation.
    """
    # Load fixed exemplar pool
    json_seed_file = "test-data/json_test.txt"
    with open(json_seed_file, "r", encoding="utf-8") as f:
        seed_lines = [ln.strip() for ln in f if ln.strip()]

    # Analyze each seed
    print(f"  Analyzing {len(seed_lines)} seeds...")
    seed_analyses = []
    for prompt in seed_lines:
        analysis = arm.analyze_seed_point(prompt)
        seed_analyses.append(analysis)
    
    # Compute descriptors based on type
    print(f"  Computing {descriptor_type} descriptors...")
    descriptors = []
    for analysis in seed_analyses:
        if descriptor_type == 'spectral':
            desc = compute_descriptor_spectral_only(analysis['resonance_signature'])
        elif descriptor_type == 'topology':
            desc = compute_descriptor_topology_only(analysis['persistence_data'], arm.config.max_homology_dim)
        elif descriptor_type == 'combined':
            desc = compute_descriptor_combined(
                analysis['resonance_signature'],
                analysis['persistence_data'],
                arm.config.max_homology_dim
            )
        else:
            raise ValueError(f"Unknown descriptor_type: {descriptor_type}")
        descriptors.append(desc)
    
    descriptors = np.array(descriptors)
    print(f"  Descriptor shape: {descriptors.shape}")
    
    # Target signature: average across all seeds (using s_norm for steering)
    s_norms = [a['resonance_signature']['s_norm'] for a in seed_analyses]
    min_len = min(len(s) for s in s_norms)
    target_sig = np.mean(np.stack([s[:min_len] for s in s_norms], axis=0), axis=0)

    # Few-shot prompt prefix (KEY INGREDIENT FROM WORKING HARNESS)
    rng = np.random.default_rng(42)
    k = min(2, len(seed_lines))
    shots = rng.choice(seed_lines, size=k, replace=False)
    examples_block = "\n".join(shots)
    request_prefix = (
        f"Generate JSON similar to the following examples (JSON only):\n{examples_block}\n\n"
        "Task: Given name=Alice, age=30, city=Paris. Respond with exactly: {\"name\":\"Alice\",\"age\":30,\"city\":\"Paris\"}"
    )
    required_keys = ["name", "age", "city"]

    # Build manifold data for steering
    resonance_sigs = [a['resonance_signature'] for a in seed_analyses]
    graph_data = arm.topology_mapper.build_resonance_graph(resonance_sigs)
    
    manifold_data = {
        'seed_analyses': seed_analyses,
        'graph_data': graph_data,
        'descriptors': descriptors,
        'n_seeds': len(seed_lines),
        'prompts': seed_lines,
        'layer_to_probe': arm.config.layer_to_probe,
    }
    
    # Set manifold data on mapper for steering
    arm._last_results = manifold_data

    results = []
    for s in strengths:
        print(f"    Testing strength={s}...")
        if s > 0:
            # Manifold-signature steering
            text = arm.steer_generation_toward_signature(
                prompt=request_prefix,
                target_signature=target_sig,
                max_length=40,
                temperature=0.8,
                steering_strength=s,
            )
        else:
            # Baseline with beam search
            gen = arm.create_controlled_generator()
            text = gen.generate_with_steering(
                prompt=request_prefix,
                max_length=40,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        # Constrained JSON extraction (from working harness)
        import re
        filtered = re.sub(r"[^\{\}\[\]\:\,\"0-9A-Za-z\s]", "", text)
        m = re.search(r"\{[^{}]*\}", filtered, flags=re.DOTALL)
        text_for_scoring = m.group(0) if m else filtered
        
        score = safe_json_adherence_score(text_for_scoring, required_keys)
        results.append({
            "strength": s,
            "score": score,
            "output": text[:100],  # Truncate for display
        })
        print(f"      Score: {score}")

    return {
        'descriptor_type': descriptor_type,
        'results': results,
    }


def main():
    print("="*80)
    print("EXERCISE 2.1: ABLATION STUDY (Fixed)")
    print("Using working harness infrastructure")
    print("="*80)

    # Use same config as working harness
    arm_config = ARMConfig(
        model_name="gpt2-medium",
        layer_to_probe=3,
        n_seeds=3,
        probes_per_seed=2,
        steps_per_probe=2,
        eps=0.03,
        n_modes=4,
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {arm_config.model_name}")
    print(f"  Layer: {arm_config.layer_to_probe}")
    print(f"  Probes: {arm_config.probes_per_seed} x {arm_config.steps_per_probe}")
    
    arm = ARMMapper(arm_config)
    
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    descriptor_types = ['spectral', 'topology', 'combined']
    
    all_results = {}
    
    for desc_type in descriptor_types:
        print(f"\n{'-'*80}")
        print(f"Testing: {desc_type.upper()} descriptors")
        print(f"{'-'*80}")
        
        result = eval_json_with_descriptor_type(arm, desc_type, strengths)
        all_results[desc_type] = result
    
    # Summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    print("\nJSON Adherence by Descriptor Type and Steering Strength:")
    print("-"*80)
    print(f"{'Type':<15} ", end="")
    for s in strengths:
        print(f"s={s:<6.1f} ", end="")
    print()
    print("-"*80)
    
    for desc_type in descriptor_types:
        scores = [r['score'] for r in all_results[desc_type]['results']]
        print(f"{desc_type:<15} ", end="")
        for score in scores:
            print(f"{score:<8.2f} ", end="")
        print()
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    spectral_scores = [r['score'] for r in all_results['spectral']['results']]
    topology_scores = [r['score'] for r in all_results['topology']['results']]
    combined_scores = [r['score'] for r in all_results['combined']['results']]
    
    # Best performance at high strength
    best_spectral = max(spectral_scores[1:])  # Exclude baseline
    best_topology = max(topology_scores[1:])
    best_combined = max(combined_scores[1:])
    
    print(f"\nBest Performance (excluding baseline):")
    print(f"  Spectral only:  {best_spectral:.3f}")
    print(f"  Topology only:  {best_topology:.3f}")
    print(f"  Combined:       {best_combined:.3f}")
    
    # Verdict
    print("\n" + "-"*80)
    print("VERDICT")
    print("-"*80)
    
    improvement = best_combined - best_spectral
    
    if best_combined > best_spectral + 0.15:
        print(f"[TOPOLOGY HELPS] Combined outperforms spectral by {improvement:+.3f}")
        print("  Recommendation: Keep topology features")
    elif abs(improvement) < 0.1:
        print(f"[TOPOLOGY NEUTRAL] Similar performance (diff: {improvement:+.3f})")
        print("  Recommendation: Topology is likely vestigial - consider dropping")
    else:
        print(f"[TOPOLOGY HURTS] Spectral beats combined by {-improvement:.3f}")
        print("  Recommendation: Drop topology features")
    
    if best_topology > 0.3:
        print(f"\n  Note: Topology alone achieves {best_topology:.3f}")
        print("  Some independent signal in topology features")
    else:
        print(f"\n  Note: Topology alone only {best_topology:.3f}")
        print("  Insufficient for steering by itself")
    
    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "ablation_study_fixed_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['descriptor_type', 'strength'] + [f's={s}' for s in strengths])
        for desc_type in descriptor_types:
            scores = [r['score'] for r in all_results[desc_type]['results']]
            writer.writerow([desc_type] + [''] + scores)
    
    print(f"\n\nResults saved to: {csv_path}")
    print("="*80)

if __name__ == "__main__":
    main()

