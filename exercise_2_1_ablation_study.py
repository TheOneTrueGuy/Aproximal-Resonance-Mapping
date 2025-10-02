#!/usr/bin/env python3
"""
Exercise 2.1: Ablation Study - Topology vs Spectral Only
From critique_10-1-25.txt

CRITICAL QUESTION: Does topology actually contribute to steering performance,
or is it vestigial computational overhead?

This test will compare:
- Version A: Spectral only (s_norm, entropy, PR) 
- Version B: Topology only (persistence features)
- Version C: Combined (current implementation)

We'll test on JSON adherence and measure steering effectiveness.
"""

import sys
import numpy as np
import torch
from typing import Dict, Any, List
import json
from pathlib import Path

# Add arm_library to path
sys.path.insert(0, str(Path(__file__).parent))

from arm_library.core.arm_mapper import ARMMapper
from arm_library.core.resonance_analyzer import ResonanceAnalyzer
from arm_library.core.topology_mapper import TopologyMapper
from arm_library.utils.config import ARMConfig
from arm_library.interfaces.model_interface import TransformerModelInterface, ModelConfig

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
            # Pad with zeros if no features in this dimension
            topo_features.extend([0.0, 0.0, 0.0])
    
    return np.array(topo_features, dtype=np.float32)

def compute_descriptor_combined(resonance_signature: Dict[str, Any], 
                                persistence_data: Dict[str, Any],
                                max_homology_dim: int = 1) -> np.ndarray:
    """Version C: Combined (current implementation)."""
    spectral = compute_descriptor_spectral_only(resonance_signature)
    topology = compute_descriptor_topology_only(persistence_data, max_homology_dim)
    return np.concatenate([spectral, topology])

def build_manifold_with_descriptor_type(
    arm_mapper: ARMMapper,
    seed_prompts: List[str],
    descriptor_type: str
) -> Dict[str, Any]:
    """
    Build manifold using specified descriptor type.
    
    Args:
        arm_mapper: ARM mapper instance
        seed_prompts: List of seed prompts
        descriptor_type: 'spectral', 'topology', or 'combined'
    
    Returns:
        Results dict with descriptors of specified type
    """
    seed_analyses = []
    
    for prompt in seed_prompts:
        analysis = arm_mapper.analyze_seed_point(prompt)
        seed_analyses.append(analysis)
    
    # Compute descriptors based on type
    descriptors = []
    for analysis in seed_analyses:
        if descriptor_type == 'spectral':
            desc = compute_descriptor_spectral_only(analysis['resonance_signature'])
        elif descriptor_type == 'topology':
            desc = compute_descriptor_topology_only(
                analysis['persistence_data'], 
                arm_mapper.config.max_homology_dim
            )
        elif descriptor_type == 'combined':
            desc = compute_descriptor_combined(
                analysis['resonance_signature'],
                analysis['persistence_data'],
                arm_mapper.config.max_homology_dim
            )
        else:
            raise ValueError(f"Unknown descriptor_type: {descriptor_type}")
        
        descriptors.append(desc)
    
    descriptors = np.array(descriptors)
    
    # Build graph and clustering (using whatever features we have)
    resonance_sigs = [analysis['resonance_signature'] for analysis in seed_analyses]
    graph_data = arm_mapper.topology_mapper.build_resonance_graph(resonance_sigs)
    clustering_data = arm_mapper.topology_mapper.detect_attractor_basins(resonance_sigs, graph_data)
    
    return {
        'seed_analyses': seed_analyses,
        'graph_data': graph_data,
        'clustering_data': clustering_data,
        'descriptors': descriptors,
        'n_seeds': len(seed_prompts),
        'prompts': seed_prompts,
        'layer_to_probe': arm_mapper.config.layer_to_probe,
        'descriptor_type': descriptor_type,
    }

def evaluate_json_adherence(
    model_interface: TransformerModelInterface,
    test_prompts: List[str],
    manifold_data: Dict[str, Any],
    steering_strength: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate JSON adherence with manifold-signature steering.
    """
    from arm_library.core.steering import ARMControlledGenerator
    
    generator = ARMControlledGenerator(model_interface)
    
    # Compute average signature for steering
    descriptors = manifold_data['descriptors']
    target_signature = descriptors.mean(axis=0)
    
    results = []
    for prompt in test_prompts:
        # Generate with steering
        output = generator.generate_with_manifold_steering(
            prompt=prompt,
            target_signature=target_signature,
            manifold_data=manifold_data,
            max_length=50,
            temperature=0.7,
            steering_strength=steering_strength,
        )
        
        # Check for JSON structure
        try:
            # Try to parse as JSON
            parsed = json.loads(output)
            has_required = all(k in parsed for k in ['name', 'age', 'city'])
            score = 1.0 if has_required else 0.5
        except:
            # Not valid JSON
            score = 0.0
        
        results.append({
            'prompt': prompt,
            'output': output,
            'score': score
        })
    
    avg_score = np.mean([r['score'] for r in results])
    
    return {
        'results': results,
        'avg_score': avg_score,
        'descriptor_type': manifold_data['descriptor_type'],
        'steering_strength': steering_strength,
    }

def run_ablation_study():
    """
    Main ablation study comparing spectral-only, topology-only, and combined.
    """
    print("="*80)
    print("EXERCISE 2.1: ABLATION STUDY")
    print("Testing: Does topology contribute to steering performance?")
    print("="*80)
    
    # Configuration
    config = ARMConfig(
        model_name="gpt2-medium",
        device="cpu",
        n_seeds=10,  # Use reasonable number
        probes_per_seed=8,
        steps_per_probe=5,
        eps=0.05,
        layer_to_probe=3,
        n_modes=8,
        max_homology_dim=1,
        random_seed=42,
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Probes: {config.probes_per_seed}")
    print(f"  Steps: {config.steps_per_probe}")
    print(f"  Layer: {config.layer_to_probe}")
    
    # Initialize ARM mapper
    print("\nInitializing ARM mapper...")
    arm_mapper = ARMMapper(config)
    
    # Load JSON corpus for manifold building
    json_corpus_path = Path("test-data/json_test.txt")
    if json_corpus_path.exists():
        with open(json_corpus_path) as f:
            corpus_lines = [line.strip() for line in f if line.strip()]
        seed_prompts = corpus_lines[:config.n_seeds]
    else:
        # Fallback to hardcoded examples
        seed_prompts = [
            '{"name": "Alice", "age": 30, "city": "NYC"}',
            '{"name": "Bob", "age": 25, "city": "LA"}',
            '{"name": "Carol", "age": 35, "city": "Chicago"}',
            '{"name": "David", "age": 28, "city": "Boston"}',
            '{"name": "Eve", "age": 32, "city": "Seattle"}',
        ][:config.n_seeds]
    
    print(f"\nBuilding manifolds from {len(seed_prompts)} JSON examples...")
    
    # Test prompts
    test_prompts = [
        "Generate a person profile:",
        "Create user data:",
        "Make a JSON object:",
    ]
    
    # Run ablation for each descriptor type
    descriptor_types = ['spectral', 'topology', 'combined']
    results_by_type = {}
    
    for desc_type in descriptor_types:
        print(f"\n{'-'*80}")
        print(f"Testing: {desc_type.upper()} features")
        print(f"{'-'*80}")
        
        # Build manifold with this descriptor type
        print(f"  Building manifold...")
        manifold_data = build_manifold_with_descriptor_type(
            arm_mapper, seed_prompts, desc_type
        )
        
        print(f"  Descriptor shape: {manifold_data['descriptors'].shape}")
        
        # Test at different steering strengths
        strengths = [0.0, 0.5, 1.0, 2.0]
        eval_results = []
        
        for strength in strengths:
            print(f"  Testing steering strength: {strength}")
            eval_result = evaluate_json_adherence(
                arm_mapper.model_interface,
                test_prompts,
                manifold_data,
                steering_strength=strength
            )
            eval_results.append(eval_result)
            print(f"    JSON adherence score: {eval_result['avg_score']:.2f}")
        
        results_by_type[desc_type] = eval_results
    
    # Analysis and comparison
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    print("\nJSON Adherence by Descriptor Type and Steering Strength:")
    print("-"*80)
    print(f"{'Descriptor Type':<20} {'Strength=0.0':<15} {'Strength=0.5':<15} {'Strength=1.0':<15} {'Strength=2.0':<15}")
    print("-"*80)
    
    for desc_type in descriptor_types:
        scores = [r['avg_score'] for r in results_by_type[desc_type]]
        print(f"{desc_type:<20} {scores[0]:<15.2f} {scores[1]:<15.2f} {scores[2]:<15.2f} {scores[3]:<15.2f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Compare performance
    spectral_scores = [r['avg_score'] for r in results_by_type['spectral']]
    topology_scores = [r['avg_score'] for r in results_by_type['topology']]
    combined_scores = [r['avg_score'] for r in results_by_type['combined']]
    
    # Find best performer at high strength (strength=1.0)
    best_spectral = spectral_scores[2]
    best_topology = topology_scores[2]
    best_combined = combined_scores[2]
    
    print(f"\nBest Performance at Steering Strength = 1.0:")
    print(f"  Spectral only:  {best_spectral:.3f}")
    print(f"  Topology only:  {best_topology:.3f}")
    print(f"  Combined:       {best_combined:.3f}")
    
    # Determine if topology adds value
    print("\n" + "-"*80)
    print("VERDICT")
    print("-"*80)
    
    topology_improvement = best_combined - best_spectral
    topology_alone = best_topology
    
    if best_combined > best_spectral + 0.1:  # 10% improvement threshold
        print("[TOPOLOGY HELPS] Combined performs better than spectral-only")
        print(f"  Improvement: +{topology_improvement:.3f}")
        print("  Recommendation: Keep topology features")
    elif abs(best_combined - best_spectral) < 0.05:  # Within 5%
        print("[TOPOLOGY NEUTRAL] Combined and spectral-only perform similarly")
        print(f"  Difference: {topology_improvement:+.3f}")
        print("  Recommendation: Topology may be vestigial - consider dropping for speed")
    else:
        print("[TOPOLOGY HURTS] Spectral-only outperforms combined")
        print(f"  Degradation: {topology_improvement:.3f}")
        print("  Recommendation: Drop topology features")
    
    if topology_alone > 0.3:
        print(f"\n  Note: Topology-only achieves {topology_alone:.3f}")
        print("  This suggests topology captures some useful signal independently")
    else:
        print(f"\n  Note: Topology-only achieves only {topology_alone:.3f}")
        print("  This suggests topology alone is insufficient for steering")
    
    print("\n" + "="*80)
    
    # Save results
    output_path = Path("arm_output/ablation_study_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert to serializable format
        save_data = {
            desc_type: [
                {
                    'steering_strength': r['steering_strength'],
                    'avg_score': r['avg_score'],
                    'results': r['results']
                }
                for r in results_by_type[desc_type]
            ]
            for desc_type in descriptor_types
        }
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nExercise 2.1 complete!")

if __name__ == "__main__":
    run_ablation_study()

