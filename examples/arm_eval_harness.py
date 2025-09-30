#!/usr/bin/env python3
"""
Minimal evaluation harness for ARM steering.

Compares baseline (no steering) vs. control-vector ARM steering across a sweep of
steering strengths on two simple tasks:

1) JSON adherence: Given an instruction and schema, measure whether output parses as JSON
   and matches required keys/types (lenient heuristic).

2) Style transfer: Generate in the style of a target author; score via a
   shallow classifier heuristic using lexical markers and a simple keyword model.

Outputs:
- results CSV per task under arm_output/
- dose–response plots (score vs. strength) under arm_output/

This harness uses ARM's control-vector steering only; it does not require topology.
"""

import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


# ------------------------- Utilities -------------------------

OUTPUT_DIR = "arm_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_json_adherence_score(text: str, required_keys: List[str]) -> float:
    """Return 1.0 if text contains a valid JSON object with all required keys, else 0.0.
    Uses a lenient heuristic: find first {...} block and attempt eval via json.
    """
    import json
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return 0.0
    candidate = text[start:end + 1]
    try:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            return 0.0
        for k in required_keys:
            if k not in obj:
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def simple_style_score(text: str, positive_markers: List[str], negative_markers: List[str]) -> float:
    """Heuristic style score: (#pos_markers - #neg_markers) normalized by length.
    Range roughly in [-1, 1]."""
    text_lower = text.lower()
    pos = sum(text_lower.count(m.lower()) for m in positive_markers)
    neg = sum(text_lower.count(m.lower()) for m in negative_markers)
    length_norm = max(1, len(text_lower) // 50)
    return (pos - neg) / float(length_norm)


def plot_dose_response(x_values: List[float], y_values: List[float], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker='o')
    plt.title(title)
    plt.xlabel('Steering strength')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ------------------------- Harness -------------------------

@dataclass
class EvalConfig:
    model_name: str = "distilgpt2"
    layer_to_probe: int = 3
    max_tokens: int = 80
    temperature: float = 0.8
    strengths: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)


def build_arm(config: EvalConfig) -> ARMMapper:
    arm_config = ARMConfig(
        model_name=config.model_name,
        layer_to_probe=config.layer_to_probe,
        n_seeds=3,
        probes_per_seed=2,
        steps_per_probe=2,
        eps=0.03,
        n_modes=4,
    )
    return ARMMapper(arm_config)


def quick_signature(arm: ARMMapper, text: str, k: int = 2, steps: int = 3, eps: float = 0.02):
    """Compute a lightweight resonance signature for arbitrary text."""
    A = arm.collect_activation_matrix(text, k=k, steps=steps, eps=eps)
    return arm.resonance_analyzer.resonance_signature(A)


def topology_summary(manifold: Dict[str, Any]) -> Dict[str, Any]:
    """Compute simple topology/cluster summary metrics from manifold results."""
    from sklearn.metrics import silhouette_score

    out = {}
    try:
        X = manifold['graph_data']['feature_vectors']
        labels = np.array(manifold['clustering_data']['cluster_labels'])

        # Silhouette only valid if >1 cluster and each cluster has >1 sample
        if len(np.unique(labels)) > 1 and len(X) >= 3:
            out['silhouette'] = float(silhouette_score(X, labels))
        else:
            out['silhouette'] = None

        sizes = [int(np.sum(labels == i)) for i in range(manifold['clustering_data']['n_clusters'])]
        out['cluster_sizes'] = sizes
        out['n_clusters'] = int(manifold['clustering_data']['n_clusters'])
    except Exception:
        out['silhouette'] = None
    return out


def eval_manifold_signature(arm: ARMMapper, cfg: EvalConfig,
                            pos: List[str], neg: List[str], request: str,
                            target_idx: int = 0) -> Dict[str, Any]:
    """Sweep manifold-signature steering strength and measure signature similarity lift."""
    # Build manifold from exemplars
    manifold_prompts = pos + neg
    manifold = arm.map_latent_manifold(manifold_prompts)

    # Target is one of the positive exemplars by default
    target_sig = manifold['seed_analyses'][target_idx]['resonance_signature']['s_norm']

    results = []
    for s in cfg.strengths:
        if s > 0:
            generated = arm.steer_generation_toward_signature(
                prompt=request,
                target_signature=target_sig,
                max_length=cfg.max_tokens,
                temperature=cfg.temperature,
                steering_strength=s,
            )
        else:
            gen = arm.create_controlled_generator()
            generated = gen.generate_with_steering(
                prompt=request,
                max_length=cfg.max_tokens,
                temperature=cfg.temperature,
                do_sample=True,
            )

        sig = quick_signature(arm, generated)
        m = min(len(sig['s_norm']), len(target_sig))
        denom = (np.linalg.norm(sig['s_norm'][:m]) * np.linalg.norm(target_sig[:m]) + 1e-12)
        cosine = float(np.dot(sig['s_norm'][:m], target_sig[:m]) / denom)

        results.append({
            'strength': s,
            'signature_cosine': cosine,
            'output': generated,
        })

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "manifold_signature_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strength", "signature_cosine", "output"])
        writer.writeheader()
        writer.writerows(results)

    # Plot
    plot_path = plot_dose_response(
        [r['strength'] for r in results],
        [r['signature_cosine'] for r in results],
        title="Signature cosine vs. steering strength (manifold)",
        ylabel="Cosine(sim(generated, target))",
        filename="manifold_signature_dose_response.png",
    )

    # Topology/cluster summary
    topo = topology_summary(manifold)

    return {
        'results_csv': csv_path,
        'plot': plot_path,
        'rows': results,
        'topology': topo,
    }


def compute_control_vector_from_examples(arm: ARMMapper, pos_prompts: List[str], neg_prompts: List[str], strength: float):
    control = arm.compute_steering_vector_from_manifold(
        positive_region_indices=list(range(len(pos_prompts))),
        negative_region_indices=list(range(len(pos_prompts), len(pos_prompts) + len(neg_prompts))),
        layer=arm.config.layer_to_probe
    )
    control.coefficient = strength
    return control


def eval_json_adherence(arm: ARMMapper, cfg: EvalConfig) -> Dict[str, Any]:
    # Positive/negative exemplars for control-vector computation
    pos = [
        "Return valid JSON with keys name, age, city only.",
        "Output only a JSON object with fields name, age, city.",
    ]
    neg = [
        "Write a prose paragraph about a person.",
    ]

    # Build a transient manifold by analyzing the examples
    # We do not need full topology; map_latent_manifold provides storage for prompts
    manifold_prompts = pos + neg
    arm.map_latent_manifold(manifold_prompts)

    request = (
        "Given: name=Alice, age=30, city=Paris. Output only a JSON object with keys "
        "name, age, city and no extra text."
    )
    required_keys = ["name", "age", "city"]

    results = []
    strengths = list(cfg.strengths)
    for s in strengths:
        gen = arm.create_controlled_generator()
        if s > 0:
            control = arm.compute_steering_vector_from_manifold(
                positive_region_indices=list(range(len(pos))),
                negative_region_indices=list(range(len(pos), len(pos) + len(neg))),
                layer=arm.config.layer_to_probe
            )
            control.coefficient = s
            gen.set_control(control)

        text = gen.generate_with_steering(
            prompt=request,
            max_length=cfg.max_tokens,
            temperature=cfg.temperature,
            do_sample=True,
        )

        score = safe_json_adherence_score(text, required_keys)
        results.append({
            "strength": s,
            "score": score,
            "output": text,
        })

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "json_adherence_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strength", "score", "output"])
        writer.writeheader()
        writer.writerows(results)

    # Plot dose–response
    plot_path = plot_dose_response(
        [r["strength"] for r in results],
        [r["score"] for r in results],
        title="JSON adherence vs. steering strength",
        ylabel="Parse & keys present (0/1)",
        filename="json_adherence_dose_response.png",
    )

    return {
        "results_csv": csv_path,
        "plot": plot_path,
        "rows": results,
    }


def eval_style_transfer(arm: ARMMapper, cfg: EvalConfig) -> Dict[str, Any]:
    # Target style markers (example: whimsical/Jabberwocky-like)
    positive_markers = ["gyre", "gimble", "toves", "borogoves", "slithy", "frumious", "jabberwock"]
    negative_markers = ["therefore", "hence", "in conclusion"]

    pos = [
        "Continue in a whimsical, nonsensical, Carroll-like verse style with neologisms.",
        "Write in playful, rhythmic nonsense poetry with invented words.",
    ]
    neg = [
        "Write a dry technical summary with formal tone and no poetry.",
    ]

    # Build manifold for the exemplars
    manifold_prompts = pos + neg
    arm.map_latent_manifold(manifold_prompts)

    request = "The creature lurked in the tulgey wood, and I raised my eyes to speak:"

    results = []
    strengths = list(cfg.strengths)
    for s in strengths:
        gen = arm.create_controlled_generator()
        if s > 0:
            control = arm.compute_steering_vector_from_manifold(
                positive_region_indices=list(range(len(pos))),
                negative_region_indices=list(range(len(pos), len(pos) + len(neg))),
                layer=arm.config.layer_to_probe
            )
            control.coefficient = s
            gen.set_control(control)

        text = gen.generate_with_steering(
            prompt=request,
            max_length=cfg.max_tokens,
            temperature=cfg.temperature,
            do_sample=True,
        )

        score = simple_style_score(text, positive_markers, negative_markers)
        results.append({
            "strength": s,
            "score": score,
            "output": text,
        })

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "style_transfer_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strength", "score", "output"])
        writer.writeheader()
        writer.writerows(results)

    # Plot dose–response
    plot_path = plot_dose_response(
        [r["strength"] for r in results],
        [r["score"] for r in results],
        title="Style score vs. steering strength",
        ylabel="Heuristic style score",
        filename="style_transfer_dose_response.png",
    )

    return {
        "results_csv": csv_path,
        "plot": plot_path,
        "rows": results,
    }


def main():
    cfg = EvalConfig()
    arm = build_arm(cfg)

    print("Running JSON adherence evaluation...")
    json_res = eval_json_adherence(arm, cfg)
    print(f"CSV: {json_res['results_csv']}")
    print(f"Plot: {json_res['plot']}")

    print("\nRunning style transfer evaluation...")
    style_res = eval_style_transfer(arm, cfg)
    print(f"CSV: {style_res['results_csv']}")
    print(f"Plot: {style_res['plot']}")

    print("\nRunning manifold-signature evaluation...")
    # Reuse style exemplars for manifold evaluation
    pos = [
        "Continue in a whimsical, nonsensical, Carroll-like verse style with neologisms.",
        "Write in playful, rhythmic nonsense poetry with invented words.",
    ]
    neg = [
        "Write a dry technical summary with formal tone and no poetry.",
    ]
    request = "The creature lurked in the tulgey wood, and I raised my eyes to speak:"
    mani_res = eval_manifold_signature(arm, cfg, pos, neg, request, target_idx=0)
    print(f"CSV: {mani_res['results_csv']}")
    print(f"Plot: {mani_res['plot']}")
    topo = mani_res.get('topology', {})
    if topo:
        print(f"Topology: n_clusters={topo.get('n_clusters')}, sizes={topo.get('cluster_sizes')}, silhouette={topo.get('silhouette')}")

    print("\nDone. Check arm_output/ for results and plots.")


if __name__ == "__main__":
    main()


