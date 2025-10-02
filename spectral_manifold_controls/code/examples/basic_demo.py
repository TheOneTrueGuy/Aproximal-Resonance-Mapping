"""
Basic SMC Demo - Weighted Index Composition

This demonstrates the full SMC workflow:
1. Build manifold from corpus (one-time, expensive)
2. Add semantic labels
3. Mix indices with weights (instant, reusable)
4. Show the power of compositional control
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core import ManifoldBuilder, WeightedControlVectorComputer


def main():
    print("="*80)
    print("SMC BASIC DEMO: Weighted Index Composition")
    print("="*80)
    
    # Step 1: Build manifold (one-time cost)
    print("\n[1/4] Building manifold from diverse corpus...")
    print("      (This is the expensive one-time operation)")
    
    corpus = [
        # Formal examples (indices 0-2)
        "The research demonstrates significant findings in this domain.",
        "According to our analysis, the hypothesis is supported by empirical evidence.",
        "The methodology employed follows established scientific protocols.",
        
        # Friendly examples (indices 3-5)
        "Hey there! Hope you're having a great day!",
        "Thanks so much for your help, really appreciate it!",
        "That's awesome! Let me know how it goes!",
        
        # Technical examples (indices 6-8)
        "The algorithm achieves O(n log n) complexity through divide-and-conquer.",
        "Import the module using: from transformers import AutoModel",
        "Configure the hyperparameters in the config.json file.",
        
        # Casual examples (indices 9-11)
        "lol that's pretty cool ngl",
        "yeah i think it's gonna work out fine",
        "nah don't worry about it tbh",
    ]
    
    builder = ManifoldBuilder(
        model_name="distilgpt2",
        device="cpu",
        n_modes=4
    )
    
    manifold = builder.build_manifold(
        corpus=corpus,
        layer=3,
        probes_per_seed=2,  # Small for demo
        steps_per_probe=2,
        eps=0.03
    )
    
    print(f"      [OK] Built manifold with {len(corpus)} examples")
    
    # Step 2: Add semantic labels
    print("\n[2/4] Adding semantic labels to indices...")
    manifold.add_labels({
        'formal': [0, 1, 2],
        'friendly': [3, 4, 5],
        'technical': [6, 7, 8],
        'casual': [9, 10, 11],
    })
    print(f"      [OK] Labels: {list(manifold.index_labels.keys())}")
    
    # Step 3: Create control vectors with different "recipes"
    print("\n[3/4] Creating control vectors with different weight mixes...")
    print("      (These are instant - no recomputation!)")
    
    computer = WeightedControlVectorComputer(builder.model_interface)
    
    # Recipe 1: Professional (formal + technical, avoid casual)
    print("\n      Recipe 1: Professional")
    print("        - 70% formal, 50% technical, avoid casual")
    recipe_professional = {
        'formal': 0.7,
        'technical': 0.5,
        'casual': -0.2,
    }
    cv_professional = computer.compute_from_labels(
        recipe_professional,
        manifold.index_labels,
        manifold.corpus,
        layer=3
    )
    print(f"        [OK] Created control vector for layer {cv_professional.layer}")
    
    # Recipe 2: Friendly technical (balance friendly + technical)
    print("\n      Recipe 2: Friendly Technical")
    print("        - 60% friendly, 60% technical")
    recipe_friendly_tech = {
        'friendly': 0.6,
        'technical': 0.6,
    }
    cv_friendly_tech = computer.compute_from_labels(
        recipe_friendly_tech,
        manifold.index_labels,
        manifold.corpus,
        layer=3
    )
    print(f"        [OK] Created control vector for layer {cv_friendly_tech.layer}")
    
    # Recipe 3: Ultra casual (max casual, avoid formal)
    print("\n      Recipe 3: Ultra Casual")
    print("        - 90% casual, avoid formal and technical")
    recipe_casual = {
        'casual': 0.9,
        'formal': -0.3,
        'technical': -0.2,
    }
    cv_casual = computer.compute_from_labels(
        recipe_casual,
        manifold.index_labels,
        manifold.corpus,
        layer=3
    )
    print(f"        [OK] Created control vector for layer {cv_casual.layer}")
    
    # Step 4: Show interpolation
    print("\n[4/4] Bonus: Interpolating between recipes...")
    cv_blend = computer.interpolate_vectors(
        cv_professional,
        cv_friendly_tech,
        alpha=0.5
    )
    print(f"      [OK] Created 50/50 blend of professional and friendly-technical")
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nWhat we demonstrated:")
    print("  [OK] Built manifold ONCE from 12 examples")
    print("  [OK] Created 4 different control vectors INSTANTLY")
    print("  [OK] Used semantic labels (not raw indices)")
    print("  [OK] Mixed with positive and negative weights")
    print("  [OK] Interpolated between recipes")
    print("\nKey insight:")
    print("  The expensive part (manifold building) happens once.")
    print("  After that, creating new control vectors is instant!")
    print("  This is the power of SMC's compositional approach.")
    print("\nNext steps:")
    print("  - Save manifold for reuse")
    print("  - Save recipes as JSON")
    print("  - Use control vectors for generation")
    print("  - Build interactive mixing board UI")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

