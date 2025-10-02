# Spectral Manifold Control (SMC)
## Compositional Language Model Steering via Weighted Index Mixing

**Discovery Date**: October 1, 2025  
**Status**: Validated concept, ready for implementation

---

## What Is SMC?

A novel approach to steering language models using **weighted composition of spectral signatures** from a pre-built manifold.

**Key Innovation**: Build a reusable manifold once, then compose control vectors instantly by mixing indices with individual weights - like a "style mixing board."

---

## Three Core Capabilities

### 1. **Positive-Only Steering**
- No need for negative examples in many cases
- Build manifold from target examples only
- Steer toward average signature
- **Validated**: 1.0 JSON adherence with 50 positive examples

### 2. **Index-Based Composition**
- Build manifold ONCE (expensive upfront)
- Mix indices INSTANTLY (no recomputation)
- Unlike RepE which requires recomputing for each variant

### 3. **Weighted Mixing** (The Breakthrough!)
```python
# Not binary (in/out), but fine-grained weights:
recipe = {
    'formal': 0.7,      # 70% formal style
    'friendly': 0.5,    # 50% friendly tone
    'technical': 0.3,   # 30% technical language
    'casual': -0.2,     # Avoid casual language
}
```

---

## The "Style Mixing Board" Concept

**Metaphor**: Treat manifold indices like audio mixer tracks with individual volume controls

```
Style Tracks (adjustable weights):
[0] Formal      ████████░░  0.8
[1] Technical   ███░░░░░░░  0.3
[5] Casual      ░░░░░░░░░░ -0.2 (negative)
[12] Friendly   █████░░░░░  0.5

           ↓
    Generate with this mix
```

**Practical Benefits**:
- Save "recipes" as lightweight JSON
- Share recipes with team
- Version control style evolution
- Instant experimentation

---

## Why This Matters

### SMC vs Traditional RepE

| Feature | RepE | SMC |
|---------|------|-----|
| Build cost | Low | Higher (once) |
| Reusability | None - recompute each time | Infinite - build once, use forever |
| Composition | Manual - need new examples | Automatic - index mixing |
| Fine-tuning | Binary (in/out) | Weighted (0.7*this + 0.3*that) |
| Exploration | Slow - recompute each variant | Fast - instant index selection |
| Sharing | Share examples + code | Share tiny JSON recipes |

**Trade-off**: Higher upfront cost for long-term compositional power

---

## Validated Through Testing

### What Works ✅
- Spectral analysis (SVD-based signatures)
- Positive-only steering (no negatives needed)
- Multi-seed aggregation
- Index composition

### What Doesn't ❌
- Topology features (redundant, actually hurt performance)
- "Resonance" terminology (no empirical oscillations)

See `empirical_validation.md` for details.

---

## Key Documents

- **`concept.md`** - Full specification and use cases
- **`empirical_validation.md`** - Test results and findings
- **`implementation_plan.md`** - Roadmap for building SMC
- **`recipe_format.md`** - Recipe specification

---

## Quick Example

```python
# One-time: Build manifold from diverse corpus
manifold = build_manifold([
    "formal document 1", "formal document 2",
    "casual chat 1", "casual chat 2",
    "technical paper 1", "technical paper 2",
    # ... etc
])

# Daily: Experiment with recipes instantly
recipe_professional = {'formal': 0.8, 'technical': 0.5}
recipe_friendly = {'friendly': 0.9, 'casual': 0.6}
recipe_balanced = {'formal': 0.5, 'friendly': 0.5}

# Generate with different recipes (no recomputation!)
cv1 = apply_recipe(recipe_professional, manifold)
cv2 = apply_recipe(recipe_friendly, manifold)
cv3 = apply_recipe(recipe_balanced, manifold)
```

---

## Dependencies

**Note**: SMC currently imports model interface utilities from the legacy ARM codebase:
- `arm_library.interfaces.model_interface.TransformerModelInterface`
- This is infrastructure code (HuggingFace wrappers), not ARM-specific theory
- Allows SMC to work with transformer models without code duplication
- Future: May copy to SMC for full independence (see implementation_plan.md Phase 2)

## Next: Implementation

See `implementation_plan.md` for the development roadmap.

