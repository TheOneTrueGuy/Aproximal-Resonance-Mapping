# Next Steps - SMC Implementation

## When You Return

### Immediate Actions (Session Start)

1. **Create SMC structure**
   ```bash
   mkdir -p smc/{core,recipes,interfaces,utils,examples}
   ```

2. **Implement weighted composition** (~2 hours)
   - `smc/core/weighted_control.py`
   - Add `compute_weighted_control_vector()` method

3. **Build mixing board UI** (~3 hours)
   - `smc/interfaces/mixing_board.py`
   - Gradio with sliders for each label/index
   - Real-time generation preview

4. **Test end-to-end** (~1 hour)
   - Load manifold → adjust sliders → generate
   - Save/load recipes

## Key Files to Reference

- **Full spec**: `spectral_manifold_control.md`
- **Today's findings**: `session_summary_oct_1_2025.md`
- **What worked**: `exercise_2_1_ablation_fixed.py`
- **Working harness**: `examples/arm_eval_harness.py`

## The Core Innovation

**Weighted index composition** = Style Mixing Board concept:
```python
recipe = {
    'formal': 0.7,
    'friendly': 0.5,
    'casual': -0.2,
}
```

## What's Validated ✅

- Spectral analysis (SVD) works
- Positive-only steering works
- Index composition works
- Topology is redundant (drop it)
- "Resonance" not empirical (rename to spectral)

## Ready to Build! 🚀

