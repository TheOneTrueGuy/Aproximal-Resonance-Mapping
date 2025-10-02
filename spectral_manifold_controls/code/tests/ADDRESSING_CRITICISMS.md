# Addressing Criticisms for SMC Evals

**Source**: `Criticisms.txt` from ARM project  
**Purpose**: Ensure SMC evals don't repeat ARM's mistakes

---

## The Criticisms About JSON Results

### ✅ What Worked
> "The dose-response curve showing JSON adherence improving from 0.0 → 1.0 with manifold-signature steering is exactly what you'd want to see."

**Validation**: The JSON task DID work with 50 examples, showing clear dose-response.

---

## Critical Problems to Avoid in SMC

### Criticism #1: ❌ "Resonance" terminology
**ARM Issue**: Metaphorical, not mathematical  
**SMC Fix**: ✅ **ALREADY FIXED** - Renamed to "SpectralAnalyzer", documentation clarifies it's SVD

---

### Criticism #2: ❌ Topology may be vestigial
**ARM Issue**: Topology adds overhead without proven benefit  
**SMC Fix**: ✅ **ALREADY FIXED** - Dropped topology entirely based on ablation study

**Evidence**: Ablation showed topology redundant (Exercise 2.1)

---

### Criticism #3: ❌ Weak baseline comparison
**ARM Issue**: Only compared against "no steering" (strength=0.0)  
**SMC Must Do**: 
- [ ] Compare against **few-shot prompting** (very strong for JSON)
- [ ] Compare against **RepE** control vectors
- [ ] Compare against **fine-tuning** (optional, expensive)

**Why this matters**: Few-shot might beat SMC on JSON! Need to prove SMC adds value.

---

### Criticism #4: ❌ Sample size = 1 task
**ARM Issue**: Only tested JSON on gpt2-medium  
**SMC Must Do**:
- [ ] Test **multiple tasks** (JSON, style, sentiment, formality)
- [ ] Test **multiple models** (distilgpt2, gpt2, gpt2-medium)
- [ ] Run with **multiple random seeds** 
- [ ] Add **statistical significance testing**

**Why this matters**: One task + one model = anecdotal, not validated

---

### Criticism #5: ⚠️ Manifold from too few seeds
**ARM Clarification**: Actually used **50 seeds** (not 2-3 as critic thought)  
**SMC Consideration**: 
- Document seed count clearly
- Test: Does SMC work with fewer seeds than ARM? (compositional might be more data-efficient)
- Measure: Manifold quality vs seed count

---

### Criticism #6: ⚠️ Computational cost
**ARM Issue**: `n_seeds × probes × steps` = expensive  
**Example**: 50 seeds × 16 probes × 9 steps = 7,200 forward passes

**SMC Consideration**:
- One-time cost, but still expensive
- Trade-off: Build once, reuse forever
- **Must measure**: Is the upfront cost worth it vs RepE's cheap recompute?

**Test needed**: Speed benchmark (in EVAL_PLAN.md)

---

### Criticism #7: ❌ "Emergent recognition" claim not valid
**ARM Issue**: Claimed boundaries emerge without labels, but still manually providing examples  
**SMC Position**: 
- **Don't claim** boundaries emerge naturally
- **Do claim**: Positive-only steering (no need for negatives in some cases)
- **Be honest**: Labels are still manual (formal/friendly/casual categories)

---

## How SMC Should Eval Differently

### ✅ Already Fixed from ARM
1. **Terminology**: "Spectral" not "resonance" 
2. **Topology**: Dropped (proven redundant)
3. **Documentation**: Honest about what we're doing

### ❌ Still Need to Do
1. **Baseline comparisons**: Few-shot, RepE, (fine-tuning)
2. **Multiple tasks**: Not just JSON
3. **Multiple models**: Test generalization
4. **Statistical testing**: Multiple seeds, significance tests
5. **Cost analysis**: Time/memory benchmarks

---

## Specific Eval Requirements for SMC

### Tier 1: Validate Core Mechanism (Minimum Viable)
- [x] Unit tests for weighted composition
- [ ] JSON task matches ARM performance (1.0 score)
- [ ] Style task shows effect
- [ ] Speed benchmark (SMC vs RepE recompute)

### Tier 2: Prove Value Over Alternatives
- [ ] **Few-shot baseline**: Is SMC better than just better prompting?
- [ ] **RepE baseline**: Is weighted composition better than simple pos/neg?
- [ ] **Weighted vs binary**: Does fine-grained mixing help?

### Tier 3: Generalization & Rigor
- [ ] 3+ tasks tested (JSON, style, sentiment/formality)
- [ ] 3+ models tested (distilgpt2, gpt2, gpt2-medium)
- [ ] 5+ random seeds per test
- [ ] Statistical significance (t-tests, confidence intervals)

### Tier 4: SMC-Specific Claims
- [ ] Positive-only works (no negatives needed)
- [ ] Weighted mixing beats binary selection
- [ ] Recipe composition is predictable
- [ ] Interpolation works smoothly

---

## The Critical Tests We MUST Run

### Test 1: Few-Shot Baseline (High Priority!)
```python
# Baseline: Just better prompting
prompt_0_shot = "Generate JSON:"
prompt_2_shot = "Examples:\n{examples}\n\nGenerate JSON:"
prompt_5_shot = "Examples:\n{examples}\n\nGenerate JSON:"

# SMC: 2-shot + manifold steering
prompt_smc = "Examples:\n{examples}\n\nGenerate JSON:" + STEERING

# Compare: Does SMC beat 5-shot prompting?
if score_smc <= score_5_shot:
    print("WARNING: SMC doesn't beat strong prompting baseline!")
```

**Why critical**: Few-shot is **very strong** for JSON. If we can't beat it, SMC's value is unclear.

---

### Test 2: RepE Comparison
```python
# RepE: mean(positive) - mean(negative)
repe_cv = mean(formal_activations) - mean(casual_activations)

# SMC: Weighted composition
smc_cv = 0.7*formal + 0.5*technical - 0.2*casual

# Compare: Same task, which wins?
# Also measure: Time to create variants (SMC should be faster)
```

**Why critical**: RepE is the current standard. Need to show SMC is competitive or better.

---

### Test 3: Multiple Tasks (Not Just JSON)
```python
tasks = [
    'json_generation',      # Structured output
    'style_transfer',       # Jabberwocky markers
    'sentiment_control',    # Positive vs negative
    'formality_control',    # Formal vs casual
]

for task in tasks:
    smc_score = eval_smc(task)
    baseline_score = eval_baseline(task)
    
    if smc_score <= baseline_score:
        print(f"FAIL on {task}")
```

**Why critical**: One task could be a fluke. Need multiple to claim generalization.

---

### Test 4: Statistical Significance
```python
# Run 10 times with different random seeds
smc_scores = []
baseline_scores = []

for seed in range(10):
    set_seed(seed)
    smc_scores.append(eval_smc())
    baseline_scores.append(eval_baseline())

# t-test
t_stat, p_value = ttest_ind(smc_scores, baseline_scores)

if p_value > 0.05:
    print("WARNING: Difference not statistically significant!")
```

**Why critical**: Without this, we don't know if results are real or random variation.

---

## Mandatory Disclosures in SMC Evals

When reporting results, we MUST include:

1. **Sample size**: How many examples in manifold
2. **Baseline comparisons**: What we're beating (or not)
3. **Statistical tests**: p-values, confidence intervals
4. **Computational cost**: Time and memory vs baselines
5. **Task diversity**: How many different tasks tested
6. **Failure cases**: Where SMC doesn't help

**Honest reporting** builds credibility. Don't oversell.

---

## Red Flags to Avoid

### 🚩 "SMC achieves 1.0 on JSON!"
**Problem**: Without baseline comparison, meaningless  
**Better**: "SMC achieves 1.0 vs baseline 0.0, and matches 5-shot prompting (0.95)"

### 🚩 "Positive-only steering works!"
**Problem**: Maybe just fewer examples, not a win  
**Better**: "Positive-only achieves 90% of pos/neg performance with half the labeling effort"

### 🚩 "Weighted mixing enables fine control!"
**Problem**: Need to prove it's better than binary  
**Better**: "Weighted mixing achieves 0.73 formality vs binary's 0.65 (p<0.01)"

### 🚩 "Build once, use forever!"
**Problem**: If build takes 10x longer but saves 2 variants, not worth it  
**Better**: "Build cost: 120s. Creating 10 recipes: 0.5s vs RepE 45s (9x faster)"

---

## Summary: What SMC Evals Must Include

### Before claiming SMC "works":
- ✅ Spectral analysis validated (we did this)
- ✅ Topology dropped (we did this)
- ❌ **Few-shot baseline** (MUST DO)
- ❌ **RepE baseline** (MUST DO)
- ❌ Multiple tasks (at least 3)
- ❌ Statistical tests (at least 5 seeds)
- ❌ Cost analysis (time/memory)

### Before claiming SMC is "better":
- ❌ Head-to-head vs RepE on same tasks
- ❌ Quantify speed advantage
- ❌ Show where weighted beats binary
- ❌ Demonstrate composition quality

### Before claiming SMC is "practical":
- ❌ Measure build cost vs recompute savings
- ❌ Test with realistic corpus sizes
- ❌ Profile memory usage
- ❌ Identify sweet spots (when to use vs not)

---

## Implementation Priority

### Week 1: Validate Core (avoid ARM's "weak baseline" mistake)
1. Port JSON eval from ARM harness
2. Add few-shot baseline comparison
3. Add RepE baseline comparison
4. **Critical**: Ensure SMC beats or matches baselines

### Week 2: Generalization (avoid ARM's "1 task" mistake)
1. Add style transfer eval
2. Add sentiment/formality eval
3. Test on 3 models
4. Statistical testing (5+ seeds)

### Week 3: SMC-Specific
1. Weighted vs binary comparison
2. Speed benchmarks
3. Recipe composition quality
4. Cost analysis

---

## Honest Assessment Questions

Before publishing/claiming SMC works, answer:

1. **Does SMC beat few-shot prompting?** (If no, limited value)
2. **Does SMC beat RepE?** (If no, not an improvement)
3. **Does weighted beat binary?** (If no, just use RepE)
4. **Is build cost worth reuse benefit?** (If no, not practical)
5. **Does it generalize?** (If no, just a JSON trick)

**Only claim what we can prove with rigorous evals.**

---

## Next Immediate Action

**Implement Test 1**: Few-shot baseline for JSON

This is the **most critical** test. If SMC can't beat strong prompting, the whole value proposition is questionable.

File: `spectral_manifold_controls/code/tests/eval_json_vs_baselines.py`

Time: 2-3 hours

Ready to implement?
