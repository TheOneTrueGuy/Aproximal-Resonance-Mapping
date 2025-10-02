# SMC Evaluation Plan

**Current Status**: We have unit tests for mechanics, but NO evaluations of steering effectiveness.

---

## What We Need

### Level 1: Mechanism Tests (✅ HAVE)
- Weighted composition math works
- Labels convert to indices correctly
- Error handling works
- **Status**: Complete in `test_weighted_control.py`

### Level 2: Integration Tests (⚠️ PARTIAL)
- Manifold builds without errors ✓
- Control vectors can be created ✓
- **Missing**: Actually using them for generation
- **Missing**: Quality metrics on manifold

### Level 3: Task Evaluations (❌ MISSING)
- JSON adherence with weighted recipes
- Style transfer with weighted recipes
- Comparison against baselines (RepE, few-shot)
- **Status**: Not implemented yet

### Level 4: SMC-Specific Evals (❌ MISSING)
- Does weighted mixing beat binary selection?
- Do negative weights improve results?
- Recipe composition quality
- **Status**: Not implemented yet

---

## Recommended Approach

### Quick Win: Adapt ARM Harness

**We already have working eval code in `arm_eval_harness.py`!**

Adapt it for SMC:

```python
# Instead of:
manifold = arm.map_latent_manifold(seed_prompts)

# Do:
builder = ManifoldBuilder(model_name="gpt2-medium")
manifold = builder.build_manifold(seed_prompts, layer=3)

# Instead of:
arm.steer_generation_toward_signature(...)

# Do:
computer = WeightedControlVectorComputer(builder.model_interface)
cv = computer.compute_from_labels(recipe, manifold.index_labels, manifold.corpus, layer=3)
# Then generate with cv
```

### Priority Order

**Phase 1: Port Existing Evals** (2-3 hours)
1. Create `eval_json_adherence_smc.py` based on ARM harness
2. Create `eval_style_transfer_smc.py` based on ARM harness
3. Verify we get similar scores (validates SMC works as well as ARM)

**Phase 2: SMC-Specific Tests** (3-4 hours)
1. Test weighted vs binary selection
2. Test negative weights effectiveness
3. Test recipe interpolation quality
4. Speed benchmarks (build once vs RepE recompute)

**Phase 3: Baseline Comparisons** (4-5 hours)
1. Implement RepE baseline
2. Few-shot baseline
3. Head-to-head comparison
4. Document when SMC wins

---

## Specific Tests to Implement

### Test 1: JSON Adherence (Port from ARM)

**File**: `spectral_manifold_controls/code/tests/eval_json.py`

```python
def eval_json_adherence_smc():
    # Build manifold from test-data/json_test.txt
    builder = ManifoldBuilder("gpt2-medium")
    corpus = load_json_corpus()
    manifold = builder.build_manifold(corpus, layer=3)
    
    # Recipe: steer toward average JSON signature
    recipe = {'json_style': 1.0}  # All examples are positive
    manifold.add_labels({'json_style': list(range(len(corpus)))})
    
    # Test with few-shot prompt
    computer = WeightedControlVectorComputer(builder.model_interface)
    cv = computer.compute_from_labels(recipe, manifold.index_labels, manifold.corpus, layer=3)
    
    # Generate and score
    scores = []
    for strength in [0.0, 0.5, 1.0, 1.5, 2.0]:
        cv.strength = strength
        output = generate_with_control(test_prompt, cv)
        score = check_json_adherence(output, required_keys)
        scores.append((strength, score))
    
    return scores
```

**Expected Result**: Should match ARM's 1.0 score at strength ≥0.5

---

### Test 2: Weighted vs Binary Comparison

**File**: `spectral_manifold_controls/code/tests/eval_weighted_vs_binary.py`

```python
def compare_weighted_vs_binary():
    # Build manifold from diverse corpus
    corpus = load_mixed_corpus()  # formal, casual, technical, friendly
    manifold = builder.build_manifold(corpus, layer=3)
    manifold.add_labels({...})
    
    # Test Case 1: Binary selection (0 or 1)
    recipe_binary = {
        'formal': 1.0,
        'technical': 1.0,
        'casual': 0.0,  # Ignored (not included)
    }
    
    # Test Case 2: Weighted mixing (0.0 to 1.0)
    recipe_weighted = {
        'formal': 0.7,
        'technical': 0.5,
        'casual': -0.2,  # Negative!
    }
    
    # Compare outputs on same prompts
    for prompt in test_prompts:
        output_binary = generate(prompt, recipe_binary)
        output_weighted = generate(prompt, recipe_weighted)
        
        # Measure: formality, technical density, casualness
        metrics_binary = analyze_style(output_binary)
        metrics_weighted = analyze_style(output_weighted)
    
    # Question: Does weighted give finer control?
    return comparison_results
```

**Expected Result**: Weighted should show more nuanced control

---

### Test 3: Recipe Composition Quality

**File**: `spectral_manifold_controls/code/tests/eval_recipe_quality.py`

```python
def eval_recipe_composition():
    # Build manifold
    manifold = build_test_manifold()
    
    # Create recipes with different compositions
    recipes = {
        'pure_formal': {'formal': 1.0},
        'formal_friendly': {'formal': 0.6, 'friendly': 0.4},
        'balanced': {'formal': 0.5, 'friendly': 0.5},
        'avoid_casual': {'formal': 0.7, 'casual': -0.3},
    }
    
    # Generate samples
    for recipe_name, recipe in recipes.items():
        outputs = []
        for prompt in test_prompts:
            output = generate_with_recipe(prompt, recipe, manifold)
            outputs.append(output)
        
        # Analyze: Does output match recipe intent?
        analysis = analyze_recipe_adherence(outputs, recipe)
        print(f"{recipe_name}: {analysis}")
    
    # Question: Do recipes compose predictably?
    return recipe_analyses
```

**Expected Result**: Outputs should reflect recipe weights

---

### Test 4: Speed Benchmark

**File**: `spectral_manifold_controls/code/tests/benchmark_speed.py`

```python
def benchmark_smc_vs_repe():
    corpus = load_test_corpus(50)
    
    # SMC: Build once
    start = time.time()
    manifold = builder.build_manifold(corpus, layer=3)
    build_time = time.time() - start
    
    # SMC: Create 10 different recipes (instant)
    recipe_times = []
    for recipe in test_recipes:
        start = time.time()
        cv = computer.compute_from_labels(recipe, manifold, ...)
        recipe_times.append(time.time() - start)
    
    # RepE: Recompute for each variant
    repe_times = []
    for pos_indices, neg_indices in test_variants:
        start = time.time()
        cv = compute_repe_vector(corpus[pos_indices], corpus[neg_indices])
        repe_times.append(time.time() - start)
    
    print(f"SMC build: {build_time:.2f}s (one-time)")
    print(f"SMC recipe creation: {np.mean(recipe_times):.4f}s (per recipe)")
    print(f"RepE recompute: {np.mean(repe_times):.4f}s (per variant)")
    
    # Question: Is SMC faster for exploration?
    return {
        'smc_build': build_time,
        'smc_recipe': np.mean(recipe_times),
        'repe_recompute': np.mean(repe_times),
    }
```

**Expected Result**: SMC should be 10-100x faster after initial build

---

## Integration with Existing ARM Evals

We can **directly reuse** from `arm_eval_harness.py`:

1. `safe_json_adherence_score()` ✓
2. `simple_style_score()` ✓
3. `plot_dose_response()` ✓
4. Few-shot prompt construction ✓

Just need to:
- Replace ARM's steering with SMC's weighted control
- Keep all the scoring logic
- Compare results

---

## Success Criteria

### Minimum Viable Evals
- [ ] JSON task works (score ≥1.0 like ARM)
- [ ] Style task works (positive markers increase)
- [ ] Weighted beats binary on some metric
- [ ] Speed benchmark shows advantage

### Full Validation
- [ ] All tests from critique_10-1-25.txt exercises
- [ ] RepE baseline comparison
- [ ] Statistical significance testing
- [ ] Multiple models tested
- [ ] Recipe interpolation quality verified

---

## Next Immediate Step

**Port JSON eval from ARM harness** - it's the quickest win to validate SMC works.

File to create: `spectral_manifold_controls/code/tests/eval_json_smc.py`

Estimated time: 1-2 hours

Want me to implement this?

