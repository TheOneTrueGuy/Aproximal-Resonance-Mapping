# Lessons Learned & Future Possibilities

**Date**: October 1, 2025  
**Project**: Spectral Manifold Control (SMC) / Aproximal Resonance Mapping (ARM)  
**Status**: Experimental - Interesting ideas but baseline beats us  
**For**: Future exploration by LLM assistant in YOLO mode

---

## TL;DR

**What we tried**: Build a manifold once from diverse examples, then compose control vectors instantly by mixing indices with weights (like a "style mixing board").

**What we found**: Simple 2-shot prompting with sampling beats our complex approach. But there are interesting ideas worth exploring.

**Core tension**: The premise is delightful (compositional control, positive-only steering, weighted mixing) but evaluation shows simpler wins.

---

## The Journey: From ARM to SMC

### Original ARM Concept
- **Resonance**: Was metaphorical, not oscillatory (we fixed terminology to "Spectral")
- **Topology**: Persistent homology was redundant (ablation study confirmed - drop it)
- **Spectral Analysis**: SVD on activation matrices - this part is solid
- **Probing**: Local neighborhood exploration with directional perturbations

### Evolution to SMC
We stripped away the fluff and focused on:
1. **Spectral signatures** from activation patterns (SVD-based)
2. **Index-addressable manifold** - each example gets an index
3. **Weighted composition** - mix indices with individual strengths
4. **Positive-only steering** - no negative examples needed (usually)

---

## What Actually Works (The Good Parts)

### ✅ Spectral Analysis is Sound
```python
# SVD on activation matrices
U, S, Vt = np.linalg.svd(activation_matrix)
signature = {
    'singular_values': S,
    'entropy': compute_entropy(S),
    'participation_ratio': compute_pr(S),
}
```

**Why it works**: Captures mode structure in activation space. Mathematically clean.

### ✅ Control Vectors Can Be Composed
```python
# Traditional RepE
control = mean(positive_acts) - mean(negative_acts)

# SMC approach
control = 0.7*formal + 0.5*technical - 0.2*casual
```

**Why it's interesting**: Instead of binary pos/neg, you can mix multiple behaviors with weights.

### ✅ Positive-Only Steering Works Sometimes
```python
# Just average all JSON examples
json_indices = list(range(len(json_corpus)))
weights = {i: 1.0/len(json_corpus) for i in json_indices}
control_vector = weighted_compose(weights, corpus)
```

**Why it's interesting**: Don't need to collect negative examples. Simpler data collection.

### ✅ Index-Based Composition is Elegant
```python
# Build manifold once
manifold = build_from_corpus(diverse_examples)  # One-time: 65s

# Create infinite variants instantly
recipe_1 = {'formal': 0.8, 'friendly': 0.3}
recipe_2 = {'formal': 0.3, 'friendly': 0.8, 'casual': -0.2}
recipe_3 = {'technical': 0.9, 'casual': -0.5}

# Each recipe creation: ~2s (vs RepE which needs recomputation)
```

**Why it's elegant**: Reusable manifold, instant recipe creation, shareable recipes.

---

## What Didn't Work (The Harsh Truth)

### ❌ Baseline Beats Us
**Controlled test (8 samples, matched generation strategy)**:
- 2-shot + sampling: **0.500** ← Simple wins!
- SMC + sampling: **0.375** ← Complex loses!

**50-sample test (original)**:
- SMC appeared to win (0.48 vs 0.16)
- But baselines used beam search (wrong for this task)
- SMC used sampling (right for this task)
- **We compared the wrong things**

### ❌ High Variance
- Standard deviation: ±0.5 on a 0-1 scale
- Results flip with different samples
- Not reliable enough for production

### ❌ Build Cost Not Justified
- Manifold build: 65s
- If you don't beat baseline, why pay the cost?
- Break-even only if you create 100+ recipe variants

### ❌ Modest Absolute Performance
- Best SMC: ~48% success on JSON task
- Best baseline: ~50% success
- Both are mediocre in absolute terms
- gpt2-medium (355M) is too small for this task

---

## Evaluation Setup (Reproducible)

### Test: JSON Adherence

**Task**: Generate valid JSON with required keys from prompt
```
Input: "Task: Given name=Alice, age=30, city=Paris. Respond with exactly:"
Success: {"name": "Alice", "age": 30, "city": "Paris"}
Failure: "Hello Alice, I am 30 years old..." (not JSON)
```

**Scoring**: Binary (1.0 if valid JSON with all keys, 0.0 otherwise)

**Why JSON**: Easy to measure, clear success criteria, structured output

### Baselines Tested
1. **Zero-shot**: Just the task prompt
2. **2-shot**: Task prompt + 2 JSON examples  
3. **5-shot**: Task prompt + 5 JSON examples

### Generation Strategies
1. **Beam search**: `num_beams=5`, deterministic
2. **Sampling**: `temperature=0.8`, stochastic

**Key finding**: For gpt2-medium, **sampling > beam search** (opposite of usual!)

### SMC Configuration
- **Corpus**: 50 JSON examples
- **Manifold**: SVD decomposition, layer 3, 2 probes × 2 steps
- **Strengths tested**: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
- **Optimal**: Strength = 1.5 (but still lost to baseline)

### Sample Sizes
- **50 samples**: More reliable, but baselines used wrong strategy
- **8 samples**: Matched strategies, but high variance

**Lesson**: Need 20-30 samples minimum for reliable results with this variance.

---

## Interesting Ideas Worth Exploring

Even though SMC didn't beat baselines, there are cool concepts here:

### 1. Recipe Composition
**Idea**: Mix multiple behavioral dimensions with weights

```python
# Recipe format
recipe = {
    'formal': 0.7,
    'technical': 0.5,
    'friendly': 0.3,
    'casual': -0.2,  # Negative weight = avoid
}

# Apply recipe
control = compute_weighted_control(recipe, manifold)
output = generate(prompt, control)
```

**Why interesting**: 
- Fine-grained control over multiple axes
- Recipes are shareable (JSON files)
- Interpolate between recipes for smooth transitions

**Unexplored**:
- Does composition actually work predictably?
- Can you discover novel combinations?
- Do weights compose linearly or have interaction effects?

### 2. Meta-Model for Recipe Selection
**Idea**: Train a small model to predict optimal recipes

```python
# User describes intent
task = "Write professional but approachable blog post"

# Model predicts recipe
recipe_model = RecipePredictor()
recipe = recipe_model.predict(task)
# Returns: {'formal': 0.6, 'friendly': 0.7, 'technical': 0.2}
```

**Why interesting**: Automates recipe creation, learns what works

**Unexplored**: 
- Supervised learning from labeled (task → recipe) pairs
- RL with quality reward signal
- Meta-learning across manifolds

### 3. Activation-Based Corpus Generation
**Insight**: SMC works on activation patterns, not prompt text

```python
# Different prompts, same activation → equivalent
prompts_a = ["Create JSON", "Generate JSON", "Make JSON"]
prompts_b = ["Output JSON format", "Provide JSON data"]

# If activations similar, can use interchangeably
```

**Why interesting**: Could auto-generate diverse prompts with LLM

**Unexplored**:
- Test activation similarity across paraphrases
- Auto-generate corpus for target behaviors
- Cluster prompts by activation patterns

### 4. Weighted Index Mixing Board
**Idea**: GUI with sliders for each behavioral dimension

```
[Formal]     [========70%========]
[Friendly]   [=====50%=====------]
[Technical]  [===30%=====---------]
[Casual]     [----(-20%)----------]  (negative)

Generate: "The research findings suggest..." (70% formal tone)
```

**Why interesting**: Intuitive, real-time, interactive control

**Unexplored**: 
- Does visual mixing help users discover good recipes?
- Can users "feel" what different weights do?
- Real-time preview as sliders move?

### 5. Style Transfer via Composition
**Idea**: Analyze source text, extract style weights, apply to new content

```python
# Analyze Shakespeare
shakespeare_recipe = analyze_style(shakespeare_corpus)
# Returns: {'archaic': 0.9, 'poetic': 0.8, 'formal': 0.7}

# Apply to modern text
modern_prompt = "Explain quantum physics"
output = generate(modern_prompt, shakespeare_recipe)
# Result: "Behold! The quanta doth exhibit..."
```

**Why interesting**: Automatic style extraction + transfer

**Unexplored**: 
- Can you actually extract style recipes from arbitrary text?
- Does transfer preserve content while changing style?
- Cross-domain style transfer?

### 6. Manifold Interpolation
**Idea**: Smoothly transition between behavioral modes

```python
# Start: Very formal
recipe_start = {'formal': 1.0}

# End: Very casual  
recipe_end = {'casual': 1.0}

# Interpolate
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    recipe_mid = interpolate(recipe_start, recipe_end, alpha)
    output = generate(prompt, recipe_mid)
    # Observe gradual formality → casualness
```

**Why interesting**: Explore the space between styles

**Unexplored**:
- Is interpolation smooth or jumpy?
- Non-linear interpolation (spherical, etc)?
- Multi-dimensional interpolation?

### 7. Hierarchical Recipes
**Idea**: Compose recipes from sub-recipes

```python
# Base recipes
technical_writing = {'technical': 0.8, 'formal': 0.6}
friendly_tone = {'friendly': 0.9, 'casual': 0.5}

# Composed recipe
friendly_technical = 0.6 * technical_writing + 0.4 * friendly_tone
# Automatically computes weighted blend
```

**Why interesting**: Modular, reusable recipe building blocks

**Unexplored**: 
- Recipe algebra (addition, scaling, composition)
- Recipe libraries (share across projects)
- Recipe inheritance?

### 8. Per-Token Control Strength
**Idea**: Vary steering strength across sequence

```python
# Stronger steering at start, weaker at end
strengths = [2.0, 1.5, 1.0, 0.5, 0.0]  # Per token
output = generate_variable_strength(prompt, control, strengths)
```

**Why interesting**: Fine-grained temporal control

**Unexplored**:
- Does this even work with current architecture?
- Optimal strength schedules?
- Attention-based strength modulation?

---

## Why We Think This MIGHT Still Work

### Hypothesis 1: Wrong Model Size
- gpt2-medium (355M params) is small
- Control vectors might work better with larger models
- Larger models have more capacity for subtle steering

**Test**: Try with gpt2-large (774M) or gpt2-xl (1.5B)

### Hypothesis 2: Wrong Task
- JSON generation is too structured
- Maybe steering helps more with style/tone tasks
- Style is continuous, JSON is binary (works/doesn't)

**Test**: Try style transfer, sentiment control, formality tuning

### Hypothesis 3: Wrong Layer
- We tested layer 3
- Maybe other layers are more steerable
- Earlier layers: syntax; later layers: semantics

**Test**: Sweep across layers (0-11 for gpt2-medium)

### Hypothesis 4: Need More Manifold Data
- 50 examples might be too few
- Richer manifold = better representation
- Diminishing returns, but where's the threshold?

**Test**: Build manifold with 200, 500, 1000 examples

### Hypothesis 5: Recipe Search Problem
- We manually designed recipes
- Maybe optimal recipes are non-obvious
- Need search/optimization to find them

**Test**: 
- Grid search over weight space
- Genetic algorithm for recipe discovery
- RL to learn recipes for specific tasks

### Hypothesis 6: Composition Actually Helps
- We didn't rigorously test composition
- Single behaviors tested, not combinations
- Maybe mixing is where the value is

**Test**: 
- Compare single-behavior vs multi-behavior recipes
- Test if composition is predictable
- Measure interaction effects

---

## Recommended Next Steps (YOLO Mode)

If you want to keep exploring despite baseline winning:

### Quick Wins (< 1 day each)
1. **Try larger model**: gpt2-large or gpt2-xl
2. **Test style task**: Jabberwocky marker injection (we have corpus!)
3. **Layer sweep**: Test layers 0, 3, 6, 9, 11
4. **Composition test**: Compare single vs mixed recipes
5. **Negative weight test**: Do negative weights actually help?

### Medium Experiments (1-3 days)
1. **Recipe search**: Grid search to find optimal weights
2. **Multi-task manifold**: Build from diverse tasks, test transfer
3. **Activation clustering**: Visualize manifold structure
4. **Recipe interpolation**: Test smoothness
5. **Beam + SMC**: Try beam search with steering (quality vs speed)

### Big Swings (1-2 weeks)
1. **Meta-model**: Train recipe predictor
2. **Auto-corpus generation**: LLM generates training examples
3. **GUI mixer board**: Interactive Gradio interface
4. **Recipe library**: Build + share recipe collection
5. **Cross-model transfer**: Do recipes work across model sizes?

---

## Technical Details for Reproducibility

### Manifold Building
```python
from core import ManifoldBuilder

builder = ManifoldBuilder(
    model_name="gpt2-medium",
    device="cpu",
    n_modes=4  # SVD modes to keep
)

manifold = builder.build_manifold(
    corpus=your_examples,
    layer=3,  # Which transformer layer
    probes_per_seed=2,  # Directional probes
    steps_per_probe=2,  # Steps along each probe
    eps=0.03  # Perturbation magnitude
)
```

**Time**: ~1.3s per example (65s for 50 examples)

### Control Vector Creation
```python
from core import WeightedControlVectorComputer

computer = WeightedControlVectorComputer(builder.model_interface)

# Option 1: From index weights
index_weights = {0: 0.7, 1: 0.5, 2: -0.2}
cv = computer.compute_from_weights(index_weights, corpus, layer=3)

# Option 2: From semantic labels
label_weights = {'formal': 0.7, 'friendly': 0.5}
index_labels = {'formal': [0,1,2], 'friendly': [3,4,5]}
cv = computer.compute_from_labels(label_weights, index_labels, corpus, layer=3)
```

**Time**: ~2s per control vector

### Generation with Control
```python
from arm_library.core.steering import ARMControlledGenerator

generator = ARMControlledGenerator(model_interface)
generator.set_control(cv)

output = generator.generate_with_steering(
    prompt="Your prompt here",
    max_length=40,
    do_sample=True,  # Use sampling, not beam!
    temperature=0.8,
)
```

**Key**: Use **sampling** not beam search for gpt2-medium!

### Evaluation
```python
import json, re

def json_score(text, required_keys):
    """1.0 if valid JSON with keys, else 0.0"""
    candidates = re.findall(r"\{[^{}]*\}", text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if all(k in obj for k in required_keys):
                return 1.0
        except:
            pass
    return 0.0
```

---

## Open Questions

### Scientific
1. **Why does sampling beat beam for gpt2-medium?** Unusual!
2. **Why is variance so high?** (±0.5 on 0-1 scale)
3. **Do control vectors compose linearly?** Or are there interaction effects?
4. **What's the optimal layer for steering?** We only tested layer 3
5. **Does manifold quality improve with more data?** Diminishing returns?

### Practical
1. **Can you beat few-shot on ANY task?** We only tested JSON
2. **Does this work with larger models?** Scale up behavior?
3. **Can recipes transfer across models?** gpt2-medium → gpt2-large?
4. **Is the build cost ever justified?** Need 100+ variants?
5. **Can you auto-discover good recipes?** Or need manual tuning?

### Philosophical
1. **Is this solving the right problem?** Maybe prompting is fine?
2. **Is compositional control actually useful?** Or just cool?
3. **Does "style mixing board" have real use cases?** Or novelty?
4. **Should we just use larger models?** Does capability > control?

---

## Files & Code Structure

### Documentation
- `README.md` - Overview of SMC concept
- `concept.md` - Full specification
- `stats.md` - Evaluation results (50 samples)
- `lessons_learned_and_possibilities.md` - This document
- `FUTURE_DIRECTIONS.md` - Danny's meta-model idea + activation-based generation

### Core Code
- `spectral_manifold_controls/code/core/`
  - `spectral_analyzer.py` - SVD-based signature extraction
  - `manifold_builder.py` - Build manifolds from corpora
  - `weighted_control.py` - Weighted index composition
  
### Tests & Evals
- `code/tests/test_weighted_control.py` - Unit tests (8 tests, all pass)
- `code/tests/eval_json_baseline.py` - Main evaluation (50 samples)
- `code/tests/eval_generation_strategy.py` - Controlled comparison (8 samples)
- `code/tests/check_model_cache.py` - Verify HF models cached

### Legacy ARM (mostly ignore)
- `arm_library/` - Original ARM code
- `resonance_audit.md` - Why we dropped "resonance" terminology  
- `critique_10-1-25.txt` - Initial criticisms that led to SMC

---

## What We'd Tell Our Past Selves

### Do Different
1. **Test generation strategies first** - Could have saved days
2. **Start with simple baselines** - Establish floor before fancy stuff
3. **Larger sample sizes earlier** - 8 samples too variable
4. **Test composition explicitly** - We assumed it worked
5. **Try other tasks** - Don't anchor on JSON

### Do Same
1. **Honest evaluation** - Accepted results even when they sucked
2. **Controlled comparisons** - Matched strategies properly (eventually)
3. **Drop bad terminology** - "Resonance" → "Spectral" was right
4. **Ablation studies** - Proved topology was useless
5. **Document everything** - Future us says thanks!

---

## Final Thoughts

**The premise is delightful**: Compositional control, weighted mixing, positive-only steering, shareable recipes. It's elegant and has real potential.

**The results are humbling**: Simple few-shot prompting beats us. The added complexity isn't justified.

**The question remains**: Is this fundamentally flawed, or did we just not find the right application/model/task combo?

**For a YOLO assistant**: There are interesting ideas here worth exploring. Don't expect miracles, but don't give up without trying:
- Larger models
- Different tasks  
- Composition tests
- Recipe optimization

Maybe you'll find the use case where this shines. Or maybe you'll confirm simpler is better. Either way, you'll learn something.

**Good luck!** 🚀

---

*"Not all who wander are lost, but some of us definitely took a wrong turn at the resonance metaphor."*

---

## Appendix: Quick Reference

### Command to run eval
```bash
cd K:\projects_annex\ARM
arm_env\Scripts\Activate.ps1
python spectral_manifold_controls/code/tests/eval_json_baseline.py
```

### Expected results (baseline to beat)
- **2-shot + sampling**: ~0.500 on JSON task (8 samples)
- **5-shot + sampling**: ~0.125 on JSON task (8 samples)
- **Zero-shot**: 0.000 (complete failure)

### Model cache location
`C:\Users\Z440\.cache\huggingface\hub\`
- gpt2-medium: 2.9GB (cached ✓)

### Key hyperparameters
- Layer: 3
- Probes: 2
- Steps: 2
- Epsilon: 0.03
- Strength (optimal): 1.5
- Generation: **sampling** not beam!
- Temperature: 0.8

### Time estimates
- Model load: ~2s
- Manifold build (50 examples): ~65s
- Control vector: ~2s
- Generation: ~5s/sample (sampling), ~12s/sample (beam)

### Dependencies
- Python 3.12
- PyTorch
- Transformers (HuggingFace)
- NumPy, SciPy
- See `arm_library/interfaces/model_interface.py` for model loading

---

*Last updated: October 1, 2025 - After realizing baseline beats us*

