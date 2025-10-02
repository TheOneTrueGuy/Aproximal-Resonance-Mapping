# Future Directions for SMC

**Purpose**: Document ideas and extensions for Spectral Manifold Control  
**Date**: October 1, 2025

---

## Key Insight: Activation-Based, Not Prompt-Based

**Important clarification**: SMC works on activation patterns, not prompt text.

```python
# SMC analyzes:
activation = model.get_hidden_at_layer(prompt, layer=3)  # ← This
signature = spectral_analyzer.compute_signature(activation_matrix)

# NOT analyzing:
prompt_text = "Generate JSON"  # ← Not this directly
```

**Implication**: Different prompts that produce similar activations are functionally equivalent in SMC.

### Consequence 1: Prompt Variants

Multiple phrasings can target the same activation pattern:
```python
# These might produce similar activations:
json_prompts = [
    "Create a JSON object",
    "Generate JSON data", 
    "Output structured JSON",
    # All targeting same activation pattern
]
```

**Test**: Build manifolds from paraphrased prompts, compare signatures.

### Consequence 2: Automated Corpus Generation

**Idea**: Use LLM to generate diverse prompts for target behaviors

```python
def generate_corpus_for_behavior(behavior_description, n_examples=50):
    """
    Use meta-LLM to generate prompts that should elicit specific behavior.
    
    Args:
        behavior_description: "valid JSON output", "whimsical text", etc.
        n_examples: How many prompt variants to generate
    
    Returns:
        List of prompts that should produce similar activations
    """
    meta_prompt = f"""
    Generate {n_examples} different prompts that would elicit {behavior_description}.
    Make them diverse in phrasing but similar in the behavior they induce.
    """
    
    return meta_llm.generate(meta_prompt)

# Then build manifold automatically
json_corpus = generate_corpus_for_behavior("valid JSON output", 50)
manifold = builder.build_manifold(json_corpus, layer=3)
```

**Advantages**:
- No manual prompt engineering
- Ensures diversity in phrasing
- Scales to arbitrary corpus sizes
- Can target specific activation patterns

**Challenges**:
- Need to verify activations actually similar
- Meta-LLM might introduce unwanted variation
- Quality control on generated prompts

**Next Step**: Test activation similarity across paraphrases.

---

## Danny's Suggestion: Meta-Model for Recipe Selection

**Source**: Conversation with Danny  
**Idea**: Train a small model to predict optimal recipe weights for tasks

### Current Workflow (Manual)
```python
# User manually creates recipe
recipe = {
    'formal': 0.7,
    'friendly': 0.5,
    'technical': 0.2,
    'casual': -0.1,
}

cv = compute_from_labels(recipe, manifold, ...)
output = generate(prompt, cv)
```

**Problem**: Requires trial and error to find good recipes.

### Proposed Workflow (Automated)
```python
# Train model to predict recipes
recipe_model = RecipePredictor()

# User just describes task
recipe = recipe_model.predict(
    task_description="Professional but approachable blog post",
    target_metrics={'readability': 0.8, 'engagement': 0.7}
)
# Returns: {'formal': 0.6, 'friendly': 0.7, 'casual': -0.1}

# Use predicted recipe
cv = compute_from_labels(recipe, manifold, ...)
```

**Advantage**: From task description to recipe, no manual tuning.

---

## Architecture Options for Recipe Meta-Model

### Option 1: Supervised Learning (Simplest)

**Training Data**:
```python
examples = [
    {
        'task': "Write formal email",
        'metrics': {'formality': 0.9, 'friendliness': 0.3},
        'recipe': {'formal': 0.9, 'friendly': 0.2, 'casual': -0.3},
        'quality_score': 0.85,  # Human rating
    },
    {
        'task': "Casual chat message",
        'metrics': {'formality': 0.1, 'friendliness': 0.9},
        'recipe': {'casual': 0.9, 'friendly': 0.8, 'formal': -0.2},
        'quality_score': 0.90,
    },
    # ... collect hundreds of examples
]
```

**Model**: Simple feedforward network
```python
class RecipePredictor(nn.Module):
    def __init__(self, n_labels):
        self.task_encoder = TextEncoder()  # BERT, etc.
        self.metric_encoder = nn.Linear(n_metrics, 64)
        self.recipe_head = nn.Linear(128, n_labels)
    
    def forward(self, task_text, target_metrics):
        task_emb = self.task_encoder(task_text)
        metric_emb = self.metric_encoder(target_metrics)
        combined = torch.cat([task_emb, metric_emb])
        recipe_weights = self.recipe_head(combined)
        return recipe_weights  # {label: weight}
```

**Training**: Minimize difference between predicted and optimal recipes
```python
loss = MSE(predicted_recipe, optimal_recipe) + quality_score_as_reward
```

**Advantages**:
- Simple to implement
- Requires labeled data (task → good recipe)
- Fast inference

**Challenges**:
- Need to collect training data (human-labeled good recipes)
- Limited to seen task types
- Doesn't learn interaction effects well

---

### Option 2: Reinforcement Learning (More Adaptive)

**Setup**: Treat recipe creation as sequential decision-making

```python
# State: Task description + current recipe + manifold info
# Action: Adjust weight for one label (+0.1, -0.1, etc.)
# Reward: Quality metrics on generated output

class RecipeRL:
    def __init__(self, manifold):
        self.policy = PolicyNetwork()
        self.manifold = manifold
    
    def select_recipe(self, task_description):
        state = encode_state(task_description, current_recipe={})
        
        # Iteratively refine recipe
        for step in range(max_steps):
            action = self.policy(state)  # Which label to adjust, how much
            current_recipe = apply_action(current_recipe, action)
            
            # Test current recipe
            output = generate_with_recipe(current_recipe)
            reward = evaluate_output(output, task_description)
            
            # Learn from reward
            update_policy(reward)
            
            if converged(recipe):
                break
        
        return current_recipe
```

**Training**: Learn through trial and error

**Advantages**:
- Learns from outcomes, not labeled recipes
- Discovers interaction effects (which combos work)
- Adapts to new manifolds

**Challenges**:
- Expensive (many generations per training step)
- Reward function design critical
- Slow convergence

---

### Option 3: Meta-Learning (Learn to Learn Recipes)

**Idea**: Learn a general strategy for recipe creation that transfers

```python
class RecipeMetaLearner:
    """
    Learn from examples of successful recipe creation processes.
    Builds intuition about composition patterns.
    """
    
    def meta_train(self, manifolds, tasks):
        for manifold, task in zip(manifolds, tasks):
            # Inner loop: Learn good recipe for this task
            recipe = learn_recipe_for_task(task, manifold)
            
            # Outer loop: Update meta-parameters
            # "What makes a good recipe?" knowledge
            self.update_meta_knowledge(recipe, task, manifold)
    
    def predict(self, new_task, new_manifold):
        # Apply learned recipe-creation strategy
        recipe = self.meta_knowledge.apply(new_task, new_manifold)
        return recipe
```

**Advantages**:
- Transfers across manifolds
- Learns "recipe intuition"
- Few-shot adaptation to new tasks

**Challenges**:
- Complex training setup
- Requires diverse task/manifold pairs
- Computationally intensive

---

## Implementation Roadmap

### Phase 1: Data Collection (Manual)
1. Use SMC with manual recipes
2. Record: (task, recipe, output quality)
3. Collect 100-500 examples
4. Label which recipes worked well

**Time**: Ongoing as we use SMC  
**Output**: Training dataset for meta-model

### Phase 2: Simple Predictor (Option 1)
1. Implement supervised recipe predictor
2. Train on collected data
3. Test: Does it predict decent recipes?
4. Iterate on architecture

**Time**: 1-2 weeks  
**Output**: Basic recipe recommendation system

### Phase 3: Reinforcement Learning (Option 2)
1. Define reward function (quality metrics)
2. Implement policy network
3. Train on multiple tasks
4. Compare to supervised approach

**Time**: 3-4 weeks  
**Output**: Adaptive recipe learner

### Phase 4: Meta-Learning (Option 3)
1. Collect multiple manifolds
2. Implement meta-learning framework
3. Train across diverse tasks
4. Test transfer to new manifolds

**Time**: 4-6 weeks  
**Output**: General-purpose recipe strategy

---

## Evaluation Metrics for Meta-Model

### Success Criteria
- **Recipe quality**: Does predicted recipe achieve target metrics?
- **Efficiency**: How many iterations to find good recipe?
- **Transfer**: Does it work on new manifolds?
- **Interpretability**: Can we understand why it chooses recipes?

### Tests
```python
# Test 1: Predict known-good recipes
holdout_tasks = ["formal email", "casual chat", ...]
for task in holdout_tasks:
    predicted = recipe_model.predict(task)
    actual = known_good_recipes[task]
    similarity = compare_recipes(predicted, actual)
    assert similarity > 0.7

# Test 2: Zero-shot on new tasks
new_task = "technical documentation with friendly tone"
recipe = recipe_model.predict(new_task)
output = generate_with_recipe(recipe, ...)
quality = human_eval(output, new_task)
assert quality > 0.6  # Reasonable even without fine-tuning

# Test 3: Efficiency
n_iterations_manual = 10  # Human trial and error
n_iterations_model = 1    # Direct prediction
assert n_iterations_model < n_iterations_manual
```

---

## Related Ideas

### Idea 1: Recipe Library with Recommendations

Build collaborative filtering on recipes:
```python
# Users create and rate recipes
library = RecipeLibrary()
library.add_recipe("professional_blog", recipe1, rating=4.5)
library.add_recipe("casual_email", recipe2, rating=4.8)

# Recommend based on similarity
task = "semi-formal blog post"
recommended = library.recommend_similar(task)
# Returns: "professional_blog" (most similar)
```

### Idea 2: Interactive Recipe Tuning

AI assists human in real-time:
```python
# Human sets initial recipe
recipe = {'formal': 0.7, 'friendly': 0.5}

# Generate sample
output = generate(recipe)

# Human feedback: "More friendly, less formal"
recipe_adjusted = recipe_model.adjust(
    current_recipe=recipe,
    feedback="More friendly, less formal"
)
# Returns: {'formal': 0.5, 'friendly': 0.7}

# Iterate until satisfied
```

### Idea 3: Task-to-Recipe Translation

Direct natural language to recipe:
```python
recipe = nlp_to_recipe(
    "I want formal text that's still accessible, "
    "with some technical depth but not overwhelming, "
    "and definitely avoid being too casual"
)
# Returns: {'formal': 0.7, 'technical': 0.5, 'casual': -0.3, 'accessible': 0.6}
```

---

## Open Questions

### For Activation-Based Corpus Generation
1. How similar must activations be to be "equivalent"?
2. Can we cluster prompts by activation similarity?
3. Does paraphrasing preserve activation patterns?
4. Can we visualize activation clusters?

### For Meta-Model
1. What's the minimum training data needed?
2. Does it transfer across models (GPT-2 → GPT-3)?
3. Can it discover novel useful combinations?
4. How to handle manifold-specific labels?

### For Both
1. How to validate automatically (without human eval)?
2. What's the failure mode (bad recipes → what output)?
3. Can we provide confidence scores?
4. How to handle conflicting objectives?

---

## Next Steps

### Immediate (Before Meta-Model)
1. Test activation similarity across paraphrases
2. Collect manual recipes with quality ratings
3. Build recipe library for pattern analysis

### Short-term (3 months)
1. Implement supervised recipe predictor (Option 1)
2. Test on held-out tasks
3. Compare to manual recipe creation

### Long-term (6+ months)
1. Explore RL-based recipe optimization (Option 2)
2. Meta-learning across manifolds (Option 3)
3. Integration with mixing board UI

---

## Credits

**Danny's suggestion**: Meta-model for recipe selection  
**Key insight**: Activations matter, not prompt text  
**Status**: Documented for future implementation

---

*Last updated: October 1, 2025*

