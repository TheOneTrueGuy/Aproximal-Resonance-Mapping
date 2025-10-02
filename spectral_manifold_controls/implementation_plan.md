# SMC Implementation Plan

**Goal**: Build Spectral Manifold Control system with weighted composition and mixing board UI

---

## Phase 1: Core Implementation (Week 1)

### Directory Structure
```
spectral_manifold_controls/
├── README.md                    # Overview
├── concept.md                   # Full specification
├── empirical_validation.md      # Test results
├── implementation_plan.md       # This file
├── recipe_format.md            # Recipe spec
└── code/                       # Implementation (TBD)
    ├── core/
    │   ├── spectral_analyzer.py
    │   ├── manifold_builder.py
    │   └── weighted_control.py
    ├── recipes/
    │   ├── recipe_manager.py
    │   └── schema.py
    └── interfaces/
        └── mixing_board.py
```

### Core Components to Build

#### 1. SpectralAnalyzer (Rename from ResonanceAnalyzer)
**File**: `code/core/spectral_analyzer.py`

**Key changes**:
- Rename class: ResonanceAnalyzer → SpectralAnalyzer
- Rename method: resonance_signature → spectral_signature
- Update docstrings to use "spectral" terminology
- Keep all SVD math unchanged (it works!)

**Estimated time**: 2 hours

---

#### 2. WeightedControlVectorComputer
**File**: `code/core/weighted_control.py`

**New functionality**:
```python
def compute_weighted_control_vector(
    self,
    index_weights: Dict[int, float],
    layer: int
) -> ControlVector:
    """
    Create control vector from weighted index combination.
    
    Args:
        index_weights: {index: weight} where:
            - Positive weights steer TOWARD
            - Negative weights steer AWAY
        layer: Transformer layer
    
    Returns:
        Control vector for steering
    """
    # Get activation for each index
    weighted_activations = []
    for idx, weight in index_weights.items():
        prompt = self.manifold['corpus'][idx]
        hidden = self.model.get_hidden_at_layer(prompt, layer)
        activation = hidden.mean(dim=0)
        weighted_activations.append(activation * weight)
    
    # Sum and normalize
    direction = torch.stack(weighted_activations).sum(dim=0)
    direction = direction / (torch.norm(direction) + 1e-8)
    
    return ControlVector(direction, layer)
```

**Estimated time**: 3 hours

---

#### 3. Recipe System
**File**: `code/recipes/recipe_manager.py`

**Functionality**:
- Save recipes as JSON
- Load recipes from JSON
- Validate recipe format
- Convert label weights to index weights
- Recipe versioning

**Recipe Format** (see `recipe_format.md`):
```json
{
  "name": "professional_friendly",
  "manifold_id": "company_v1",
  "layer": 3,
  "weights": {
    "formal": 0.6,
    "friendly": 0.7,
    "casual": -0.1
  }
}
```

**Estimated time**: 2 hours

---

## Phase 2: Mixing Board UI (Week 2)

### Interactive Interface
**File**: `code/interfaces/mixing_board.py`

**Technology**: Gradio (already familiar from ARM interface)

**Features**:
1. **Slider controls** for each semantic label
2. **Real-time preview** as sliders adjust
3. **Recipe presets** (load with one click)
4. **Save custom recipes**
5. **Solo/Mute** buttons per track
6. **Master strength** slider

**Interface Layout**:
```
┌─────────────────────────────────────────────────┐
│ SMC Mixing Board                                │
├─────────────────────────────────────────────────┤
│                                                 │
│ Manifold: [Load .pkl]                          │
│                                                 │
│ Style Tracks:                                   │
│ ┌─────────────────────────────────────────────┐ │
│ │ Formal     [━━━━━━━●━━] 0.7  [Solo] [Mute] │ │
│ │ Friendly   [━━━●━━━━━━] 0.4  [Solo] [Mute] │ │
│ │ Technical  [━●━━━━━━━━] 0.2  [Solo] [Mute] │ │
│ │ Casual     [━━━━━━━━━●] -0.1 [Solo] [Mute] │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ Recipes: [Blog] [Docs] [Email] [Custom]        │
│ [Save Recipe] [Load Recipe]                     │
│                                                 │
│ Preview:                                        │
│ ┌─────────────────────────────────────────────┐ │
│ │ Prompt: Write about AI                      │ │
│ │                                             │ │
│ │ Output: AI systems require careful...      │ │
│ │                                             │ │
│ │ [Generate] [Copy]                           │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ Layer: 3  Temp: 0.8  Length: 100               │
└─────────────────────────────────────────────────┘
```

**Estimated time**: 6-8 hours

---

## Phase 3: Integration & Testing (Week 3)

### End-to-End Workflow
1. Build manifold from corpus
2. Label semantic categories
3. Load in mixing board
4. Adjust sliders, generate
5. Save successful recipes

### Testing Tasks
- [ ] Load pre-built manifold
- [ ] Adjust weights via sliders
- [ ] Generate with recipe
- [ ] Save recipe as JSON
- [ ] Load recipe back
- [ ] Compare outputs with different recipes

### Example Demonstration
```python
# 1. Build manifold (one-time, expensive)
corpus = load_diverse_examples()
manifold = build_manifold(corpus)
manifold.label_indices({
    'formal': [0-9],
    'casual': [10-19],
    'technical': [20-29],
    'friendly': [30-39],
})
save_manifold("demo_manifold.pkl", manifold)

# 2. Use in mixing board (fast, interactive)
# - Load demo_manifold.pkl
# - Adjust sliders: formal=0.7, friendly=0.5
# - Generate
# - Save as "professional_friendly.json"

# 3. Programmatic use
recipe = load_recipe("professional_friendly.json")
cv = apply_recipe(recipe, manifold)
generate_with_control(prompt, cv)
```

**Estimated time**: 4-5 hours

---

## Phase 4: Validation (Week 4)

### Comparison Against RepE

**Test scenarios**:
1. **Speed**: RepE recomputation vs SMC index selection
2. **Quality**: Steering effectiveness
3. **Usability**: Time to find good recipe

**Metrics**:
- Computation time (build vs recompute)
- Steering effectiveness (task-specific scores)
- Iteration speed (how fast to try variants)

### Expected Results:
- SMC slower to build (one-time cost)
- SMC much faster for experimentation (instant)
- SMC better for exploration workflows
- RepE better for one-off tasks

**Estimated time**: 6 hours

---

## Phase 5: Documentation & Examples (Week 5)

### Documentation
- [ ] Tutorial: Building your first manifold
- [ ] Tutorial: Creating recipes
- [ ] Tutorial: Using the mixing board
- [ ] API reference
- [ ] Best practices guide

### Example Notebooks
- [ ] `basic_usage.ipynb` - Simple end-to-end
- [ ] `recipe_creation.ipynb` - Recipe workflow
- [ ] `team_library.ipynb` - Shared manifold use case
- [ ] `style_exploration.ipynb` - Interactive discovery

### Demo Materials
- [ ] Video walkthrough (5-10 min)
- [ ] Example manifolds (ready to use)
- [ ] Recipe library (10+ recipes)

**Estimated time**: 8-10 hours

---

## Milestone Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Core implementation | SpectralAnalyzer, WeightedControl, Recipes |
| 2 | Mixing board UI | Interactive Gradio interface |
| 3 | Integration | End-to-end workflow, testing |
| 4 | Validation | RepE comparison, benchmarks |
| 5 | Documentation | Tutorials, examples, demos |

**Total estimated time**: 30-40 hours over 5 weeks

---

## Critical Path Items

### Must Have (MVP)
- [x] SpectralAnalyzer (rename from existing)
- [ ] WeightedControlVectorComputer
- [ ] Recipe save/load
- [ ] Basic mixing board with sliders
- [ ] End-to-end demo

### Should Have
- [ ] Solo/mute buttons
- [ ] Recipe presets
- [ ] Real-time preview
- [ ] Multiple manifold support

### Nice to Have
- [ ] Recipe version control integration
- [ ] Cloud storage for manifolds
- [ ] Team collaboration features
- [ ] Advanced visualization

---

## Technical Decisions

### Keep from ARM
✅ Model interface (works well)  
✅ Spectral analyzer (rename only)  
✅ Evaluation harness (proven)  
✅ Test infrastructure  

### Drop from ARM
❌ Topology mapper (redundant)  
❌ "Resonance" terminology  
❌ Persistent homology  

### Add for SMC
➕ Weighted composition  
➕ Recipe system  
➕ Mixing board UI  
➕ Semantic labeling  

---

## Success Criteria

### Technical
- [ ] Index composition works correctly
- [ ] Recipes save/load without errors
- [ ] Mixing board generates as expected
- [ ] Performance acceptable (generation <5s)

### Usability
- [ ] Non-technical user can use mixing board
- [ ] Recipes are shareable (just JSON)
- [ ] Finding good recipe takes <5 minutes

### Validation
- [ ] SMC matches or exceeds RepE quality
- [ ] SMC enables faster exploration than RepE
- [ ] Weighted mixing provides finer control

---

## Risk Mitigation

**Risk**: Weighted composition doesn't work as expected  
**Mitigation**: Start with binary (0/1 weights) to validate, then add negative weights, then fractional

**Risk**: UI too complex for users  
**Mitigation**: Start simple (just sliders), add features incrementally based on feedback

**Risk**: Recipe format inadequate  
**Mitigation**: Version recipes, allow schema evolution

**Risk**: Performance issues with large manifolds  
**Mitigation**: Profile early, optimize hot paths, consider caching

---

## Next Immediate Steps

1. Create `code/` directory structure
2. Copy and rename ResonanceAnalyzer → SpectralAnalyzer
3. Implement WeightedControlVectorComputer
4. Test weighted composition on simple example
5. Build minimal mixing board UI

**Start here when ready to code!** 🚀

