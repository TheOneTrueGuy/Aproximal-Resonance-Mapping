# SMC Recipe Format Specification

**Version**: 1.0  
**Purpose**: Define standard format for shareable style recipes

---

## Basic Recipe Format

```json
{
  "name": "recipe_name",
  "description": "Human-readable description",
  "version": "1.0",
  "manifold_id": "identifier_of_manifold",
  "layer": 3,
  "weights": {
    "label1": 0.7,
    "label2": 0.5,
    "label3": -0.2
  },
  "metadata": {
    "author": "username",
    "created": "2025-10-01",
    "tested_on": ["example1", "example2"]
  }
}
```

---

## Field Specifications

### Required Fields

#### `name` (string)
- Unique identifier for the recipe
- Use snake_case or kebab-case
- Example: `"professional_friendly"`, `"technical-docs"`

#### `version` (string)
- Semantic version of recipe format
- Current: `"1.0"`
- Allows future schema evolution

#### `manifold_id` (string)
- Identifier of manifold this recipe was designed for
- Allows validation (warn if using with different manifold)
- Example: `"company_style_v1"`, `"blog_corpus_2025"`

#### `layer` (integer)
- Transformer layer to apply control at
- Typically 2-4 for GPT-2 style models
- Example: `3`

#### `weights` (object)
- Mapping of semantic labels to weight values
- Labels must exist in manifold's `index_labels`
- Weights can be positive (steer toward) or negative (steer away)
- Range typically -1.0 to 1.0, but no hard limits
- Example:
  ```json
  {
    "formal": 0.7,
    "friendly": 0.5,
    "technical": 0.3,
    "casual": -0.2
  }
  ```

### Optional Fields

#### `description` (string)
- Human-readable explanation of recipe purpose
- Example: `"Balanced professional and friendly tone for blog posts"`

#### `metadata` (object)
- Extensible metadata for tracking and documentation
- Common fields:
  - `author`: Who created this recipe
  - `created`: ISO date string
  - `modified`: ISO date string (for updates)
  - `tested_on`: Array of test cases
  - `performance`: Metrics or notes
  - `tags`: Array of searchable tags

---

## Weight Semantics

### Positive Weights (Steer Toward)
- `0.0`: No influence
- `0.1-0.3`: Slight influence
- `0.4-0.6`: Moderate influence
- `0.7-0.9`: Strong influence
- `1.0+`: Very strong influence

### Negative Weights (Steer Away)
- `0.0`: No influence
- `-0.1 to -0.3`: Slight avoidance
- `-0.4 to -0.6`: Moderate avoidance
- `-0.7 to -0.9`: Strong avoidance
- `-1.0-`: Very strong avoidance

### Normalization
- Weights are used as-is (not automatically normalized)
- Larger absolute values = stronger influence
- Consider total "energy" when mixing many labels

---

## Example Recipes

### Professional Blog Post
```json
{
  "name": "professional_blog",
  "description": "Professional yet approachable tone for blog posts",
  "version": "1.0",
  "manifold_id": "company_blog_v1",
  "layer": 3,
  "weights": {
    "formal": 0.6,
    "friendly": 0.7,
    "technical": 0.2,
    "marketing": 0.3,
    "casual": -0.1
  },
  "metadata": {
    "author": "content_team",
    "created": "2025-10-01",
    "tested_on": ["product_launch", "feature_announcement"],
    "tags": ["blog", "marketing", "professional"]
  }
}
```

### Technical Documentation
```json
{
  "name": "technical_docs",
  "description": "Clear technical documentation with minimal fluff",
  "version": "1.0",
  "manifold_id": "company_docs_v1",
  "layer": 3,
  "weights": {
    "technical": 0.9,
    "formal": 0.7,
    "precise": 0.8,
    "friendly": 0.2,
    "marketing": -0.5,
    "casual": -0.3
  },
  "metadata": {
    "author": "docs_team",
    "created": "2025-10-01",
    "tags": ["documentation", "technical", "formal"]
  }
}
```

### Customer Support Email
```json
{
  "name": "support_email",
  "description": "Helpful and empathetic customer support tone",
  "version": "1.0",
  "manifold_id": "support_v1",
  "layer": 3,
  "weights": {
    "friendly": 0.9,
    "helpful": 0.9,
    "empathetic": 0.7,
    "professional": 0.5,
    "technical": 0.3,
    "formal": -0.2,
    "marketing": -0.4
  },
  "metadata": {
    "author": "support_team",
    "created": "2025-10-01",
    "performance": "High customer satisfaction scores",
    "tags": ["support", "customer-facing", "friendly"]
  }
}
```

### Creative Writing
```json
{
  "name": "creative_writing",
  "description": "Imaginative and expressive creative writing",
  "version": "1.0",
  "manifold_id": "creative_corpus_v1",
  "layer": 2,
  "weights": {
    "creative": 0.9,
    "expressive": 0.8,
    "vivid": 0.7,
    "casual": 0.5,
    "formal": -0.3,
    "technical": -0.5
  },
  "metadata": {
    "author": "writing_team",
    "created": "2025-10-01",
    "tags": ["creative", "fiction", "expressive"]
  }
}
```

---

## Recipe Libraries

### Directory Structure
```
recipes/
├── blog/
│   ├── professional_friendly.json
│   ├── technical_deep_dive.json
│   └── casual_update.json
├── documentation/
│   ├── api_reference.json
│   ├── user_guide.json
│   └── tutorial.json
├── customer_facing/
│   ├── support_email.json
│   ├── sales_email.json
│   └── announcement.json
└── creative/
    ├── short_story.json
    ├── blog_narrative.json
    └── descriptive.json
```

### Library Metadata
```json
{
  "library_name": "Company Style Library",
  "version": "1.0",
  "manifold_id": "company_unified_v1",
  "recipes": [
    "blog/professional_friendly.json",
    "documentation/api_reference.json",
    "customer_facing/support_email.json"
  ],
  "updated": "2025-10-01"
}
```

---

## Version Control Integration

### Git-Friendly Format
- JSON is text-based (good diffs)
- Meaningful field ordering
- Indent with 2 spaces
- No trailing commas

### Recipe Evolution
```bash
# Track changes
git diff professional_blog.json

# See evolution
- "weights": {"formal": 0.8, "friendly": 0.4}
+ "weights": {"formal": 0.6, "friendly": 0.7}

# Commit with context
git commit -m "Adjusted blog recipe: less formal, more friendly based on user feedback"
```

### Branching for Experiments
```bash
# Try variant
git checkout -b experiment/more_technical
# Edit recipe: technical: 0.2 → 0.6
git commit -m "Testing more technical tone"

# Merge if successful
git checkout main
git merge experiment/more_technical
```

---

## Validation Rules

### Required Validations
1. **Format**: Valid JSON
2. **Required fields**: name, version, manifold_id, layer, weights
3. **Weight types**: All values must be numbers
4. **Layer range**: Reasonable for model (0-11 for GPT-2)

### Warning Validations
1. **Manifold mismatch**: Recipe manifold_id ≠ loaded manifold
2. **Unknown labels**: Weight label not in manifold's index_labels
3. **Extreme weights**: Absolute value > 2.0 (might be error)
4. **Empty weights**: No weights specified

### Example Validation Code
```python
def validate_recipe(recipe, manifold):
    errors = []
    warnings = []
    
    # Required fields
    required = ['name', 'version', 'manifold_id', 'layer', 'weights']
    for field in required:
        if field not in recipe:
            errors.append(f"Missing required field: {field}")
    
    # Manifold match
    if recipe.get('manifold_id') != manifold.id:
        warnings.append(
            f"Recipe for {recipe['manifold_id']}, "
            f"but using {manifold.id}"
        )
    
    # Label existence
    for label in recipe.get('weights', {}):
        if label not in manifold.index_labels:
            warnings.append(f"Unknown label: {label}")
    
    # Extreme weights
    for label, weight in recipe.get('weights', {}).items():
        if abs(weight) > 2.0:
            warnings.append(f"Extreme weight for {label}: {weight}")
    
    return errors, warnings
```

---

## Recipe Sharing

### Sharing Formats

#### Minimal (Just Recipe)
```json
{
  "name": "my_style",
  "version": "1.0",
  "manifold_id": "generic_v1",
  "layer": 3,
  "weights": {"formal": 0.7, "friendly": 0.5}
}
```

#### With Context (Include Manifold Labels)
```json
{
  "recipe": {
    "name": "my_style",
    "weights": {"formal": 0.7, "friendly": 0.5}
  },
  "manifold_labels": {
    "formal": [0, 1, 2, 3, 4],
    "friendly": [10, 11, 12, 13, 14]
  }
}
```

#### Full Package (Recipe + Manifold)
- Share recipe JSON + manifold .pkl
- Include README with instructions
- Specify model compatibility

---

## Future Extensions (v2.0+)

### Potential Additions
- **Interpolation**: Smooth transitions between recipes
- **Conditionals**: Different weights based on context
- **Composition**: Recipes that reference other recipes
- **Optimization**: Auto-tune weights for objectives

### Backwards Compatibility
- Version field allows schema evolution
- Old recipes work with new system
- Warnings for deprecated features

---

## Usage Examples

### Load and Apply
```python
# Load recipe
with open('recipes/professional_blog.json') as f:
    recipe = json.load(f)

# Apply to manifold
cv = apply_recipe(recipe, manifold)

# Generate
output = generate_with_control(prompt, cv)
```

### Create and Save
```python
# Create recipe
recipe = {
    'name': 'my_custom_style',
    'version': '1.0',
    'manifold_id': manifold.id,
    'layer': 3,
    'weights': {
        'formal': 0.6,
        'friendly': 0.8,
        'casual': -0.1,
    },
    'metadata': {
        'author': 'me',
        'created': '2025-10-01',
    }
}

# Save
with open('recipes/my_custom_style.json', 'w') as f:
    json.dump(recipe, f, indent=2)
```

### Edit Existing
```python
# Load
recipe = load_recipe('professional_blog.json')

# Modify
recipe['weights']['friendly'] = 0.8  # More friendly
recipe['metadata']['modified'] = '2025-10-02'
recipe['metadata']['changes'] = 'Increased friendliness'

# Save
save_recipe(recipe, 'professional_blog.json')
```

---

## Summary

**Recipe format enables**:
- ✅ Lightweight sharing (just JSON)
- ✅ Version control (git-friendly)
- ✅ Validation (catch errors early)
- ✅ Evolution (extensible schema)
- ✅ Organization (library structure)

**Best practices**:
- Use descriptive names
- Include metadata
- Test before sharing
- Document changes
- Version manifolds

