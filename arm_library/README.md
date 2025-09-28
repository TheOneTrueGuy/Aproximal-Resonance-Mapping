# ARM Library: Aproximal Resonance Mapping

A modular, testable implementation of Aproximal Resonance Mapping (ARM) - a topological generalization of linear algorithmic tomography for mapping latent manifolds in generative models.

## Overview

ARM maps a model's latent manifold by:
1. **Probing local neighborhoods** with proximal ensembles of directions
2. **Measuring resonance spectra** (co-activation/mode structure) across probes
3. **Building topological graphs/fields** of attractors and transitions using spectral and persistent-homology tools

The result is a control surface richer than single linear control vectors as found in Representation Engineering.

## Architecture

```
arm_library/
├── core/                    # Core ARM algorithms
│   ├── arm_mapper.py       # Main orchestrator class
│   ├── probe_generator.py  # Directional probe generation
│   ├── resonance_analyzer.py # Spectral analysis
│   └── topology_mapper.py  # Persistent homology
├── interfaces/             # Model interfaces
│   └── model_interface.py  # Transformer model wrapper
├── utils/                  # Utilities and configuration
│   └── config.py          # Configuration management
└── __init__.py
```

## Installation

```bash
# Install core dependencies
pip install torch transformers numpy scikit-learn

# Install topology dependencies
pip install ripser

# Install testing dependencies
pip install pytest pytest-cov pytest-mock

# Optional: visualization
pip install matplotlib umap-learn
```

## Quick Start

```python
from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig

# Configure ARM parameters
config = ARMConfig(
    model_name="distilgpt2",
    n_seeds=50,
    probes_per_seed=16,
    steps_per_probe=9,
    eps=0.03,
    layer_to_probe=6,
)

# Initialize ARM mapper
arm_mapper = ARMMapper(config)

# Define seed prompts
seed_prompts = [
    "The cat sat on the mat",
    "Machine learning is powerful",
    "The weather is beautiful",
]

# Map latent manifold
result = arm_mapper.map_latent_manifold(seed_prompts)

# Access results
descriptors = result['descriptors']  # Combined feature vectors
analyses = result['seed_analyses']   # Individual seed analyses
```

## Configuration

ARM behavior is controlled through the `ARMConfig` class:

```python
from arm_library.utils.config import ARMConfig

config = ARMConfig(
    # Model settings
    model_name="distilgpt2",     # or "gpt2", "gpt2-medium", etc.
    device="cuda",               # or "cpu"

    # ARM hyperparameters
    n_seeds=200,                 # Number of seed points
    probes_per_seed=16,          # Directional probes per seed
    steps_per_probe=9,           # Steps along each probe
    eps=0.03,                    # Perturbation magnitude
    layer_to_probe=6,            # Transformer layer to analyze

    # Analysis settings
    n_modes=8,                   # Resonance modes to track
    feature_type="hidden_pooled", # "hidden_pooled" or "logits_last"

    # Topology settings
    max_homology_dim=1,          # Maximum homology dimension
    topology_neighbors=10,       # KNN neighbors for topology

    # Reproducibility
    random_seed=42,
)
```

## Core Components

### ProbeGenerator
Generates directional perturbations around seed points in latent space.

```python
from arm_library.core.probe_generator import ProbeGenerator

generator = ProbeGenerator(config)
probes, directions = generator.generate_probe_batch(hidden_state)
```

### ResonanceAnalyzer
Performs spectral decomposition of activation matrices to extract resonance patterns.

```python
from arm_library.core.resonance_analyzer import ResonanceAnalyzer

analyzer = ResonanceAnalyzer(config)
signature = analyzer.resonance_signature(activation_matrix)
```

### TopologyMapper
Computes persistent homology and builds topological graphs of the latent manifold.

```python
from arm_library.core.topology_mapper import TopologyMapper

mapper = TopologyMapper(config)
persistence = mapper.local_persistence(activations)
graph = mapper.build_resonance_graph(resonance_signatures)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arm_library --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only

# Run tests for specific module
pytest tests/unit/test_probe_generator.py
```

## Examples

See `examples/basic_arm_usage.py` for a complete example of using the ARM library.

## Key Features

- **Modular Design**: Clean separation between model interfaces, core algorithms, and utilities
- **Comprehensive Testing**: 80%+ code coverage with unit and integration tests
- **Configurable**: Extensive configuration options for different analysis scenarios
- **Reproducible**: Deterministic results with random seed control
- **Extensible**: Easy to add new models, analysis methods, or features

## Performance Considerations

- **Memory**: Scales with `n_seeds × probes_per_seed × steps_per_probe × model_size`
- **Compute**: SVD and persistent homology are the main computational bottlenecks
- **Optimization**: Use smaller models (distilgpt2) for initial experiments

## Applications

- **Latent Space Analysis**: Understand structure of transformer representations
- **Model Steering**: Richer control surfaces than single linear vectors
- **Interpretability**: Topological analysis of learned manifolds
- **Safety Research**: Identify and navigate different behavioral regions

## Citation

If you use ARM in your research, please cite the original work:

```
Aproximal Resonance Mapping: A Latent Space Topological Generalization of Linear Algorithmic Tomography
```

## License

This implementation is provided for research and educational purposes.
