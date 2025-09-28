# ARM Web Interface

A comprehensive web interface for exploring Aproximal Resonance Mapping (ARM) through an interactive Gradio application.

## Features

### 🔬 ARM Analysis Tab
- **Model Selection**: Choose from distilgpt2, gpt2, gpt2-medium
- **Parameter Control**:
  - Number of seeds (1-10)
  - Probes per seed (1-8)
  - Steps per probe (1-10)
  - Epsilon perturbation (0.001-0.5)
  - Layer to probe (0-11 for GPT-2 models)
  - Resonance modes (1-8)
- **Multi-Prompt Input**: Enter multiple prompts (one per line)
- **Live Visualizations**:
  - Resonance signature plots
  - Topology analysis graphs
  - Descriptor space heatmaps
- **Detailed Summary**: Markdown report with all metrics

### 🎭 Text Generation Tab
- **Steered Generation**: Framework for ARM-guided text generation
- **Temperature Control**: Adjust randomness (0.1-2.0)
- **Token Limits**: Control output length (10-200 tokens)
- **ARM Steering Toggle**: Enable/disable behavioral steering (coming soon)

### 📚 Help & Documentation Tab
- **Parameter Guide**: Detailed explanations of all settings
- **Best Practices**: Tips for getting good results
- **Troubleshooting**: Common issues and solutions
- **Limitations**: Current constraints and future plans

## Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
arm_env\Scripts\Activate.ps1

# Install requirements
pip install -r requirements-test.txt
```

### 2. Launch Interface
```bash
python launch_interface.py
```

### 3. Access Interface
- Open browser to: http://localhost:7860
- The interface will load automatically

## Usage Guide

### Basic ARM Analysis

1. **Choose Model**: Start with `distilgpt2` (fastest)
2. **Set Parameters**:
   - Seeds: 3-5 (number of prompts to analyze)
   - Probes: 2-4 (directions to explore)
   - Steps: 3-5 (resolution per direction)
   - Epsilon: 0.01-0.1 (perturbation strength)
3. **Enter Prompts**:
   ```
   The cat sat on the mat
   Once upon a time
   In the beginning
   ```
4. **Click "Run ARM Analysis"**
5. **Explore Results**:
   - Resonance plots show activation patterns
   - Topology graphs reveal behavioral clustering
   - Descriptor heatmaps visualize latent signatures

### Understanding Results

#### Resonance Analysis
- **Entropy**: How complex/diverse the activation patterns are
- **Participation Ratio**: How evenly spectral energy is distributed
- **Singular Values**: Strength of different activation modes

#### Topology Analysis
- **Spectral Embedding**: 2D projection of behavioral relationships
- **Clusters**: Groups of prompts with similar latent behavior
- **Effective Neighbors**: How many connections each prompt has

#### Descriptor Space
- **Heatmap**: Visual representation of each prompt's signature
- **Dimensions**: Different aspects of behavioral patterns
- **Similarity**: How prompts cluster in mathematical space

## Advanced Usage

### Parameter Optimization

#### For Style Analysis
```python
# Jabberwocky-style detection
n_seeds: 5-10
probes_per_seed: 4-8
eps: 0.05-0.1
layer_to_probe: 2-4  # Middle layers capture semantics
```

#### For Speed
```python
# Quick experimentation
n_seeds: 2-3
probes_per_seed: 2
steps_per_probe: 2-3
eps: 0.01
```

#### For Detail
```python
# Comprehensive analysis
n_seeds: 10
probes_per_seed: 8
steps_per_probe: 7
eps: 0.03
```

### Interpreting Visualizations

#### Resonance Plots
- **High Entropy**: Complex, diverse activation patterns
- **Low Entropy**: Simple, focused activation patterns
- **Uniform Participation**: Well-distributed spectral energy
- **Concentrated Participation**: Dominant activation modes

#### Topology Graphs
- **Tight Clusters**: Similar behavioral patterns
- **Spread Out**: Diverse behavioral patterns
- **Central Points**: Representative behaviors
- **Isolated Points**: Unique behavioral signatures

## Technical Details

### Performance
- **distilgpt2**: ~30-60 seconds for 3 seeds
- **gpt2**: ~2-3 minutes for 3 seeds
- **gpt2-medium**: ~5-10 minutes for 3 seeds

### Memory Usage
- **CPU Mode**: ~300-800MB depending on parameters
- **GPU Mode**: Requires CUDA installation for acceleration

### Limitations
- Text generation steering not yet implemented
- Large models (>1B parameters) may be slow
- Topology analysis requires 3+ prompts for meaningful results
- GPU acceleration requires additional setup

## Save/Load Results (NEW!)

### Save Formats

**JSON Format (.json):**
- Human-readable text format
- Compatible with any programming language
- Slightly larger file size (~380KB for typical analysis)
- Good for sharing and version control
- Contains: metadata, configuration, prompts, and results

**Pickle Format (.pkl):**
- Python-specific binary format
- Preserves exact numpy arrays and data types
- Smaller file size (~100KB for typical analysis)
- Faster loading
- Only works with Python

### Usage
1. **Save**: After running analysis, go to "Save/Load Results" tab and click "Save Results"
2. **Load**: Upload a saved .json or .pkl file to restore previous analysis
3. **Share**: JSON files can be shared with other researchers
4. **Backup**: Save important results for reproducibility

### Programmatic Access
```python
from demo_save_load import demo_save_load
demo_save_load()  # Creates demo_results.json and demo_results.pkl
```

### Advanced Features

#### Reproducibility
- All random seeds are saved with results
- Exact configurations preserved
- Same prompts and parameters reproducible
- Timestamped filenames for version tracking

#### Data Preservation
- Numpy arrays converted to/from lists automatically
- Spectral embeddings and graph data preserved
- Resonance signatures and topology features maintained
- Clustering results and centroids saved

#### Performance
- JSON: Readable but larger files
- Pickle: Compact but Python-only
- Load times: Pickle (~0.1s) vs JSON (~0.5s)
- Memory usage: Both formats efficient

### Future Enhancements
- **Batch Export**: Save multiple analyses at once
- **Cloud Storage**: Direct upload to cloud services
- **Comparison Mode**: Load and compare multiple saved results
- **Metadata Tags**: Add custom labels and descriptions

## Troubleshooting

### Common Issues

#### "CUDA out of memory"
**Solution**: Reduce parameters or use CPU mode
```python
# In ARMConfig
device='cpu'  # Force CPU usage
n_seeds=2     # Reduce seeds
probes_per_seed=2  # Reduce probes
```

#### "Analysis takes too long"
**Solution**: Use smaller model or reduce parameters
```python
model_name="distilgpt2"  # Fastest model
n_seeds=2
steps_per_probe=3
```

#### "Poor results" / "No patterns visible"
**Solution**: Adjust parameters for your use case
```python
eps=0.05      # Increase perturbation
layer_to_probe=3  # Try different layers
n_seeds=5     # More prompts for comparison
```

#### "Interface won't load"
**Solution**: Check port availability and dependencies
```bash
# Kill existing processes on port 7860
netstat -ano | findstr :7860

# Check Python environment
python -c "import gradio; print('Gradio OK')"
```

### Getting Help

1. **Check Logs**: Look for error messages in terminal
2. **Parameter Guide**: Use the Help tab for detailed explanations
3. **Start Small**: Begin with minimal parameters and increase gradually
4. **Compare Results**: Try different parameter combinations

## Future Features

### Coming Soon
- ✅ **ARM-Steered Text Generation**: Generate text following discovered patterns
- 🔄 **Model Upload**: Support for custom fine-tuned models
- 📊 **Advanced Metrics**: More detailed behavioral analysis
- 🎨 **Custom Visualizations**: Interactive exploration tools
- 📈 **Batch Processing**: Analyze multiple prompt sets
- 💾 **Result Export**: Save and share analysis results

### Research Directions
- **Cross-Model Comparison**: Analyze same prompts across different models
- **Temporal Analysis**: How behavior changes during generation
- **Safety Applications**: Detect and control harmful behavioral patterns
- **Creative Steering**: Guide models toward specific artistic styles

## Contributing

The ARM interface is built with modularity in mind. Key extension points:

### Adding New Models
```python
# In ARMInterface.__init__
self.available_models.append("your-custom-model")
```

### New Visualizations
```python
# Add methods to ARMInterface
def create_custom_plot(self, results):
    # Your plotting logic
    return plot_path
```

### Analysis Metrics
```python
# Extend resonance_analyzer.py
def custom_metric(self, activations):
    # Your metric calculation
    return score
```

## License & Attribution

This interface builds on the ARM research framework. See main repository for licensing and attribution requirements.

---

**Interface Version**: 1.0.0
**Last Updated**: December 2025
**Compatibility**: Python 3.8+, PyTorch 2.0+, Gradio 4.0+
