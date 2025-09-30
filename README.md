# Aproximal Resonance Mapping
A Latent Space Topological Generalization of Linear Algorithmic Tomography

ARM/ApReMap maps a model’s latent manifold by:

(1) probing local neighborhoods with proximal ensembles of directions, 

(2) measuring resonance spectra (co-activation / mode structure) across probes, and 

(3) building a topological graph/field of attractors and transitions using spectral and persistent-homology tools. 

The result is a control surface richer than single linear control vectors as might be found in Representation Engineering.

Swap out your one-dimensional Control Vectors for rich, poly-dimensional Control Manifolds.

## Quick Start

### ARM Analysis Interface
Run manifold analysis on prompts:
```bash
python launch_interface.py
```
Open http://127.0.0.1:7860

### ARM Chat Application
Chat with ARM-steered generation (requires saved manifold):
```bash
python launch_chat.py
```
Open http://127.0.0.1:7861

## Installation
```bash
pip install -r requirements-test.txt
```

See [INTERFACE_README.md](INTERFACE_README.md) for detailed usage.