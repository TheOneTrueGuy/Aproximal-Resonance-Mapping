# Moth Balling this project
turns out I can do the same thing more efficiently through a couple other means.
Also, don't trust LLMs when it comes to writing your evals. It's okay to have them write the eval, but just be sure to double check the code manually yourself
I wasted several days because GPT5 decided to compare beam search on the baseline against sampling on the experimental. Beam search with 5 beams is of course much slower.
 
 Aproximal Resonance Mapping


A Latent Space Topological Generalization of Linear Algorithmic Tomography

ARM/ApReMap maps a model’s latent manifold by:

(1) probing local neighborhoods with proximal ensembles of directions, 

(2) measuring resonance spectra (co-activation / mode structure) across probes, and 

(3) building a topological graph/field of attractors and transitions using spectral and persistent-homology tools. 

The result is a control surface richer than single linear control vectors as might be found in Representation Engineering.

Swap out your one-dimensional Control Vectors for rich, poly-dimensional Control Manifolds.

Instructions:
Clone the repo

make and activate a venv

pip install -r requirements-test.txt

python launch_interface.py



