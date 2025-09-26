ARM for Transformers
Where “latent space” lives in LLMs

Residual stream vectors at each layer (common in representation engineering).

Attention head outputs (queries/keys/values).

MLP activations (often hold rich features).

Token embedding space (input/output).

When people do LAT with control vectors, they’re usually probing the residual stream or embeddings using linear directions. ARM generalizes this to nonlinear, topology-aware resonance probes.

Step-by-step adaptation
1. Seeds

Choose anchor states in the model:

A specific prompt (e.g. “The capital of France is …”).

Capture the hidden representations at some layer(s).

These are your “seed vectors” analogous to latent z.

2. Proximal probes

Instead of perturbing a Gaussian latent vector, perturb the representation:

Add small perturbations along known directions (e.g. known control vectors like “truthful vs. hallucination”).

Add random orthogonal perturbations in the residual stream.

Scale by ε (tiny, like 1–5% of vector norm).

Now you’ve got an ensemble of perturbed hidden states around the seed.

3. Resonance mapping

Run these perturbed hidden states forward through the rest of the Transformer. Measure:

Logit changes at output layer.

Hidden activations at future layers.

Vector field divergence: how much small perturbations amplify (resonance) or dampen.

Resonance signature here =

SVD of the matrix of responses to probes.

Ratio of singular values = “how many strong modes does this neighborhood have.”

Entropy / participation ratio = “spread” of directions that matter.

4. Topology

Collect perturbed hidden states and measure geometry:

Cluster / manifold structure (e.g. do certain perturbations collapse to the same outcome?).

Persistent homology on distances between perturbed states in representation space — detects whether responses form loops, branches, cavities.

That gives you a topological signature of the local concept representation.

5. Global atlas

Repeat across different prompts/seeds → build a resonance-topology map of the model’s hidden space.

You’ll find basins (robust attractors like “France → Paris”).

You’ll find barriers (loops or bifurcations where the model flips between interpretations).

You’ll identify resonant modes that generalize across prompts (concepts).

6. Steering

Instead of a single linear vector, ARM would let you:

Steer by following a resonance flow toward a basin.

Use proximal operators that nudge representations toward desired attractors without destabilizing them.

Potentially discover multi-vector “chords” (resonant combinations) that shift meaning more robustly than one direction.

Analogy vs Diffusion/VAEs

In diffusion/VAEs: latent = a vector z you can directly perturb.

In LLMs: latent = hidden states (residuals/MLPs), where probes = small perturbations.

The math (probe ensembles, SVD spectra, persistent homology) is the same.

The interpretation shifts: instead of “images change this way,” you’re seeing “tokens/logits/representations change this way.”
