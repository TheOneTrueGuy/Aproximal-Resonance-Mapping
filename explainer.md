ARM for diffusion models

ARM maps a model’s latent manifold by (1) probing local neighborhoods with proximal ensembles of directions, (2) measuring resonance spectra (co-activation / mode structure) across probes, and (3) building a topological graph/field of attractors and transitions using spectral and persistent-homology tools. The result is a control surface richer than single linear control vectors.

Ingredients (what you need)

A model with a latent space: an encoder/decoder (VAE/AE) or generator (GAN/diffusion) with an accessible latent vector z and a way to synthesize x = G(z) or to get z = E(x).

Access to intermediate activations (optional but highly useful).

Standard libraries: PyTorch, NumPy, SciPy, scikit-learn. For topology: ripser or gudhi (python). For visualizations: UMAP / matplotlib.

Enough compute to sample many local neighborhoods and run SVD/CCA / Laplacian eigendecomposition.

High-level algorithm (step-by-step)
Step 0 — Notation

Z ⊂ R^d latent space. Points z ∈ Z.

G(z) maps to observation space.

A(z; p) activation matrix for a set of probes p at z (rows = probes, columns = features or units, or vice-versa).

N_r(z) neighborhood (radius r) around z.

P(z) a probe ensemble — a small set of direction vectors or perturbation functions applied at z.

R(z) resonance signature: spectral decomposition of A(z; P(z)) (e.g., singular values, principal modes, coherence).

Step 1 — Seed sampling

Choose a set of seed latents S = {z_i}. These can be:

random samples from prior,

encodings of dataset examples,

or targeted z’s (e.g., known control locations from LAT).

Parameter: number of seeds n_seeds (start with 500–2000).

Step 2 — Proximal neighborhoods

For each seed z define N_r(z) = { z + δ | ||δ|| ≤ r }. But instead of uniform perturbation, use proximal probe ensembles:

Construct P(z) = {p_1, ..., p_k} where each p_j is a small perturbation generator:

linear vectors sampled from a local PCA of samples in N_r(z),

random Gaussian directions scaled by local covariance,

learned proximal operators (see later).

Choose k small (8–32). Perturb magnitude ε ≪ r (e.g., 1e-2..1e-1 of latent scale).

For each probe p_j generate a path/curve γ_j(t) = z + t * p_j for t ∈ [-τ, τ] sampled at m steps (e.g., m=9).

Step 3 — Build activation/resonance matrices

For each probe path, synthesize or forward through model and collect:

latent vectors γ_j(t),

optionally intermediate activations h_l(γ_j(t)) for layers l,

final outputs G(γ_j(t)).

Construct an activation matrix A(z; P) of shape (k*m) × f where f is feature dimensionality (units or pooled features). You can also create stacked matrices per-layer.

Step 4 — Resonance signature extraction

Compute spectral statistics of A to get R(z):

SVD: A = U Σ V^T. Use singular values σ_1..σ_r as resonance magnitudes.

Principal angles / CCA between probe-subspaces and canonical basis.

Auto-correlation across t for each probe (detect oscillatory or non-monotone responses).

Phase coherence (if activations signed/complex) — measure alignment across probes.

A compact choice: R(z) = (σ_1/σ_sum, σ_2/σ_sum, entropy(σ), participation_ratio, top-k_modes_vectors). This is the local resonance spectrum.

Step 5 — Local topology: persistence & geometry

From the set of points produced by all probe sample points {γ_j(t)} (or their activations), compute neighborhood geometry:

Local PCA to estimate tangent plane, curvature.

Pairwise distances between probe-sampled activations; build a Vietoris–Rips filtration and compute persistent homology (0D & 1D features). Use ripser or gudhi. The birth/death pairs give local topological invariants (holes, loops).

Combine the persistence diagram PD(z) with R(z) to form the aproximal resonance-topology descriptor D(z) = [R(z); topological features].

Step 6 — Build global resonance graph / atlas

Treat seeds z_i as nodes. Define similarity between nodes via:

Distance in descriptor space dist(D(z_i), D(z_j)) (e.g., cosine or Mahalanobis),

Or measurement of mode alignment: e.g., correlation between top singular vectors or subspace angles.

Build a weighted graph G_res (kNN or ε-graph). Compute:

Graph Laplacian eigenmaps (spectral embedding) to get global coordinates.

Identify clusters / attractor basins (community detection, e.g., Leiden/ Louvain).

Identify edges that correspond to topological transitions (large changes in PD signatures).

Step 7 — Vector field and proximal operators

Derive a latent vector field v(z) that points toward local attractors:

Estimate directional derivatives by comparing resonance spectra along small displacements.

Fit a smooth vector field (e.g., by kernel regression or neural net) v(z) ≈ argmin_v Σ ||D(z + α v(z)) - attractor_signature||^2 + smoothness.

Alternatively, train a small surrogate model P_op that given z outputs a proximal update z' = z + P_op(z) that moves z toward a chosen resonance basin.

Step 8 — Control / steering

Use the ARM map to:

Find z whose resonance matches a desired R*.

Use the learned P_op iteratively to steer z into a basin rather than applying a single linear control vector.

Use topological barriers (persistent cycles) to avoid unwanted transitions when steering.

Concrete diagnostics to compute & visualize

For each seed: singular value scree plot, entropy(σ), and 0D/1D persistence diagrams (plot).

Global: UMAP/ spectral embedding of descriptor vectors D(z) colored by cluster/latent attribute.

Graph: visualize the resonance graph with attractor centers and edges annotated with topological-change scores.

Animate probe paths in latent→image space to inspect behaviour.

Ablation: compare LAT (single vector) steering vs ARM steering on target attributes; report success, fidelity, and off-manifold artifacts.

Losses & Training objectives (if you learn operators)

If you train a proximal operator P_op (neural net) that maps z -> z':

Attraction loss: L_attr = ||R(z') - R_target||^2.

Smoothness: L_smooth = ||P_op(z) - P_op(z + δ)||^2 for small δ.

Manifold-consistency: L_manifold = ||G(z') - recon(G(z'))||^2 (reconstruction or discriminator penalty to keep images realistic).

Topology-preservation (optional): encourage PD(z') to share high-persistence features with the target basin.

Use multi-task weighting; start small.

Complexity & hyperparameters (practical defaults)

n_seeds: 500–2000

k (probes per seed): 12–24

m (steps per probe): 7–15

ε perturbation magnitude: 0.01–0.1 (relative)

r neighborhood radius: 0.05–0.5 (tune per model scale)

Use SVD on matrices where (k*m) × f is manageable; reduce f with average pooling or PCA on features.

Libraries & code hints

PyTorch for forward passes and gradient-based experiments.

ripser or gudhi for persistent homology.

scikit-learn for PCA, spectral embedding, CCA.

umap-learn for visualization.

Optional: geomstats for manifold ops.
