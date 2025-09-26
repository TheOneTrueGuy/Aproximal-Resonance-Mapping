# pseudocode (not copy/paste guaranteed — but very close)
import torch
import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding

def sample_probes(z, model, k=16, m=9, eps=0.03):
    # z: torch tensor shape (d,)
    # sample k directions from local PCA of N_r(z) or random normals
    # here use Gaussian for simplicity
    d = z.shape[0]
    dirs = torch.randn(k, d)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)
    ts = torch.linspace(-1, 1, steps=m)
    samples = []
    for j in range(k):
        for t in ts:
            zt = z + (eps * t) * dirs[j]
            samples.append(zt)
    samples = torch.stack(samples, dim=0)  # (k*m, d)
    return samples, dirs, ts

def build_activation_matrix(samples, model, layer_hook=None):
    # Forward samples through model and collect features/activations
    model.eval()
    with torch.no_grad():
        # Either get latent features or activation from hook
        feats = []
        for zt in samples:
            x = model.decode(zt.unsqueeze(0))  # or model(zt) depending on API
            feat = extract_feature_vector(x)   # e.g. pooled image features or intermediate act
            feats.append(feat)
    A = np.stack(feats, axis=0)  # (k*m, f)
    return A

def resonance_signature(A, n_modes=6):
    # compute SVD
    U, s, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    s = np.maximum(s, 1e-12)
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    participation = (s**2).sum()**2 / ((s**4).sum() + 1e-12)
    return dict(singular_values=s[:n_modes], entropy=entropy, participation=participation)

def local_persistence(A, maxdim=1):
    # use ripser on the activation points
    D = pairwise_distances(A)
    r = ripser(D, distance_matrix=True, maxdim=maxdim)
    return r['dgms']  # diagrams

# High-level
descriptors = []
for z in seeds:
    samples, dirs, ts = sample_probes(z, model, k=16, m=9, eps=eps)
    A = build_activation_matrix(samples, model)
    R = resonance_signature(A)
    PD = local_persistence(A)
    D = pack_descriptor(R, PD)
    descriptors.append(D)

# Build graph
X = np.array([flatten_descriptor(d) for d in descriptors])
W = kneighbors_graph(X, n_neighbors=10, mode='distance', include_self=False)
L_emb = spectral_embedding(W, n_components=3)
# cluster, visualize, etc.
