# ARM_transformer_scaffold.py
# Requires: torch, transformers, numpy, scikit-learn, ripser, scipy (install via pip)
# pip install torch transformers scikit-learn ripser scipy umap-learn

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from ripser import ripser
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Dict, Any
import math

# -----------------------
# Configuration / defaults
# -----------------------
MODEL_NAME = "distilgpt2"   # small, efficient; switch to "gpt2" if you prefer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ARM hyperparams (safe defaults)
N_SEEDS = 200
PROBES_PER_SEED = 16
STEPS_PER_PROBE = 9
EPS = 0.03                 # perturbation magnitude (relative to hidden vector norm)
LAYER_TO_PROBE = 6         # index of transformer block to inject perturbations (0-based)
NEIGHBOR_PCA_SAMPLES = 128 # for local PCA when available

# -----------------------
# Utilities: load model
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
model.eval()

# Helper: get token ids and attention mask
def encode_prompt(prompt: str):
    toks = tokenizer(prompt, return_tensors="pt")
    return toks["input_ids"].to(DEVICE), toks["attention_mask"].to(DEVICE)

# -----------------------
# Core: run forward from a chosen layer (block-wise)
# -----------------------
# We'll use the model.transformer.* components directly so we can inject altered hidden states.
# For distilgpt2/gpt2 HF models, the transformer body is model.transformer consisting of:
# - wte (token embeddings), wpe (position embeddings), drop, and h = list of blocks, ln_f.
#
# Strategy:
# 1) Build initial hidden states (token embeddings + positions) up to the layer to probe.
# 2) Optionally modify the residual stream at that layer (add delta).
# 3) Run remaining transformer blocks from that layer onward to get final logits/hidden states.

def build_initial_hidden(input_ids: torch.LongTensor):
    # returns hidden states BEFORE block 0 (embedding+pos), shape (batch, seq_len, d_model)
    wte = model.transformer.wte(input_ids)        # token embeddings
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)
    wpe = model.transformer.wpe(position_ids)
    hidden = wte + wpe  # shape batch x seq x d_model
    hidden = model.transformer.drop(hidden)
    return hidden

def forward_from_layer(hidden: torch.Tensor, start_layer: int, attention_mask: torch.Tensor=None):
    """
    hidden: (batch, seq, d_model) hidden state to feed to block start_layer
    returns: final logits, final hidden, and list of intermediate hidden states (per layer)
    """
    h = hidden
    intermediates = []
    # blocks are modules in model.transformer.h (list-like)
    for i, block in enumerate(model.transformer.h):
        if i < start_layer:
            continue
        h = block(h)[0] if isinstance(block(h), tuple) else block(h)
        intermediates.append(h)
    # final layer norm
    h = model.transformer.ln_f(h)
    # lm head (tie weights with wte)
    # reshape for lm head: (batch*seq, d_model)
    logits = F.linear(h, model.transformer.wte.weight)  # tied weights
    return logits, h, intermediates

# -----------------------
# Seed / probe generation
# -----------------------
def get_seed_hidden(prompt: str, layer_idx: int) -> torch.Tensor:
    """
    Returns hidden state at layer_idx just BEFORE running block layer_idx.
    shape: (seq_len, d_model) - batch dim removed for simplicity
    """
    input_ids, attn_mask = encode_prompt(prompt)
    hidden = build_initial_hidden(input_ids)  # batch x seq x d
    # run blocks up to layer_idx-1 to get hidden state to modify
    h = hidden
    for i, block in enumerate(model.transformer.h):
        if i >= layer_idx:
            break
        h = block(h)[0] if isinstance(block(h), tuple) else block(h)
    # h is batch x seq x d; return squeeze(0)
    return h.squeeze(0).detach().cpu()  # move to CPU numpy-friendly

def sample_probes_for_hidden(hidden_vec: np.ndarray, k: int = PROBES_PER_SEED, eps: float = EPS):
    """
    hidden_vec: (seq_len, d) array (we'll flatten sequence dimension to treat as a single vector or pool)
    Return: probe_deltas shape (k, d) or (k, seq_len, d)
    Approach: get global direction sampling in hidden-space.
    - For simplicity start with isotropic Gaussian directions normalized,
      then scale to magnitude eps * ||hidden_vec|| (per token or pooled).
    """
    # pool hidden to a single vector per seed (mean over tokens) for direction construction,
    # but we will expand deltas per token when injecting.
    pooled = hidden_vec.mean(axis=0)   # (d,)
    d = pooled.shape[0]
    rng = np.random.default_rng()
    dirs = rng.normal(size=(k, d))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    hidden_norm = np.linalg.norm(pooled) + 1e-12
    scale = eps * hidden_norm
    dirs = dirs * scale
    return dirs  # (k, d)

def expand_delta_to_sequence(delta_vec: np.ndarray, seq_len: int):
    # replicate delta_vec for each token position (simple approach)
    return np.tile(delta_vec[None, :], (seq_len, 1))  # (seq_len, d)

# -----------------------
# Probe path: generate small path along a direction
# -----------------------
def build_probe_path(hidden_base: np.ndarray, dir_vec: np.ndarray, steps: int = STEPS_PER_PROBE, tau: float = 1.0):
    """
    hidden_base: (seq_len, d)
    dir_vec: (d,) pooled direction; will be expanded across seq positions
    Returns: list of perturbed hidden tensors (steps long)
    """
    seq_len = hidden_base.shape[0]
    dir_seq = expand_delta_to_sequence(dir_vec, seq_len)  # (seq_len, d)
    ts = np.linspace(-tau, tau, steps)
    path = [hidden_base + (t * dir_seq) for t in ts]
    return path, ts

# -----------------------
# Activation / response collection
# -----------------------
def activation_matrix_for_seed(prompt: str, layer_idx: int, k: int = PROBES_PER_SEED, m: int = STEPS_PER_PROBE, eps: float = EPS):
    """
    For one seed prompt, sample k probes, each with m steps; forward from layer_idx
    Collect features for each sample (e.g., final logits pooled, or final hidden pooled)
    Return: A matrix of shape (k*m, f) for downstream analysis.
    """
    hidden_base = get_seed_hidden(prompt, layer_idx).numpy()  # (seq_len, d)
    seq_len, d = hidden_base.shape
    deltas = sample_probes_for_hidden(hidden_base, k=k, eps=eps)
    rows = []
    for j in range(k):
        path, ts = build_probe_path(hidden_base, deltas[j], steps=m)
        for hidden_pert in path:
            # run from layer_idx with this perturbed hidden
            # convert to tensor with batch dim
            h_t = torch.tensor(hidden_pert[None, :, :], dtype=torch.float32, device=DEVICE)
            logits, final_h, intermediates = forward_from_layer(h_t, start_layer=layer_idx, attention_mask=None)
            # choose feature vector to represent response:
            # Option A: pooled logits over last token
            # last_token_logits = logits[0, -1, :].detach().cpu().numpy()  # (vocab,)
            # Option B (more compact): mean-pooled final hidden representation
            feat = final_h.squeeze(0).mean(dim=0).detach().cpu().numpy()  # (d,)
            rows.append(feat)
    A = np.stack(rows, axis=0)  # (k*m, f) where f == d in this choice
    return A

# -----------------------
# Resonance signature (SVD-based)
# -----------------------
def resonance_signature(A: np.ndarray, n_modes: int = 8) -> Dict[str, Any]:
    """
    Compute SVD stats and compact resonance signature for activation matrix A (n_samples x f).
    Returns dict with normalized singular values, entropy, participation ratio, top modes.
    """
    # center
    A0 = A - A.mean(axis=0, keepdims=True)
    # SVD (economy)
    U, s, Vt = np.linalg.svd(A0, full_matrices=False)
    s = np.maximum(s, 1e-12)
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    # participation ratio (measure of mode concentration)
    pr = (s**2).sum()**2 / (np.sum(s**4) + 1e-12)
    sig = {
        "singular_values": s[:n_modes],
        "s_norm": s_norm[:n_modes],
        "entropy": float(entropy),
        "participation": float(pr),
        # optionally return top singular vectors (Vt[:n_modes,:]) if needed
    }
