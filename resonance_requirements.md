# What Would Make ARM True "Resonance" Analysis?
**Date**: October 1, 2025  
**Question**: What needs to change to justify "resonance" terminology mathematically?

---

## Current State: Static Spectral Analysis

**What we do now**:
```python
# Probe neighborhood → collect activations → SVD
A = collect_activations(prompt, probes)  # Static snapshot
U, s, Vt = np.linalg.svd(A)              # Decompose
# s = singular values (magnitudes, not frequencies)
```

**Why it's NOT resonance**: No dynamics, no oscillations, no frequency domain

---

## Path to True Resonance Analysis

### Option 1: Temporal Dynamics (Recurrent Resonance)

**Concept**: Analyze how activations evolve through transformer layers (layer-to-layer dynamics)

**Implementation**:
```python
def layer_resonance_analysis(prompt, layers=[0,2,4,6,8,10,12]):
    """
    Track activation evolution through layers to detect resonant patterns.
    """
    activations_by_layer = []
    
    # Get activations at each layer for the same prompt
    for layer in layers:
        h = model.get_hidden_at_layer(prompt, layer)
        activations_by_layer.append(h.mean(dim=0))  # Pool over sequence
    
    # Now we have a time series: activation(layer_0) → activation(layer_12)
    X = np.stack(activations_by_layer)  # Shape: (n_layers, hidden_dim)
    
    # Frequency domain analysis (FFT across layers)
    from scipy.fft import fft, fftfreq
    
    resonance_spectrum = {}
    for feature_idx in range(X.shape[1]):
        signal = X[:, feature_idx]
        
        # Compute FFT to find dominant frequencies
        freq_spectrum = fft(signal)
        frequencies = fftfreq(len(signal))
        
        # Detect peaks (resonant frequencies)
        power = np.abs(freq_spectrum) ** 2
        resonance_spectrum[f'feature_{feature_idx}'] = {
            'dominant_frequency': frequencies[np.argmax(power[1:])+1],
            'power': np.max(power[1:]),
            'harmonics': detect_harmonics(freq_spectrum)
        }
    
    return resonance_spectrum
```

**What this gives us**:
- ✅ Actual frequencies (cycles per layer)
- ✅ Harmonic content
- ✅ Oscillatory behavior detection
- ✅ Phase relationships between features

**Mathematical justification**: 
- Transformer layers as discrete time steps
- Activation evolution as dynamical system
- Resonance = dominant frequencies in layer-wise evolution

---

### Option 2: Perturbation Response (Impulse Resonance)

**Concept**: Measure how the model responds to perturbations over multiple generation steps

**Implementation**:
```python
def impulse_resonance_analysis(prompt, perturbation, steps=20):
    """
    Apply perturbation at layer L, measure response through generation.
    """
    responses = []
    
    # Get initial activation
    h0 = model.get_hidden_at_layer(prompt, layer=6)
    
    # Apply impulse perturbation
    h_perturbed = h0 + perturbation * 0.1
    
    # Generate tokens and track how perturbation propagates
    current_prompt = prompt
    for step in range(steps):
        # Generate next token with perturbation injected
        token, hidden = model.generate_next(
            current_prompt, 
            layer_intervention={6: h_perturbed}
        )
        
        # Measure response magnitude
        h_current = model.get_hidden_at_layer(current_prompt + token, layer=6)
        response = np.linalg.norm(h_current - h0)
        responses.append(response)
        
        current_prompt += token
    
    # Analyze response pattern
    responses = np.array(responses)
    
    # Look for oscillatory decay (classic resonance signature)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(responses)
    
    if len(peaks) > 1:
        # Compute damped frequency
        period = np.mean(np.diff(peaks))
        damping = -np.log(responses[peaks[-1]] / responses[peaks[0]]) / len(peaks)
        
        return {
            'resonant': True,
            'period': period,
            'damping_coefficient': damping,
            'quality_factor': period / (2 * damping) if damping > 0 else np.inf
        }
    else:
        return {
            'resonant': False,
            'monotonic_decay': True
        }
```

**What this gives us**:
- ✅ Impulse response function
- ✅ Damped oscillations (if they exist)
- ✅ Quality factor (Q) - classic resonance metric
- ✅ Natural frequencies of the system

**Mathematical justification**:
- Perturbations as impulse inputs
- Response tracking as transfer function
- Resonance = oscillatory response to impulses

---

### Option 3: Multi-Probe Phase Coherence

**Concept**: Analyze phase relationships between different probe directions

**Implementation**:
```python
def phase_coherence_analysis(prompt, n_probes=16, steps=20):
    """
    Look for phase-locked oscillations across probe directions.
    """
    from scipy.signal import hilbert
    
    h0 = model.get_hidden_at_layer(prompt, layer=6)
    
    # Generate multiple probe directions
    probe_directions = generate_random_directions(h0.shape[-1], n_probes)
    
    # Walk along each probe and record activation trajectory
    probe_trajectories = []
    for direction in probe_directions:
        trajectory = []
        for step in range(steps):
            h_perturbed = h0 + direction * (step * 0.01)
            
            # Project activation onto some observable (e.g., specific neuron)
            activation_value = model.forward_from_layer(h_perturbed)[0].mean()
            trajectory.append(activation_value.item())
        
        probe_trajectories.append(np.array(trajectory))
    
    # Compute phase coherence using Hilbert transform
    phases = []
    amplitudes = []
    for traj in probe_trajectories:
        analytic_signal = hilbert(traj - traj.mean())
        phase = np.angle(analytic_signal)
        amplitude = np.abs(analytic_signal)
        phases.append(phase)
        amplitudes.append(amplitude)
    
    phases = np.array(phases)  # Shape: (n_probes, steps)
    
    # Compute phase locking value (PLV)
    phase_diffs = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
    plv = np.abs(np.mean(np.exp(1j * phase_diffs), axis=2))
    
    # Detect resonant modes (high phase coherence)
    coherence_threshold = 0.7
    resonant_pairs = np.where(plv > coherence_threshold)
    
    return {
        'phase_locking_matrix': plv,
        'n_resonant_pairs': len(resonant_pairs[0]) // 2,
        'mean_coherence': np.mean(plv[np.triu_indices_from(plv, k=1)]),
        'coherent_modes': resonant_pairs
    }
```

**What this gives us**:
- ✅ Phase relationships between probes
- ✅ Phase locking (hallmark of resonance)
- ✅ Coherent vs. incoherent modes
- ✅ Synchronization detection

**Mathematical justification**:
- Phase coherence = resonant coupling
- PLV (Phase Locking Value) is standard in neuroscience
- Synchronized oscillations = resonance phenomenon

---

### Option 4: Graph Eigenmode Resonance

**Concept**: Analyze resonance in the graph structure itself

**Implementation**:
```python
def graph_resonance_analysis(resonance_graph):
    """
    Analyze eigenvalue resonance of the manifold graph.
    """
    # Get graph Laplacian
    from scipy.sparse import csgraph
    
    W = resonance_graph['adjacency_matrix']
    L = csgraph.laplacian(W, normed=True)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Resonance detection: gaps in eigenvalue spectrum
    gaps = np.diff(eigenvalues)
    significant_gaps = np.where(gaps > np.mean(gaps) + 2*np.std(gaps))[0]
    
    # Each gap indicates a resonant frequency scale
    resonant_scales = eigenvalues[significant_gaps + 1]
    
    # Compute spectral gap (largest eigenvalue gap)
    spectral_gap = np.max(gaps)
    
    # Cheeger constant (measures "resonance" of graph cuts)
    cheeger_constant = estimate_cheeger_constant(W, eigenvalues[1])
    
    return {
        'eigenvalue_spectrum': eigenvalues,
        'resonant_frequencies': resonant_scales,
        'spectral_gap': spectral_gap,
        'cheeger_constant': cheeger_constant,
        'n_resonant_modes': len(significant_gaps)
    }

def estimate_cheeger_constant(W, lambda_2):
    """
    Cheeger constant relates to resonance quality of graph partitions.
    """
    # Cheeger inequality: h/2 <= lambda_2 <= 2h
    return lambda_2 / 2  # Lower bound estimate
```

**What this gives us**:
- ✅ Graph eigenvalue spectrum (actual frequencies)
- ✅ Spectral gaps (resonant modes of network)
- ✅ Cheeger constant (resonance quality)
- ✅ Connection to wave equation on graphs

**Mathematical justification**:
- Graph Laplacian eigenvalues = resonant frequencies of network
- Well-established in spectral graph theory
- Direct connection to physical resonance on networks

---

### Option 5: Wavelet Resonance Analysis

**Concept**: Use wavelets to detect multi-scale oscillatory patterns

**Implementation**:
```python
def wavelet_resonance_analysis(activation_matrix):
    """
    Apply continuous wavelet transform to detect resonant patterns.
    """
    from scipy import signal
    
    # For each feature dimension, analyze across probe samples
    resonance_features = []
    
    for feature_idx in range(activation_matrix.shape[1]):
        signal_1d = activation_matrix[:, feature_idx]
        
        # Continuous wavelet transform
        widths = np.arange(1, 31)  # Scale range
        cwt_matrix = signal.cwt(signal_1d, signal.ricker, widths)
        
        # Find dominant scales (resonant frequencies)
        power = np.abs(cwt_matrix) ** 2
        scale_power = np.mean(power, axis=1)
        
        dominant_scale = widths[np.argmax(scale_power)]
        
        # Compute quality factor (sharpness of resonance)
        peak_power = np.max(scale_power)
        bandwidth = np.sum(scale_power > peak_power * 0.5)
        quality_factor = dominant_scale / bandwidth
        
        resonance_features.append({
            'dominant_scale': dominant_scale,
            'resonance_strength': peak_power,
            'quality_factor': quality_factor,
            'wavelet_spectrum': cwt_matrix
        })
    
    return resonance_features
```

**What this gives us**:
- ✅ Multi-scale frequency analysis
- ✅ Time-frequency localization
- ✅ Resonance bandwidth and Q-factor
- ✅ Robust to non-stationary signals

**Mathematical justification**:
- Wavelets detect oscillations at multiple scales
- Q-factor = classic resonance quality metric
- Used in signal processing for resonance detection

---

## Comparison: Current vs. Resonance-Based

| Aspect | Current (SVD) | True Resonance |
|--------|--------------|----------------|
| **Domain** | Spatial (feature space) | Temporal/Frequency |
| **Analysis** | Static snapshot | Dynamic response |
| **Output** | Singular values | Frequencies, phases |
| **Interpretation** | Principal components | Oscillatory modes |
| **Metrics** | Entropy, participation ratio | Q-factor, damping, PLV |
| **Math Foundation** | Linear algebra | Dynamical systems, signal processing |

---

## Recommended Hybrid Approach

Combine current spectral analysis with genuine resonance metrics:

### Phase 1: Keep Spectral Core (Current)
```python
# This is still valuable - rename it properly
spectral_signature = spectral_analyzer.compute_signature(A)
```

### Phase 2: Add Layer-Wise Resonance
```python
# NEW: Track evolution through layers
layer_resonance = resonance_analyzer.layer_dynamics(prompt, layers=range(12))
```

### Phase 3: Add Perturbation Response  
```python
# NEW: Measure impulse response
impulse_response = resonance_analyzer.perturbation_response(
    prompt, perturbation, steps=20
)
```

### Phase 4: Combined Descriptor
```python
# Combine spectral + resonance features
descriptor = {
    'spectral': spectral_signature,           # SVD features (current)
    'layer_resonance': layer_resonance,       # Frequency content (NEW)
    'impulse_response': impulse_response,     # Dynamic response (NEW)
    'phase_coherence': phase_coherence,       # Synchronization (NEW)
}
```

---

## Implementation Roadmap

### Minimal Change (Quick Win)
**Goal**: Add layer-wise frequency analysis while keeping current spectral analysis

**Effort**: ~8 hours
**Files to modify**:
- Create `arm_library/core/resonance_dynamics.py` (NEW)
- Keep `resonance_analyzer.py` but rename methods to `spectral_*`
- Update `arm_mapper.py` to call both spectral + resonance

**Benefits**:
- ✅ Adds genuine frequency analysis
- ✅ Minimal disruption to existing code
- ✅ Can compare spectral vs. resonance features

### Full Resonance Overhaul (Rigorous)
**Goal**: Replace SVD-based analysis with comprehensive resonance framework

**Effort**: ~40 hours
**Components**:
1. Layer dynamics analysis (temporal resonance)
2. Impulse response tracking (perturbation resonance)
3. Phase coherence measurement (synchronization)
4. Graph eigenmode analysis (structural resonance)
5. Validation on synthetic resonant systems

**Benefits**:
- ✅ Fully justified "resonance" terminology
- ✅ Novel contribution to field
- ✅ Richer behavioral signatures

**Risks**:
- ⚠️ May not find actual oscillations (transformers might not resonate)
- ⚠️ Increased computational cost
- ⚠️ More complex to interpret

---

## Key Question to Answer First

**Before investing in full resonance implementation, we need to test**:

### Experiment: Do Transformers Actually Resonate?

```python
# Quick test: Track activation through layers
prompt = "The cat sat on the"
activations = []
for layer in range(12):
    h = model.get_hidden_at_layer(prompt, layer)
    activations.append(h.mean().item())

# Plot and look for oscillations
import matplotlib.pyplot as plt
plt.plot(activations)
plt.xlabel('Layer')
plt.ylabel('Mean Activation')
plt.title('Do we see oscillations?')

# Statistical test
from scipy.stats import find_peaks
peaks, _ = find_peaks(activations)
print(f"Number of peaks: {len(peaks)}")
print(f"Evidence of oscillation: {len(peaks) > 2}")
```

**If oscillations exist**: Full resonance framework is justified  
**If monotonic/random**: Stick with spectral analysis, rename terminology

---

## Recommendation

**Short term** (2 hours):
1. Run the quick experiment above to test for layer-wise oscillations
2. If oscillations found → pursue resonance framework
3. If not found → rename to spectral analysis (Option A from audit)

**Medium term** (if oscillations exist):
1. Implement layer dynamics resonance (Option 1)
2. Add as complementary to existing spectral features
3. Run ablation: spectral only vs. resonance only vs. combined

**Long term** (if resonance proves valuable):
1. Full resonance framework with all 5 options
2. Publish as novel "Resonance Mapping" methodology
3. Theoretical paper on transformer resonance

---

## Bottom Line

**To justify "resonance" terminology, we need**:
1. ✅ **Temporal/dynamic analysis** (not static SVD)
2. ✅ **Frequency domain** (FFT, wavelets, eigenvalues)
3. ✅ **Oscillatory patterns** (empirically demonstrated)
4. ✅ **Phase relationships** (coherence, synchronization)

**Current status**: We have none of these ❌

**Path forward**: Test for oscillations first, then decide:
- Found oscillations? → Build resonance framework
- No oscillations? → Rename to spectral analysis

Would you like me to run the quick oscillation test right now? (2-3 minutes)

