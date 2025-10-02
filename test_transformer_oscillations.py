#!/usr/bin/env python3
"""
Quick Oscillation Test for Transformers
Exercise 1.1 follow-up: Do transformers actually exhibit oscillatory behavior?

This will determine if "resonance" terminology has any empirical basis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def test_layer_wise_oscillations(model_name="distilgpt2", device="cpu"):
    """
    Track activations through transformer layers to detect oscillatory patterns.
    """
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test prompts spanning different content
    test_prompts = [
        "The cat sat on the mat",
        "Machine learning is powerful",
        "Once upon a time in a galaxy",
        "The quick brown fox jumps",
        "In the beginning there was",
    ]
    
    results = {}
    
    for prompt in test_prompts:
        print(f"\nAnalyzing: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Get number of layers
        if hasattr(model, 'transformer'):
            n_layers = len(model.transformer.h)
            layer_modules = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
            layer_modules = model.model.layers
        else:
            print(f"Unknown architecture for {model_name}")
            return None
        
        # Track activations through layers
        layer_activations = []
        
        with torch.no_grad():
            # Get embeddings
            if hasattr(model, 'transformer'):
                hidden = model.transformer.wte(inputs['input_ids'])
                if hasattr(model.transformer, 'wpe'):
                    positions = torch.arange(0, inputs['input_ids'].size(1), device=device).unsqueeze(0)
                    hidden = hidden + model.transformer.wpe(positions)
            else:
                hidden = model.model.embed_tokens(inputs['input_ids'])
            
            # Pass through each layer and record mean activation
            for layer_idx, layer_module in enumerate(layer_modules):
                hidden = layer_module(hidden)[0]  # Get hidden states
                
                # Record mean activation magnitude
                mean_activation = hidden.abs().mean().item()
                layer_activations.append(mean_activation)
                
        layer_activations = np.array(layer_activations)
        
        # Analyze for oscillations
        analysis = analyze_oscillations(layer_activations, prompt)
        results[prompt] = {
            'activations': layer_activations,
            'analysis': analysis
        }
    
    return results, n_layers

def analyze_oscillations(signal, label):
    """
    Statistical analysis to detect oscillatory behavior.
    """
    # Normalize signal
    signal_norm = (signal - signal.mean()) / (signal.std() + 1e-8)
    
    # 1. Peak detection
    peaks, peak_properties = find_peaks(signal_norm, prominence=0.3)
    troughs, _ = find_peaks(-signal_norm, prominence=0.3)
    
    # 2. FFT to detect dominant frequencies
    fft_vals = fft(signal_norm)
    freqs = fftfreq(len(signal_norm))
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    power = np.abs(fft_vals[:len(freqs)//2])**2
    
    # Find dominant frequency (excluding DC component)
    if len(power) > 1:
        dominant_freq_idx = np.argmax(power[1:]) + 1
        dominant_freq = positive_freqs[dominant_freq_idx]
        dominant_power = power[dominant_freq_idx]
        dc_power = power[0]
        
        # Signal-to-noise: dominant frequency vs. mean of others
        noise_power = np.mean(np.concatenate([power[1:dominant_freq_idx], 
                                               power[dominant_freq_idx+1:]]))
        snr = dominant_power / (noise_power + 1e-8)
    else:
        dominant_freq = 0
        dominant_power = 0
        dc_power = power[0]
        snr = 0
    
    # 3. Autocorrelation to detect periodicity
    autocorr = np.correlate(signal_norm, signal_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first significant peak in autocorrelation (excluding lag 0)
    if len(autocorr) > 1:
        ac_peaks, _ = find_peaks(autocorr[1:], height=0.3)
        if len(ac_peaks) > 0:
            period_estimate = ac_peaks[0] + 1
            periodicity_strength = autocorr[period_estimate]
        else:
            period_estimate = None
            periodicity_strength = 0
    else:
        period_estimate = None
        periodicity_strength = 0
    
    # 4. Monotonicity test
    diffs = np.diff(signal)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    monotonic = sign_changes < 2
    
    # 5. Overall oscillation score
    oscillation_score = (
        (len(peaks) + len(troughs)) / len(signal) * 10 +  # Number of peaks/troughs
        snr * 0.5 +                                        # Frequency domain strength
        periodicity_strength * 2                           # Autocorrelation strength
    )
    
    return {
        'n_peaks': len(peaks),
        'n_troughs': len(troughs),
        'dominant_frequency': float(dominant_freq),
        'frequency_power': float(dominant_power),
        'dc_power': float(dc_power),
        'snr': float(snr),
        'period_estimate': period_estimate,
        'periodicity_strength': float(periodicity_strength),
        'monotonic': monotonic,
        'oscillation_score': float(oscillation_score),
    }

def visualize_results(results, n_layers, save_path='oscillation_test_results.png'):
    """
    Create comprehensive visualization of oscillation test.
    """
    n_prompts = len(results)
    fig, axes = plt.subplots(n_prompts, 3, figsize=(15, 4*n_prompts))
    
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (prompt, data) in enumerate(results.items()):
        signal = data['activations']
        analysis = data['analysis']
        
        # Truncate prompt for display
        display_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
        
        # Plot 1: Raw activation through layers
        ax1 = axes[idx, 0]
        ax1.plot(range(len(signal)), signal, 'b-', linewidth=2)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Mean Activation Magnitude')
        ax1.set_title(f'{display_prompt}\nLayer-wise Activation')
        ax1.grid(True, alpha=0.3)
        
        # Highlight peaks and troughs
        signal_norm = (signal - signal.mean()) / (signal.std() + 1e-8)
        peaks, _ = find_peaks(signal_norm, prominence=0.3)
        troughs, _ = find_peaks(-signal_norm, prominence=0.3)
        if len(peaks) > 0:
            ax1.plot(peaks, signal[peaks], 'ro', markersize=8, label='Peaks')
        if len(troughs) > 0:
            ax1.plot(troughs, signal[troughs], 'go', markersize=8, label='Troughs')
        if len(peaks) > 0 or len(troughs) > 0:
            ax1.legend()
        
        # Plot 2: Frequency spectrum
        ax2 = axes[idx, 1]
        fft_vals = fft(signal_norm)
        freqs = fftfreq(len(signal_norm))
        positive_freqs = freqs[:len(freqs)//2]
        power = np.abs(fft_vals[:len(freqs)//2])**2
        
        ax2.plot(positive_freqs, power, 'r-', linewidth=2)
        ax2.set_xlabel('Frequency (cycles/layer)')
        ax2.set_ylabel('Power')
        ax2.set_title(f'Frequency Spectrum\nDominant: {analysis["dominant_frequency"]:.3f} cycles/layer')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Autocorrelation
        ax3 = axes[idx, 2]
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        lags = range(len(autocorr))
        
        ax3.plot(lags, autocorr, 'g-', linewidth=2)
        ax3.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax3.set_xlabel('Lag (layers)')
        ax3.set_ylabel('Autocorrelation')
        ax3.set_title(f'Autocorrelation\nPeriodicity: {analysis["periodicity_strength"]:.3f}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    return fig

def print_summary(results):
    """
    Print summary statistics.
    """
    print("\n" + "="*80)
    print("OSCILLATION TEST SUMMARY")
    print("="*80)
    
    all_scores = []
    all_peaks = []
    all_monotonic = []
    
    for prompt, data in results.items():
        analysis = data['analysis']
        all_scores.append(analysis['oscillation_score'])
        all_peaks.append(analysis['n_peaks'] + analysis['n_troughs'])
        all_monotonic.append(analysis['monotonic'])
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Peaks/Troughs: {analysis['n_peaks']}/{analysis['n_troughs']}")
        print(f"  Dominant Frequency: {analysis['dominant_frequency']:.4f} cycles/layer")
        print(f"  Frequency SNR: {analysis['snr']:.2f}")
        print(f"  Periodicity Strength: {analysis['periodicity_strength']:.3f}")
        print(f"  Monotonic: {analysis['monotonic']}")
        print(f"  Oscillation Score: {analysis['oscillation_score']:.2f}")
    
    print("\n" + "-"*80)
    print("AGGREGATE STATISTICS")
    print("-"*80)
    print(f"Average Oscillation Score: {np.mean(all_scores):.2f}")
    print(f"Average Peaks+Troughs: {np.mean(all_peaks):.1f}")
    print(f"Monotonic Signals: {sum(all_monotonic)}/{len(all_monotonic)}")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    avg_score = np.mean(all_scores)
    avg_peaks = np.mean(all_peaks)
    pct_monotonic = sum(all_monotonic) / len(all_monotonic) * 100
    
    if avg_score > 5 and avg_peaks > 3:
        print("[PASS] OSCILLATIONS DETECTED: Evidence of oscillatory behavior through layers")
        print("   Recommendation: 'Resonance' terminology may be justified")
        print("   Next step: Implement full resonance framework")
    elif avg_score > 2 and avg_peaks > 1:
        print("[WEAK] WEAK OSCILLATIONS: Some periodic patterns, but not strong")
        print("   Recommendation: 'Resonance' is a stretch, but not entirely unjustified")
        print("   Next step: Decide if metaphorical usage is acceptable")
    else:
        print("[FAIL] NO OSCILLATIONS: Activations are mostly monotonic or random")
        print("   Recommendation: Rename to 'Spectral Analysis' - current term unjustified")
        print("   Next step: Proceed with Option A (rename) or Option C (hybrid)")
    
    if pct_monotonic > 50:
        print(f"\n   Note: {pct_monotonic:.0f}% of signals are monotonic (no direction changes)")
        print("   This suggests layer-wise progression, not oscillation")
    
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("TRANSFORMER OSCILLATION TEST")
    print("Testing whether transformers exhibit oscillatory behavior through layers")
    print("="*80)
    
    # Run test
    results, n_layers = test_layer_wise_oscillations(model_name="distilgpt2", device="cpu")
    
    if results is not None:
        # Print analysis
        print_summary(results)
        
        # Create visualization
        visualize_results(results, n_layers)
        
        print("\n✅ Test complete!")
        print("   Results saved to: oscillation_test_results.png")
        print("   Review plots to visually assess oscillatory behavior")
    else:
        print("❌ Test failed - could not analyze model architecture")

