import numpy as np
import pytest

from arm_library.core.resonance_analyzer import ResonanceAnalyzer
from arm_library.utils.config import ARMConfig


@pytest.mark.unit
def test_compare_resonance_signatures_metrics():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 16)).astype(np.float32)
    B = (rng.normal(size=(40, 16)) * 0.5 + A * 0.5).astype(np.float32)

    analyzer = ResonanceAnalyzer(ARMConfig(n_modes=4))
    sigA = analyzer.resonance_signature(A)
    sigB = analyzer.resonance_signature(B)

    cos = analyzer.compare_resonance_signatures(sigA, sigB, metric="cosine")
    dist = analyzer.compare_resonance_signatures(sigA, sigB, metric="euclidean")
    ediff = analyzer.compare_resonance_signatures(sigA, sigB, metric="entropy_diff")

    assert -1.0 <= cos <= 1.0
    assert dist >= 0.0
    assert ediff >= 0.0


@pytest.mark.unit
def test_detect_resonance_modes_thresholding():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(60, 20)).astype(np.float32)
    analyzer = ResonanceAnalyzer(ARMConfig(n_modes=6))
    modes = analyzer.detect_resonance_modes(A, threshold=0.05)
    assert modes['n_significant_modes'] >= 0
    assert len(modes['significant_mode_indices']) == modes['n_significant_modes']

