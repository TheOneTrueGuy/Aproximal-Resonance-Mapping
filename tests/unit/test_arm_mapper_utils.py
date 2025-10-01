import numpy as np
import pytest

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


@pytest.mark.unit
def test_find_similar_seeds_sorting(monkeypatch):
    cfg = ARMConfig(n_modes=3)
    mapper = ARMMapper(cfg)

    # Fake resonance analyzer compare to be deterministic
    def fake_compare(sig1, sig2, metric="cosine"):
        # Return higher value for larger first component if cosine
        if metric == "cosine":
            return float(sig2['s_norm'][0])
        else:
            return float(abs(sig2['s_norm'][0] - sig1['s_norm'][0]))

    mapper.resonance_analyzer.compare_resonance_signatures = fake_compare

    target = {'s_norm': np.array([0.9, 0.05, 0.05], dtype=np.float32)}
    seeds = [
        {'resonance_signature': {'s_norm': np.array([x, 0.1, 0.1], dtype=np.float32)}}
        for x in (0.1, 0.5, 0.8)
    ]

    top = mapper.find_similar_seeds(target, seeds, top_k=2, metric='cosine')
    # Expect highest s_norm[0] first
    assert top[0][0] == 2 and len(top) == 2


@pytest.mark.unit
def test_compute_steering_vector_from_manifold_invalid_indices(monkeypatch):
    cfg = ARMConfig()
    mapper = ARMMapper(cfg)
    # Inject last results without prompts
    mapper._last_results = {'prompts': ["a", "b"]}
    with pytest.raises(ValueError):
        mapper.compute_steering_vector_from_manifold([5], [6])


