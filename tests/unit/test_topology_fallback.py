import numpy as np
import pytest

from arm_library.core.topology_mapper import TopologyMapper
from arm_library.utils.config import ARMConfig


@pytest.mark.unit
def test_clustering_kmeans_fallback(monkeypatch):
    # Force SpectralClustering failure to exercise KMeans fallback
    cfg = ARMConfig(topology_neighbors=2, random_seed=42)
    mapper = TopologyMapper(cfg)

    # Create small feature set that will trigger exception path for spectral clustering
    sigs = [{
        's_norm': np.array([0.7, 0.2, 0.1], dtype=np.float32),
        'entropy': 0.4,
        'participation_ratio_normalized': 0.3,
    } for _ in range(4)]

    g = mapper.build_resonance_graph(sigs)

    class Boom(Exception):
        pass

    # Monkeypatch SpectralClustering.fit_predict to raise
    import sklearn.cluster as skc
    orig_cls = skc.SpectralClustering

    class BadSC(orig_cls):
        def fit_predict(self, X, y=None):  # noqa: N802 (sklearn API)
            raise Boom("force fallback")

    monkeypatch.setattr(skc, 'SpectralClustering', BadSC)

    out = mapper.detect_attractor_basins(sigs, g, n_clusters=2)
    assert out['n_clusters'] == 2
    assert len(out['cluster_labels']) == len(sigs)

