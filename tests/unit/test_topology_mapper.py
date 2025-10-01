import numpy as np
import pytest

from arm_library.core.topology_mapper import TopologyMapper
from arm_library.utils.config import ARMConfig


@pytest.mark.unit
def test_local_persistence_shapes():
    cfg = ARMConfig(max_homology_dim=1)
    mapper = TopologyMapper(cfg)
    A = np.random.RandomState(0).randn(20, 8).astype(np.float32)
    out = mapper.local_persistence(A)
    assert 'diagrams' in out and isinstance(out['diagrams'], list)
    assert 'persistence_features' in out
    # Check finite metrics if present
    for k, v in out['persistence_features'].items():
        assert v['n_features'] >= 0
        assert np.isfinite(v['max_persistence'])
        assert np.isfinite(v['mean_persistence'])


@pytest.mark.unit
def test_build_resonance_graph_shapes():
    cfg = ARMConfig(topology_neighbors=2)
    mapper = TopologyMapper(cfg)
    # Create three synthetic signatures
    sigs = []
    for i in range(3):
        sigs.append({
            's_norm': np.array([0.6, 0.3, 0.1], dtype=np.float32),
            'entropy': 0.5 + 0.1 * i,
            'participation_ratio_normalized': 0.2 + 0.1 * i,
        })
    g = mapper.build_resonance_graph(sigs)
    X = g['feature_vectors']
    W = g['adjacency_matrix']
    E = g['spectral_embedding']
    assert X.shape[0] == 3 and X.shape[1] >= 5
    assert W.shape == (3, 3)
    # For very small n, spectral embedding may return up to n-1 components
    assert E.shape[1] in (2, 3)


@pytest.mark.unit
def test_detect_attractor_basins_small_n():
    cfg = ARMConfig(topology_neighbors=2)
    mapper = TopologyMapper(cfg)
    sigs = []
    for i in range(3):
        sigs.append({
            's_norm': np.array([0.7, 0.2, 0.1], dtype=np.float32),
            'entropy': 0.4 + 0.1 * i,
            'participation_ratio_normalized': 0.3 + 0.1 * i,
        })
    g = mapper.build_resonance_graph(sigs)
    c = mapper.detect_attractor_basins(sigs, g)
    assert 'cluster_labels' in c and len(c['cluster_labels']) == 3
    assert 1 <= c['n_clusters'] <= 3
    assert sum(c['cluster_sizes']) == 3
"""
Unit tests for TopologyMapper class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from arm_library.core.topology_mapper import TopologyMapper
from arm_library.utils.config import ARMConfig


class TestTopologyMapper:
    """Test cases for TopologyMapper functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ARMConfig(max_homology_dim=1, topology_neighbors=5, random_seed=42)

    @pytest.fixture
    def topology_mapper(self, config):
        """Create TopologyMapper instance."""
        return TopologyMapper(config)

    @pytest.fixture
    def sample_activation_matrix(self):
        """Create sample activation matrix for testing."""
        np.random.seed(42)
        n_points, n_features = 50, 64
        # Create points arranged in two clusters for topological interest
        points = []

        # Cluster 1: circle-like
        for i in range(25):
            angle = 2 * np.pi * i / 25
            x = np.cos(angle) + 0.1 * np.random.randn()
            y = np.sin(angle) + 0.1 * np.random.randn()
            point = np.random.randn(n_features)
            point[0] = x
            point[1] = y
            points.append(point)

        # Cluster 2: different region
        for i in range(25):
            point = np.random.randn(n_features)
            point[0] += 3.0  # shift x coordinate
            point[1] += 3.0  # shift y coordinate
            points.append(point)

        return np.array(points).astype(np.float32)

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_initialization(self, config):
        """Test TopologyMapper initialization."""
        mapper = TopologyMapper(config)
        assert mapper.config == config

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_local_persistence_basic(self, topology_mapper, sample_activation_matrix):
        """Test basic persistent homology computation."""
        result = topology_mapper.local_persistence(sample_activation_matrix)

        # Check required keys
        required_keys = ['diagrams', 'persistence_features', 'n_points', 'n_features']
        for key in required_keys:
            assert key in result

        # Check data integrity
        assert result['n_points'] == len(sample_activation_matrix)
        assert result['n_features'] == sample_activation_matrix.shape[1]
        assert len(result['diagrams']) == topology_mapper.config.max_homology_dim + 1

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_local_persistence_custom_maxdim(self, topology_mapper, sample_activation_matrix):
        """Test persistent homology with custom max dimension."""
        maxdim = 2
        result = topology_mapper.local_persistence(sample_activation_matrix, maxdim=maxdim)

        assert len(result['diagrams']) == maxdim + 1

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_build_resonance_graph(self, topology_mapper):
        """Test building resonance graph from signatures."""
        # Create mock resonance signatures
        n_signatures = 10
        signatures = []
        for i in range(n_signatures):
            sig = {
                's_norm': np.random.randn(6).astype(np.float32),
                'entropy': float(np.random.rand()),
                'participation_ratio_normalized': float(np.random.rand()),
            }
            signatures.append(sig)

        result = topology_mapper.build_resonance_graph(signatures)

        # Check required keys
        required_keys = ['adjacency_matrix', 'feature_vectors', 'spectral_embedding', 'n_nodes']
        for key in required_keys:
            assert key in result

        # Check shapes
        assert result['n_nodes'] == n_signatures
        assert result['adjacency_matrix'].shape == (n_signatures, n_signatures)
        assert result['feature_vectors'].shape == (n_signatures, 8)  # 6 s_norm + 2 features
        assert result['spectral_embedding'].shape == (n_signatures, 3)

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("sklearn", reason="sklearn not available"),
                       reason="sklearn not available")
    def test_detect_attractor_basins(self, topology_mapper):
        """Test attractor basin detection."""
        # Create mock data
        n_signatures = 20
        signatures = []
        feature_vectors = []

        for i in range(n_signatures):
            sig = {
                's_norm': np.random.randn(6).astype(np.float32),
                'entropy': float(np.random.rand()),
                'participation_ratio_normalized': float(np.random.rand()),
            }
            signatures.append(sig)
            features = np.concatenate([sig['s_norm'], [sig['entropy'], sig['participation_ratio_normalized']]])
            feature_vectors.append(features)

        graph_data = {
            'feature_vectors': np.array(feature_vectors),
        }

        result = topology_mapper.detect_attractor_basins(signatures, graph_data)

        # Check required keys
        required_keys = ['cluster_labels', 'n_clusters', 'centroids', 'cluster_sizes']
        for key in required_keys:
            assert key in result

        # Check data integrity
        assert len(result['cluster_labels']) == n_signatures
        assert result['n_clusters'] >= 2  # Should find at least 2 clusters
        assert len(result['cluster_sizes']) == result['n_clusters']
        assert len(result['centroids']) == result['n_clusters']

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_compute_topological_descriptor(self, topology_mapper):
        """Test computation of combined topological descriptor."""
        # Mock resonance signature
        resonance_sig = {
            's_norm': np.array([0.5, 0.3, 0.2], dtype=np.float32),
            'entropy': 1.5,
            'participation_ratio_normalized': 0.8,
        }

        # Mock persistence data
        persistence_data = {
            'persistence_features': {
                'h0_features': {
                    'max_persistence': 2.1,
                    'mean_persistence': 1.2,
                    'n_features': 5,
                },
                'h1_features': {
                    'max_persistence': 1.8,
                    'mean_persistence': 0.9,
                    'n_features': 2,
                }
            }
        }

        descriptor = topology_mapper.compute_topological_descriptor(resonance_sig, persistence_data)

        # Check descriptor combines resonance and topological features
        expected_length = len(resonance_sig['s_norm']) + 2 + (3 * 2)  # s_norm + 2 resonance + 3*2 topo features
        assert len(descriptor) == expected_length

        # Check resonance features are included
        np.testing.assert_array_equal(descriptor[:3], resonance_sig['s_norm'])
        assert descriptor[3] == resonance_sig['entropy']
        assert descriptor[4] == resonance_sig['participation_ratio_normalized']

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_compute_topological_descriptor_missing_topo_features(self, topology_mapper):
        """Test descriptor computation when some topological features are missing."""
        resonance_sig = {
            's_norm': np.array([0.6, 0.4], dtype=np.float32),
            'entropy': 1.0,
            'participation_ratio_normalized': 0.7,
        }

        # Persistence data with only H0 features
        persistence_data = {
            'persistence_features': {
                'h0_features': {
                    'max_persistence': 1.5,
                    'mean_persistence': 0.8,
                    'n_features': 3,
                }
            }
        }

        descriptor = topology_mapper.compute_topological_descriptor(resonance_sig, persistence_data)

        # Should pad missing H1 features with zeros
        expected_length = len(resonance_sig['s_norm']) + 2 + (3 * 2)  # s_norm + 2 resonance + 3*2 topo features
        assert len(descriptor) == expected_length

        # H1 features should be zero
        assert descriptor[-3:] == [0.0, 0.0, 0.0]  # max, mean, n_features for H1

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_missing_dependencies_raises_error(self):
        """Test that missing dependencies raise appropriate errors."""
        config = ARMConfig()

        # Mock missing ripser
        with patch.dict('sys.modules', {'ripser': None}):
            with pytest.raises(ImportError, match="ripser package required"):
                TopologyMapper(config)
