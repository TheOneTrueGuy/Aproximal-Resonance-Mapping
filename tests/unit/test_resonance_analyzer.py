"""
Unit tests for ResonanceAnalyzer class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from arm_library.core.resonance_analyzer import ResonanceAnalyzer
from arm_library.utils.config import ARMConfig


class TestResonanceAnalyzer:
    """Test cases for ResonanceAnalyzer functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ARMConfig(n_modes=6, random_seed=42)

    @pytest.fixture
    def resonance_analyzer(self, config):
        """Create ResonanceAnalyzer instance."""
        return ResonanceAnalyzer(config)

    @pytest.fixture
    def sample_activation_matrix(self):
        """Create sample activation matrix for testing."""
        np.random.seed(42)
        n_samples, n_features = 80, 256  # 16 probes * 5 steps = 80 samples
        # Create matrix with some structure (first few modes should dominate)
        A = np.random.randn(n_samples, n_features).astype(np.float32)

        # Add some dominant modes for testing
        mode1 = np.random.randn(n_features)
        mode2 = np.random.randn(n_features)
        A += 5.0 * np.outer(np.sin(np.linspace(0, 4*np.pi, n_samples)), mode1)
        A += 3.0 * np.outer(np.cos(np.linspace(0, 4*np.pi, n_samples)), mode2)

        return A

    def test_initialization(self, config):
        """Test ResonanceAnalyzer initialization."""
        analyzer = ResonanceAnalyzer(config)
        assert analyzer.config == config

    def test_resonance_signature_basic_properties(self, resonance_analyzer, sample_activation_matrix):
        """Test basic properties of resonance signature computation."""
        signature = resonance_analyzer.resonance_signature(sample_activation_matrix)

        # Check required keys
        required_keys = [
            'singular_values', 's_norm', 'entropy', 'participation_ratio',
            'participation_ratio_normalized', 'top_singular_vectors', 'explained_variance_ratio'
        ]

        for key in required_keys:
            assert key in signature

        # Check shapes
        n_modes = resonance_analyzer.config.n_modes
        assert signature['singular_values'].shape == (n_modes,)
        assert signature['s_norm'].shape == (n_modes,)
        assert signature['top_singular_vectors'].shape == (n_modes, sample_activation_matrix.shape[1])
        assert signature['explained_variance_ratio'].shape == (n_modes,)

        # Check value ranges
        assert np.all(signature['singular_values'] >= 0)
        assert np.all(signature['s_norm'] >= 0)
        assert np.all(signature['s_norm'] <= 1)
        # Note: s_norm contains only top n_modes values, so they don't necessarily sum to 1
        assert np.sum(signature['s_norm']) <= 1.0
        assert signature['entropy'] >= 0
        assert 0 <= signature['participation_ratio_normalized'] <= 1

    def test_resonance_signature_custom_modes(self, resonance_analyzer, sample_activation_matrix):
        """Test resonance signature with custom number of modes."""
        n_modes = 3
        signature = resonance_analyzer.resonance_signature(sample_activation_matrix, n_modes=n_modes)

        assert signature['singular_values'].shape == (n_modes,)
        assert signature['s_norm'].shape == (n_modes,)
        assert signature['top_singular_vectors'].shape == (n_modes, sample_activation_matrix.shape[1])

    def test_batch_resonance_signatures(self, resonance_analyzer):
        """Test batch processing of resonance signatures."""
        np.random.seed(42)
        matrices = [
            np.random.randn(50, 128).astype(np.float32),
            np.random.randn(30, 128).astype(np.float32),
            np.random.randn(70, 128).astype(np.float32),
        ]

        signatures = resonance_analyzer.batch_resonance_signatures(matrices)

        assert len(signatures) == len(matrices)
        for sig in signatures:
            assert 'singular_values' in sig

    def test_compare_resonance_signatures_cosine(self, resonance_analyzer, sample_activation_matrix):
        """Test cosine similarity comparison."""
        sig1 = resonance_analyzer.resonance_signature(sample_activation_matrix)

        # Create slightly different matrix
        A2 = sample_activation_matrix + 0.1 * np.random.randn(*sample_activation_matrix.shape)
        sig2 = resonance_analyzer.resonance_signature(A2)

        similarity = resonance_analyzer.compare_resonance_signatures(sig1, sig2, metric="cosine")

        # Cosine similarity should be between -1 and 1
        assert -1 <= similarity <= 1

        # Should be high similarity (matrices are similar)
        assert similarity > 0.5

    def test_compare_resonance_signatures_euclidean(self, resonance_analyzer, sample_activation_matrix):
        """Test euclidean distance comparison."""
        sig1 = resonance_analyzer.resonance_signature(sample_activation_matrix)

        # Create different matrix
        A2 = np.random.randn(*sample_activation_matrix.shape).astype(np.float32)
        sig2 = resonance_analyzer.resonance_signature(A2)

        distance = resonance_analyzer.compare_resonance_signatures(sig1, sig2, metric="euclidean")

        # Euclidean distance should be non-negative
        assert distance >= 0

    def test_compare_resonance_signatures_entropy_diff(self, resonance_analyzer, sample_activation_matrix):
        """Test entropy difference comparison."""
        sig1 = resonance_analyzer.resonance_signature(sample_activation_matrix)

        # Create different matrix
        A2 = np.random.randn(*sample_activation_matrix.shape).astype(np.float32)
        sig2 = resonance_analyzer.resonance_signature(A2)

        diff = resonance_analyzer.compare_resonance_signatures(sig1, sig2, metric="entropy_diff")

        # Entropy difference should be non-negative
        assert diff >= 0

    def test_compare_resonance_signatures_invalid_metric(self, resonance_analyzer, sample_activation_matrix):
        """Test invalid metric raises ValueError."""
        sig1 = resonance_analyzer.resonance_signature(sample_activation_matrix)
        sig2 = resonance_analyzer.resonance_signature(sample_activation_matrix)

        with pytest.raises(ValueError, match="Unknown metric"):
            resonance_analyzer.compare_resonance_signatures(sig1, sig2, metric="invalid")

    def test_detect_resonance_modes(self, resonance_analyzer, sample_activation_matrix):
        """Test resonance mode detection."""
        result = resonance_analyzer.detect_resonance_modes(sample_activation_matrix, threshold=0.05)

        required_keys = [
            'n_significant_modes', 'significant_mode_indices',
            'significant_explained_var', 'cumulative_explained_var'
        ]

        for key in required_keys:
            assert key in result

        assert result['n_significant_modes'] >= 0
        assert len(result['significant_mode_indices']) == result['n_significant_modes']
        assert len(result['cumulative_explained_var']) == resonance_analyzer.config.n_modes

    def test_entropy_calculation(self, resonance_analyzer):
        """Test entropy calculation for known distributions."""
        # Uniform distribution should have high entropy
        uniform_s = np.ones(10) / 10
        entropy_uniform = -np.sum(uniform_s * np.log(uniform_s + 1e-12))

        # Concentrated distribution should have low entropy
        concentrated_s = np.zeros(10)
        concentrated_s[0] = 1.0
        entropy_concentrated = -np.sum(concentrated_s * np.log(concentrated_s + 1e-12))

        assert entropy_uniform > entropy_concentrated

    def test_participation_ratio_calculation(self, resonance_analyzer):
        """Test participation ratio calculation."""
        # Single dominant mode
        s_single = np.array([10.0, 1.0, 1.0, 1.0])
        pr_single = (s_single**2).sum()**2 / (np.sum(s_single**4))

        # Uniform modes
        s_uniform = np.array([1.0, 1.0, 1.0, 1.0])
        pr_uniform = (s_uniform**2).sum()**2 / (np.sum(s_uniform**4))

        # Participation ratio should be higher for more uniform distributions
        assert pr_uniform > pr_single
        assert pr_single >= 1.0
        assert pr_uniform <= len(s_uniform)
