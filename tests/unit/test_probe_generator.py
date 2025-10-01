import numpy as np
import pytest

from arm_library.core.probe_generator import ProbeGenerator
from arm_library.utils.config import ARMConfig


@pytest.mark.unit
def test_sample_probes_and_expand_sequence():
    cfg = ARMConfig(eps=0.05, random_seed=123)
    gen = ProbeGenerator(cfg)

    # Hidden sequence: seq_len x d_model
    hidden = np.ones((6, 8), dtype=np.float32)

    dirs = gen.sample_probes_for_hidden(hidden, k=4, eps=0.05)
    assert dirs.shape == (4, 8)
    # Non-zero and finite
    assert np.all(np.isfinite(dirs))
    assert np.any(np.abs(dirs) > 0)

    expanded = gen.expand_delta_to_sequence(dirs[0], seq_len=6)
    assert expanded.shape == (6, 8)
    # All rows equal to the delta
    assert np.allclose(expanded[0], expanded[1])


@pytest.mark.unit
def test_build_path_and_batch_generation():
    cfg = ARMConfig(eps=0.03, steps_per_probe=5, random_seed=42)
    gen = ProbeGenerator(cfg)
    hidden = np.random.RandomState(0).randn(4, 10).astype(np.float32)

    # Single direction path
    dir_vec = np.random.RandomState(1).randn(10).astype(np.float32)
    path, ts = gen.build_probe_path(hidden, dir_vec, steps=5, tau=1.0)
    assert len(path) == 5
    assert ts.shape == (5,)
    assert path[0].shape == hidden.shape

    # Batch
    paths, dirs = gen.generate_probe_batch(hidden, k=3, steps=4, eps=0.02)
    assert len(paths) == 3
    assert dirs.shape == (3, 10)
    # Each path has 4 steps of same shape as hidden
    assert all(len(p) == 4 and p[0].shape == hidden.shape for p in paths)

"""
Unit tests for ProbeGenerator class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from arm_library.core.probe_generator import ProbeGenerator
from arm_library.utils.config import ARMConfig


class TestProbeGenerator:
    """Test cases for ProbeGenerator functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ARMConfig(
            probes_per_seed=8,
            steps_per_probe=5,
            eps=0.1,
            random_seed=42
        )

    @pytest.fixture
    def probe_generator(self, config):
        """Create ProbeGenerator instance."""
        return ProbeGenerator(config)

    @pytest.fixture
    def sample_hidden_vec(self):
        """Create sample hidden vector for testing."""
        np.random.seed(42)
        return np.random.randn(10, 768).astype(np.float32)  # seq_len=10, d_model=768

    def test_initialization(self, config):
        """Test ProbeGenerator initialization."""
        generator = ProbeGenerator(config)
        assert generator.config == config
        assert generator.rng is not None

    def test_sample_probes_for_hidden_default_params(self, probe_generator, sample_hidden_vec):
        """Test probe sampling with default parameters."""
        probes = probe_generator.sample_probes_for_hidden(sample_hidden_vec)

        # Check shape
        assert probes.shape == (probe_generator.config.probes_per_seed, sample_hidden_vec.shape[1])

        # Check that probes are properly normalized and scaled
        pooled = sample_hidden_vec.mean(axis=0)
        expected_scale = probe_generator.config.eps * np.linalg.norm(pooled)

        # Check that probe magnitudes are approximately correct
        probe_norms = np.linalg.norm(probes, axis=1)
        np.testing.assert_allclose(probe_norms, expected_scale, rtol=1e-5)

    def test_sample_probes_for_hidden_custom_params(self, probe_generator, sample_hidden_vec):
        """Test probe sampling with custom parameters."""
        k = 12
        eps = 0.05
        probes = probe_generator.sample_probes_for_hidden(sample_hidden_vec, k=k, eps=eps)

        assert probes.shape == (k, sample_hidden_vec.shape[1])

        pooled = sample_hidden_vec.mean(axis=0)
        expected_scale = eps * np.linalg.norm(pooled)
        probe_norms = np.linalg.norm(probes, axis=1)
        np.testing.assert_allclose(probe_norms, expected_scale, rtol=1e-5)

    def test_expand_delta_to_sequence(self, probe_generator):
        """Test delta expansion to sequence positions."""
        delta_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq_len = 5

        expanded = probe_generator.expand_delta_to_sequence(delta_vec, seq_len)

        assert expanded.shape == (seq_len, len(delta_vec))
        expected = np.tile(delta_vec[None, :], (seq_len, 1))
        np.testing.assert_array_equal(expanded, expected)

    def test_build_probe_path(self, probe_generator, sample_hidden_vec):
        """Test building probe paths along directions."""
        dir_vec = np.array([1.0] * sample_hidden_vec.shape[1], dtype=np.float32)
        steps = 7
        tau = 2.0

        path, ts = probe_generator.build_probe_path(sample_hidden_vec, dir_vec, steps=steps, tau=tau)

        # Check path length
        assert len(path) == steps
        assert len(ts) == steps

        # Check step values
        expected_ts = np.linspace(-tau, tau, steps, dtype=np.float32)
        np.testing.assert_array_equal(ts, expected_ts)

        # Check that path points are correctly offset
        seq_len = sample_hidden_vec.shape[0]
        expanded_dir = probe_generator.expand_delta_to_sequence(dir_vec, seq_len)

        for i, (hidden_state, t) in enumerate(zip(path, ts)):
            expected = sample_hidden_vec + t * expanded_dir
            np.testing.assert_array_equal(hidden_state, expected)

    def test_generate_probe_batch(self, probe_generator, sample_hidden_vec):
        """Test generating batch of probe paths."""
        k = 6
        steps = 4
        eps = 0.08

        probe_paths, probe_directions = probe_generator.generate_probe_batch(
            sample_hidden_vec, k=k, steps=steps, eps=eps
        )

        # Check number of paths
        assert len(probe_paths) == k
        assert probe_directions.shape == (k, sample_hidden_vec.shape[1])

        # Check each path
        for path in probe_paths:
            assert len(path) == steps
            assert path[0].shape == sample_hidden_vec.shape

    def test_reproducibility(self, config, sample_hidden_vec):
        """Test that results are reproducible with same random seed."""
        generator1 = ProbeGenerator(config)
        generator2 = ProbeGenerator(config)

        probes1 = generator1.sample_probes_for_hidden(sample_hidden_vec)
        probes2 = generator2.sample_probes_for_hidden(sample_hidden_vec)

        np.testing.assert_array_equal(probes1, probes2)

    def test_different_seeds_give_different_results(self, sample_hidden_vec):
        """Test that different random seeds give different results."""
        config1 = ARMConfig(random_seed=42)
        config2 = ARMConfig(random_seed=123)

        generator1 = ProbeGenerator(config1)
        generator2 = ProbeGenerator(config2)

        probes1 = generator1.sample_probes_for_hidden(sample_hidden_vec)
        probes2 = generator2.sample_probes_for_hidden(sample_hidden_vec)

        # Should be different (with very high probability)
        assert not np.allclose(probes1, probes2)
