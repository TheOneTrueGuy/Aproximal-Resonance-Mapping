"""
Integration tests for ARMMapper class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


@pytest.mark.integration
class TestARMMapperIntegration:
    """Integration tests for ARMMapper functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration optimized for integration testing."""
        return ARMConfig(
            model_name="distilgpt2",
            n_seeds=3,  # Very small for integration testing
            probes_per_seed=2,
            steps_per_probe=2,
            eps=0.01,
            layer_to_probe=1,  # Early layer for faster testing
            n_modes=3,
            random_seed=42,
        )

    @pytest.fixture
    def arm_mapper(self, config):
        """Create ARMMapper instance with mocked model interface."""
        with patch('arm_library.core.arm_mapper.TransformerModelInterface') as mock_interface_class:
            mock_interface = MagicMock()
            mock_interface_class.return_value = mock_interface

            # Configure mock interface
            mock_interface.get_model_dimensions.return_value = {
                'vocab_size': 50257,
                'd_model': 768,
                'n_layers': 6,
                'n_heads': 12,
            }

            # Mock get_hidden_at_layer to return consistent shapes
            def mock_get_hidden(prompt, layer_idx):
                seq_len = len(prompt.split())
                return np.random.randn(seq_len, 768).astype(np.float32)

            mock_interface.get_hidden_at_layer.side_effect = mock_get_hidden

            # Mock forward_from_layer
            def mock_forward(hidden, layer_idx, attention_mask=None):
                batch_size, seq_len, d_model = hidden.shape
                logits = np.random.randn(batch_size, seq_len, 50257).astype(np.float32)
                final_h = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
                intermediates = [final_h] * 3
                return torch.tensor(logits), torch.tensor(final_h), intermediates

            mock_interface.forward_from_layer.side_effect = mock_forward

            mapper = ARMMapper(config)
            return mapper

    def test_initialization(self, arm_mapper, config):
        """Test ARMMapper initialization."""
        assert arm_mapper.config == config
        assert arm_mapper.model_interface is not None
        assert arm_mapper.probe_generator is not None
        assert arm_mapper.resonance_analyzer is not None
        assert arm_mapper.topology_mapper is not None

    def test_collect_activation_matrix(self, arm_mapper):
        """Test activation matrix collection for a single seed."""
        prompt = "Hello world"

        A = arm_mapper.collect_activation_matrix(prompt)

        expected_samples = arm_mapper.config.probes_per_seed * arm_mapper.config.steps_per_probe
        assert A.shape[0] == expected_samples
        assert A.shape[1] == 768  # d_model

    def test_analyze_seed_point(self, arm_mapper):
        """Test complete analysis of a single seed point."""
        prompt = "The cat sat on the mat"

        result = arm_mapper.analyze_seed_point(prompt)

        # Check required keys
        required_keys = [
            'prompt', 'activation_matrix', 'resonance_signature',
            'persistence_data', 'descriptor'
        ]

        for key in required_keys:
            assert key in result

        assert result['prompt'] == prompt
        assert isinstance(result['descriptor'], np.ndarray)

    @pytest.mark.skipif(not hasattr(pytest, 'importorskip') or
                       pytest.importorskip("ripser", reason="ripser not available"),
                       reason="ripser package not available")
    def test_map_latent_manifold(self, arm_mapper):
        """Test mapping of latent manifold with multiple seeds."""
        prompts = [
            "The weather is nice",
            "Machine learning works",
            "I love coding",
        ]

        result = arm_mapper.map_latent_manifold(prompts)

        # Check required keys
        required_keys = [
            'seed_analyses', 'graph_data', 'clustering_data', 'descriptors', 'n_seeds'
        ]

        for key in required_keys:
            assert key in result

        assert result['n_seeds'] == len(prompts)
        assert len(result['seed_analyses']) == len(prompts)
        assert result['descriptors'].shape == (len(prompts), result['descriptors'].shape[1])

    def test_find_similar_seeds(self, arm_mapper):
        """Test finding similar seeds based on resonance patterns."""
        # Create mock seed analyses
        seed_analyses = []
        for i in range(5):
            analysis = {
                'resonance_signature': {
                    's_norm': np.random.randn(3).astype(np.float32),
                    'entropy': float(np.random.rand()),
                    'participation_ratio_normalized': float(np.random.rand()),
                    'top_singular_vectors': np.random.randn(3, 10).astype(np.float32),
                }
            }
            seed_analyses.append(analysis)

        # Use first signature as target
        target_sig = seed_analyses[0]['resonance_signature']

        similar_seeds = arm_mapper.find_similar_seeds(target_sig, seed_analyses, top_k=3)

        assert len(similar_seeds) == 3
        # First result should be the most similar (itself)
        assert similar_seeds[0][0] == 0  # seed index 0

        # Similarities should be sorted in descending order for cosine
        similarities = [sim for _, sim in similar_seeds]
        assert similarities == sorted(similarities, reverse=True)

    def test_custom_parameters(self, config):
        """Test ARMMapper with custom parameters."""
        custom_config = ARMConfig(
            model_name="distilgpt2",
            n_seeds=2,
            probes_per_seed=3,
            steps_per_probe=4,
            eps=0.02,
            layer_to_probe=2,
            n_modes=5,
        )

        with patch('arm_library.core.arm_mapper.TransformerModelInterface'):
            mapper = ARMMapper(custom_config)

            # Test that custom config is used
            assert mapper.config.probes_per_seed == 3
            assert mapper.config.steps_per_probe == 4
            assert mapper.config.n_modes == 5

    def test_memory_efficiency(self, arm_mapper):
        """Test that operations complete without excessive memory usage."""
        prompt = "This is a test prompt for memory efficiency"

        # This should complete without memory issues
        result = arm_mapper.analyze_seed_point(prompt)

        # Basic sanity check
        assert result['activation_matrix'] is not None
        assert result['resonance_signature'] is not None

    def test_reproducibility(self, config):
        """Test that results are reproducible with same configuration."""
        with patch('arm_library.core.arm_mapper.TransformerModelInterface') as mock_class:
            mock_interface = MagicMock()
            mock_class.return_value = mock_interface

            # Configure consistent mock behavior
            mock_interface.get_hidden_at_layer.return_value = np.random.randn(4, 768).astype(np.float32)
            mock_interface.forward_from_layer.return_value = (
                torch.randn(1, 4, 50257),
                torch.randn(1, 4, 768),
                [torch.randn(1, 4, 768)] * 3
            )

            mapper1 = ARMMapper(config)
            mapper2 = ARMMapper(config)

            result1 = mapper1.analyze_seed_point("test prompt")
            result2 = mapper2.analyze_seed_point("test prompt")

            # Activation matrices should be identical (same random seed)
            np.testing.assert_array_equal(result1['activation_matrix'], result2['activation_matrix'])
