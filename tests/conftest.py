"""
Shared pytest fixtures and configuration for ARM library tests.
"""

import pytest
import numpy as np
import torch

from arm_library.utils.config import ARMConfig, ModelConfig
from arm_library.core.probe_generator import ProbeGenerator
from arm_library.core.resonance_analyzer import ResonanceAnalyzer


@pytest.fixture(scope="session")
def torch_device():
    """Get appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config(torch_device):
    """Default ARM configuration for testing."""
    return ARMConfig(
        model_name="distilgpt2",
        device=torch_device,
        n_seeds=10,  # Smaller for testing
        probes_per_seed=4,  # Smaller for testing
        steps_per_probe=3,  # Smaller for testing
        eps=0.05,
        layer_to_probe=2,  # Use earlier layer for faster testing
        n_modes=4,
        random_seed=42,
    )


@pytest.fixture
def sample_hidden_state():
    """Sample hidden state tensor for testing."""
    np.random.seed(42)
    seq_len, d_model = 8, 768
    return np.random.randn(seq_len, d_model).astype(np.float32)


@pytest.fixture
def sample_activation_matrix():
    """Sample activation matrix for testing."""
    np.random.seed(42)
    n_samples, n_features = 12, 256  # 4 probes * 3 steps = 12 samples
    return np.random.randn(n_samples, n_features).astype(np.float32)


@pytest.fixture
def mock_model_interface():
    """Mock model interface for testing without actual model loading."""
    class MockModelInterface:
        def __init__(self, config):
            self.config = config

        def encode_prompt(self, prompt):
            # Mock tokenization
            input_ids = torch.randint(0, 1000, (1, len(prompt.split())))
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask

        def get_hidden_at_layer(self, prompt, layer_idx):
            # Return mock hidden state
            seq_len = len(prompt.split())
            d_model = 768
            return torch.randn(seq_len, d_model)

        def get_model_dimensions(self):
            return {
                'vocab_size': 50257,
                'd_model': 768,
                'n_layers': 6,
                'n_heads': 12,
            }

    return MockModelInterface


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


# Test data fixtures
@pytest.fixture
def simple_prompt():
    """Simple test prompt."""
    return "Hello world"


@pytest.fixture
def test_prompts():
    """List of test prompts."""
    return [
        "The cat sat on the mat",
        "Machine learning is fascinating",
        "The weather is nice today",
        "I love programming",
        "Natural language processing",
    ]
