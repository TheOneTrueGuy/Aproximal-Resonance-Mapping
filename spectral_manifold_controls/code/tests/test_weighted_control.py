"""
Tests for weighted control vector computation.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pytest

from core.weighted_control import WeightedControlVector, WeightedControlVectorComputer


class MockModelInterface:
    """Mock model interface for testing."""
    
    def __init__(self, hidden_size=768):
        self.hidden_size = hidden_size
    
    def get_hidden_at_layer(self, prompt: str, layer: int) -> torch.Tensor:
        """Return deterministic fake activations based on prompt hash."""
        # Use prompt hash for reproducibility
        seed = hash(prompt) % 10000
        torch.manual_seed(seed)
        
        # Return fake hidden states: [seq_len=5, hidden_size]
        return torch.randn(5, self.hidden_size)


def test_weighted_control_vector_creation():
    """Test basic weighted control vector creation."""
    direction = torch.randn(768)
    layer = 3
    weights = {0: 0.7, 1: 0.3}
    
    cv = WeightedControlVector(direction, layer, weights)
    
    assert cv.layer == 3
    assert cv.weights == {0: 0.7, 1: 0.3}
    assert cv.strength == 1.0
    assert cv.direction.shape == (768,)


def test_weighted_control_vector_apply():
    """Test applying control vector to hidden states."""
    direction = torch.ones(768)
    cv = WeightedControlVector(direction, layer=3)
    
    # Create fake hidden states
    hidden_states = torch.zeros(2, 10, 768)  # [batch, seq, hidden]
    
    # Apply control vector
    modified = cv.apply(hidden_states)
    
    # Check that control was added
    assert modified.shape == hidden_states.shape
    assert not torch.allclose(modified, hidden_states)
    
    # Check that the difference is the control vector
    diff = modified - hidden_states
    expected_diff = direction.unsqueeze(0).unsqueeze(0)
    assert torch.allclose(diff, expected_diff)


def test_compute_from_weights():
    """Test computing control vector from index weights."""
    mock_model = MockModelInterface(hidden_size=768)
    computer = WeightedControlVectorComputer(mock_model)
    
    corpus = [
        "This is formal text",
        "This is friendly text",
        "This is casual text",
    ]
    
    # Mix: 70% formal, 30% friendly, avoid casual
    weights = {0: 0.7, 1: 0.3, 2: -0.2}
    
    cv = computer.compute_from_weights(weights, corpus, layer=3)
    
    assert cv.layer == 3
    assert cv.weights == weights
    assert cv.direction.shape == (768,)
    # Should be normalized
    assert abs(torch.norm(cv.direction).item() - 1.0) < 0.01


def test_compute_from_labels():
    """Test computing control vector from semantic labels."""
    mock_model = MockModelInterface(hidden_size=768)
    computer = WeightedControlVectorComputer(mock_model)
    
    corpus = [
        "formal 1", "formal 2",          # indices 0-1
        "friendly 1", "friendly 2",      # indices 2-3
        "casual 1", "casual 2",          # indices 4-5
    ]
    
    index_labels = {
        'formal': [0, 1],
        'friendly': [2, 3],
        'casual': [4, 5],
    }
    
    label_weights = {
        'formal': 0.7,
        'friendly': 0.5,
        'casual': -0.2,
    }
    
    cv = computer.compute_from_labels(label_weights, index_labels, corpus, layer=3)
    
    assert cv.layer == 3
    assert cv.direction.shape == (768,)
    # Check that weights were distributed across indices
    assert len(cv.weights) == 6  # All indices


def test_interpolate_vectors():
    """Test interpolating between two control vectors."""
    mock_model = MockModelInterface(hidden_size=768)
    computer = WeightedControlVectorComputer(mock_model)
    
    corpus = ["text 1", "text 2", "text 3"]
    
    cv1 = computer.compute_from_weights({0: 1.0}, corpus, layer=3)
    cv2 = computer.compute_from_weights({2: 1.0}, corpus, layer=3)
    
    # Interpolate
    cv_mid = computer.interpolate_vectors(cv1, cv2, alpha=0.5)
    
    assert cv_mid.layer == 3
    assert cv_mid.direction.shape == (768,)
    
    # Should be halfway between
    expected = (cv1.direction + cv2.direction) / 2
    assert torch.allclose(cv_mid.direction, expected)


def test_empty_weights_error():
    """Test that empty weights raises error."""
    mock_model = MockModelInterface()
    computer = WeightedControlVectorComputer(mock_model)
    
    corpus = ["text 1"]
    
    with pytest.raises(ValueError, match="cannot be empty"):
        computer.compute_from_weights({}, corpus, layer=3)


def test_out_of_range_index_error():
    """Test that out of range index raises error."""
    mock_model = MockModelInterface()
    computer = WeightedControlVectorComputer(mock_model)
    
    corpus = ["text 1", "text 2"]
    
    with pytest.raises(ValueError, match="out of range"):
        computer.compute_from_weights({5: 1.0}, corpus, layer=3)


def test_unknown_label_error():
    """Test that unknown label raises error."""
    mock_model = MockModelInterface()
    computer = WeightedControlVectorComputer(mock_model)
    
    corpus = ["text 1"]
    index_labels = {'formal': [0]}
    
    with pytest.raises(ValueError, match="Unknown label"):
        computer.compute_from_labels(
            {'unknown': 1.0},
            index_labels,
            corpus,
            layer=3
        )


if __name__ == "__main__":
    # Run tests
    print("Running SMC weighted control tests...")
    
    test_weighted_control_vector_creation()
    print("[PASS] Control vector creation")
    
    test_weighted_control_vector_apply()
    print("[PASS] Control vector application")
    
    test_compute_from_weights()
    print("[PASS] Compute from weights")
    
    test_compute_from_labels()
    print("[PASS] Compute from labels")
    
    test_interpolate_vectors()
    print("[PASS] Vector interpolation")
    
    test_empty_weights_error()
    print("[PASS] Empty weights error")
    
    test_out_of_range_index_error()
    print("[PASS] Out of range error")
    
    test_unknown_label_error()
    print("[PASS] Unknown label error")
    
    print("\n[SUCCESS] All tests passed!")

