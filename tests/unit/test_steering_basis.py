import numpy as np
import pytest

from arm_library.core.steering import ARMControlVectorComputer
from arm_library.interfaces.model_interface import TransformerModelInterface
from arm_library.utils.config import ModelConfig


class DummyModelInterface(TransformerModelInterface):
    def __init__(self):
        # Create a minimal fake with required attributes and dimensions
        class Dummy:
            pass
        self.config = ModelConfig(model_name="distilgpt2")
        # Bypass real model loading by stubbing required API
        self.tokenizer = type("T", (), {"pad_token_id": 0, "eos_token_id": 0, "vocab_size": 10})()
        self.model = type("M", (), {"config": type("C", (), {"hidden_size": 8, "num_hidden_layers": 3, "num_attention_heads": 2})()})()
        # Minimal methods used by tests
        def get_hidden_at_layer(prompt, layer):
            import torch
            seq_len = 4
            return torch.ones((seq_len, 8), dtype=torch.float32)
        self.get_hidden_at_layer = get_hidden_at_layer

    def get_model_dimensions(self):
        return {"vocab_size": 10, "d_model": 8, "n_layers": 3, "n_heads": 2}


@pytest.mark.unit
def test_dimension_validation():
    interface = DummyModelInterface()
    computer = ARMControlVectorComputer(interface)
    # Positive/negative prompts don't matter; dummy returns fixed hidden size
    cv = computer.compute_control_vector(["a"], ["b"], layer=0)
    assert cv.direction.shape[0] == interface.get_model_dimensions()["d_model"]


