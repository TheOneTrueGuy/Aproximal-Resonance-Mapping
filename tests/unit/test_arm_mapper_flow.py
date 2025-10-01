import numpy as np
import pytest

from arm_library.core.arm_mapper import ARMMapper
from arm_library.utils.config import ARMConfig


@pytest.mark.unit
def test_map_latent_manifold_shapes(monkeypatch):
    cfg = ARMConfig(
        model_name="distilgpt2",
        n_seeds=2,
        probes_per_seed=1,
        steps_per_probe=2,
        eps=0.01,
        n_modes=2,
        layer_to_probe=1,
        random_seed=42,
    )

    # Monkeypatch TransformerModelInterface to avoid heavy loads
    from arm_library.core import arm_mapper as am

    class DummyInterface:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_hidden_at_layer(self, prompt, layer_idx):
            import torch
            return torch.ones((5, 8), dtype=torch.float32)

        def forward_from_layer(self, h_t, start_layer):
            import torch
            logits = torch.zeros((1, h_t.shape[1], 16), dtype=torch.float32)
            final_h = torch.ones((1, h_t.shape[1], 8), dtype=torch.float32)
            return logits, final_h, []

        def get_model_dimensions(self):
            return {"vocab_size": 16, "d_model": 8, "n_layers": 2, "n_heads": 2}

    monkeypatch.setattr(am, 'TransformerModelInterface', lambda *_a, **_k: DummyInterface())

    mapper = ARMMapper(cfg)
    prompts = ["a", "b", "c"]
    res = mapper.map_latent_manifold(prompts)

    assert res['descriptors'].shape[0] == len(prompts)
    assert 'graph_data' in res and 'clustering_data' in res
    assert res['n_seeds'] == len(prompts)


