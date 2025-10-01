import numpy as np
import torch
import pytest

from arm_library.core.steering import ARMControlledGenerator, ARMControlVectorComputer


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "attention_mask": torch.tensor([[1, 1]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "DUMMY_OUT"


class DummyBlock:
    def __init__(self):
        self._hooks = []
    def __call__(self, h, attention_mask=None):
        return h
    class _Handle:
        def __init__(self, hooks):
            self._hooks = hooks
        def remove(self):
            if self._hooks:
                self._hooks.pop()
    def register_forward_pre_hook(self, hook):
        self._hooks.append(hook)
        return DummyBlock._Handle(self._hooks)


class DummyTransformer:
    def __init__(self, hidden_size=8):
        self.h = [DummyBlock(), DummyBlock()]
        self.ln_f = torch.nn.Identity()
        class WTE(torch.nn.Module):
            def __init__(self, vs, hs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(vs, hs))
            def forward(self, x):
                return torch.zeros((x.shape[0], x.shape[1], self.weight.shape[1]))
        class WPE(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], x.shape[1], hidden_size))
        self.wte = WTE(vs=16, hs=hidden_size)
        self.wpe = WPE()
        self.drop = torch.nn.Identity()


class DummyModel:
    def __init__(self):
        class C: pass
        self.config = C()
        self.config.hidden_size = 8
        self.transformer = DummyTransformer(hidden_size=8)
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return torch.cat([input_ids, torch.tensor([[3, 4]])], dim=1)


class DummyInterface:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel()
    def encode_prompt(self, prompt):
        toks = self.tokenizer(prompt, return_tensors="pt")
        return toks["input_ids"], toks["attention_mask"]
    def get_model_dimensions(self):
        return {"vocab_size": 16, "d_model": 8, "n_layers": 2, "n_heads": 2}


@pytest.mark.unit
def test_generate_with_manifold_steering_blend_modes():
    iface = DummyInterface()
    gen = ARMControlledGenerator(iface)

    # Manifold with one seed having resonance_signature
    top_vecs = np.stack([np.eye(8)[0], np.eye(8)[1]], axis=0).astype(np.float32)
    manifold = {
        'seed_analyses': [{ 'resonance_signature': { 's_norm': np.array([0.7,0.3], dtype=np.float32), 'top_singular_vectors': top_vecs } }],
        'layer_to_probe': 0
    }
    target_sig = np.array([0.7,0.3], dtype=np.float32)

    out = gen.generate_with_manifold_steering(
        prompt="Hello",
        target_signature=target_sig,
        manifold_data=manifold,
        max_length=10,
        temperature=0.8,
        steering_strength=0.5,
        mode_indices=[0,1],
        mode_weights=[0.6,0.4]
    )
    assert isinstance(out, str) and len(out) >= 0


@pytest.mark.unit
def test_compute_manifold_control_vector_errors():
    iface = DummyInterface()
    computer = ARMControlVectorComputer(iface)
    top_vecs = np.ones((2, 8), dtype=np.float32)
    manifold = {'seed_analyses': [{ 'resonance_signature': { 's_norm': np.array([0.5,0.5], dtype=np.float32), 'top_singular_vectors': top_vecs } }]}
    # Out-of-range mode index
    with pytest.raises(ValueError):
        computer.compute_manifold_control_vector(np.array([0.5,0.5], dtype=np.float32), layer=0, manifold_data=manifold, mode_indices=[10])


