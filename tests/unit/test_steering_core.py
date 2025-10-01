import torch
import pytest

from arm_library.core.steering import ARMControlledGenerator, ARMControlVector


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        # Return a simple two-token input
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "attention_mask": torch.tensor([[1, 1]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "DUMMY_OUTPUT"


class DummyBlock:
    def __init__(self):
        self.hooks = []

    class _Handle:
        def __init__(self, hooks):
            self.hooks = hooks
        def remove(self):
            if self.hooks:
                self.hooks.pop()

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return DummyBlock._Handle(self.hooks)


class DummyTransformer:
    def __init__(self, n_layers=2):
        self.h = [DummyBlock() for _ in range(n_layers)]


class DummyModel:
    def __init__(self):
        class C: pass
        self.config = C()
        self.config.hidden_size = 8
        self.transformer = DummyTransformer(2)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Return the input + two extra tokens
        extra = torch.tensor([[3, 4]], dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


class DummyInterface:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel()

    def encode_prompt(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt")
        return tokens["input_ids"], tokens["attention_mask"]

    def get_model_dimensions(self):
        return {"vocab_size": 10, "d_model": 8, "n_layers": 2, "n_heads": 2}


@pytest.mark.unit
def test_hook_registration_and_cleanup():
    iface = DummyInterface()
    gen = ARMControlledGenerator(iface)

    # Valid control vector
    direction = torch.ones(8)
    cv = ARMControlVector(direction=direction, layer=0, coefficient=0.5)
    gen.set_control(cv)

    assert len(gen._hook_handles) == 0
    _ = gen.generate_with_steering("Hello", max_length=5, temperature=0.8)
    # Hooks must be removed after generation
    assert len(gen._hook_handles) == 0


@pytest.mark.unit
def test_control_vector_dimension_guard():
    iface = DummyInterface()
    gen = ARMControlledGenerator(iface)
    bad_vec = torch.ones(6)  # wrong size
    cv = ARMControlVector(direction=bad_vec, layer=0, coefficient=1.0)
    with pytest.raises(ValueError):
        gen.set_control(cv)


