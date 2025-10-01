import torch
import pytest

from arm_library.interfaces.model_interface import TransformerModelInterface
from arm_library.utils.config import ModelConfig


@pytest.mark.unit
def test_forward_and_hidden_paths(monkeypatch):
    # Build a minimal GPT-2-like model to exercise get_hidden_at_layer and forward_from_layer
    class Block(torch.nn.Module):
        def forward(self, h, attention_mask=None):
            return h + 1

    class Dummy:
        pass

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Dummy()
            self.transformer.h = torch.nn.ModuleList([Block(), Block(), Block()])
            self.transformer.ln_f = torch.nn.Identity()
            self.transformer.drop = torch.nn.Identity()
            class WTE(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.zeros(32, 8))
                def forward(self, x):
                    return torch.zeros((x.shape[0], x.shape[1], 8))
            class WPE(torch.nn.Module):
                def forward(self, x):
                    return torch.zeros((x.shape[0], x.shape[1], 8))
            self.transformer.wte = WTE()
            self.transformer.wpe = WPE()
        def eval(self):
            return self
        def to(self, device):
            return self

    def fake_from_pretrained(name, **kwargs):
        return DummyModel()

    def fake_tokenizer(name):
        class T:
            pad_token = None
            eos_token = '</s>'
            vocab_size = 32
            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1,2,3]], dtype=torch.long), "attention_mask": torch.tensor([[1,1,1]], dtype=torch.long)}
        return T()

    import arm_library.interfaces.model_interface as mi
    monkeypatch.setattr(mi, 'AutoModelForCausalLM', type('X', (), {'from_pretrained': staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(mi, 'AutoTokenizer', type('T', (), {'from_pretrained': staticmethod(fake_tokenizer)}))

    iface = TransformerModelInterface(ModelConfig(model_name='distilgpt2'))
    hid = iface.get_hidden_at_layer("Hello", layer_idx=1)
    assert hid.shape[1] == 8
    logits, final_h, inter = iface.forward_from_layer(hid.unsqueeze(0), start_layer=1)
    assert logits.shape[-1] == 32
    assert final_h.shape[-1] == 8

