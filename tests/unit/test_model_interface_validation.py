import pytest

from arm_library.interfaces.model_interface import TransformerModelInterface
from arm_library.utils.config import ModelConfig


class _BadModel:
    pass


@pytest.mark.unit
def test_gpt2_structure_validation_passes(monkeypatch):
    # Monkeypatch to load a minimal GPT-2 like structure without downloading
    class DummyTransformer:
        def __init__(self):
            self.h = [object(), object(), object()]
            self.wte = object()
            self.wpe = object()
            self.ln_f = object()

    class DummyModel:
        def __init__(self):
            class C: pass
            self.config = C()
            self.config.hidden_size = 8
            self.config.num_hidden_layers = 3
            self.config.num_attention_heads = 2
            self.transformer = DummyTransformer()
            self._device = None

        def to(self, device):
            self._device = device
            return self

        def eval(self):
            return self

    def fake_from_pretrained(name, **kwargs):
        return DummyModel()

    # Patch HF loader and tokenizer
    import arm_library.interfaces.model_interface as mi
    monkeypatch.setattr(mi, 'AutoModelForCausalLM', type('X', (), {'from_pretrained': staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(mi, 'AutoTokenizer', type('T', (), {'from_pretrained': staticmethod(lambda name: type('TT', (), {'pad_token': None, 'eos_token': '</s>', 'vocab_size': 10})())}))

    iface = TransformerModelInterface(ModelConfig(model_name='distilgpt2'))
    dims = iface.get_model_dimensions()
    assert dims['d_model'] == 8
    assert dims['n_layers'] == 3


@pytest.mark.unit
def test_device_move_called_cpu(monkeypatch):
    # Ensure .to() is invoked with expected device
    import torch
    class DummyTransformer:
        def __init__(self):
            self.h = [object(), object(), object()]
            self.wte = object()
            self.wpe = object()
            self.ln_f = object()

    class DummyModel:
        def __init__(self):
            class C: pass
            self.config = C()
            self.config.hidden_size = 8
            self.config.num_hidden_layers = 3
            self.config.num_attention_heads = 2
            self.transformer = DummyTransformer()
            self._device = None
        def to(self, device):
            self._device = device
            return self
        def eval(self):
            return self

    def fake_from_pretrained(name, **kwargs):
        return DummyModel()

    import arm_library.interfaces.model_interface as mi
    monkeypatch.setattr(mi, 'AutoModelForCausalLM', type('X', (), {'from_pretrained': staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(mi, 'AutoTokenizer', type('T', (), {'from_pretrained': staticmethod(lambda name: type('TT', (), {'pad_token': None, 'eos_token': '</s>', 'vocab_size': 10})())}))

    cfg = ModelConfig(model_name='distilgpt2', device=torch.device('cpu'))
    iface = TransformerModelInterface(cfg)
    assert getattr(iface.model, '_device', None) == torch.device('cpu')


@pytest.mark.unit
def test_unsupported_architecture_raises(monkeypatch):
    class BadModel:
        def __init__(self):
            class C: pass
            self.config = C()
            self.config.hidden_size = 8
            # Missing transformer
        def to(self, device):
            return self
        def eval(self):
            return self

    def fake_from_pretrained(name, **kwargs):
        return BadModel()

    import arm_library.interfaces.model_interface as mi
    monkeypatch.setattr(mi, 'AutoModelForCausalLM', type('X', (), {'from_pretrained': staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(mi, 'AutoTokenizer', type('T', (), {'from_pretrained': staticmethod(lambda name: type('TT', (), {'pad_token': None, 'eos_token': '</s>', 'vocab_size': 10})())}))

    with pytest.raises(ValueError):
        TransformerModelInterface(ModelConfig(model_name='distilgpt2'))


@pytest.mark.unit
def test_quantization_kwargs_passed(monkeypatch):
    # Verify that quantization_config and device_map are passed when load_in_8bit=True
    captured = {}

    class DummyModel:
        def __init__(self):
            class C: pass
            self.config = C()
            self.config.hidden_size = 8
            self.config.num_hidden_layers = 2
            self.config.num_attention_heads = 2
            # Provide a minimal GPT-2-like transformer to pass validation
            class T:
                def __init__(self):
                    self.h = [object(), object()]
                    self.wte = object()
                    self.wpe = object()
                    self.ln_f = object()
            self.transformer = T()
        def eval(self):
            return self

    def fake_bnb_config(**kwargs):
        captured['bnb'] = kwargs
        return object()

    def fake_from_pretrained(name, **kwargs):
        captured['from_pretrained_kwargs'] = kwargs
        return DummyModel()

    import transformers as tr
    import arm_library.interfaces.model_interface as mi
    monkeypatch.setattr(tr, 'BitsAndBytesConfig', staticmethod(fake_bnb_config))
    monkeypatch.setattr(mi, 'AutoModelForCausalLM', type('X', (), {'from_pretrained': staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(mi, 'AutoTokenizer', type('T', (), {'from_pretrained': staticmethod(lambda name: type('TT', (), {'pad_token': None, 'eos_token': '</s>', 'vocab_size': 10})())}))

    cfg = ModelConfig(model_name='distilgpt2', load_in_8bit=True)
    # Should not raise
    TransformerModelInterface(cfg)
    fp_kwargs = captured.get('from_pretrained_kwargs', {})
    assert 'quantization_config' in fp_kwargs
    assert fp_kwargs.get('device_map') == 'auto'


