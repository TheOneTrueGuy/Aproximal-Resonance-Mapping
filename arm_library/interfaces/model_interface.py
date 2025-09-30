"""
Model interface for ARM operations on transformer models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from ..utils.config import ModelConfig


class TransformerModelInterface:
    """Interface for transformer model operations used by ARM."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Handle missing pad token for some models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model loading kwargs
        model_kwargs = {
            "output_hidden_states": config.output_hidden_states,
        }
        
        # Add quantization configuration if requested
        if config.load_in_8bit or config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=config.load_in_8bit,
                    load_in_4bit=config.load_in_4bit,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"  # Required for quantization
                
                print(f"🔧 Loading model with {'8-bit' if config.load_in_8bit else '4-bit'} quantization...")
                
            except ImportError:
                print("⚠️ Warning: bitsandbytes not installed. Falling back to full precision.")
                print("   Install with: pip install bitsandbytes")
                # Fall back to regular loading
                if config.torch_dtype:
                    model_kwargs["torch_dtype"] = config.torch_dtype
        else:
            # Regular loading without quantization
            if config.torch_dtype:
                model_kwargs["torch_dtype"] = config.torch_dtype

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        # Move to device only if not using quantization (device_map handles it)
        if not (config.load_in_8bit or config.load_in_4bit):
            self.model = self.model.to(config.device)

        self.model.eval()

    def encode_prompt(self, prompt: str) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Encode prompt to input_ids and attention_mask."""
        tokens = self.tokenizer(prompt, return_tensors="pt")
        return tokens["input_ids"].to(self.config.device), tokens["attention_mask"].to(self.config.device)

    def build_initial_hidden(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Build initial hidden states BEFORE block 0 (embedding + position).

        Args:
            input_ids: Token ids, shape (batch, seq_len)

        Returns:
            Hidden states, shape (batch, seq_len, d_model)
        """
        wte = self.model.transformer.wte(input_ids)  # token embeddings
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.config.device).unsqueeze(0)
        wpe = self.model.transformer.wpe(position_ids)
        hidden = wte + wpe  # shape batch x seq x d_model
        hidden = self.model.transformer.drop(hidden)
        return hidden

    def forward_from_layer(
        self,
        hidden: torch.Tensor,
        start_layer: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass starting from a specific layer.

        Args:
            hidden: Hidden state to feed to start_layer, shape (batch, seq, d_model)
            start_layer: Layer index to start from (0-based)
            attention_mask: Attention mask, shape (batch, seq_len)

        Returns:
            Tuple of (logits, final_hidden, intermediate_hidden_states)
        """
        # Convert attention mask to correct dtype for transformers
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float32)

        h = hidden
        intermediates = []

        # Run from start_layer to end
        for i, block in enumerate(self.model.transformer.h):
            if i < start_layer:
                continue
            # Handle different block output formats
            block_output = block(h, attention_mask=attention_mask)
            h = block_output[0] if isinstance(block_output, tuple) else block_output
            intermediates.append(h)

        # Final layer norm
        h = self.model.transformer.ln_f(h)

        # LM head (tied weights)
        logits = F.linear(h, self.model.transformer.wte.weight)

        return logits, h, intermediates

    def get_hidden_at_layer(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Get hidden state at a specific layer for a prompt.

        Args:
            prompt: Input prompt
            layer_idx: Layer index to extract hidden state from (before running that layer)

        Returns:
            Hidden state, shape (seq_len, d_model)
        """
        input_ids, attn_mask = self.encode_prompt(prompt)

        # Convert attention mask to correct dtype for transformers
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.float32)  # Convert to float32 as expected

        hidden = self.build_initial_hidden(input_ids)  # batch x seq x d

        # Run blocks up to layer_idx-1 to get hidden state to modify
        h = hidden
        for i, block in enumerate(self.model.transformer.h):
            if i >= layer_idx:
                break
            block_output = block(h, attention_mask=attn_mask)
            h = block_output[0] if isinstance(block_output, tuple) else block_output

        # Return without batch dimension for easier numpy conversion
        return h.squeeze(0).detach().cpu()

    def get_model_dimensions(self) -> Dict[str, int]:
        """Get model architecture dimensions."""
        return {
            'vocab_size': self.tokenizer.vocab_size,
            'd_model': self.model.config.hidden_size,
            'n_layers': self.model.config.num_hidden_layers,
            'n_heads': self.model.config.num_attention_heads,
        }
