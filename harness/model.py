"""Transformer model spec, read from a Hugging Face config.json."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool = True
    qk_norm: bool = False          # Qwen3: RMSNorm on q and k per head
    sub_norms: bool = False        # BitNet: extra norms before o_proj and down_proj
    activation: str = "silu"       # "silu" (SwiGLU) or "relu2" (BitNet)
    kv_layers: int = None          # layers that keep a KV cache (hybrid models); None = all

    @property
    def q_dim(self):
        return self.heads * self.head_dim

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    @property
    def n_kv_layers(self):
        return self.layers if self.kv_layers is None else self.kv_layers

    def kv_bytes_per_token(self, kv_bits=16):
        """K and V for one token across every layer that keeps a cache."""
        return 2 * self.n_kv_layers * self.kv_dim * kv_bits / 8

    def layer_weight_shapes(self):
        """(name, out_dim M, in_dim N) for each static-weight MVM in one layer."""
        D, F = self.hidden, self.intermediate
        return [
            ("q_proj", self.q_dim, D),
            ("k_proj", self.kv_dim, D),
            ("v_proj", self.kv_dim, D),
            ("o_proj", D, self.q_dim),
            ("gate_proj", F, D),
            ("up_proj", F, D),
            ("down_proj", D, F),
        ]

    def weight_params(self, include_lm_head=True):
        per_layer = sum(m * n for _, m, n in self.layer_weight_shapes())
        total = per_layer * self.layers
        if include_lm_head:
            total += self.vocab * self.hidden
        return total

    @classmethod
    def from_hf_config(cls, path, name=None):
        path = Path(path)
        cfg = json.loads((path / "config.json" if path.is_dir() else path).read_text())
        arch = (cfg.get("architectures") or [""])[0]
        layer_types = cfg.get("layer_types")
        kv_layers = None
        if layer_types:
            kv_layers = sum(t == "full_attention" for t in layer_types)
        return cls(
            name=name or cfg.get("_name_or_path") or path.name,
            hidden=cfg["hidden_size"],
            intermediate=cfg["intermediate_size"],
            layers=cfg["num_hidden_layers"],
            heads=cfg["num_attention_heads"],
            kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            vocab=cfg["vocab_size"],
            tied_embeddings=cfg.get("tie_word_embeddings", False),
            qk_norm=arch.startswith("Qwen3"),
            kv_layers=kv_layers,
        )


# BitNet b1.58 2B-4T, as hardcoded in TERINT-GEMV's soc_inference_estimator.py.
BITNET_2B = ModelSpec(
    name="bitnet-b1.58-2B-4T", hidden=2560, intermediate=6912, layers=30,
    heads=20, kv_heads=5, head_dim=128, vocab=128256,
    tied_embeddings=True, sub_norms=True, activation="relu2",
)
