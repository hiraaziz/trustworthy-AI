"""Expose independent Q/K/V LoRA targets and restore Phi's native layout."""

import torch
from torch import nn

class SplitQKV(nn.Module):
    def __init__(self, fused, widths):
        super().__init__()
        if not isinstance(fused, nn.Linear) or sum(widths) != fused.out_features:
            raise ValueError("Q/K/V widths do not match the fused projection.")

        # Each new linear owns a distinct Parameter backed by a disjoint
        # view of the frozen base weight. No full base-weight copy is needed.
        offset = 0
        for name, width in zip(("q_proj", "k_proj", "v_proj"), widths, strict=True):
            linear = nn.Linear(
                fused.in_features, width, bias=fused.bias is not None,
                device="meta", dtype=fused.weight.dtype,
            )
            linear.weight = nn.Parameter(
                fused.weight.detach().narrow(0, offset, width), requires_grad=False
            )
            if fused.bias is not None:
                linear.bias = nn.Parameter(
                    fused.bias.detach().narrow(0, offset, width), requires_grad=False
                )
            setattr(self, name, linear)
            offset += width

    def forward(self, x):
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)

def split_qkv(model):
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    q = cfg.num_attention_heads * head_dim
    kv = cfg.num_key_value_heads * head_dim
    widths = (q, kv, kv)
    layers = model.model.layers
    for layer in layers:
        fused = layer.self_attn.qkv_proj
        if not isinstance(fused, nn.Linear) or fused.out_features != sum(widths):
            raise ValueError("Unexpected attention projection in Phi.")
    for layer in layers:
        attn = layer.self_attn
        attn.qkv_proj = SplitQKV(attn.qkv_proj, widths)
    return widths

def restore_qkv(model):
    # Call after PEFT merge_and_unload(), when each child is an ordinary Linear.
    for layer in model.model.layers:
        attn = layer.self_attn
        split = attn.qkv_proj
        if not isinstance(split, SplitQKV):
            raise TypeError("Expected a split QKV projection.")
        parts = (split.q_proj, split.k_proj, split.v_proj)
        if not all(type(p) is nn.Linear for p in parts):
            raise TypeError("Merge and unload the LoRA adapters before restoring QKV.")
        weight = torch.cat([p.weight.detach() for p in parts], dim=0)
        has_bias = parts[0].bias is not None
        fused = nn.Linear(
            weight.shape[1], weight.shape[0], bias=has_bias,
            device="meta", dtype=weight.dtype,
        )
        fused.weight = nn.Parameter(weight, requires_grad=False)
        if has_bias:
            fused.bias = nn.Parameter(
                torch.cat([p.bias.detach() for p in parts]), requires_grad=False
            )
        attn.qkv_proj = fused
