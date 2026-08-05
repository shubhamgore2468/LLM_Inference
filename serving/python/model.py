# py/model.py
#
# Qwen2.5-0.5B. Minimum model that accepts a block table.
#
# Deliberately absent: tensor parallelism, CUDA graphs, torch.compile, fused
# kernels. The point of this file is to let the C++ scheduler control KV
# placement, not to be fast.
#
# Qwen2 vs Qwen3 (nano-vllm ships Qwen3 — do not copy its layer shape):
#   - Qwen2 HAS bias on q/k/v_proj, none on o_proj
#   - Qwen2 has NO q_norm/k_norm before RoPE
#   - rope_theta is 1e6 here, not 10000

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import paged_attention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt) * self.weight)


class RoPE:
    """Precomputed cos/sin, indexed by absolute position.

    Positions come from the BatchPlan, so a chunked-prefill chunk or a decode
    step gets its true absolute position without the model tracking any state.
    """

    def __init__(self, head_dim, max_pos, theta, device, dtype):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_pos, device=device).float()
        freqs = torch.outer(t, inv)
        self.cos = freqs.cos().to(dtype)
        self.sin = freqs.sin().to(dtype)

    def apply(self, x, positions):
        # HF "half-split" convention: rotate first half against second half.
        # The paper's interleaved-pairs layout is NOT equivalent, and mixing
        # them produces almost-right output. Match HF, since that is what the
        # weights were trained with.
        cos = self.cos[positions].unsqueeze(1)   # (T, 1, D/2)
        sin = self.sin[positions].unsqueeze(1)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_q = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.d = cfg.hidden_size // cfg.num_attention_heads

        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_q * self.d, bias=True)
        self.k_proj = nn.Linear(h, self.n_kv * self.d, bias=True)
        self.v_proj = nn.Linear(h, self.n_kv * self.d, bias=True)
        self.o_proj = nn.Linear(self.n_q * self.d, h, bias=False)

    def forward(self, x, rope, plan, slab):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_q, self.d)
        k = self.k_proj(x).view(T, self.n_kv, self.d)
        v = self.v_proj(x).view(T, self.n_kv, self.d)

        pos = plan["positions"]
        q = rope.apply(q, pos)
        k = rope.apply(k, pos)

        o = paged_attention(q, k, v, slab, self.layer_idx,
                            plan, self.n_q, self.n_kv)
        return self.o_proj(o.reshape(T, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, rope, plan, slab):
        x = x + self.self_attn(self.input_layernorm(x), rope, plan, slab)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2(nn.Module):
    def __init__(self, cfg, device="cuda", dtype=torch.float16):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.to(device=device, dtype=dtype)

        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.rope = RoPE(head_dim, cfg.max_position_embeddings,
                         cfg.rope_theta, device, dtype)

    def forward(self, plan, slab):
        x = self.embed_tokens(plan["token_ids"])
        for layer in self.layers:
            x = layer(x, self.rope, plan, slab)
        return self.norm(x)

    def logits(self, h):
        # tie_word_embeddings: the output projection IS the embedding matrix.
        # At 151936 x 896 that is ~136M params, over a quarter of the model —
        # which is also why this GEMM is not cheap.
        return F.linear(h, self.embed_tokens.weight)

    def last_token_indices(self, plan):
        """Logits are only needed for the last token of each sequence."""
        cu = plan["cu_seqlens_q"]
        return torch.tensor([cu[i + 1] - 1 for i in range(len(cu) - 1)],
                            device=self.embed_tokens.weight.device)
