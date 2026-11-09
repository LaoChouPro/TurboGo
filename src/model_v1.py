#!/usr/bin/env python3
"""
TurboGo Hybrid Conv-Transformer (b12c384, ~30.87M parameters)
Designed for:
1. High-throughput Supervised Pretraining on KataGo 19x19 self-play data (BFloat16 + FlashAttention-2 + torch.compile).
2. Seamless transition to MCTS + Self-Play Reinforcement Learning (Value/Score/Ownership/Q-value heads + 2D-RoPE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_rope_cache(h: int = 19, w: int = 19, head_dim: int = 48, device: str = "cpu"):
    """Precompute 2D Rotary Position Embedding (cos, sin) tables for 19x19 Go board."""
    d_quarter = head_dim // 4
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_quarter, device=device).float() / d_quarter))
    ys = torch.arange(h, device=device).float()
    xs = torch.arange(w, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    freqs_y = torch.outer(grid_y.reshape(-1), inv_freq)  # (361, head_dim // 4)
    freqs_x = torch.outer(grid_x.reshape(-1), inv_freq)  # (361, head_dim // 4)
    freqs = torch.cat([freqs_y, freqs_x], dim=-1)        # (361, head_dim // 2)
    cos = torch.cos(freqs)[None, None, :, :]             # (1, 1, 361, head_dim // 2)
    sin = torch.sin(freqs)[None, None, :, :]             # (1, 1, 361, head_dim // 2)
    return cos, sin


def apply_2d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D Rotary Position Embedding to Q or K tensor of shape (B, H, 361, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class KataGPoolConvBlock(nn.Module):
    """
    Local 3x3 Residual Convolution Block with KataGo-style Global Pooling (mean + max) bias.
    Captures exact 3x3 liberties, cuts, hane, and local life-and-death while conditioning on global board state.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.gpool_fc = nn.Linear(d_model * 2, d_model, bias=False)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(d_model)

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(self.conv1(x_2d)))
        g = torch.cat([h.mean(dim=(2, 3)), h.flatten(2).max(dim=-1).values], dim=1)
        h = F.silu(h + self.gpool_fc(g)[:, :, None, None])
        h = self.bn2(self.conv2(h))
        return F.silu(x_2d + h)


class FlashAttn2DRoPEBlock(nn.Module):
    """
    Full-Board 361x361 Multi-Head Self-Attention with 2D-RoPE + SwiGLU FFN.
    Uses PyTorch SDPA FlashAttention-2 hardware kernel (attn_mask=None) for 1-hop full-board ladders/ko/global balance.
    """

    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: float = 2.67):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = (int(d_model * mlp_ratio) // 32) * 32
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x_2d: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, D, H_b, W_b = x_2d.shape
        x = x_2d.flatten(2).transpose(1, 2)  # (B, 361, D)

        qkv = self.qkv(self.norm1(x)).reshape(B, 361, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2].contiguous()
        q = apply_2d_rope(q, cos, sin)
        k = apply_2d_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, 361, D))

        u, v_ffn = self.w12(self.norm2(x)).chunk(2, dim=-1)
        x = x + self.w3(F.silu(u) * v_ffn)
        return x.transpose(1, 2).reshape(B, D, H_b, W_b)


class TurboGoHybridTransformer(nn.Module):
    """
    b12c384 Hybrid Conv-Transformer (30.87M parameters):
    - 4 Stages x (2 KataGPoolConvBlocks + 1 FlashAttn2DRoPEBlock) = 12 Blocks, d_model=384, n_heads=8.
    - Heads:
      1. policy_logits: (B, 2, 362) [ch 0: main MCTS policy, ch 1: opponent reply policy]
      2. wld_logits:    (B, 3)      [Win, Loss, NoResult]
      3. score_mean:    (B,)        [Expected score lead / 20.0]
      4. ownership:     (B, 19, 19) [Final territory ownership in [-1, 1]]
    """

    def __init__(
        self,
        d_model: int = 384,
        n_stages: int = 4,
        convs_per_stage: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(22, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )
        self.glob_fc = nn.Linear(19, d_model, bias=False)

        self.stages = nn.ModuleList()
        for _ in range(n_stages):
            convs = nn.ModuleList([KataGPoolConvBlock(d_model) for _ in range(convs_per_stage)])
            attn = FlashAttn2DRoPEBlock(d_model, n_heads=n_heads)
            self.stages.append(nn.ModuleDict({"convs": convs, "attn": attn}))

        cos, sin = build_2d_rope_cache(19, 19, d_model // n_heads, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Output Heads
        self.pol_conv = nn.Conv2d(d_model, 2, kernel_size=1)
        self.pass_fc = nn.Linear(d_model, 2)
        self.val_fc = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),  # 3 WLD logits + 1 expected score mean
        )
        self.own_conv = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, spatial_x: torch.Tensor, global_x: torch.Tensor):
        x = self.stem(spatial_x) + self.glob_fc(global_x)[:, :, None, None]
        cos = self.rope_cos.to(dtype=x.dtype)
        sin = self.rope_sin.to(dtype=x.dtype)

        for st in self.stages:
            for c in st["convs"]:
                x = c(x)
            x = st["attn"](x, cos, sin)

        pol_board = self.pol_conv(x).flatten(2)  # (B, 2, 361)
        g_mean = x.mean(dim=(2, 3))
        g_max = x.flatten(2).max(dim=-1).values
        g_pool = torch.cat([g_mean, g_max], dim=1)  # (B, 2 * d_model)

        pol_pass = self.pass_fc(g_mean).unsqueeze(-1)  # (B, 2, 1)
        pol_logits = torch.cat([pol_board, pol_pass], dim=-1)  # (B, 2, 362)

        val_out = self.val_fc(g_pool)
        wld_logits = val_out[:, :3]
        score_mean = val_out[:, 3]
        ownership = torch.tanh(self.own_conv(x).squeeze(1))  # (B, 19, 19)

        return pol_logits, wld_logits, score_mean, ownership
