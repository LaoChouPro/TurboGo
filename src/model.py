#!/usr/bin/env python3
"""
TurboGo-v2: 22-Layer Nested Bottleneck Transformer (`b11c384-nbt22`, ~20.15M parameters)

Designed from the ground up on 60,000,000 KataGo 19x19 positions, fixing all v1 bottlenecks:
1. High-Efficiency Parameter Allocation (~20.15M params, zero bloat):
   - Replaces v1's heavy 384x384 3x3 convolutions (which consumed ~20M params for only 4 Attention layers)
     with 11 Nested Bottleneck Transformer (NBT) macro-blocks (`c_main=384 -> c_mid=256 -> c_main=384`),
     each containing 2 full Global Self-Attention + SwiGLU FFN layers = **22 Global Self-Attention Layers
     + 22 SwiGLU FFN Layers (44 Transformer sub-layers)** plus Depthwise 3x3 + KataGo Global Pool mixing.
2. 100% Pure Pre-Norm Identity Residual Highway (`x = x + sub(x)`):
   - Eliminates v1's `F.silu(x + h)` half-wave rectification on the residual stream completely.
   - Uses `SpatialRMSNorm` & `nn.RMSNorm` throughout the trunk to eliminate BatchNorm train/eval running-stat drift.
3. Learnable Multi-Scale 2D Rotary Position Embeddings (`rope_freqs`) in all 22 Attention Layers:
   - Each of the 176 attention heads (22 layers x 8 heads) has its own learnable `(x, y)` 2D frequency bank
     initialized across 8 compass/diagonal orientations and wavelengths from 2 to 38 intersections.
4. 32-Channel Explicit Tactical Input Stem + Built-in Illegal Move Masking + Global-Context Policy & Value Heads.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def augment_tactical_planes_32(spatial_x: torch.Tensor) -> torch.Tensor:
    """
    Expand 22 raw KataGo binary planes (B, 22, 19, 19) to 32 channels by appending
    10 explicit multiplicative tactical planes (own/opp x 1/2/3-liberty & ladder + immediate capture/escape points).
    If spatial_x already has 32 channels, returns it unchanged.
    """
    if spatial_x.shape[1] == 32:
        return spatial_x
    onboard = spatial_x[:, 0:1]
    own = spatial_x[:, 1:2]
    opp = spatial_x[:, 2:3]
    lib1 = spatial_x[:, 3:4]
    lib2 = spatial_x[:, 4:5]
    lib3 = spatial_x[:, 5:6]
    ladder = spatial_x[:, 14:15]

    own_lib1 = own * lib1
    opp_lib1 = opp * lib1
    own_lib2 = own * lib2
    opp_lib2 = opp * lib2
    own_lib3 = own * lib3
    opp_lib3 = opp * lib3
    own_lad = own * ladder
    opp_lad = opp * ladder

    # 4-connected neighborhood dilation of 1-liberty stones to highlight exact capture / save intersections
    atari_pair = torch.cat([opp_lib1, own_lib1], dim=1)  # (B, 2, 19, 19)
    p = F.pad(atari_pair, (1, 1, 1, 1))
    nbr = (p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:] + p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1]).clamp(max=1.0)
    empty = (onboard * (1.0 - own) * (1.0 - opp)).clamp(min=0.0)
    imm_capture = empty * nbr[:, 0:1]
    imm_escape = empty * nbr[:, 1:2]

    return torch.cat(
        [
            spatial_x,
            own_lib1,
            opp_lib1,
            own_lib2,
            opp_lib2,
            own_lib3,
            opp_lib3,
            own_lad,
            opp_lad,
            imm_capture,
            imm_escape,
        ],
        dim=1,
    )


def kata_gpool_19x19(x: torch.Tensor) -> torch.Tensor:
    """KataGo Global Pooling for 19x19 board: [mean, mean * 0.5, max] -> (B, 3 * C)."""
    B, C, H, W = x.shape
    xf = x.float()
    mean = xf.mean(dim=(2, 3))
    mx = xf.reshape(B, C, H * W).max(dim=2).values
    return torch.cat([mean, mean * 0.5, mx], dim=1).to(dtype=x.dtype)


def kata_value_gpool_19x19(x: torch.Tensor) -> torch.Tensor:
    """KataGo Value Global Pooling for 19x19 board: [mean, mean * 0.5, mean * 0.15] -> (B, 3 * C)."""
    mean = x.float().mean(dim=(2, 3)).to(dtype=x.dtype)
    return torch.cat([mean, mean * 0.5, mean * 0.15], dim=1)


class FastRMSNorm(nn.Module):
    """Fused-kernel compatible RMSNorm that casts weight to input dtype under BF16 autocast."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight.to(dtype=x.dtype), self.eps)


class SpatialRMSNorm(nn.Module):
    """KataGo-style 2D Spatial RMSNorm over (C, H, W) with per-channel affine scale & bias."""

    def __init__(self, c_in: int, eps: float = 1e-6):
        super().__init__()
        self.c_in = c_in
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean_sq = torch.mean(xf * xf, dim=(1, 2, 3), keepdim=True)
        rms = torch.rsqrt(mean_sq + self.eps)
        return ((xf * rms) * self.gamma + self.beta).to(dtype=x.dtype)


class Learnable2DRoPEAttention(nn.Module):
    """
    Global 19x19 Self-Attention with Learnable Multi-Scale 2D Rotary Position Embeddings (`rope_freqs`).
    Operates on flattened 361 tokens `(B, 361, c_mid)` using FlashAttention-2 (`F.scaled_dot_product_attention`).
    """

    def __init__(self, c_mid: int = 256, num_heads: int = 8, head_dim: int = 32, pos_len: int = 19):
        super().__init__()
        assert c_mid == num_heads * head_dim
        self.c_mid = c_mid
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2

        self.norm = FastRMSNorm(c_mid, eps=1e-6)
        self.qkv_proj = nn.Linear(c_mid, 3 * c_mid, bias=False)
        self.out_proj = nn.Linear(c_mid, c_mid, bias=False)

        # Initialize learnable 2D RoPE frequencies (num_heads, head_dim // 2, 2) across 8 compass/diagonal angles
        # and log-spaced wavelengths from 2.0 intersections (adjacent/keima) to 38.0 intersections (whole board).
        freqs_init = torch.zeros(num_heads, self.half_dim, 2, dtype=torch.float32)
        wavelengths = torch.exp(
            torch.linspace(math.log(2.0), math.log(38.0), self.half_dim, dtype=torch.float32)
        )
        omegas = (2.0 * math.pi) / wavelengths  # (half_dim,)
        for h_idx in range(num_heads):
            theta = (h_idx * math.pi) / num_heads
            # Alternate primary direction and orthogonal direction across frequency pairs
            for p_idx in range(self.half_dim):
                ang = theta + (0.5 * math.pi if (p_idx % 2 == 1) else 0.0)
                freqs_init[h_idx, p_idx, 0] = omegas[p_idx] * math.cos(ang)
                freqs_init[h_idx, p_idx, 1] = omegas[p_idx] * math.sin(ang)

        self.rope_freqs = nn.Parameter(freqs_init)

        # Centered board coordinates [-9.0 .. +9.0] for 19x19 intersections
        s_idx = torch.arange(pos_len * pos_len, dtype=torch.float32)
        center = (pos_len - 1) * 0.5
        self.register_buffer("s_y", (s_idx // pos_len) - center, persistent=False)
        self.register_buffer("s_x", (s_idx % pos_len) - center, persistent=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, 361, c_mid) -> returns residual delta (B, 361, c_mid)."""
        B, S, C = x_seq.shape
        xn = self.norm(x_seq)
        qkv = self.qkv_proj(xn).view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, 361, D)

        # Compute 2D rotary angles directly in (H, 361, P) layout
        angles = (
            self.s_x.view(1, S, 1) * self.rope_freqs[:, :, 0].unsqueeze(1)
            + self.s_y.view(1, S, 1) * self.rope_freqs[:, :, 1].unsqueeze(1)
        )
        cos = torch.cos(angles).to(dtype=q.dtype).unsqueeze(0)  # (1, H, 361, P)
        sin = torch.sin(angles).to(dtype=q.dtype).unsqueeze(0)  # (1, H, 361, P)

        q0, q1 = q.reshape(B, self.num_heads, S, self.half_dim, 2).unbind(dim=-1)
        q = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1).reshape(
            B, self.num_heads, S, self.head_dim
        )

        k0, k1 = k.reshape(B, self.num_heads, S, self.half_dim, 2).unbind(dim=-1)
        k = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1).reshape(
            B, self.num_heads, S, self.head_dim
        )

        attn_out = F.scaled_dot_product_attention(q, k, v.contiguous())
        attn_out = attn_out.transpose(1, 2).reshape(B, S, C)
        return self.out_proj(attn_out)


class SwiGLUFFNBlock(nn.Module):
    """Pre-RMSNorm SwiGLU Feed-Forward Network operating on `(B, 361, c_mid)`."""

    def __init__(self, c_mid: int = 256, ffn_dim: int = 640):
        super().__init__()
        self.norm = FastRMSNorm(c_mid, eps=1e-6)
        self.w12 = nn.Linear(c_mid, 2 * ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, c_mid, bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x_seq)
        x1, x_gate = self.w12(xn).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x_gate)


class TurboNBTBlock(nn.Module):
    """
    Nested Bottleneck Transformer (NBT) Macro-Block with Local Depthwise + KataGo Global Pool Mixing:
      1. Bottleneck Down-Projection (`c_main=384 -> c_mid=256`)
      2. Residual Depthwise 3x3 Conv + KataGo Global Pool (`mean, mean*0.5, max`) for local shape & global ko context
      3. `internal_length=2` pairs of (Learnable2DRoPEAttention + SwiGLUFFNBlock) on `(B, 361, c_mid)`
      4. Bottleneck Up-Projection (`c_mid=256 -> c_main=384`) added to pure identity residual stream `x = x + out`
    """

    def __init__(
        self,
        c_main: int = 384,
        c_mid: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 640,
        internal_length: int = 2,
        gpool_c: int = 64,
    ):
        super().__init__()
        self.c_main = c_main
        self.c_mid = c_mid
        self.gpool_c = gpool_c

        self.norm_down = SpatialRMSNorm(c_main)
        self.conv_down = nn.Conv2d(c_main, c_mid, kernel_size=1, bias=False)

        # Lightweight local 3x3 depthwise conv + KataGo global pool inside bottleneck
        self.norm_local = SpatialRMSNorm(c_mid)
        self.dw_conv = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, groups=c_mid, bias=False)
        self.gpool_proj = nn.Linear(3 * gpool_c, c_mid, bias=False)
        self.pw_local = nn.Conv2d(c_mid, c_mid, kernel_size=1, bias=False)

        self.attn_layers = nn.ModuleList(
            [Learnable2DRoPEAttention(c_mid, num_heads, c_mid // num_heads) for _ in range(internal_length)]
        )
        self.ffn_layers = nn.ModuleList([SwiGLUFFNBlock(c_mid, ffn_dim) for _ in range(internal_length)])

        self.norm_up = FastRMSNorm(c_mid, eps=1e-6)
        self.conv_up = nn.Linear(c_mid, c_main, bias=False)

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Sequence-resident forward pass on `(B, 361, c_main)` that eliminates 4x NCHW<->NLC transposes
        and cuDNN 1x1 conv layout conversions per block while preserving 100% weight & math equivalence.
        """
        B = x_seq.shape[0]
        xf = x_seq.float()
        rms_d = torch.rsqrt(torch.mean(xf * xf, dim=(1, 2), keepdim=True) + self.norm_down.eps)
        xd = F.silu(
            ((xf * rms_d) * self.norm_down.gamma.view(1, 1, -1) + self.norm_down.beta.view(1, 1, -1)).to(
                dtype=x_seq.dtype
            )
        )
        h_seq = F.linear(xd, self.conv_down.weight.view(self.c_mid, self.c_main))

        hf = h_seq.float()
        rms_l = torch.rsqrt(torch.mean(hf * hf, dim=(1, 2), keepdim=True) + self.norm_local.eps)
        hl_seq = F.silu(
            ((hf * rms_l) * self.norm_local.gamma.view(1, 1, -1) + self.norm_local.beta.view(1, 1, -1)).to(
                dtype=h_seq.dtype
            )
        )

        hl_g = hl_seq[:, :, : self.gpool_c].float()
        gp_mean = hl_g.mean(dim=1)
        gp_max = hl_g.max(dim=1).values
        gp = torch.cat([gp_mean, gp_mean * 0.5, gp_max], dim=1).to(dtype=h_seq.dtype)

        hl_2d = hl_seq.view(B, 19, 19, self.c_mid).permute(0, 3, 1, 2)
        dw_out = self.dw_conv(hl_2d).flatten(2).transpose(1, 2)
        local_in = dw_out + self.gpool_proj(gp).unsqueeze(1)
        h_seq = h_seq + F.linear(local_in, self.pw_local.weight.view(self.c_mid, self.c_mid))

        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            h_seq = h_seq + attn(h_seq)
            h_seq = h_seq + ffn(h_seq)

        return x_seq + self.conv_up(F.silu(self.norm_up(h_seq)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, c_main, 19, 19) -> returns (B, c_main, 19, 19) with pure identity residual."""
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2).contiguous()
        x_seq = self.forward_seq(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


class KataGPoolPolicyHeadV2(nn.Module):
    """
    Deep KataGo-Style Policy Head with Global Pooling + 3x3 Tactical Refinement:
    Produces `(B, 2, 362)` logits for `[current_player_policy, opponent_next_move_policy]`.
    """

    def __init__(self, c_in: int = 384, c_p1: int = 96, c_g1: int = 64, num_out: int = 2):
        super().__init__()
        self.num_out = num_out
        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv3p = nn.Conv2d(c_p1, c_p1, kernel_size=3, padding=1, bias=False)
        self.norm1p = SpatialRMSNorm(c_p1)

        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.norm1g = SpatialRMSNorm(c_g1)
        self.linear_g = nn.Linear(3 * c_g1 + 19, c_p1, bias=False)

        self.norm2 = SpatialRMSNorm(c_p1)
        self.conv3p_refine = nn.Conv2d(c_p1, c_p1, kernel_size=3, padding=1, bias=False)
        self.norm3 = SpatialRMSNorm(c_p1)
        self.conv2p = nn.Conv2d(c_p1, num_out, kernel_size=1, bias=True)

        self.linear_pass = nn.Linear(3 * c_g1 + 19, c_p1, bias=True)
        self.linear_pass2 = nn.Linear(c_p1, num_out, bias=True)

    def forward(self, x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        outp = self.conv1p(x)
        outp = outp + self.conv3p(F.silu(self.norm1p(outp)))

        outg = F.silu(self.norm1g(self.conv1g(x)))
        gpool = torch.cat([kata_gpool_19x19(outg), global_x.to(dtype=x.dtype)], dim=1)

        outp = outp + self.linear_g(gpool).unsqueeze(-1).unsqueeze(-1)
        outp_act = F.silu(self.norm2(outp))
        outp = outp + self.conv3p_refine(outp_act)
        board_logits = self.conv2p(F.silu(self.norm3(outp))).reshape(x.shape[0], self.num_out, 361)

        pass_logits = self.linear_pass2(F.silu(self.linear_pass(gpool))).unsqueeze(-1)
        return torch.cat([board_logits, pass_logits], dim=2)


class KataGPoolValueHeadV2(nn.Module):
    """
    KataGo-Style Value, Score Lead & Ownership Head:
      - Value branch uses `kata_value_gpool_19x19` (`[mean, mean*0.5, mean*0.15]`) + `global_x` (komi/ko/rules).
      - Ownership branch uses dedicated 3x3 + 1x1 convolutions for crisp life-and-death territory prediction.
    """

    def __init__(self, c_in: int = 384, c_v1: int = 96, c_v2: int = 160, c_own: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, bias=False)
        self.norm1 = SpatialRMSNorm(c_v1)

        self.linear2 = nn.Linear(3 * c_v1 + 19, c_v2, bias=True)
        self.linear_wld = nn.Linear(c_v2, 3, bias=True)
        self.linear_score = nn.Linear(c_v2, 1, bias=True)

        self.own_conv1 = nn.Conv2d(c_in + c_v1, c_own, kernel_size=3, padding=1, bias=False)
        self.own_norm = SpatialRMSNorm(c_own)
        self.own_conv2 = nn.Conv2d(c_own, 1, kernel_size=1, bias=True)

    def forward(
        self, x: torch.Tensor, global_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v1 = F.silu(self.norm1(self.conv1(x)))
        gpool = torch.cat([kata_value_gpool_19x19(v1), global_x.to(dtype=x.dtype)], dim=1)
        v2 = F.silu(self.linear2(gpool))
        wld_logits = self.linear_wld(v2)
        score_pred = self.linear_score(v2).squeeze(-1)

        own_feat = F.silu(self.own_norm(self.own_conv1(torch.cat([x, v1], dim=1))))
        own_pred = torch.tanh(self.own_conv2(own_feat).squeeze(1))
        return wld_logits, score_pred, own_pred


class TurboGoHybridTransformer(nn.Module):
    """
    TurboGo-v2 (`b11c384-nbt22`, ~20.15M parameters):
      - 32-channel explicit tactical stem + 19 global features -> `c_main = 384`
      - 11 Nested Bottleneck Transformer (`TurboNBTBlock`) macro-blocks (`c_mid = 256`, `internal_length = 2`)
        = **22 Global Self-Attention Layers + 22 SwiGLU FFN Layers** + 11 Depthwise 3x3 / GPool Local-Global Mixers
      - 100% pure identity residual stream (`x = x + sub(x)`) with DeepNorm residual scaling
      - Learnable 2D-RoPE frequencies across all 176 attention heads
      - `KataGPoolPolicyHeadV2` + `KataGPoolValueHeadV2` + built-in illegal move masking
    """

    def __init__(
        self,
        d_model: int = 384,
        c_mid: int = 256,
        n_blocks: int = 11,
        internal_length: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 640,
        in_channels: int = 32,
        global_channels: int = 19,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.c_mid = c_mid
        self.n_blocks = n_blocks
        self.internal_length = internal_length
        self.n_heads = n_heads
        self.in_channels = in_channels

        self.conv_spatial = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False)
        self.linear_global = nn.Linear(global_channels, d_model, bias=False)

        self.blocks = nn.ModuleList(
            [
                TurboNBTBlock(
                    c_main=d_model,
                    c_mid=c_mid,
                    num_heads=n_heads,
                    ffn_dim=ffn_dim,
                    internal_length=internal_length,
                )
                for _ in range(n_blocks)
            ]
        )

        self.norm_trunkfinal = SpatialRMSNorm(d_model)
        self.policy_head = KataGPoolPolicyHeadV2(c_in=d_model, c_p1=96, c_g1=64, num_out=2)
        self.value_head = KataGPoolValueHeadV2(c_in=d_model, c_v1=96, c_v2=160, c_own=64)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with GPT-2 / DeepNorm residual scaling for immediate 22-layer stability."""
        total_sublayers = self.n_blocks * (2 * self.internal_length + 1)
        res_std = 0.02 / math.sqrt(2.0 * total_sublayers)

        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if any(k in name for k in ("out_proj", "w3", "conv_up", "pw_local")):
                    nn.init.normal_(m.weight, mean=0.0, std=res_std)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spatial_x: torch.Tensor, global_x: torch.Tensor, mask_illegal: bool = True):
        """
        spatial_x: (B, 22 or 32, 19, 19)
        global_x:  (B, 19)
        Returns:
          pol_logits: (B, 2, 362)
          wld_logits: (B, 3)
          score_pred: (B,)
          own_pred:   (B, 19, 19)
        """
        B = spatial_x.shape[0]
        raw_22 = spatial_x[:, :22]
        x_in = augment_tactical_planes_32(spatial_x) if spatial_x.shape[1] == 22 else spatial_x

        x = self.conv_spatial(x_in) + self.linear_global(global_x).unsqueeze(-1).unsqueeze(-1)
        x_seq = x.flatten(2).transpose(1, 2).contiguous()
        for blk in self.blocks:
            x_seq = blk.forward_seq(x_seq)
        x = x_seq.transpose(1, 2).reshape(B, self.d_model, 19, 19).contiguous()

        trunk_feat = F.silu(self.norm_trunkfinal(x))
        pol_logits = self.policy_head(trunk_feat, global_x)
        wld_logits, score_pred, own_pred = self.value_head(trunk_feat, global_x)

        if mask_illegal:
            # Plane 0 = on-board (1=valid), Plane 1 = own stones, Plane 2 = opp stones, Plane 6 = ko-ban
            illegal_board = (
                (raw_22[:, 0] < 0.5)
                | (raw_22[:, 1] > 0.5)
                | (raw_22[:, 2] > 0.5)
                | (raw_22[:, 6] > 0.5)
            ).reshape(B, 361)
            p0_board = pol_logits[:, 0, :361].masked_fill(illegal_board, -1e4)
            p0 = torch.cat([p0_board, pol_logits[:, 0, 361:362]], dim=-1)
            pol_logits = torch.stack([p0, pol_logits[:, 1]], dim=1)

        return pol_logits, wld_logits, score_pred, own_pred
