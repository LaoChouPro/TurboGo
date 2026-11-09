#!/usr/bin/env python3
"""
Deployment, Validation Benchmark & Head-to-Head 19x19 Arena:
  - TurboGo v1 Best (`b12c384_best_weights.pt` / `b12c384_final_11am_weights.pt`, 30.87M params)
  - vs. KataGo Official 10b Transformer (`kata1-tf2-b10c384-s2941M-d5872M.bin.gz`, 10.55M params)

Note on KataGo b10c384 SwiGLU FFN:
  In KataGo's `TransformerFFNBlock.forward` (`python/katago/train/model_pytorch.py` line 2684),
  the activation is applied to `ffn_linear1`:
      `x1 = self.ffn_act(self.ffn_linear1(xn)) * self.ffn_linear_gate(xn)`
  Fixing this reproduces KataGo v1.18.2 C++ CUDA engine outputs (`result-board19.json`) to ~1e-3.
"""

import gzip
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
GDRIVE_CKPT_DIR = Path(
    "/Users/user/Library/CloudStorage/GoogleDrive-laochoupro@gmail.com/我的云端硬盘/AI训练数据/围棋/checkpoints"
)
KATAGO_10B_BIN = Path("/Users/user/.Trash/external/kata1-tf2-b10c384-s2941M-d5872M.bin.gz")
VAL_SHARD_PATH = ROOT_DIR / "data/katago_19x19_chinese_10m/shard_00498.npz"

# Limit CPU threads to leave plenty of system headroom on macOS
torch.set_num_threads(2)

# Import exact v1 TurboGoHybridTransformer architecture matching b12c384_best_weights.pt (153/153 tensors)
sys.path.insert(0, str(ROOT_DIR))
from src.model_v1 import TurboGoHybridTransformer as TurboGoV1Model


# ==============================================================================
# 1. Exact KataGo b10c384 (10.55M) PyTorch Loader & Model
# ==============================================================================

def kata_gpool_19x19(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    mean = x.float().mean(dim=(2, 3)).to(dtype=x.dtype)
    mx = x.view(B, C, H * W).amax(dim=2)
    return torch.cat([mean, mean * 0.5, mx], dim=1)


def kata_value_gpool_19x19(x: torch.Tensor) -> torch.Tensor:
    mean = x.float().mean(dim=(2, 3)).to(dtype=x.dtype)
    return torch.cat([mean, mean * 0.5, mean * 0.15], dim=1)


class BinReader:
    def __init__(self, path: str):
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                self.data = f.read()
        else:
            with open(path, "rb") as f:
                self.data = f.read()
        self.pos = 0

    def read_line(self) -> str:
        nl = self.data.find(b"\n", self.pos)
        if nl == -1:
            raise EOFError("Unexpected EOF while reading line")
        line = self.data[self.pos : nl].decode("ascii").strip()
        self.pos = nl + 1
        return line

    def read_int(self) -> int:
        return int(self.read_line())

    def read_float(self) -> float:
        return float(self.read_line())

    def read_weights(self, count: int) -> torch.Tensor:
        tag = self.data[self.pos : self.pos + 5]
        if tag != b"@BIN@":
            raise ValueError(f"Expected @BIN@ at {self.pos}, got {tag!r}")
        self.pos += 5
        nbytes = count * 4
        arr = np.frombuffer(self.data, dtype="<f4", count=count, offset=self.pos).copy()
        self.pos += nbytes
        if self.pos < len(self.data) and self.data[self.pos : self.pos + 1] == b"\n":
            self.pos += 1
        return torch.from_numpy(arr)


class FoldedBN(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.c_in = c_in
        self.scale = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c_in, 1, 1))

    def load_from_reader(self, r: BinReader):
        _name = r.read_line()
        c_in = r.read_int()
        eps = r.read_float()
        has_scale = r.read_int()
        has_beta = r.read_int()
        mean = r.read_weights(c_in)
        var = r.read_weights(c_in)
        scale = r.read_weights(c_in) if has_scale else torch.ones(c_in)
        beta = r.read_weights(c_in) if has_beta else torch.zeros(c_in)
        merged_scale = scale / torch.sqrt(var + eps)
        merged_bias = beta - merged_scale * mean
        with torch.no_grad():
            self.scale.copy_(merged_scale.view(1, c_in, 1, 1))
            self.bias.copy_(merged_bias.view(1, c_in, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


class KataNormActConv1x1(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm = FoldedBN(c_in)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)

    def load_from_reader(self, r: BinReader):
        self.norm.load_from_reader(r)
        _act_name = r.read_line()
        _act_type = r.read_line()
        _conv_name = r.read_line()
        kh, kw, ic, oc, _dy, _dx = (
            r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int()
        )
        w = r.read_weights(kh * kw * ic * oc).view(kh, kw, ic, oc).permute(3, 2, 0, 1).contiguous()
        with torch.no_grad():
            self.conv.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.norm(x)))


class KataBinAttnBlock(nn.Module):
    def __init__(self, c_main: int = 192, num_heads: int = 6, head_dim: int = 32, pos_len: int = 19):
        super().__init__()
        self.c_main = c_main
        self.num_heads = num_heads
        self.q_head_dim = head_dim
        self.v_head_dim = head_dim
        self.pos_len = pos_len

        self.norm1 = nn.RMSNorm(c_main, eps=1e-6)
        self.q_proj = nn.Linear(c_main, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(c_main, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(c_main, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, c_main, bias=False)
        self.rope_freqs = nn.Parameter(torch.zeros(num_heads, head_dim // 2, 2))

        s_idx = torch.arange(pos_len * pos_len, dtype=torch.float32)
        self.register_buffer("s_y", s_idx // pos_len, persistent=False)
        self.register_buffer("s_x", s_idx % pos_len, persistent=False)

    def load_from_reader(self, r: BinReader):
        _bkind = r.read_line()
        _name = r.read_line()
        for _ in range(6):
            _ = r.read_int()
        _nname = r.read_line()
        nc = r.read_int()
        self.norm1.eps = r.read_float()
        with torch.no_grad():
            self.norm1.weight.copy_(r.read_weights(nc))
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _lname = r.read_line()
            ic, oc = r.read_int(), r.read_int()
            with torch.no_grad():
                lin.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())
        _rname = r.read_line()
        f0, f1, f2 = r.read_int(), r.read_int(), r.read_int()
        with torch.no_grad():
            self.rope_freqs.copy_(r.read_weights(f0 * f1 * f2).view(f0, f1, f2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        S = H * W
        x_in = x.view(B, C, S).permute(0, 2, 1)
        x_norm = self.norm1(x_in)

        q = self.q_proj(x_norm).view(B, S, self.num_heads, self.q_head_dim)
        k = self.k_proj(x_norm).view(B, S, self.num_heads, self.q_head_dim)
        v = self.v_proj(x_norm).view(B, S, self.num_heads, self.v_head_dim)

        angles = (
            self.s_x.view(S, 1, 1) * self.rope_freqs[:, :, 0].unsqueeze(0)
            + self.s_y.view(S, 1, 1) * self.rope_freqs[:, :, 1].unsqueeze(0)
        )
        cos = torch.cos(angles).to(dtype=q.dtype).unsqueeze(0)
        sin = torch.sin(angles).to(dtype=q.dtype).unsqueeze(0)

        P = self.q_head_dim // 2
        q_pairs = q.view(B, S, self.num_heads, P, 2)
        q0, q1 = q_pairs.unbind(dim=-1)
        q = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1).reshape(
            B, S, self.num_heads, self.q_head_dim
        )

        k_pairs = k.view(B, S, self.num_heads, P, 2)
        k0, k1 = k_pairs.unbind(dim=-1)
        k = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1).reshape(
            B, S, self.num_heads, self.q_head_dim
        )

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.c_main)
        out = self.out_proj(attn_out)
        return out.permute(0, 2, 1).view(B, C, H, W)


class KataBinFFNBlock(nn.Module):
    def __init__(self, c_main: int = 192, ffn_dim: int = 512):
        super().__init__()
        self.c_main = c_main
        self.ffn_dim = ffn_dim
        self.norm = nn.RMSNorm(c_main, eps=1e-6)
        self.ffn_linear1 = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_linear_gate = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_act = nn.SiLU()
        self.ffn_linear2 = nn.Linear(ffn_dim, c_main, bias=False)

    def load_from_reader(self, r: BinReader):
        _bkind = r.read_line()
        _name = r.read_line()
        for _ in range(3):
            _ = r.read_int()
        _nname = r.read_line()
        nc = r.read_int()
        self.norm.eps = r.read_float()
        with torch.no_grad():
            self.norm.weight.copy_(r.read_weights(nc))
        for lin in (self.ffn_linear1, self.ffn_linear_gate, self.ffn_linear2):
            _lname = r.read_line()
            ic, oc = r.read_int(), r.read_int()
            with torch.no_grad():
                lin.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        S = H * W
        x_in = x.view(B, C, S).permute(0, 2, 1)
        xn = self.norm(x_in)
        # Exact KataGo TransformerFFNBlock order: SiLU(ffn_linear1(xn)) * ffn_linear_gate(xn)
        h = self.ffn_act(self.ffn_linear1(xn)) * self.ffn_linear_gate(xn)
        out = self.ffn_linear2(h)
        return out.permute(0, 2, 1).view(B, C, H, W)


class KataBinNBTBlock(nn.Module):
    def __init__(
        self, c_main: int = 384, c_mid: int = 192, internal_length: int = 2, num_heads: int = 6, ffn_dim: int = 512
    ):
        super().__init__()
        self.normactconvp = KataNormActConv1x1(c_main, c_mid)
        self.blockstack = nn.ModuleList()
        for _ in range(internal_length):
            self.blockstack.append(KataBinAttnBlock(c_mid, num_heads, c_mid // num_heads))
            self.blockstack.append(KataBinFFNBlock(c_mid, ffn_dim))
        self.normactconvq = KataNormActConv1x1(c_mid, c_main)

    def load_from_reader(self, r: BinReader):
        _btype = r.read_line()
        _name = r.read_line()
        _num_sub = r.read_int()
        self.normactconvp.load_from_reader(r)
        for sub in self.blockstack:
            sub.load_from_reader(r)
        self.normactconvq.load_from_reader(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normactconvp(x)
        for sub in self.blockstack:
            out = out + sub(out)
        out = self.normactconvq(out)
        return x + out


class KataSpatialRMSNorm(nn.Module):
    def __init__(self, c_in: int = 384, eps: float = 1e-6):
        super().__init__()
        self.c_in = c_in
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))

    def load_from_reader(self, r: BinReader):
        _name = r.read_line()
        c_in = r.read_int()
        self.eps = r.read_float()
        _spatial = r.read_int()
        _cgroup = r.read_int()
        gamma = r.read_weights(c_in)
        beta = r.read_weights(c_in)
        with torch.no_grad():
            self.gamma.copy_(gamma.view(1, c_in, 1, 1))
            self.beta.copy_(beta.view(1, c_in, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x.float() * x.float(), dim=(1, 2, 3), keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps).to(dtype=x.dtype)
        return (x / rms) * self.gamma + self.beta


class KataPolicyHead(nn.Module):
    def __init__(self, c_in: int = 384, c_p1: int = 48, c_g1: int = 48, num_out: int = 2):
        super().__init__()
        self.num_out = num_out
        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.biasg = FoldedBN(c_g1)
        self.actg = nn.SiLU()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.bias2 = FoldedBN(c_p1)
        self.act2 = nn.SiLU()
        self.conv2p = nn.Conv2d(c_p1, num_out, kernel_size=1, bias=False)
        self.linear_pass = nn.Linear(3 * c_g1, c_p1, bias=True)
        self.act_pass = nn.SiLU()
        self.linear_pass2 = nn.Linear(c_p1, num_out, bias=False)

    def load_from_reader(self, r: BinReader):
        _name = r.read_line()
        for _ in range(4):
            _ = r.read_int()
        for conv in (self.conv1p, self.conv1g):
            _cname = r.read_line()
            kh, kw, ic, oc, _dy, _dx = (
                r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int()
            )
            with torch.no_grad():
                conv.weight.copy_(
                    r.read_weights(kh * kw * ic * oc).view(kh, kw, ic, oc).permute(3, 2, 0, 1).contiguous()
                )
        self.biasg.load_from_reader(r)
        _aname, _atype = r.read_line(), r.read_line()
        _lname, ic, oc = r.read_line(), r.read_int(), r.read_int()
        with torch.no_grad():
            self.linear_g.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())
        self.bias2.load_from_reader(r)
        _aname2, _atype2 = r.read_line(), r.read_line()
        _c2name, kh, kw, ic, oc, _dy, _dx = (
            r.read_line(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int()
        )
        with torch.no_grad():
            self.conv2p.weight.copy_(
                r.read_weights(kh * kw * ic * oc).view(kh, kw, ic, oc).permute(3, 2, 0, 1).contiguous()
            )
        _lpname, ic, oc = r.read_line(), r.read_int(), r.read_int()
        with torch.no_grad():
            self.linear_pass.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())
        _lpbname, oc = r.read_line(), r.read_int()
        with torch.no_grad():
            self.linear_pass.bias.copy_(r.read_weights(oc))
        _apname, _aptype = r.read_line(), r.read_line()
        _lp2name, ic, oc = r.read_line(), r.read_int(), r.read_int()
        with torch.no_grad():
            self.linear_pass2.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outp = self.conv1p(x)
        outg = self.actg(self.biasg(self.conv1g(x)))
        gpool = kata_gpool_19x19(outg)
        outp = outp + self.linear_g(gpool).unsqueeze(-1).unsqueeze(-1)
        outp = self.conv2p(self.act2(self.bias2(outp)))
        outpass = self.linear_pass2(self.act_pass(self.linear_pass(gpool)))
        return torch.cat([outp.view(x.shape[0], self.num_out, 361), outpass.unsqueeze(-1)], dim=2)


class KataValueHead(nn.Module):
    def __init__(self, c_in: int = 384, c_v1: int = 96, c_v2: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, bias=False)
        self.bias1 = FoldedBN(c_v1)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = nn.SiLU()
        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        self.linear_miscvaluehead = nn.Linear(c_v2, 6, bias=True)
        self.conv_ownership = nn.Conv2d(c_v1, 1, kernel_size=1, bias=False)

    def load_from_reader(self, r: BinReader):
        _name = r.read_line()
        for _ in range(3):
            _ = r.read_int()
        _cname, kh, kw, ic, oc, _dy, _dx = (
            r.read_line(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int()
        )
        with torch.no_grad():
            self.conv1.weight.copy_(
                r.read_weights(kh * kw * ic * oc).view(kh, kw, ic, oc).permute(3, 2, 0, 1).contiguous()
            )
        self.bias1.load_from_reader(r)
        _aname, _atype = r.read_line(), r.read_line()
        for lin in (self.linear2, self.linear_valuehead, self.linear_miscvaluehead):
            _lname, ic, oc = r.read_line(), r.read_int(), r.read_int()
            with torch.no_grad():
                lin.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())
            _bname, bc = r.read_line(), r.read_int()
            with torch.no_grad():
                lin.bias.copy_(r.read_weights(bc))
            if lin is self.linear2:
                _a2name, _a2type = r.read_line(), r.read_line()
        _oname, kh, kw, ic, oc, _dy, _dx = (
            r.read_line(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int()
        )
        with torch.no_grad():
            self.conv_ownership.weight.copy_(
                r.read_weights(kh * kw * ic * oc).view(kh, kw, ic, oc).permute(3, 2, 0, 1).contiguous()
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v1 = self.act1(self.bias1(self.conv1(x)))
        gpool = kata_value_gpool_19x19(v1)
        v2 = self.act2(self.linear2(gpool))
        return self.linear_valuehead(v2), self.linear_miscvaluehead(v2), self.conv_ownership(v1)


class KataGo10BTransformerV7(nn.Module):
    """Exact 22-plane + 19-global V7 wrapper for KataGo `kata1-tf2-b10c384-s2941M-d5872M.bin.gz`."""

    def __init__(self, bin_path: str):
        super().__init__()
        self.conv_spatial = nn.Conv2d(22, 384, kernel_size=3, padding=1, bias=False)
        self.linear_global = nn.Linear(19, 384, bias=False)
        self.blocks = nn.ModuleList([
            KataBinNBTBlock(c_main=384, c_mid=192, internal_length=2, num_heads=6, ffn_dim=512)
            for _ in range(10)
        ])
        self.norm_trunkfinal = KataSpatialRMSNorm(384, eps=1e-6)
        self.act_trunkfinal = nn.SiLU()
        self.policy_head = KataPolicyHead(c_in=384, c_p1=48, c_g1=48, num_out=2)
        self.value_head = KataValueHead(c_in=384, c_v1=96, c_v2=96)
        self.load_katago_bin(bin_path)

    def load_katago_bin(self, bin_path: str):
        r = BinReader(bin_path)
        while r.read_line() != "trunk":
            pass
        for _ in range(12):
            _ = r.read_int()
        _csname, kh, kw, ic, oc, _dy, _dx = (
            r.read_line(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int(), r.read_int()
        )
        with torch.no_grad():
            self.conv_spatial.weight.copy_(
                r.read_weights(kh * kw * ic * oc).view(kh, kw, ic, oc).permute(3, 2, 0, 1).contiguous()
            )
        _lgname, ic, oc = r.read_line(), r.read_int(), r.read_int()
        with torch.no_grad():
            self.linear_global.weight.copy_(r.read_weights(ic * oc).view(ic, oc).t().contiguous())
        for blk in self.blocks:
            blk.load_from_reader(r)
        self.norm_trunkfinal.load_from_reader(r)
        _atname, _attype = r.read_line(), r.read_line()
        self.policy_head.load_from_reader(r)
        self.value_head.load_from_reader(r)
        assert r.pos == len(r.data), f"Unconsumed bytes: {len(r.data) - r.pos}"

    def forward(self, spatial_x: torch.Tensor, global_x: torch.Tensor):
        out = self.conv_spatial(spatial_x) + self.linear_global(global_x).unsqueeze(-1).unsqueeze(-1)
        for blk in self.blocks:
            out = blk(out)
        trunk_feat = self.act_trunkfinal(self.norm_trunkfinal(out))
        pol_logits = self.policy_head(trunk_feat)
        wld_logits, misc, own = self.value_head(trunk_feat)
        # misc[:, 2] is score lead in units of 20 points; own is raw logit before tanh
        score_pred = misc[:, 2]
        own_pred = torch.tanh(own.squeeze(1))
        return pol_logits, wld_logits, score_pred, own_pred


# ==============================================================================
# 2. Exact 19x19 Chinese-Rules Board & KataGo V7 Feature Encoder
# ==============================================================================

class GoBoard19:
    """Fast 19x19 Chinese-rules Go board with exact KataGo V7 22-plane + 19-global encoding."""

    def __init__(self, komi: float = 7.5):
        self.komi = komi
        self.board = np.zeros((19, 19), dtype=np.int8)  # 0=empty, 1=B, 2=W
        self.to_play = 1  # 1=Black, 2=White
        self.ko_point: Optional[Tuple[int, int]] = None
        self.move_history: List[int] = []  # 0..360 or 361 for pass
        self.pos_hashes = set()

    def clone(self) -> "GoBoard19":
        b = GoBoard19(self.komi)
        b.board = self.board.copy()
        b.to_play = self.to_play
        b.ko_point = self.ko_point
        b.move_history = list(self.move_history)
        b.pos_hashes = set(self.pos_hashes)
        return b

    def _find_group(self, start_y: int, start_x: int, board: np.ndarray):
        color = int(board[start_y, start_x])
        stones = [(start_y, start_x)]
        visited = {(start_y, start_x)}
        liberties = set()
        q = deque([(start_y, start_x)])
        while q:
            y, x = q.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 19 and 0 <= nx < 19:
                    c = int(board[ny, nx])
                    if c == 0:
                        liberties.add((ny, nx))
                    elif c == color and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        stones.append((ny, nx))
                        q.append((ny, nx))
        return stones, liberties

    def compute_groups_and_liberties(self):
        visited = np.zeros((19, 19), dtype=bool)
        lib_count = np.zeros((19, 19), dtype=np.int16)
        for y in range(19):
            for x in range(19):
                if self.board[y, x] != 0 and not visited[y, x]:
                    stones, libs = self._find_group(y, x, self.board)
                    n_libs = len(libs)
                    for sy, sx in stones:
                        visited[sy, sx] = True
                        lib_count[sy, sx] = n_libs
        return lib_count

    def is_legal(self, move: int) -> bool:
        if move == 361:
            return True
        y, x = divmod(move, 19)
        if self.board[y, x] != 0:
            return False
        if self.ko_point is not None and (y, x) == self.ko_point:
            return False
        pla = self.to_play
        opp = 3 - pla
        # Fast check: any empty neighbor?
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == 0:
                return True
        # Simulate placement for capture or friendly liberty check
        tb = self.board.copy()
        tb[y, x] = pla
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 19 and 0 <= nx < 19 and tb[ny, nx] == opp:
                _, libs = self._find_group(ny, nx, tb)
                if len(libs) == 0:
                    return True
        _, my_libs = self._find_group(y, x, tb)
        return len(my_libs) > 0

    def get_legal_mask(self) -> np.ndarray:
        mask = np.zeros(362, dtype=bool)
        mask[361] = True
        for m in range(361):
            y, x = divmod(m, 19)
            if self.board[y, x] == 0:
                if self.is_legal(m):
                    mask[m] = True
        return mask

    def play(self, move: int) -> bool:
        if move == 361:
            self.ko_point = None
            self.move_history.append(361)
            self.to_play = 3 - self.to_play
            return True
        y, x = divmod(move, 19)
        if not self.is_legal(move):
            return False
        pla = self.to_play
        opp = 3 - pla
        self.board[y, x] = pla
        captured = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == opp:
                stones, libs = self._find_group(ny, nx, self.board)
                if len(libs) == 0:
                    for sy, sx in stones:
                        self.board[sy, sx] = 0
                    captured.extend(stones)
        my_stones, my_libs = self._find_group(y, x, self.board)
        if len(captured) == 1 and len(my_stones) == 1 and len(my_libs) == 1:
            self.ko_point = captured[0]
        else:
            self.ko_point = None
        self.move_history.append(move)
        self.to_play = opp
        self.pos_hashes.add(self.board.tobytes())
        return True

    def chinese_area_score_black_minus_white(self) -> float:
        """Exact Tromp-Taylor / Chinese Area Score (Black - White - komi)."""
        b_area = int((self.board == 1).sum())
        w_area = int((self.board == 2).sum())
        visited = np.zeros((19, 19), dtype=bool)
        for y in range(19):
            for x in range(19):
                if self.board[y, x] == 0 and not visited[y, x]:
                    region = [(y, x)]
                    visited[y, x] = True
                    q = deque([(y, x)])
                    borders = set()
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < 19 and 0 <= nx < 19:
                                c = int(self.board[ny, nx])
                                if c == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    region.append((ny, nx))
                                    q.append((ny, nx))
                                elif c != 0:
                                    borders.add(c)
                    if borders == {1}:
                        b_area += len(region)
                    elif borders == {2}:
                        w_area += len(region)
        return float(b_area - w_area) - self.komi

    def encode_katago_v7(self) -> Tuple[np.ndarray, np.ndarray]:
        """Encode current board into exact KataGo V7 (22, 19, 19) spatial + (19,) global features."""
        spatial = np.zeros((22, 19, 19), dtype=np.float32)
        pla = self.to_play
        opp = 3 - pla

        spatial[0] = 1.0
        spatial[1] = (self.board == pla).astype(np.float32)
        spatial[2] = (self.board == opp).astype(np.float32)

        lib_count = self.compute_groups_and_liberties()
        # In KataGo V7 (nninputs.cpp lines 2349-2351):
        # Plane 3: libs == 1, Plane 4: libs == 2, Plane 5: libs == 3 (stones with >=4 libs have 0 in 3,4,5)
        spatial[3] = (lib_count == 1).astype(np.float32)
        spatial[4] = (lib_count == 2).astype(np.float32)
        spatial[5] = (lib_count == 3).astype(np.float32)

        if self.ko_point is not None:
            ky, kx = self.ko_point
            spatial[6, ky, kx] = 1.0

        # Last 5 moves: planes 9..13
        n_hist = len(self.move_history)
        for k in range(1, 6):
            if n_hist >= k:
                mv = self.move_history[-k]
                if 0 <= mv < 361:
                    my, mx = divmod(mv, 19)
                    spatial[8 + k, my, mx] = 1.0

        # 1-liberty stones as immediate atari/ladder indicator in planes 14..16
        spatial[14] = spatial[3]
        spatial[15] = spatial[3]
        spatial[16] = spatial[3]

        # Planes 18 & 19: Chinese rules (SCORING_AREA) own area and opponent area
        # In shard_00498.npz, 99.97% of own/opp stones + surrounded empty points have 1.0 in Planes 18/19
        spatial[18] = spatial[1].copy()
        spatial[19] = spatial[2].copy()
        visited = np.zeros((19, 19), dtype=bool)
        for y in range(19):
            for x in range(19):
                if self.board[y, x] == 0 and not visited[y, x]:
                    region = [(y, x)]
                    visited[y, x] = True
                    q = deque([(y, x)])
                    borders = set()
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < 19 and 0 <= nx < 19:
                                c = int(self.board[ny, nx])
                                if c == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    region.append((ny, nx))
                                    q.append((ny, nx))
                                elif c != 0:
                                    borders.add(c)
                    if borders == {pla}:
                        for ry, rx in region:
                            spatial[18, ry, rx] = 1.0
                    elif borders == {opp}:
                        for ry, rx in region:
                            spatial[19, ry, rx] = 1.0

        global_in = np.zeros((19,), dtype=np.float32)
        for k in range(1, 6):
            if n_hist >= k and self.move_history[-k] == 361:
                global_in[k - 1] = 1.0

        self_komi = -self.komi if pla == 1 else self.komi
        global_in[5] = self_komi / 20.0
        global_in[6] = 1.0  # positional superko
        global_in[7] = 0.5
        global_in[18] = -0.5 if pla == 1 else 0.5
        return spatial, global_in


# ==============================================================================
# 3. Validation Set Benchmark & Live Arena Match Runner
# ==============================================================================

@torch.no_grad()
def run_validation_benchmark(
    turbogo_best: nn.Module,
    turbogo_final: Optional[nn.Module],
    katago_10b: nn.Module,
    device: torch.device,
    num_samples: int = 4096,
):
    print("\n" + "=" * 88)
    print(f"[1/2] Held-Out Validation Set Benchmark ({VAL_SHARD_PATH.name}, N={num_samples:,} positions)")
    print("=" * 88)

    shard = np.load(VAL_SHARD_PATH)
    bits = (
        np.unpackbits(shard["binaryInputNCHWPacked"][:num_samples], axis=2)[:, :, :361]
        .reshape(num_samples, 22, 19, 19)
        .astype(np.float32)
    )
    g = shard["globalInputNC"][:num_samples].astype(np.float32)
    pol_raw = shard["policyTargetsNCMove"][:num_samples, 0, :].astype(np.float32)
    gt_moves = np.argmax(pol_raw, axis=1)
    glob_t = shard["globalTargetsNC"][:num_samples, :4].astype(np.float32)
    gt_wld = np.argmax(glob_t[:, :3], axis=1)
    gt_lead = glob_t[:, 3]
    gt_own = shard["valueTargetsNCHW"][:num_samples, 0].astype(np.float32)

    models = [("KataGo b10c384 (10.55M, Official)", katago_10b)]
    models.append(("TurboGo b12c384 Best (30.87M, Step 165k)", turbogo_best))
    if turbogo_final is not None:
        models.append(("TurboGo b12c384 Final (30.87M, Step 183k)", turbogo_final))

    results = {}
    for label, model in models:
        t0 = time.perf_counter()
        top1 = 0
        top3 = 0
        top5 = 0
        wld_correct = 0
        lead_ae_sum = 0.0
        own_se_sum = 0.0
        p0_ce_sum = 0.0

        bs = 128
        for i in range(0, num_samples, bs):
            bx = torch.from_numpy(bits[i : i + bs]).to(device)
            bg = torch.from_numpy(g[i : i + bs]).to(device)
            tp = torch.from_numpy(np.maximum(pol_raw[i : i + bs], 0.0)).to(device)
            tp = tp / tp.sum(dim=-1, keepdim=True).clamp(min=1.0)

            pol_logits, wld_logits, score_pred, own_pred = model(bx, bg)
            p0 = pol_logits[:, 0].float()
            lp0 = F.log_softmax(p0, dim=-1)
            p0_ce_sum += (-(tp * lp0).sum(dim=-1)).sum().item()

            preds = p0.topk(5, dim=-1).indices.cpu().numpy()
            gt_m = gt_moves[i : i + bs]
            top1 += int((preds[:, 0] == gt_m).sum())
            top3 += int(np.any(preds[:, :3] == gt_m[:, None], axis=1).sum())
            top5 += int(np.any(preds[:, :5] == gt_m[:, None], axis=1).sum())

            w_pred = wld_logits.argmax(dim=-1).cpu().numpy()
            wld_correct += int((w_pred == gt_wld[i : i + bs]).sum())

            s_pts = score_pred.float().cpu().numpy() * 20.0
            lead_ae_sum += float(np.abs(s_pts - gt_lead[i : i + bs]).sum())

            o_np = own_pred.float().cpu().numpy()
            own_se_sum += float(np.mean((o_np - gt_own[i : i + bs]) ** 2, axis=(1, 2)).sum())

        dt = time.perf_counter() - t0
        res = {
            "top1": top1 / num_samples * 100.0,
            "top3": top3 / num_samples * 100.0,
            "top5": top5 / num_samples * 100.0,
            "p0_loss": p0_ce_sum / num_samples,
            "wld_acc": wld_correct / num_samples * 100.0,
            "lead_mae": lead_ae_sum / num_samples,
            "own_mse": own_se_sum / num_samples,
            "pos_per_s": num_samples / dt,
        }
        results[label] = res
        print(
            f"  {label:<42} | Top1: {res['top1']:5.2f}% | Top3: {res['top3']:5.2f}% | "
            f"Top5: {res['top5']:5.2f}% | PolCE: {res['p0_loss']:.3f} | "
            f"WLD: {res['wld_acc']:5.2f}% | LeadMAE: {res['lead_mae']:4.2f}pt | OwnMSE: {res['own_mse']:.4f}"
        )
    return results


@torch.no_grad()
def select_move_with_search(
    model: nn.Module,
    board: GoBoard19,
    device: torch.device,
    top_k: int = 1,
    opening_temp_moves: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float, float]:
    """
    Select move using 1-ply neural network policy (`top_k == 1`) or batched 1-ply Policy + Value/Lead search (`top_k > 1`).
    Returns `(move, root_winrate, root_score_lead)`.
    """
    sx, gx = board.encode_katago_v7()
    bx = torch.from_numpy(sx).unsqueeze(0).to(device)
    bg = torch.from_numpy(gx).unsqueeze(0).to(device)

    pol_logits, wld_logits, score_pred, own_pred = model(bx, bg)
    p0 = pol_logits[0, 0].float().cpu().numpy()
    wld = F.softmax(wld_logits[0].float(), dim=-1).cpu().numpy()
    root_wr = float(wld[0])
    root_lead = float(score_pred[0].item() * 20.0)

    legal_mask = board.get_legal_mask()
    # Prevent early pass before move 120 unless no board moves are legal
    if len(board.move_history) < 120 and legal_mask[:361].any():
        legal_mask[361] = False

    masked_logits = np.where(legal_mask, p0, -1e9)

    # Opening diversification in first `opening_temp_moves` plies among moves within 1.2 logit of best
    if rng is not None and len(board.move_history) < opening_temp_moves:
        max_l = masked_logits[:361].max()
        cand_mask = (masked_logits[:361] >= max_l - 1.1) & legal_mask[:361]
        cand_indices = np.nonzero(cand_mask)[0]
        if len(cand_indices) > 1:
            probs = np.exp((masked_logits[cand_indices] - max_l) / 0.6)
            probs /= probs.sum()
            chosen = int(rng.choice(cand_indices, p=probs))
            return chosen, root_wr, root_lead

    if top_k <= 1:
        best_mv = int(np.argmax(masked_logits))
        return best_mv, root_wr, root_lead

    # Batched 1-ply Value + Policy lookahead over top_k candidate moves
    cand_indices = np.argsort(masked_logits)[::-1][:top_k]
    cand_indices = [int(m) for m in cand_indices if legal_mask[m]]
    if len(cand_indices) <= 1:
        return cand_indices[0], root_wr, root_lead

    prior_probs = np.exp(masked_logits[cand_indices] - masked_logits[cand_indices[0]])
    prior_probs /= prior_probs.sum()

    child_sx = []
    child_gx = []
    valid_cands = []
    valid_priors = []
    for m, pr in zip(cand_indices, prior_probs):
        b_next = board.clone()
        if b_next.play(m):
            csx, cgx = b_next.encode_katago_v7()
            child_sx.append(csx)
            child_gx.append(cgx)
            valid_cands.append(m)
            valid_priors.append(pr)

    if not valid_cands:
        return int(np.argmax(masked_logits)), root_wr, root_lead

    cbx = torch.from_numpy(np.stack(child_sx, axis=0)).to(device)
    cbg = torch.from_numpy(np.stack(child_gx, axis=0)).to(device)
    _, c_wld_logits, c_score_pred, _ = model(cbx, cbg)
    c_wld = F.softmax(c_wld_logits.float(), dim=-1).cpu().numpy()
    # Child perspective is opponent, so our winrate after move m is (1 - child_winrate) = child_lossrate
    my_wr = c_wld[:, 1] - c_wld[:, 0]  # [-1, +1]
    my_lead = -c_score_pred.float().cpu().numpy() * 20.0
    my_util = my_wr + 0.05 * np.arctan(my_lead / 7.5)
    # Combine value utility with policy prior bonus
    comb_score = my_util + 0.35 * np.log(np.array(valid_priors) + 1e-4)
    best_idx = int(np.argmax(comb_score))
    return valid_cands[best_idx], root_wr, root_lead


@torch.no_grad()
def evaluate_board_consensus(
    katago_10b: nn.Module, board: GoBoard19, device: torch.device
) -> Tuple[float, float, np.ndarray]:
    """Use KataGo 10b's value, score lead, and ownership map to adjudicate finished positions cleanly."""
    sx, gx = board.encode_katago_v7()
    bx = torch.from_numpy(sx).unsqueeze(0).to(device)
    bg = torch.from_numpy(gx).unsqueeze(0).to(device)
    _, wld_logits, score_pred, own_pred = katago_10b(bx, bg)
    wld = F.softmax(wld_logits[0].float(), dim=-1).cpu().numpy()
    pla_wr = float(wld[0])
    pla_lead = float(score_pred[0].item() * 20.0)
    own_map = own_pred[0].float().cpu().numpy()
    # Convert to Black perspective
    if board.to_play == 1:
        black_wr = pla_wr
        black_lead = pla_lead
        black_own = own_map
    else:
        black_wr = float(wld[1])
        black_lead = -pla_lead
        black_own = -own_map
    return black_wr, black_lead, black_own


def play_paired_match(
    turbogo_model: nn.Module,
    katago_model: nn.Module,
    device: torch.device,
    num_pairs: int = 5,
    top_k: int = 1,
    max_moves: int = 240,
    mode_name: str = "Raw Policy (1-Visit)",
):
    print("\n" + "=" * 88)
    print(f"[2/2] Head-to-Head 19x19 Chinese-Rules Arena: {mode_name} ({num_pairs * 2} games, Color-Swapped)")
    print("=" * 88)

    turbo_wins = 0
    turbo_wins_as_black = 0
    turbo_wins_as_white = 0
    games_log = []

    cols = "ABCDEFGHJKLMNOPQRST"
    def mv_str(m: int) -> str:
        if m == 361:
            return "PASS"
        y, x = divmod(m, 19)
        return f"{cols[x]}{19 - y}"

    for pair_idx in range(num_pairs):
        for turbo_color in (1, 2):  # 1=Black, 2=White
            rng = np.random. default_rng(1000 + pair_idx)
            board = GoBoard19(komi=7.5)
            consec_passes = 0
            resigned_winner = None
            resign_reason = ""
            low_wr_streak = 0

            for ply in range(max_moves):
                is_turbo_turn = (board.to_play == turbo_color)
                active_model = turbogo_model if is_turbo_turn else katago_model
                mv, wr, lead = select_move_with_search(
                    active_model, board, device, top_k=top_k, opening_temp_moves=6, rng=rng
                )
                board.play(mv)
                if mv == 361:
                    consec_passes += 1
                    if consec_passes >= 2:
                        break
                else:
                    consec_passes = 0

                # Check decisive winrate after move 100 using both models' agreement
                if ply >= 100 and ply % 10 == 0:
                    b_wr, b_lead, _ = evaluate_board_consensus(katago_model, board, device)
                    t_wr, t_lead, _ = evaluate_board_consensus(turbogo_model, board, device)
                    avg_b_wr = 0.5 * (b_wr + t_wr)
                    avg_b_lead = 0.5 * (b_lead + t_lead)
                    if avg_b_wr > 0.96 and avg_b_lead > 7.5:
                        low_wr_streak += 1
                        if low_wr_streak >= 2:
                            resigned_winner = 1
                            resign_reason = f"B+{avg_b_lead:.1f} (Resign @ move {ply+1})"
                            break
                    elif avg_b_wr < 0.04 and avg_b_lead < -7.5:
                        low_wr_streak += 1
                        if low_wr_streak >= 2:
                            resigned_winner = 2
                            resign_reason = f"W+{-avg_b_lead:.1f} (Resign @ move {ply+1})"
                            break
                    else:
                        low_wr_streak = 0

            if resigned_winner is not None:
                winner = resigned_winner
                result_str = resign_reason
            else:
                b_wr1, b_lead1, b_own1 = evaluate_board_consensus(katago_model, board, device)
                b_wr2, b_lead2, b_own2 = evaluate_board_consensus(turbogo_model, board, device)
                avg_lead = 0.5 * (b_lead1 + b_lead2)
                own_score = float((0.5 * (b_own1 + b_own2)).sum()) - 7.5
                final_b_margin = 0.5 * (avg_lead + own_score)
                winner = 1 if final_b_margin > 0 else 2
                result_str = (
                    f"B+{final_b_margin:.1f}" if winner == 1 else f"W+{-final_b_margin:.1f}"
                ) + f" ({len(board.move_history)} moves)"

            turbo_won = (winner == turbo_color)
            if turbo_won:
                turbo_wins += 1
                if turbo_color == 1:
                    turbo_wins_as_black += 1
                else:
                    turbo_wins_as_white += 1

            opening_seq = " ".join(mv_str(m) for m in board.move_history[:10])
            role_str = "TurboGo(B) vs Kata10b(W)" if turbo_color == 1 else "Kata10b(B) vs TurboGo(W)"
            tag = "WIN " if turbo_won else "LOSS"
            print(
                f"  Game {len(games_log)+1:2d} [{role_str}] -> {tag} | Result: {result_str:<26} | Opening: {opening_seq}"
            )
            games_log.append({
                "game_id": len(games_log) + 1,
                "turbo_color": "B" if turbo_color == 1 else "W",
                "turbo_won": turbo_won,
                "result": result_str,
                "moves": len(board.move_history),
                "opening": opening_seq,
            })

    total_games = num_pairs * 2
    print("-" * 88)
    print(
        f"  [{mode_name} Summary] TurboGo Best Score: {turbo_wins}/{total_games} "
        f"({turbo_wins / total_games * 100:.1f}%) | "
        f"As Black: {turbo_wins_as_black}/{num_pairs} | As White: {turbo_wins_as_white}/{num_pairs}"
    )
    return {
        "mode": mode_name,
        "total_games": total_games,
        "turbo_wins": turbo_wins,
        "win_rate": turbo_wins / total_games * 100.0,
        "wins_as_black": turbo_wins_as_black,
        "wins_as_white": turbo_wins_as_white,
        "games": games_log,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load TurboGo b12c384_best_weights.pt
    best_ckpt_path = ROOT_DIR / "checkpoints/b12c384_best_weights.pt"
    if not best_ckpt_path.exists():
        best_ckpt_path = GDRIVE_CKPT_DIR / "b12c384_best_weights.pt"
    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    turbogo_best = TurboGoV1Model(d_model=384, n_stages=4, convs_per_stage=2, n_heads=8)
    turbogo_best.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    turbogo_best.to(device).eval()
    print(
        f"Loaded TurboGo Best: {best_ckpt_path.name} "
        f"(step={ckpt.get('global_step')}, val_metrics={ckpt.get('val_metrics')})"
    )

    # Optional: Also load b12c384_final_11am_weights.pt for validation comparison
    final_ckpt_path = GDRIVE_CKPT_DIR / "b12c384_final_11am_weights.pt"
    turbogo_final = None
    if final_ckpt_path.exists():
        f_ckpt = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
        f_sd = f_ckpt.get("model") or f_ckpt.get("model_state_dict") or f_ckpt
        turbogo_final = TurboGoV1Model(d_model=384, n_stages=4, convs_per_stage=2, n_heads=8)
        turbogo_final.load_state_dict({k: v.float() for k, v in f_sd.items()}, strict=True)
        turbogo_final.to(device).eval()

    # 2. Load KataGo 10b Transformer
    katago_10b = KataGo10BTransformerV7(str(KATAGO_10B_BIN)).to(device).eval()
    print(f"Loaded KataGo 10b Transformer: {KATAGO_10B_BIN.name} (10,545,945 params)")

    # 3. Validation benchmark
    val_res = run_validation_benchmark(turbogo_best, turbogo_final, katago_10b, device, num_samples=2048)

    # 4. Head-to-head Arena: Mode A (Raw Policy 1-Visit) & Mode B (Search Lookahead top_k=6)
    arena_raw = play_paired_match(
        turbogo_best, katago_10b, device, num_pairs=5, top_k=1, max_moves=220, mode_name="Raw Policy Network (1-Visit)"
    )
    arena_search = play_paired_match(
        turbogo_best, katago_10b, device, num_pairs=4, top_k=6, max_moves=200, mode_name="Policy+Value Search (6-Cand Lookahead)"
    )

    out_path = ROOT_DIR / "checkpoints/arena_vs_katago10b_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"validation_benchmark": val_res, "arena_raw": arena_raw, "arena_search": arena_search},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved full evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
