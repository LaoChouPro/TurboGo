#!/usr/bin/env python3
"""
High-Throughput RTX PRO 6000 Blackwell Pretraining Pipeline for TurboGo-Mini (`b6c192-nbt12`, 3.06M params)
on 100,000,000 KataGo 19x19 Chinese-Rules Positions (From Scratch)

Architecture (Identical to TurboGo-v2, scaled to 3.06M params to match TinyGo Large's 3.22M ResNet):
  - `d_model = 192`, `c_mid = 128`, `n_blocks = 6`, `internal_length = 2`, `n_heads = 4` (`head_dim = 32`), `ffn_dim = 256`
  - 6 Nested Bottleneck Transformer (`TurboNBTBlock`) macro-blocks = **12 Global Self-Attention Layers
    + 12 SwiGLU FFN Layers** + 6 Depthwise 3x3 / KataGo Global Pool local-global mixers
  - Learnable Multi-Scale 2D-RoPE (`rope_freqs`) + 100% Pure Identity Pre-RMSNorm Residual Highway
  - 32-Channel Explicit Tactical Input Stem + KataGo Global-Pool Policy, Value (WLD), Score Lead & Ownership Heads
"""

import json
import math
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import TurboGoHybridTransformer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MINI_WEIGHTS_GDRIVE_ID = "1Uh11E6zGnYGJcLSTsdFWh82hYenpJRhd"
MINI_METRICS_GDRIVE_ID = "1E3UBroUaFMY07ag8XQSbOlsqek72BiFi"


def build_d4_symmetry_tables(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(361, dtype=torch.long).reshape(19, 19)
    perms_361 = []
    perms_362 = []
    for sym_id in range(8):
        g = base.clone()
        if sym_id & 4:
            g = g.t().contiguous()
        if sym_id & 2:
            g = torch.flip(g, dims=[0])
        if sym_id & 1:
            g = torch.flip(g, dims=[1])
        p361 = g.reshape(361)
        p362 = torch.cat([p361, torch.tensor([361], dtype=torch.long)], dim=0)
        perms_361.append(p361)
        perms_362.append(p362)
    return torch.stack(perms_361, dim=0).to(device), torch.stack(perms_362, dim=0).to(device)


def unpack_and_transform_batch(
    b_packed: torch.Tensor,
    g_in: torch.Tensor,
    p_vis: torch.Tensor,
    g_tgt: torch.Tensor,
    v_own: torch.Tensor,
    bit_shifts: torch.Tensor,
    perm_361: torch.Tensor,
    perm_362: torch.Tensor,
    apply_d4: bool = True,
):
    B = b_packed.shape[0]
    bits = (b_packed.unsqueeze(-1) >> bit_shifts) & 1
    spatial = bits.reshape(B, 22, 368)[:, :, :361].reshape(B, 22, 19, 19).to(torch.bfloat16)

    global_x = g_in.to(torch.bfloat16)

    p_float = p_vis.to(torch.float32).clamp(min=0.0)
    p_sum = p_float.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    target_pol = p_float / p_sum

    target_wld = g_tgt[:, 0:3].to(torch.float32)
    target_wld = target_wld / target_wld.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    target_score = (g_tgt[:, 3].to(torch.float32) / 20.0).clamp(-3.0, 3.0)
    target_own = v_own[:, 0].to(torch.float32)

    if apply_d4 and B >= 8 and (B % 8 == 0):
        sub = B // 8
        idx_361 = perm_361.unsqueeze(1).expand(8, sub, 361).reshape(B, 361)
        idx_362 = perm_362.unsqueeze(1).expand(8, sub, 362).reshape(B, 362)

        spatial = torch.gather(
            spatial.reshape(B, 22, 361), 2, idx_361.unsqueeze(1).expand(B, 22, 361)
        ).reshape(B, 22, 19, 19)
        target_pol = torch.gather(target_pol, 2, idx_362.unsqueeze(1).expand(B, 2, 362))
        target_own = torch.gather(target_own.reshape(B, 361), 1, idx_361).reshape(B, 19, 19)

    return spatial, global_x, target_pol, target_wld, target_score, target_own


def load_single_shard(path: Path):
    for _ in range(600):
        if path.exists() and path.stat().st_size > 100_000:
            break
        time.sleep(0.5)
    with np.load(path) as d:
        return (
            d["binaryInputNCHWPacked"],
            d["globalInputNC"],
            d["policyTargetsNCMove"],
            d["globalTargetsNC"][:, :4].copy(),
            d["valueTargetsNCHW"][:, :1].copy(),
        )


def load_macro_chunk(shard_paths: List[Path], pool: ThreadPoolExecutor):
    t0 = time.perf_counter()
    results = list(pool.map(load_single_shard, shard_paths))
    b_arr = np.concatenate([r[0] for r in results], axis=0)
    gi_arr = np.concatenate([r[1] for r in results], axis=0)
    p_arr = np.concatenate([r[2] for r in results], axis=0)
    gt_arr = np.concatenate([r[3] for r in results], axis=0)
    vo_arr = np.concatenate([r[4] for r in results], axis=0)
    load_sec = time.perf_counter() - t0
    return (
        torch.from_numpy(b_arr),
        torch.from_numpy(gi_arr),
        torch.from_numpy(p_arr),
        torch.from_numpy(gt_arr),
        torch.from_numpy(vo_arr),
        load_sec,
    )


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    raw_model: nn.Module,
    ema_params: List[torch.Tensor],
    val_tensors: Tuple[torch.Tensor, ...],
    bit_shifts: torch.Tensor,
    perm_361: torch.Tensor,
    perm_362: torch.Tensor,
    batch_size: int = 1024,
) -> Dict[str, float]:
    live_params = [p.data.clone() for p in raw_model.parameters()]
    for p, ep in zip(raw_model.parameters(), ema_params):
        p.data.copy_(ep)

    model.eval()
    vb, vgi, vp, vgt, vo = val_tensors
    N = vb.shape[0]
    n_batches = N // batch_size

    tot_loss = 0.0
    tot_p0_loss = 0.0
    tot_val_loss = 0.0
    tot_score_mae = 0.0
    tot_own_mse = 0.0
    tot_top1 = 0
    tot_top3 = 0
    tot_val_acc = 0

    for i in range(n_batches):
        s = i * batch_size
        e = s + batch_size
        spatial, global_x, t_pol, t_wld, t_score, t_own = unpack_and_transform_batch(
            vb[s:e], vgi[s:e], vp[s:e], vgt[s:e], vo[s:e], bit_shifts, perm_361, perm_362, apply_d4=False
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pol_logits, wld_logits, score_pred, own_pred = model(spatial, global_x)
            lp0 = F.log_softmax(pol_logits[:, 0].float(), dim=-1)
            lp1 = F.log_softmax(pol_logits[:, 1].float(), dim=-1)
            loss_p0 = -(t_pol[:, 0] * lp0).sum(dim=-1).mean()
            loss_p1 = -(t_pol[:, 1] * lp1).sum(dim=-1).mean()
            loss_val = -(t_wld * F.log_softmax(wld_logits.float(), dim=-1)).sum(dim=-1).mean()
            loss_score = F.huber_loss(score_pred.float(), t_score)
            own_mask = (t_own.abs().amax(dim=(1, 2)) > 0.0).to(torch.float32)
            own_sq = (own_pred.float() - t_own).pow(2).mean(dim=(1, 2))
            loss_own = (own_sq * own_mask).sum() / own_mask.sum().clamp(min=1.0)
            loss = loss_p0 + 0.15 * loss_p1 + 1.2 * loss_val + 0.4 * loss_score + 0.4 * loss_own

        tot_loss += loss.item()
        tot_p0_loss += loss_p0.item()
        tot_val_loss += loss_val.item()
        tot_score_mae += (score_pred.float() - t_score).abs().mean().item() * 20.0
        tot_own_mse += loss_own.item()

        gt_move = t_pol[:, 0].argmax(dim=-1)
        pred_top3 = pol_logits[:, 0].topk(3, dim=-1).indices
        tot_top1 += (pred_top3[:, 0] == gt_move).sum().item()
        tot_top3 += (pred_top3 == gt_move.unsqueeze(-1)).any(dim=-1).sum().item()
        tot_val_acc += (wld_logits.argmax(dim=-1) == t_wld.argmax(dim=-1)).sum().item()

    for p, lp in zip(raw_model.parameters(), live_params):
        p.data.copy_(lp)

    model.train()
    total_samples = n_batches * batch_size
    return {
        "val_loss": round(tot_loss / n_batches, 4),
        "val_p0_loss": round(tot_p0_loss / n_batches, 4),
        "val_wld_loss": round(tot_val_loss / n_batches, 4),
        "val_own_mse": round(tot_own_mse / n_batches, 4),
        "val_score_mae_pts": round(tot_score_mae / n_batches, 2),
        "val_top1_acc": round(tot_top1 / total_samples * 100.0, 2),
        "val_top3_acc": round(tot_top3 / total_samples * 100.0, 2),
        "val_wld_acc": round(tot_val_acc / total_samples * 100.0, 2),
    }


def cloud_sync_to_gdrive(metrics_path: Path, best_weights_path: Path | None = None):
    """Directly PATCH turbogo_mini_metrics.jsonl and turbogo_mini_best_weights.pt to Google Drive."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://172.28.0.1:8008/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        tok = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())["access_token"]
        if metrics_path.exists():
            m_req = urllib.request.Request(
                f"https://www.googleapis.com/upload/drive/v3/files/{MINI_METRICS_GDRIVE_ID}?uploadType=media",
                data=metrics_path.read_bytes(),
                method="PATCH",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            )
            urllib.request.urlopen(m_req, timeout=15).read()
        if best_weights_path is not None and best_weights_path.exists():
            t0 = time.perf_counter()
            bw_bytes = best_weights_path.read_bytes()
            b_req = urllib.request.Request(
                f"https://www.googleapis.com/upload/drive/v3/files/{MINI_WEIGHTS_GDRIVE_ID}?uploadType=media",
                data=bw_bytes,
                method="PATCH",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/octet-stream"},
            )
            urllib.request.urlopen(b_req, timeout=60).read()
            print(
                f"  >>> [Cloud-GDrive] Uploaded turbogo_mini_best_weights.pt ({len(bw_bytes)/1e6:.2f} MB) + metrics.jsonl in {time.perf_counter()-t0:.2f}s",
                flush=True,
            )
    except Exception as e:
        print(f"  [Cloud-GDrive Warn] {e}", flush=True)


def main():
    data_dir_10m = Path("/content/data/katago_19x19_chinese_10m")
    data_dir_50m = Path("/content/data/katago_19x19_chinese_50m")
    data_dir_40m = Path("/content/data/katago_19x19_chinese_40m")
    ckpt_dir = Path("/content/checkpoints/turbogo_mini")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    shards_10m = [data_dir_10m / f"shard_{i:05d}.npz" for i in range(500)]
    shards_50m = [data_dir_50m / f"shard_{i:05d}.npz" for i in range(2500)]
    shards_40m = [data_dir_40m / f"shard_{i:05d}.npz" for i in range(2000)]

    # Exact same 40,000-position validation split (shards 498..499 of 10M) for 100% benchmark comparability
    val_shards = shards_10m[498:500]
    train_shards = shards_10m[:498] + shards_50m + shards_40m  # 4,998 shards = 99,960,000 unique positions

    batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
    num_epochs = int(os.environ.get("EPOCHS", "1"))
    max_lr = float(os.environ.get("MAX_LR", "1.60e-3"))
    min_lr = float(os.environ.get("MIN_LR", "1.50e-5"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "1000"))
    shards_per_chunk = 16  # 320,000 positions per GPU-resident macro-chunk (313 chunks = 97,656 steps at BS=1024)

    device = torch.device("cuda:0")
    print("=" * 92, flush=True)
    print("TurboGo-Mini (b6c192-nbt12, 12-Layer Global Self-Attention + SwiGLU + 2D-RoPE, 3.06M params)", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(device)} | VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GiB", flush=True)
    print(f"Dataset: {len(train_shards):,} train shards ({len(train_shards)*20000:,} pos) + {len(val_shards)} val shards ({len(val_shards)*20000:,} pos)", flush=True)
    print(f"Hyperparams: BS={batch_size} | One-Cycle Cosine LR={max_lr:.2e}->{min_lr:.2e} | EMA=0.9992->0.9995", flush=True)
    print("=" * 92, flush=True)

    raw_model = TurboGoHybridTransformer(
        d_model=192, c_mid=128, n_blocks=6, internal_length=2, n_heads=4, ffn_dim=256, in_channels=32
    ).to(device)
    num_params = sum(p.numel() for p in raw_model.parameters())
    n_attn = sum(len(b.attn_layers) for b in raw_model.blocks)
    n_ffn = sum(len(b.ffn_layers) for b in raw_model.blocks)
    print(
        f"[Init] TurboGo-Mini Parameters: {num_params:,} ({num_params / 1e6:.3f}M) | "
        f"NBT Macro-Blocks: {len(raw_model.blocks)} | Global Attention Layers: {n_attn} | SwiGLU FFN Layers: {n_ffn}",
        flush=True,
    )

    bit_shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    perm_361, perm_362 = build_d4_symmetry_tables(device)

    decay_params = []
    no_decay_params = []
    rope_params = []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if "rope_freqs" in name:
            rope_params.append(p)
        elif p.ndim < 2 or "norm" in name or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    print(
        f"[Optim] Decay Params (wd=1e-4): {sum(p.numel() for p in decay_params):,} | "
        f"No-Decay Params (wd=0.0): {sum(p.numel() for p in no_decay_params):,} | "
        f"2D-RoPE Params (2.0x LR, wd=0.0): {sum(p.numel() for p in rope_params):,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "lr": max_lr, "weight_decay": 1e-4},
            {"params": no_decay_params, "lr": max_lr, "weight_decay": 0.0},
            {"params": rope_params, "lr": max_lr * 2.0, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
        fused=True,
    )

    # Exact total steps across all 313 chunks of the 100M dataset at BS=1024:
    # 312 full chunks * (320,000 // 1024 = 312 steps) + 1 tail chunk (120,000 // 1024 = 117 steps) = 97,461 steps
    total_steps = int(os.environ.get("TOTAL_STEPS", "97461"))

    start_epoch = 0
    start_chunk = 0
    global_step = 0
    best_val_loss = float("inf")
    best_top1_acc = 0.0

    model_params = list(raw_model.parameters())
    ema_params = [p.detach().clone() for p in model_params]

    latest_ckpt = ckpt_dir / "turbogo_mini_latest.pt"
    best_ckpt = ckpt_dir / "turbogo_mini_best.pt"
    best_weights_ckpt = ckpt_dir / "turbogo_mini_best_weights.pt"

    if latest_ckpt.exists():
        print(f"[Resume] Loading TurboGo-Mini checkpoint from {latest_ckpt}...", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
        raw_model.load_state_dict(sd)
        if "ema_model" in ckpt:
            ema_sd = ckpt["ema_model"]
            for (name, _), ep in zip(raw_model.named_parameters(), ema_params):
                if name in ema_sd:
                    ep.copy_(ema_sd[name].to(device))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        start_chunk = ckpt.get("chunk", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_top1_acc = ckpt.get("val_metrics", {}).get("val_top1_acc", 0.0)
        print(
            f"[Resume] Resumed TurboGo-Mini at epoch={start_epoch+1}, chunk={start_chunk}, "
            f"global_step={global_step}, best_val_loss={best_val_loss:.4f}, best_top1_acc={best_top1_acc:.2f}%",
            flush=True,
        )

    print("[Compile] Compiling TurboGo-Mini (12-Layer NBT, 3.06M) with TorchInductor (Triton)...", flush=True)
    model = torch.compile(raw_model)

    io_pool = ThreadPoolExecutor(max_workers=24)
    prefetch_executor = ThreadPoolExecutor(max_workers=2)
    gdrive_sync_executor = ThreadPoolExecutor(max_workers=1)

    print("[Data] Preloading 40,000 validation positions into GPU VRAM...", flush=True)
    vb, vgi, vp, vgt, vo, v_sec = load_macro_chunk(val_shards, io_pool)
    val_gpu_tensors = (
        vb.to(device, non_blocking=True),
        vgi.to(device, non_blocking=True),
        vp.to(device, non_blocking=True),
        vgt.to(device, non_blocking=True),
        vo.to(device, non_blocking=True),
    )
    del vb, vgi, vp, vgt, vo
    print(f"[Data] Validation set ready in VRAM ({v_sec:.2f}s).", flush=True)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return max_lr * ((step + 1) / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cos_f = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cos_f

    t_train_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        init_chunk = start_chunk if epoch == start_epoch else 0
        rng = random.Random(202602 + epoch)
        epoch_shards = list(train_shards)
        rng.shuffle(epoch_shards)

        chunks = [epoch_shards[i : i + shards_per_chunk] for i in range(0, len(epoch_shards), shards_per_chunk)]

        # Double-buffered async macro-chunk prefetch queue (prefetches 2 chunks = 640,000 positions into RAM)
        prefetch_q = deque()
        for c_i in range(init_chunk, min(init_chunk + 2, len(chunks))):
            prefetch_q.append((c_i, prefetch_executor.submit(load_macro_chunk, chunks[c_i], io_pool)))

        for chunk_idx in range(init_chunk, len(chunks)):
            _, fut = prefetch_q.popleft()
            cb, cgi, cp, cgt, cvo, load_sec = fut.result()

            if chunk_idx + 2 < len(chunks):
                prefetch_q.append(
                    (chunk_idx + 2, prefetch_executor.submit(load_macro_chunk, chunks[chunk_idx + 2], io_pool))
                )

            g_bin = cb.to(device, non_blocking=True)
            g_gin = cgi.to(device, non_blocking=True)
            g_pol = cp.to(device, non_blocking=True)
            g_gt = cgt.to(device, non_blocking=True)
            g_own = cvo.to(device, non_blocking=True)
            del cb, cgi, cp, cgt, cvo

            N_chunk = g_bin.shape[0]
            perm = torch.randperm(N_chunk, device=device)
            n_chunk_steps = N_chunk // batch_size

            model.train()
            chunk_t0 = time.perf_counter()
            running_loss_t = torch.zeros((), device=device, dtype=torch.float32)
            running_p0_t = torch.zeros((), device=device, dtype=torch.float32)
            running_val_t = torch.zeros((), device=device, dtype=torch.float32)
            running_own_t = torch.zeros((), device=device, dtype=torch.float32)
            running_top1_t = torch.zeros((), device=device, dtype=torch.int64)

            for s_idx in range(n_chunk_steps):
                lr = get_lr(global_step)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr
                optimizer.param_groups[2]["lr"] = lr * 2.0

                idx = perm[s_idx * batch_size : (s_idx + 1) * batch_size]
                spatial, global_x, t_pol, t_wld, t_score, t_own = unpack_and_transform_batch(
                    g_bin[idx],
                    g_gin[idx],
                    g_pol[idx],
                    g_gt[idx],
                    g_own[idx],
                    bit_shifts,
                    perm_361,
                    perm_362,
                    apply_d4=True,
                )

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pol_logits, wld_logits, score_pred, own_pred = model(spatial, global_x)
                    lp0 = F.log_softmax(pol_logits[:, 0].float(), dim=-1)
                    lp1 = F.log_softmax(pol_logits[:, 1].float(), dim=-1)

                    loss_p0 = -(t_pol[:, 0] * lp0).sum(dim=-1).mean()
                    loss_p1 = -(t_pol[:, 1] * lp1).sum(dim=-1).mean()
                    loss_val = -(t_wld * F.log_softmax(wld_logits.float(), dim=-1)).sum(dim=-1).mean()
                    loss_score = F.huber_loss(score_pred.float(), t_score)
                    own_mask = (t_own.abs().amax(dim=(1, 2)) > 0.0).to(torch.float32)
                    own_sq = (own_pred.float() - t_own).pow(2).mean(dim=(1, 2))
                    loss_own = (own_sq * own_mask).sum() / own_mask.sum().clamp(min=1.0)

                    loss = loss_p0 + 0.15 * loss_p1 + 1.2 * loss_val + 0.4 * loss_score + 0.4 * loss_own

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                optimizer.step()

                global_step += 1

                if epoch == start_epoch and chunk_idx == init_chunk and s_idx == 0:
                    torch.cuda.synchronize()
                    comp_dt = time.perf_counter() - chunk_t0
                    vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    print(
                        f"[Compile] Step {global_step} compiled & executed in {comp_dt:.1f}s | "
                        f"Loss: {loss.item():.4f} (Pol:{loss_p0.item():.3f} Val:{loss_val.item():.3f} Own:{loss_own.item():.3f}) | "
                        f"VRAM: {vram_gb:.1f}G",
                        flush=True,
                    )
                    chunk_t0 = time.perf_counter()

                if global_step % 5 == 0:
                    ema_cap = 0.9995 if global_step > int(0.75 * total_steps) else 0.9992
                    ema_decay = min(ema_cap, (1.0 + global_step) / (20.0 + global_step))
                    torch._foreach_lerp_(ema_params, model_params, 1.0 - ema_decay)

                with torch.no_grad():
                    running_loss_t += loss.detach()
                    running_p0_t += loss_p0.detach()
                    running_val_t += loss_val.detach()
                    running_own_t += loss_own.detach()
                    running_top1_t += (pol_logits[:, 0].argmax(dim=-1) == t_pol[:, 0].argmax(dim=-1)).sum()

                if (s_idx + 1) % 104 == 0 or (s_idx + 1) == n_chunk_steps:
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - chunk_t0
                    done_steps = s_idx + 1
                    timed_steps = max(1, done_steps - 1) if (epoch == start_epoch and chunk_idx == init_chunk) else done_steps
                    sps = (timed_steps * batch_size) / max(dt, 1e-4)
                    avg_l = running_loss_t.item() / done_steps
                    avg_p0 = running_p0_t.item() / done_steps
                    avg_v = running_val_t.item() / done_steps
                    avg_o = running_own_t.item() / done_steps
                    avg_top1 = running_top1_t.item() / (done_steps * batch_size) * 100.0
                    vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    print(
                        f"[Ep {epoch+1}/{num_epochs} | Chunk {chunk_idx+1:03d}/{len(chunks):03d} | Step {global_step:06d}/{total_steps:06d}] "
                        f"Loss: {avg_l:.4f} (Pol:{avg_p0:.3f} Val:{avg_v:.3f} Own:{avg_o:.3f}) | "
                        f"Top1: {avg_top1:5.2f}% | LR: {lr:.2e} | Speed: {sps:,.0f} pos/s | VRAM: {vram_gb:.1f}G",
                        flush=True,
                    )

            del g_bin, g_gin, g_pol, g_gt, g_own

            if chunk_idx == init_chunk or chunk_idx == 0 or (chunk_idx + 1) % 3 == 0 or (chunk_idx + 1) == len(chunks):
                val_metrics = evaluate_validation(
                    model, raw_model, ema_params, val_gpu_tensors, bit_shifts, perm_361, perm_362, batch_size=batch_size
                )
                rope_rms = float(
                    torch.stack([a.rope_freqs.detach().norm() for b in raw_model.blocks for a in b.attn_layers]).mean().item()
                )
                print(
                    f"  >>> [VAL-EMA @ Step {global_step:06d}] "
                    f"Loss: {val_metrics['val_loss']:.4f} (Pol:{val_metrics['val_p0_loss']:.3f} WLD:{val_metrics['val_wld_loss']:.3f} OwnMSE:{val_metrics['val_own_mse']:.3f}) | "
                    f"Move Top1: {val_metrics['val_top1_acc']:.2f}% Top3: {val_metrics['val_top3_acc']:.2f}% | "
                    f"WLD Acc: {val_metrics['val_wld_acc']:.2f}% | Score MAE: {val_metrics['val_score_mae_pts']:.2f} pts | RoPE_Norm: {rope_rms:.3f}",
                    flush=True,
                )

                record = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": "turbogo_mini_b6c192_nbt12_100m",
                    "epoch": epoch + 1,
                    "chunk": chunk_idx + 1,
                    "global_step": global_step,
                    "lr": round(lr, 7),
                    "lr_trunk": round(lr, 7),
                    "res_gate": round(rope_rms, 4),
                    "elapsed_min": round((time.perf_counter() - t_train_start) / 60.0, 2),
                    **val_metrics,
                }
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(record) + "\n")

                composite_score = round(
                    val_metrics["val_p0_loss"] + 0.5 * val_metrics["val_wld_loss"] + 2.0 * val_metrics["val_own_mse"], 4
                )
                cfg_meta = {
                    "arch": "turbogo_v2_nbt22",
                    "version": "turbogo_mini_b6c192_nbt12",
                    "dataset_size": "100m",
                    "d_model": 192,
                    "c_mid": 128,
                    "n_blocks": 6,
                    "internal_length": 2,
                    "n_heads": 4,
                    "ffn_dim": 256,
                    "in_channels": 32,
                    "params": num_params,
                }
                ema_state_dict = {}
                param_map = {name: ep for (name, _), ep in zip(raw_model.named_parameters(), ema_params)}
                for k, v in raw_model.state_dict().items():
                    ema_state_dict[k] = param_map[k].clone() if k in param_map else v.clone()

                ckpt_payload = {
                    "model": raw_model.state_dict(),
                    "ema_model": ema_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "chunk": chunk_idx + 1,
                    "global_step": global_step,
                    "best_val_loss": min(best_val_loss, composite_score),
                    "val_metrics": val_metrics,
                    "config": cfg_meta,
                }
                torch.save(ckpt_payload, latest_ckpt)
                is_new_best = (composite_score <= best_val_loss) or (val_metrics["val_top1_acc"] >= best_top1_acc)
                if is_new_best:
                    best_val_loss = min(best_val_loss, composite_score)
                    best_top1_acc = max(best_top1_acc, val_metrics["val_top1_acc"])
                    torch.save(ckpt_payload, best_ckpt)
                    bf16_ema = {
                        k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu())
                        for k, v in ema_state_dict.items()
                    }
                    weights_payload = {
                        "model": bf16_ema,
                        "epoch": epoch + 1,
                        "chunk": chunk_idx + 1,
                        "global_step": global_step,
                        "composite_score": composite_score,
                        "val_metrics": val_metrics,
                        "config": cfg_meta,
                    }
                    tmp_bw = best_weights_ckpt.with_suffix(".pt.tmp")
                    torch.save(weights_payload, tmp_bw)
                    os.replace(tmp_bw, best_weights_ckpt)

                if (chunk_idx + 1) < len(chunks):
                    gdrive_sync_executor.submit(
                        cloud_sync_to_gdrive,
                        metrics_path,
                        best_weights_ckpt if is_new_best else None,
                    )

        torch.save(ckpt_payload, ckpt_dir / "turbogo_mini_final.pt")
        cloud_sync_to_gdrive(metrics_path, best_weights_ckpt)
        print(f"[Completed] Finished TurboGo-Mini 100M Pretraining at Step {global_step}!", flush=True)


if __name__ == "__main__":
    main()
