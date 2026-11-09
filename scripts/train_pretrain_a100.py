#!/usr/bin/env python3
"""
Full-Scale A100-SXM4-40GB Pretraining Pipeline for TurboGo-v2 (`b11c384-nbt22`, ~20.15M params)
on 60,000,000 KataGo 19x19 Chinese-Rules Positions (From Scratch)

Key Engineering Optimizations:
1. Architecture: 11 Nested Bottleneck Transformer (`TurboNBTBlock`) macro-blocks = **22 Global Self-Attention
   Layers + 22 SwiGLU FFN Layers** (`c_main=384, c_mid=256, n_heads=8`) + 11 Depthwise 3x3 / KataGo Global Pool
   local-global mixers + Learnable Multi-Scale 2D-RoPE (`rope_freqs`) + 100% Pure Identity Residual Highway.
2. Zero-Idle Async Macro-Chunk Prefetching: While the A100 GPU trains on Chunk N, a background worker thread
   decompresses Chunk N+1 (16 `.npz` shards = 320,000 positions) into RAM, eliminating inter-chunk GPU idle time.
3. GPU-Resident Bit-Unpacking & Balanced 8x D4 Symmetry Augmentation (`torch.gather` across all 8 symmetries per batch).
4. Fused GPU Exponential Moving Average (`torch._foreach_lerp_`, `ema_decay=0.9995`) evaluated on the 40,000-position
   validation benchmark and exported to `b12c384_best_weights.pt`.
5. Proper Transformer Weight Decay Separation: `weight_decay=1e-4` on 2D/4D Linear/Conv weights, `0.0` on norms,
   biases, and learnable 2D-RoPE frequencies (`rope_freqs`).
"""

import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import TurboGoHybridTransformer, augment_tactical_planes_32

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def build_d4_symmetry_tables(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute index gather permutations for all 8 dihedral (D4) symmetries of a 19x19 Go board.
    Returns:
      perm_361: (8, 361) int64 index tensor for 19x19 spatial tensors
      perm_362: (8, 362) int64 index tensor for 362-dim policy vectors (361 board + 1 pass)
    """
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
    """
    Unpack bit-packed binaryInputNCHWPacked (B, 22, 46) on GPU, expand to 32 tactical planes,
    normalize policy/value/ownership targets, and apply balanced 8-way D4 symmetry augmentation.
    """
    B = b_packed.shape[0]
    bits = (b_packed.unsqueeze(-1) >> bit_shifts) & 1
    spatial = bits.reshape(B, 22, 368)[:, :, :361].reshape(B, 22, 19, 19).to(torch.bfloat16)

    global_x = g_in.to(torch.bfloat16)

    p_float = p_vis.to(torch.float32).clamp(min=0.0)
    p_sum = p_float.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    target_pol = p_float / p_sum  # (B, 2, 362)

    target_wld = g_tgt[:, 0:3].to(torch.float32)
    target_wld = target_wld / target_wld.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    target_score = (g_tgt[:, 3].to(torch.float32) / 20.0).clamp(-3.0, 3.0)
    target_own = v_own[:, 0].to(torch.float32)  # (B, 19, 19)

    if apply_d4 and B >= 8 and (B % 8 == 0):
        # Assign each 1/8th slice of the batch to one of the 8 D4 symmetries in a single fused gather on 22 channels
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
    """Load `len(shard_paths)` shards in parallel on CPU threads and concatenate into pinned-ready tensors."""
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
    batch_size: int = 512,
) -> Dict[str, float]:
    """Run full-metrics validation across all 40,000 validation positions using EMA weights."""
    # Swap EMA weights into raw_model for evaluation
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

    # Restore live training weights back into raw_model
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
    """Directly PATCH metrics.jsonl and b12c384_best_weights.pt from Colab VM to Google Drive."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://172.28.0.1:8008/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        tok = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())["access_token"]
        if metrics_path.exists():
            m_req = urllib.request.Request(
                "https://www.googleapis.com/upload/drive/v3/files/12z6xEZHx0b0ObExXVt2_XbIXYDQcz_3f?uploadType=media",
                data=metrics_path.read_bytes(),
                method="PATCH",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            )
            urllib.request.urlopen(m_req, timeout=15).read()
        if best_weights_path is not None and best_weights_path.exists():
            t0 = time.perf_counter()
            bw_bytes = best_weights_path.read_bytes()
            b_req = urllib.request.Request(
                "https://www.googleapis.com/upload/drive/v3/files/1rN7Qvh9Oq3gTjkTkzuSpY-ho91Ctj8Ge?uploadType=media",
                data=bw_bytes,
                method="PATCH",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/octet-stream"},
            )
            urllib.request.urlopen(b_req, timeout=60).read()
            print(
                f"  >>> [Cloud-GDrive] Uploaded b12c384_best_weights.pt ({len(bw_bytes)/1e6:.1f} MB) + metrics.jsonl in {time.perf_counter()-t0:.2f}s",
                flush=True,
            )
    except Exception as e:
        print(f"  [Cloud-GDrive Warn] {e}", flush=True)


def main():
    data_dir_10m = Path("/content/data/katago_19x19_chinese_10m")
    data_dir_50m = Path("/content/data/katago_19x19_chinese_50m")
    data_dir_40m = Path("/content/data/katago_19x19_chinese_40m")
    ckpt_dir = Path("/content/checkpoints/b12c384")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    shards_10m = [data_dir_10m / f"shard_{i:05d}.npz" for i in range(500)]
    shards_50m = [data_dir_50m / f"shard_{i:05d}.npz" for i in range(2500)]
    shards_40m = [data_dir_40m / f"shard_{i:05d}.npz" for i in range(2000)]

    # Keep the exact same 40,000-position validation split (shards 498..499 of 10M) for 100% benchmark comparability
    val_shards = shards_10m[498:500]
    train_shards_60m = shards_10m[:498] + shards_50m
    train_shards = train_shards_60m + shards_40m  # 4,998 shards = 99,960,000 unique positions (313 chunks)

    batch_size = int(os.environ.get("BATCH_SIZE", "768"))
    num_epochs = int(os.environ.get("EPOCHS", "2"))
    max_lr = float(os.environ.get("MAX_LR", "1.15e-3"))
    min_lr = float(os.environ.get("MIN_LR", "1.5e-5"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "1000"))
    stop_at_utc = os.environ.get("STOP_AT_UTC", "2026-10-29 14:30:00")  # 2026-10-29 22:30:00 CST
    shards_per_chunk = 16  # 320,000 positions per GPU-resident macro-chunk (313 chunks per 100M epoch)

    device = torch.device("cuda:0")
    print("=" * 88, flush=True)
    print("TurboGo-v2 (b11c384-nbt22, 22-Layer Global Self-Attention + SwiGLU + 2D-RoPE, 20.82M params)", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(device)} | VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GiB", flush=True)
    print(f"Dataset: {len(train_shards):,} train shards ({len(train_shards)*20000:,} pos) + {len(val_shards)} val shards ({len(val_shards)*20000:,} pos)", flush=True)
    print(f"Hyperparams: BS={batch_size} | StopAt={stop_at_utc} UTC (10:30 CST) | EMA=0.9992", flush=True)
    print("=" * 88, flush=True)

    raw_model = TurboGoHybridTransformer(
        d_model=384, c_mid=256, n_blocks=11, internal_length=2, n_heads=8, ffn_dim=640, in_channels=32
    ).to(device)
    num_params = sum(p.numel() for p in raw_model.parameters())
    n_attn = sum(len(b.attn_layers) for b in raw_model.blocks)
    n_ffn = sum(len(b.ffn_layers) for b in raw_model.blocks)
    print(
        f"[Init] TurboGo-v2 Parameters: {num_params:,} ({num_params / 1e6:.2f}M) | "
        f"NBT Macro-Blocks: {len(raw_model.blocks)} | Global Attention Layers: {n_attn} | SwiGLU FFN Layers: {n_ffn}",
        flush=True,
    )

    bit_shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    perm_361, perm_362 = build_d4_symmetry_tables(device)

    # Proper Transformer weight decay separation + 2.0x LR on 2D-RoPE frequencies (`rope_freqs`)
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

    # Target step count calibrated to reach min_lr right at the end of Epoch 2 (Chunk 313/313 = Step 212,454)
    total_steps = int(os.environ.get("TOTAL_STEPS", "212454"))

    start_epoch = 0
    start_chunk = 0
    global_step = 0
    resume_step = 0
    resumed_without_optim = False
    best_val_loss = float("inf")
    best_top1_acc = 0.0

    # Initialize GPU-resident EMA parameter list
    model_params = list(raw_model.parameters())
    ema_params = [p.detach().clone() for p in model_params]

    latest_ckpt = ckpt_dir / "b12c384_latest.pt"
    best_ckpt = ckpt_dir / "b12c384_best.pt"
    best_weights_ckpt = ckpt_dir / "b12c384_best_weights.pt"
    best_p0_loss = float("inf")
    # For fast cooldown, unconditionally resume from latest_ckpt so we preserve the latest step & optimizer state
    load_ckpt_path = latest_ckpt if latest_ckpt.exists() else (best_ckpt if best_ckpt.exists() else best_weights_ckpt)

    if load_ckpt_path.exists():
        print(f"[Resume] Loading v2 checkpoint from {load_ckpt_path}...", flush=True)
        ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("config", {}).get("arch") == "turbogo_v2_nbt22":
            sd = {
                k: (v.float() if v.is_floating_point() else v)
                for k, v in ckpt["model"].items()
            }
            raw_model.load_state_dict(sd)
            if "ema_model" in ckpt:
                ema_sd = ckpt["ema_model"]
                for (name, _), ep in zip(raw_model.named_parameters(), ema_params):
                    if name in ema_sd:
                        ep.copy_(ema_sd[name].to(device))
            else:
                ema_params = [p.detach().clone() for p in model_params]
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt.get("epoch", 0)
            else:
                resumed_without_optim = True
                start_epoch = max(0, ckpt.get("epoch", 1) - 1)
            start_chunk = ckpt.get("chunk", 0)
            total_chunks_now = (len(train_shards) + shards_per_chunk - 1) // shards_per_chunk
            if start_chunk >= total_chunks_now:
                start_epoch += 1
                start_chunk = 0
            global_step = ckpt.get("global_step", 0)
            resume_step = global_step
            vm = ckpt.get("val_metrics", {})
            best_p0_loss = vm.get("val_p0_loss", float("inf"))
            best_val_loss = round(
                vm.get("val_p0_loss", 2.133) + 0.5 * vm.get("val_wld_loss", 0.693) + 2.0 * vm.get("val_own_mse", 0.493), 4
            )
            best_top1_acc = vm.get("val_top1_acc", 0.0)
            print(
                f"[Resume] Resumed TurboGo-v2 on 100M dataset for 6-Chunk Fast Cooldown at epoch={start_epoch+1}, "
                f"chunk={start_chunk}/{total_chunks_now}, global_step={global_step}, "
                f"init_composite={best_val_loss:.4f}, init_p0={best_p0_loss:.4f}, init_top1={best_top1_acc:.2f}%",
                flush=True,
            )

    print("[Compile] Compiling TurboGo-v2 (22-Layer NBT) with TorchInductor (Triton)...", flush=True)
    model = torch.compile(raw_model)

    io_pool = ThreadPoolExecutor(max_workers=16)
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    gdrive_sync_executor = ThreadPoolExecutor(max_workers=1)

    # Preload 40,000 validation positions directly to GPU VRAM (~115 MB)
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

    # 6-Chunk (2,496-step, ~13-min) Fast Cosine Cooldown: LR 2.50e-4 -> 1.00e-5
    cooldown_chunks = int(os.environ.get("COOLDOWN_CHUNKS", "6"))
    cooldown_steps = cooldown_chunks * ((shards_per_chunk * 20000) // batch_size)
    total_steps = resume_step + cooldown_steps
    cooldown_start_lr = 2.50e-4
    cooldown_min_lr = 1.00e-5

    def get_lr(step: int) -> float:
        cd_progress = min(1.0, max(0.0, (step - resume_step) / max(1, total_steps - resume_step)))
        cos_f = 0.5 * (1.0 + math.cos(math.pi * cd_progress))
        return cooldown_min_lr + (cooldown_start_lr - cooldown_min_lr) * cos_f

    t_train_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        init_chunk = start_chunk if epoch == start_epoch else 0
        if epoch == 1 and len(shards_40m) > 0:
            # Keep the exact 34-chunk split point from Epoch 2 (when 100M was introduced) so chunks[120..312]
            # are 100% identical to last night's permutation with 0% overlap with chunks[0..119]!
            rng_60m = random.Random(202602 + epoch)
            ep_60m = list(train_shards_60m)
            rng_60m.shuffle(ep_60m)
            split_idx = 34 * shards_per_chunk
            done_shards = ep_60m[:split_idx]
            rem_shards = ep_60m[split_idx:] + list(shards_40m)
            rng_100m = random.Random(2026100 + epoch)
            rng_100m.shuffle(rem_shards)
            epoch_shards = done_shards + rem_shards
        else:
            rng = random.Random(202602 + epoch)
            epoch_shards = list(train_shards)
            rng.shuffle(epoch_shards)

        chunks = [epoch_shards[i : i + shards_per_chunk] for i in range(0, len(epoch_shards), shards_per_chunk)]

        # Kick off prefetch of the first chunk
        next_future = prefetch_executor.submit(load_macro_chunk, chunks[init_chunk], io_pool)

        for chunk_idx in range(init_chunk, len(chunks)):
            cb, cgi, cp, cgt, cvo, load_sec = next_future.result()

            # Immediately launch background prefetch for chunk_idx + 1 while GPU trains on chunk_idx!
            if chunk_idx + 1 < len(chunks):
                next_future = prefetch_executor.submit(load_macro_chunk, chunks[chunk_idx + 1], io_pool)

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

                # Fused GPU EMA update every 5 steps (responsive 0.994 -> 0.997 during 6-chunk cooldown)
                if global_step % 5 == 0:
                    cd_prog = min(1.0, max(0.0, (global_step - resume_step) / max(1, total_steps - resume_step)))
                    ema_decay = 0.994 + 0.003 * cd_prog
                    torch._foreach_lerp_(ema_params, model_params, 1.0 - ema_decay)

                with torch.no_grad():
                    running_loss_t += loss.detach()
                    running_p0_t += loss_p0.detach()
                    running_val_t += loss_val.detach()
                    running_own_t += loss_own.detach()
                    running_top1_t += (pol_logits[:, 0].argmax(dim=-1) == t_pol[:, 0].argmax(dim=-1)).sum()

                if (s_idx + 1) % 120 == 0 or (s_idx + 1) == n_chunk_steps:
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

            reached_deadline = (global_step >= total_steps) or (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) >= stop_at_utc)

            if True:
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
                    "phase": "turbogo_v2_nbt22_60m",
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
                    "version": "v2_100m",
                    "dataset_size": "100m",
                    "d_model": 384,
                    "c_mid": 256,
                    "n_blocks": 11,
                    "internal_length": 2,
                    "n_heads": 8,
                    "ffn_dim": 640,
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
                is_new_best = (
                    (composite_score <= best_val_loss)
                    or (val_metrics["val_p0_loss"] <= best_p0_loss)
                    or (val_metrics["val_top1_acc"] >= best_top1_acc)
                )
                if is_new_best:
                    best_val_loss = min(best_val_loss, composite_score)
                    best_p0_loss = min(best_p0_loss, val_metrics["val_p0_loss"])
                    best_top1_acc = max(best_top1_acc, val_metrics["val_top1_acc"])
                    torch.save(ckpt_payload, ckpt_dir / "b12c384_best.pt")
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
                    tmp_bw = ckpt_dir / "b12c384_best_weights.pt.tmp"
                    torch.save(weights_payload, tmp_bw)
                    os.replace(tmp_bw, ckpt_dir / "b12c384_best_weights.pt")

                if not reached_deadline:
                    gdrive_sync_executor.submit(
                        cloud_sync_to_gdrive,
                        metrics_path,
                        (ckpt_dir / "b12c384_best_weights.pt") if is_new_best else None,
                    )

            if reached_deadline:
                torch.save(ckpt_payload, ckpt_dir / "b12c384_v2_final.pt")
                cloud_sync_to_gdrive(metrics_path, ckpt_dir / "b12c384_best_weights.pt")
                print(f"[Completed] Finished 6-Chunk Fast Cooldown at Step {global_step}!", flush=True)
                return

        torch.save(ckpt_payload, ckpt_dir / f"b12c384_epoch_{epoch+1:02d}.pt")


if __name__ == "__main__":
    main()
