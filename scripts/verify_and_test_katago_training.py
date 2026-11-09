#!/usr/bin/env python3
"""
Unpack and verify KataGo 10M self-play dataset shards on NVIDIA A100-SXM4-40GB.
Tests:
1. Archive & array structure verification (7 standard KataGo v3 fields).
2. Spatial bit-unpacking verification: binaryInputNCHWPacked (N, 22, 46) uint8 -> (N, 22, 19, 19) float32.
3. Target scaling & normalization verification (Policy MCTS visits, Value W/L/D, Score lead, Score PDF, Ownership, Q-values).
4. Live PyTorch + A100 GPU BFloat16 training benchmark (DataLoader + on-GPU bit-unpacking + multi-head loss + backward step).
"""

import time
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KataGoNPZDataset(torch.utils.data.Dataset):
    """Memory-efficient Dataset for a single unpacked KataGo .npz shard (20,000 positions)."""

    def __init__(self, npz_path: Path):
        t0 = time.perf_counter()
        with np.load(npz_path) as data:
            self.binary_packed = data["binaryInputNCHWPacked"]      # (N, 22, 46) uint8
            self.global_input = data["globalInputNC"]               # (N, 19) float32
            self.policy_targets = data["policyTargetsNCMove"]       # (N, 2, 362) int16
            self.global_targets = data["globalTargetsNC"]           # (N, 80) float32
            self.score_distr = data["scoreDistrN"]                  # (N, 842) int8
            self.value_spatial = data["valueTargetsNCHW"]           # (N, 5, 19, 19) int8
            self.q_targets = data["qValueTargetsNCMove"]            # (N, 3, 362) int16
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

    def __len__(self):
        return len(self.binary_packed)


def unpack_binary_planes_gpu(packed_uint8: torch.Tensor, bit_shifts: torch.Tensor) -> torch.Tensor:
    """
    Fast on-GPU bit-unpacking of KataGo binaryInputNCHWPacked:
      Input:  (B, 22, 46) uint8 on CUDA
      Output: (B, 22, 19, 19) bfloat16/float32 on CUDA
    """
    # (B, 22, 46, 1) >> (8,) -> (B, 22, 46, 8) -> (B, 22, 368) -> slice :361 -> (B, 22, 19, 19)
    bits = ((packed_uint8.unsqueeze(-1) >> bit_shifts) & 1).reshape(packed_uint8.shape[0], 22, 368)
    return bits[:, :, :361].reshape(packed_uint8.shape[0], 22, 19, 19)


class KataGoPretrainProbeNet(nn.Module):
    """KataGo-style multi-head network (Spatial + Global fusion -> Policy, Value, Score, Ownership)."""

    def __init__(self, channels: int = 256, num_blocks: int = 6):
        super().__init__()
        self.conv_in = nn.Conv2d(22, channels, kernel_size=3, padding=1, bias=False)
        self.fc_global_in = nn.Linear(19, channels, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Policy head: outputs 2 channels (0: main policy, 1: opponent reply auxiliary policy) over 362 moves
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 19 * 19, 2 * 362)

        # Value & Score head: outputs W/L/NoResult logits (3) + expected score lead (1)
        self.value_fc1 = nn.Linear(channels * 2, 128)
        self.value_fc2 = nn.Linear(128, 3)
        self.score_fc = nn.Linear(128, 1)

        # Ownership head: outputs (B, 1, 19, 19) in [-1, 1] via tanh
        self.ownership_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, spatial_x: torch.Tensor, global_x: torch.Tensor):
        h = F.silu(self.bn_in(self.conv_in(spatial_x) + self.fc_global_in(global_x)[:, :, None, None]))
        for blk in self.blocks:
            h = F.silu(h + blk(h))

        # Policy
        p = F.silu(self.policy_conv(h)).flatten(1)
        policy_logits = self.policy_fc(p).view(-1, 2, 362)

        # Global pooling (mean + max) for Value & Score
        g_pool = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        v_hidden = F.silu(self.value_fc1(g_pool))
        value_logits = self.value_fc2(v_hidden)
        score_pred = self.score_fc(v_hidden).squeeze(-1)

        # Ownership
        ownership_pred = torch.tanh(self.ownership_conv(h)).squeeze(1)
        return policy_logits, value_logits, score_pred, ownership_pred


def main():
    shard_path = Path("/content/data/katago_19x19_chinese_10m/shard_00000.npz")
    print(f"=== 1. Unpacking & Inspecting {shard_path} ===")
    with zipfile.ZipFile(shard_path, "r") as zf:
        uncompressed_bytes = sum(i.file_size for i in zf.infolist())
        compressed_bytes = shard_path.stat().st_size
    print(
        f"Shard size: {compressed_bytes / 1e6:.2f} MB compressed -> "
        f"{uncompressed_bytes / 1e6:.2f} MB uncompressed ({uncompressed_bytes / compressed_bytes:.1f}x ratio)"
    )

    ds = KataGoNPZDataset(shard_path)
    print(f"Unpacked {len(ds):,} positions into RAM in {ds.load_time_ms:.1f} ms")

    # CPU verification of bit-unpacking vs GPU bit-unpacking
    cpu_bits = np.unpackbits(ds.binary_packed[:100], axis=-1)[:, :, :361].reshape(100, 22, 19, 19)
    on_board_ones = (cpu_bits[:, 0] == 1).all()
    trailing_pad_zeros = (np.unpackbits(ds.binary_packed[:100], axis=-1)[:, :, 361:] == 0).all()
    chinese_rules_ok = (ds.global_input[:, 9] == 0.0).all() and (ds.global_input[:, 10] == 0.0).all()
    pol0_sums = ds.policy_targets[:, 0, :].astype(np.int32).sum(axis=1)
    pol1_sums = ds.policy_targets[:, 1, :].astype(np.int32).sum(axis=1)
    wld_sums = ds.global_targets[:, 0:3].sum(axis=1)
    score_distr_sums = ds.score_distr.astype(np.int32).sum(axis=1)

    print("\n=== 2. Data Integrity & Normalization Checks ===")
    print(f"  - 19x19 On-Board Mask (Plane 0 all 1s):      {on_board_ones}")
    print(f"  - Bit-Packing 7 Trailing Bits all 0s:        {trailing_pad_zeros}")
    print(f"  - Standard Chinese Rules (Area + No Tax):    {chinese_rules_ok}")
    print(f"  - Main Policy Visits (ch 0) sum > 0:         {(pol0_sums > 0).all()} (min={pol0_sums.min()}, max={pol0_sums.max()})")
    print(f"  - Aux Policy Visits (ch 1) sum > 0:          {(pol1_sums > 0).all()} (min={pol1_sums.min()}, max={pol1_sums.max()})")
    print(f"  - Value Target W/L/NR (ch 0..2) sum == 1.0:  {np.allclose(wld_sums, 1.0, atol=1e-3)} (mean W={ds.global_targets[:,0].mean():.3f}, L={ds.global_targets[:,1].mean():.3f})")
    print(f"  - Score Distr PDF (842 bins) sum == 100:     {(score_distr_sums == 100).all()}")
    print(f"  - Ownership Target (valueTargets ch 0):      values in {np.unique(ds.value_spatial[:, 0]).tolist()}")

    print("\n=== 3. Live PyTorch A100-SXM4-40GB Training Benchmark ===")
    device = torch.device("cuda:0")
    print(f"  - GPU Device: {torch.cuda.get_device_name(device)}")

    # Move entire shard to GPU or stream batches
    packed_t = torch.from_numpy(ds.binary_packed).to(device, non_blocking=True)
    glob_in_t = torch.from_numpy(ds.global_input).to(device, non_blocking=True)
    pol_tgt_t = torch.from_numpy(ds.policy_targets).to(device, non_blocking=True)
    glob_tgt_t = torch.from_numpy(ds.global_targets).to(device, non_blocking=True)
    own_tgt_t = torch.from_numpy(ds.value_spatial[:, 0]).to(device, non_blocking=True)
    bit_shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)

    # Verify GPU bit-unpacking matches numpy.unpackbits 100%
    gpu_bits = unpack_binary_planes_gpu(packed_t[:100], bit_shifts).cpu().numpy()
    assert np.array_equal(cpu_bits, gpu_bits), "GPU bit-unpacking mismatch!"
    print("  - GPU vs CPU Bit-Unpacking Exact Match:      True")

    model = KataGoPretrainProbeNet(channels=256, num_blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch_size = 512
    num_steps = 30
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    initial_loss = None
    final_loss = None

    for step in range(num_steps):
        idx = torch.randint(0, len(ds), (batch_size,), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            spatial_x = unpack_binary_planes_gpu(packed_t[idx], bit_shifts).to(torch.bfloat16)
            global_x = glob_in_t[idx].to(torch.bfloat16)

            # Normalize policy targets from int16 visit counts to float probabilities
            raw_pol = pol_tgt_t[idx].float().clamp(min=0)
            target_pol = raw_pol / raw_pol.sum(dim=-1, keepdim=True).clamp(min=1.0)

            target_wld = glob_tgt_t[idx, 0:3].float()
            target_score = (glob_tgt_t[idx, 3].float() / 20.0)  # scaled by 20 points
            target_own = own_tgt_t[idx].float()

            pol_logits, val_logits, score_pred, own_pred = model(spatial_x, global_x)

            loss_p0 = -(target_pol[:, 0] * F.log_softmax(pol_logits[:, 0].float(), dim=-1)).sum(dim=-1).mean()
            loss_p1 = -(target_pol[:, 1] * F.log_softmax(pol_logits[:, 1].float(), dim=-1)).sum(dim=-1).mean()
            loss_val = -(target_wld * F.log_softmax(val_logits.float(), dim=-1)).sum(dim=-1).mean()
            loss_score = F.huber_loss(score_pred.float(), target_score)
            loss_own = F.mse_loss(own_pred.float(), target_own)

            total_loss = loss_p0 + 0.15 * loss_p1 + 1.5 * loss_val + 0.5 * loss_score + 0.5 * loss_own

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = (total_loss.item(), loss_p0.item(), loss_val.item(), loss_own.item())
        if step == num_steps - 1:
            final_loss = (total_loss.item(), loss_p0.item(), loss_val.item(), loss_own.item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    throughput = (num_steps * batch_size) / elapsed
    max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    print(f"  - Step 01 Loss: total={initial_loss[0]:.4f} (policy={initial_loss[1]:.4f}, value={initial_loss[2]:.4f}, own={initial_loss[3]:.4f})")
    print(f"  - Step 30 Loss: total={final_loss[0]:.4f} (policy={final_loss[1]:.4f}, value={final_loss[2]:.4f}, own={final_loss[3]:.4f})")
    print(f"  - A100 Throughput (BS={batch_size}, BF16): {throughput:,.0f} positions/sec | Peak VRAM: {max_vram_gb:.2f} GB / 40.00 GB")


if __name__ == "__main__":
    main()
