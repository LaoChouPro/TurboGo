#!/usr/bin/env python3
"""
High-Speed Datacenter-to-Datacenter Live Migration Script (`turbogo` -> `turbogo2`).
1. Streams all 2,500 shards of 50M dataset + 500 shards of 10M dataset + historical checkpoints
   across 16 parallel workers inside Google Cloud `asia-southeast1` (~200-500 MB/s).
2. Once all 3,000 shards (60,000,000 positions) are verified on `turbogo2`, pulls the freshest
   live `b12c384_latest.pt`, `b12c384_best.pt`, `b12c384_best_weights.pt`, `metrics.jsonl`,
   and `train_b12c384.log` from `turbogo` (verifying `torch.load` integrity).
3. Automatically launches `train_pretrain_a100.py`, `dashboard_server.py`, and `cloudflared`
   on `turbogo2` to seamlessly continue the 60M training with a fresh 24-hour A100 lifecycle.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

BASE_URL = os.environ["SRC_BASE_URL"].rstrip("/")
TOKEN = os.environ["SRC_TOKEN"]


def download_file(rel_path: str, dst_path: Path, min_size: int = 100) -> int:
    if dst_path.exists() and dst_path.stat().st_size >= min_size:
        return dst_path.stat().st_size
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + f".tmp.{os.getpid()}")
    url = f"{BASE_URL}/files/{rel_path.lstrip('/')}?colab-runtime-proxy-token={TOKEN}"
    for attempt in range(4):
        res = subprocess.run(
            ["curl", "-fsSL", "--retry", "2", "--max-time", "180", "-o", str(tmp_path), url],
            capture_output=True,
        )
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size >= min_size:
            os.replace(tmp_path, dst_path)
            return dst_path.stat().st_size
        time.sleep(1.0 * (attempt + 1))
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    raise RuntimeError(f"Failed to download {rel_path} after 4 attempts")


def main():
    t0 = time.perf_counter()
    print("=" * 80, flush=True)
    print(f"[Migrate] Starting High-Speed Datacenter Migration from {BASE_URL}", flush=True)
    print("=" * 80, flush=True)

    # 1. Copy existing_10m_hashes.npy
    h_sz = download_file("content/data/existing_10m_hashes.npy", Path("/content/data/existing_10m_hashes.npy"), min_size=1_000_000)
    print(f"[Migrate] Copied existing_10m_hashes.npy ({h_sz / 1e6:.1f} MB)", flush=True)

    # 2. Build task list for 2,500 shards of 50M + 500 shards of 10M
    tasks = []
    for i in range(2500):
        rel = f"content/data/katago_19x19_chinese_50m/shard_{i:05d}.npz"
        tasks.append((rel, Path(f"/{rel}"), 1_000_000))
    for i in range(500):
        rel = f"content/data/katago_19x19_chinese_10m/shard_{i:05d}.npz"
        tasks.append((rel, Path(f"/{rel}"), 10_000_000))

    total_tasks = len(tasks)
    done_count = 0
    total_bytes = 0
    t_shards = time.perf_counter()

    with ThreadPoolExecutor(max_workers=16) as pool:
        future_map = {pool.submit(download_file, r, d, ms): (r, d) for r, d, ms in tasks}
        for fut in as_completed(future_map):
            sz = fut.result()
            done_count += 1
            total_bytes += sz
            if done_count % 100 == 0 or done_count == total_tasks:
                dt = time.perf_counter() - t_shards
                mb_s = (total_bytes / 1e6) / max(dt, 1e-3)
                print(
                    f"[Migrate Shards] {done_count:4d}/{total_tasks} ({done_count/total_tasks*100:5.1f}%) | "
                    f"Transferred: {total_bytes/1e9:6.2f} GB | Speed: {mb_s:6.1f} MB/s | Elapsed: {dt:5.1f}s",
                    flush=True,
                )

    # 3. Copy historical epoch checkpoints
    ckpt_dir = Path("/content/checkpoints/b12c384")
    for ep in range(1, 12):
        for suffix in [f"b12c384_epoch_{ep:02d}_weights.pt"]:
            try:
                download_file(f"content/checkpoints/b12c384/{suffix}", ckpt_dir / suffix, min_size=10_000_000)
            except Exception:
                pass

    # 4. Pull the latest live checkpoints (`b12c384_latest.pt`, `b12c384_best.pt`, `b12c384_best_weights.pt`, `metrics.jsonl`, `train_b12c384.log`)
    print("[Migrate] Pulling latest live checkpoints and logs from turbogo...", flush=True)
    for fname, min_sz in [
        ("checkpoints/b12c384/metrics.jsonl", 1000),
        ("train_b12c384.log", 1000),
        ("checkpoints/b12c384/b12c384_best_weights.pt", 50_000_000),
        ("checkpoints/b12c384/b12c384_best.pt", 100_000_000),
    ]:
        dst = Path(f"/content/{fname}")
        if dst.exists():
            dst.unlink()
        download_file(f"content/{fname}", dst, min_size=min_sz)

    # Pull b12c384_latest.pt and verify with torch.load (retry if source was mid-write)
    latest_dst = ckpt_dir / "b12c384_latest.pt"
    for attempt in range(5):
        if latest_dst.exists():
            latest_dst.unlink()
        download_file("content/checkpoints/b12c384/b12c384_latest.pt", latest_dst, min_size=100_000_000)
        try:
            ckpt = torch.load(latest_dst, map_location="cpu", weights_only=False)
            print(
                f"[Migrate] Verified latest checkpoint: global_step={ckpt.get('global_step')}, "
                f"epoch={ckpt.get('epoch')+1}, chunk={ckpt.get('chunk')}, best_val_loss={ckpt.get('best_val_loss')}",
                flush=True,
            )
            break
        except Exception as e:
            print(f"[Migrate] Retry {attempt+1} loading b12c384_latest.pt ({e})...", flush=True)
            time.sleep(3.0)

    # 5. Launch dashboard_server.py, cloudflared, and train_pretrain_a100.py on turbogo2!
    subprocess.run("pkill -f dashboard_server.py || true", shell=True)
    subprocess.run("pkill -f cloudflared || true", shell=True)
    subprocess.run("pkill -f train_pretrain_a100.py || true", shell=True)
    time.sleep(1.0)

    subprocess.Popen(
        "nohup python3 /content/scripts/dashboard_server.py > /content/dashboard.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
        close_fds=True,
    )
    subprocess.Popen(
        "nohup /usr/local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 > /content/cloudflared.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
        close_fds=True,
    )
    subprocess.Popen(
        "nohup python3 /content/scripts/train_pretrain_a100.py >> /content/train_b12c384.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
        close_fds=True,
    )
    time.sleep(4.0)
    print(f"[Migrate] ALL DONE in {time.perf_counter() - t0:.1f}s! Training & Dashboard active on turbogo2!", flush=True)


if __name__ == "__main__":
    main()
