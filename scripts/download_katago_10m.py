#!/usr/bin/env python3
"""
Stream, filter, and pack 10,000,000 positions (data rows) from KataGo's latest
strongest neural network self-play archives (kata1).

Filter criteria (Plan A):
1. Latest strongest models (reverse chronological order from 2026-10-22 backwards).
2. 19x19 standard board size:
   - binaryInputNCHWPacked has shape (N, 22, 46)
   - Channel 0 (on-board mask) has all 361 bits set to 1:
     bytes 0..44 == 255 (360 bits) and byte 45 == 128 (361st bit).
3. Standard Chinese scoring system (Area Scoring + Tax None):
   - globalInputNC[:, 9] == 0.0 (Rules::SCORING_AREA)
   - globalInputNC[:, 10] == 0.0 (Rules::TAX_NONE, which also implies globalInputNC[:, 11] == 0.0)

Output format:
- 500 compressed NumPy shards (shard_00000.npz .. shard_00499.npz),
  each containing exactly 20,000 rows (total = 10,000,000 rows).
- Preserves all standard KataGo training fields (matching KataGo shuffle.py v3 spec):
  binaryInputNCHWPacked: (N, 22, 46) uint8
  globalInputNC:         (N, 19) float32
  policyTargetsNCMove:   (N, 2, 362) int16
  globalTargetsNC:       (N, 80) float32 (auto zero-padded from 64 if v2)
  scoreDistrN:           (N, 842) int8
  valueTargetsNCHW:      (N, 5, 19, 19) int8
  qValueTargetsNCMove:   (N, 3, 362) int16
"""

import os
import sys
import io
import json
import time
import tarfile
import re
import subprocess
import multiprocessing as mp
from pathlib import Path
import numpy as np

ARCHIVE_INDEX_URL = "https://katagoarchive.org/kata1/trainingdata/index.html"
ARCHIVE_BASE_URL = "https://katagoarchive.org/kata1/trainingdata/"

TARGET_ROWS = 10_000_000
ROWS_PER_SHARD = 20_000
TARGET_SHARDS = TARGET_ROWS // ROWS_PER_SHARD  # 500 shards

REQUIRED_BASE_KEYS = (
    "binaryInputNCHWPacked",
    "globalInputNC",
    "policyTargetsNCMove",
    "globalTargetsNC",
    "scoreDistrN",
    "valueTargetsNCHW",
)

ALL_OUTPUT_KEYS = REQUIRED_BASE_KEYS + ("qValueTargetsNCMove",)

EXPECTED_SHAPES = {
    "binaryInputNCHWPacked": (22, 46),
    "globalInputNC": (19,),
    "policyTargetsNCMove": (2, 362),
    "globalTargetsNC": (80,),
    "scoreDistrN": (842,),
    "valueTargetsNCHW": (5, 19, 19),
    "qValueTargetsNCMove": (3, 362),
}


def fetch_available_archives():
    """Fetch list of YYYY-MM-DDnpzs.tgz filenames in reverse chronological order."""
    out = subprocess.check_output(
        ["curl", "-4", "--interface", "en0", "-sL", "--retry", "3", ARCHIVE_INDEX_URL],
        text=True,
    )
    matches = re.findall(r'href="\./(\d{4}-\d{2}-\d{2}npzs\.tgz)"', out)
    unique_sorted = sorted(set(matches), reverse=True)
    return unique_sorted


def save_shard(output_dir: Path, shard_id: int, buffers: dict, n_rows: int):
    """Slice exactly ROWS_PER_SHARD rows from buffers, save atomically to shard_XXXXX.npz."""
    shard_data = {}
    for k in ALL_OUTPUT_KEYS:
        arr = np.concatenate(buffers[k], axis=0)
        shard_data[k] = arr[:n_rows]
        remainder = arr[n_rows:]
        buffers[k] = [remainder] if len(remainder) > 0 else []

    final_path = output_dir / f"shard_{shard_id:05d}.npz"
    tmp_path = output_dir / f".tmp_shard_{shard_id:05d}_{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **shard_data)
    os.replace(tmp_path, final_path)
    return final_path.stat().st_size


def worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    next_shard_id: mp.Value,
    completed_shards: mp.Value,
    total_scanned_rows: mp.Value,
    total_matched_rows: mp.Value,
    total_bytes_saved: mp.Value,
    start_time: float,
    lock: mp.Lock,
    output_dir_str: str,
):
    output_dir = Path(output_dir_str)
    buffers = {k: [] for k in ALL_OUTPUT_KEYS}
    buffered_rows = 0

    while True:
        with lock:
            if completed_shards.value >= TARGET_SHARDS or next_shard_id.value >= TARGET_SHARDS:
                break

        try:
            archive_name = task_queue.get_nowait()
        except Exception:
            break

        url = ARCHIVE_BASE_URL + archive_name
        proc = subprocess.Popen(
            [
                "curl",
                "-4",
                "--interface",
                "en0",
                "-sL",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--speed-limit",
                "20000",
                "--speed-time",
                "45",
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1024 * 1024,
        )

        try:
            tf = tarfile.open(fileobj=proc.stdout, mode="r|gz")
            for member in tf:
                if completed_shards.value >= TARGET_SHARDS:
                    break
                if not (member.isfile() and member.name.endswith(".npz")):
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue
                raw_bytes = f.read()
                try:
                    npz = np.load(io.BytesIO(raw_bytes))
                except Exception:
                    continue

                if not all(k in npz for k in REQUIRED_BASE_KEYS):
                    continue

                bin_in = npz["binaryInputNCHWPacked"]
                glob_in = npz["globalInputNC"]
                pol_tgt = npz["policyTargetsNCMove"]
                glob_tgt = npz["globalTargetsNC"]
                score_d = npz["scoreDistrN"]
                val_tgt = npz["valueTargetsNCHW"]

                if (
                    bin_in.shape[1:] != (22, 46)
                    or glob_in.shape[1:] != (19,)
                    or pol_tgt.shape[1:] != (2, 362)
                    or score_d.shape[1:] != (842,)
                    or val_tgt.shape[1:] != (5, 19, 19)
                    or glob_tgt.shape[1] not in (64, 80)
                ):
                    continue

                n = len(bin_in)

                # 1. 19x19 standard board check (all 361 bits in channel 0 are 1)
                m_19x19 = (bin_in[:, 0, :45] == 255).all(axis=1) & (
                    bin_in[:, 0, 45] == 128
                )
                if not m_19x19.any():
                    with lock:
                        total_scanned_rows.value += n
                    continue

                # 2. Chinese rules scoring check: SCORING_AREA (g[9]==0) and TAX_NONE (g[10]==0)
                m_chinese_area = (
                    m_19x19 & (glob_in[:, 9] == 0.0) & (glob_in[:, 10] == 0.0)
                )
                matched = int(m_chinese_area.sum())

                with lock:
                    total_scanned_rows.value += n
                    total_matched_rows.value += matched

                if matched == 0:
                    continue

                buffers["binaryInputNCHWPacked"].append(bin_in[m_chinese_area])
                buffers["globalInputNC"].append(glob_in[m_chinese_area])
                buffers["policyTargetsNCMove"].append(pol_tgt[m_chinese_area])

                gt_sliced = glob_tgt[m_chinese_area]
                if gt_sliced.shape[1] == 64:
                    gt_sliced = np.pad(gt_sliced, ((0, 0), (0, 16)), mode="constant")
                buffers["globalTargetsNC"].append(gt_sliced)

                buffers["scoreDistrN"].append(score_d[m_chinese_area])
                buffers["valueTargetsNCHW"].append(val_tgt[m_chinese_area])

                if "qValueTargetsNCMove" in npz and npz["qValueTargetsNCMove"].shape[1:] == (3, 362):
                    buffers["qValueTargetsNCMove"].append(npz["qValueTargetsNCMove"][m_chinese_area])
                else:
                    buffers["qValueTargetsNCMove"].append(
                        np.zeros((matched, 3, 362), dtype=np.int16)
                    )

                buffered_rows += matched

                while buffered_rows >= ROWS_PER_SHARD:
                    with lock:
                        if completed_shards.value >= TARGET_SHARDS or next_shard_id.value >= TARGET_SHARDS:
                            break
                        my_shard_id = next_shard_id.value
                        next_shard_id.value += 1

                    shard_bytes = save_shard(
                        output_dir, my_shard_id, buffers, ROWS_PER_SHARD
                    )
                    buffered_rows -= ROWS_PER_SHARD

                    with lock:
                        completed_shards.value += 1
                        total_bytes_saved.value += shard_bytes
                        done = completed_shards.value
                        scanned = total_scanned_rows.value
                        saved_mb = total_bytes_saved.value / (1024 * 1024)
                        elapsed = time.time() - start_time
                        rate = (done * ROWS_PER_SHARD) / max(elapsed, 1e-3)
                        eta_min = ((TARGET_SHARDS - done) * ROWS_PER_SHARD / max(rate, 1e-3)) / 60.0
                        print(
                            f"[Shard {done:03d}/{TARGET_SHARDS}] "
                            f"Rows: {done * ROWS_PER_SHARD:,}/{TARGET_ROWS:,} ({done / TARGET_SHARDS * 100:.1f}%) | "
                            f"Scanned: {scanned:,} | "
                            f"Saved: {saved_mb:.1f} MB | "
                            f"Speed: {rate:,.0f} rows/s | "
                            f"ETA: {eta_min:.1f}m | "
                            f"Worker {worker_id:02d} ({archive_name})",
                            flush=True,
                        )
        except Exception:
            pass
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def main():
    output_dir = Path("/Users/user/Desktop/TurboGo/data/katago_19x19_chinese_10m")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover .tmp files from interrupted runs
    for tmp in output_dir.glob(".tmp_shard_*.npz"):
        try:
            tmp.unlink()
        except Exception:
            pass

    existing_shards = sorted(output_dir.glob("shard_*.npz"))
    existing_ids = set()
    existing_bytes = 0
    for p in existing_shards:
        m = re.match(r"shard_(\d{5})\.npz$", p.name)
        if m:
            existing_ids.add(int(m.group(1)))
            existing_bytes += p.stat().st_size

    start_shard_id = 0
    while start_shard_id in existing_ids:
        start_shard_id += 1

    if start_shard_id >= TARGET_SHARDS:
        print(f"All {TARGET_SHARDS} shards ({TARGET_ROWS:,} rows) already exist in {output_dir}!")
        return

    print("Fetching archive list from katagoarchive.org...", flush=True)
    archives = fetch_available_archives()
    print(f"Found {len(archives)} daily archives. Newest: {archives[0]}, Oldest: {archives[-1]}", flush=True)

    task_queue = mp.Queue()
    for a in archives:
        task_queue.put(a)

    num_workers = int(os.environ.get("NUM_WORKERS", "16"))
    next_shard_id = mp.Value("i", start_shard_id)
    completed_shards = mp.Value("i", len(existing_ids))
    total_scanned_rows = mp.Value("q", 0)
    total_matched_rows = mp.Value("q", len(existing_ids) * ROWS_PER_SHARD)
    total_bytes_saved = mp.Value("q", existing_bytes)
    lock = mp.Lock()
    t0 = time.time()

    print(
        f"Starting {num_workers} parallel streaming workers to collect "
        f"{TARGET_ROWS:,} rows ({TARGET_SHARDS} shards x {ROWS_PER_SHARD:,} rows)...",
        flush=True,
    )

    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_loop,
            args=(
                wid,
                task_queue,
                next_shard_id,
                completed_shards,
                total_scanned_rows,
                total_matched_rows,
                total_bytes_saved,
                t0,
                lock,
                str(output_dir),
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    elapsed = time.time() - t0
    final_shards = sorted(output_dir.glob("shard_*.npz"))
    final_bytes = sum(p.stat().st_size for p in final_shards)
    final_rows = len(final_shards) * ROWS_PER_SHARD

    manifest = {
        "dataset": "KataGo kata1 self-play training data (filtered)",
        "network_series": "kata1-tf3-b11c768 & kata1-zhizi-b40c768nbt (latest strongest mainline, Elo ~14500-14603)",
        "board_size": "19x19",
        "rules": "Chinese Area Scoring (SCORING_AREA + TAX_NONE)",
        "total_rows": final_rows,
        "num_shards": len(final_shards),
        "rows_per_shard": ROWS_PER_SHARD,
        "total_size_bytes": final_bytes,
        "total_size_gb": round(final_bytes / (1024**3), 3),
        "elapsed_seconds": round(elapsed, 1),
        "fields": {k: list(v) for k, v in EXPECTED_SHAPES.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"COMPLETED: {len(final_shards)}/{TARGET_SHARDS} shards ({final_rows:,} positions)")
    print(f"Total Disk Size: {final_bytes / (1024**3):.2f} GB ({final_bytes / (1024**2):.1f} MB)")
    print(f"Output Directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
