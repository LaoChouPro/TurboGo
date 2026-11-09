#!/usr/bin/env python3
"""
Stream, filter, deduplicate, and pack 50,000,000 additional positions (data rows)
from KataGo's latest strongest neural network self-play archives (kata1) via
direct physical interface (en0, bypassing VPN).

Features:
1. Direct connection (--interface en0 -4) with ResumableCurlStream:
   If any HTTP stream drops or stalls mid-archive, it transparently reconnects
   using HTTP Range (bytes=offset-) so 100% of every .tgz archive is read.
2. Exact 64-bit game-hash + turn-index deduplication against the first 10M dataset
   (/Users/user/Desktop/TurboGo/data/katago_19x19_chinese_10m), guaranteeing 0% overlap.
3. Strict 19x19 + Standard Chinese Area Scoring (SCORING_AREA + TAX_NONE) filter.
4. Outputs 2,500 shards (shard_00000.npz .. shard_02499.npz, 20,000 rows/shard = 50M rows)
   to /Users/user/Desktop/TurboGo/data/katago_19x19_chinese_50m.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ARCHIVE_INDEX_URL = "https://katagoarchive.org/kata1/trainingdata/index.html"
ARCHIVE_BASE_URL = "https://katagoarchive.org/kata1/trainingdata/"

TARGET_ROWS = 50_000_000
ROWS_PER_SHARD = 20_000
TARGET_SHARDS = TARGET_ROWS // ROWS_PER_SHARD  # 2,500 shards

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


def compute_row_hashes(gt: np.ndarray) -> np.ndarray:
    """Compute a 64-bit fingerprint from KataGo's 128-bit game hash (cols 41..44) and turnIdx (col 51)."""
    h1 = gt[:, 41].astype(np.uint64)
    h2 = gt[:, 42].astype(np.uint64)
    h3 = gt[:, 43].astype(np.uint64)
    h4 = gt[:, 44].astype(np.uint64)
    turn = gt[:, 51].astype(np.uint64)
    return (h1 << 42) ^ (h2 << 21) ^ (h3 << 7) ^ (h4 << 13) ^ turn


def build_or_load_dedup_hashes(first_10m_dir: Path, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading existing 10M deduplication hashes from {cache_path}...", flush=True)
        return np.load(cache_path)

    shards = sorted(first_10m_dir.glob("shard_*.npz"))
    if not shards:
        return np.array([], dtype=np.uint64)

    print(f"Building deduplication index from {len(shards)} shards in {first_10m_dir}...", flush=True)
    t0 = time.time()

    def _read_one(p: Path):
        with np.load(p) as npz:
            return compute_row_hashes(npz["globalTargetsNC"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        chunks = list(ex.map(_read_one, shards))

    all_hashes = np.unique(np.concatenate(chunks))
    np.save(cache_path, all_hashes)
    print(
        f"Indexed {len(all_hashes):,} unique position hashes in {time.time() - t0:.1f}s "
        f"({cache_path.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    return all_hashes


class ResumableCurlStream(io.RawIOBase):
    """
    A streaming file-like object backed by `curl -4 --interface en0` that automatically
    reconnects with HTTP Range (`--range offset-`) if a connection stalls or drops,
    allowing `tarfile.open(..., mode='r|gz')` to read multi-GB .tgz archives completely.
    """

    def __init__(self, url: str, byte_counter: mp.Value, lock: mp.Lock, max_resumes: int = 12):
        super().__init__()
        self.url = url
        self.byte_counter = byte_counter
        self.lock = lock
        self.max_resumes = max_resumes
        self.offset = 0
        self.resumes = 0
        self.proc = None
        self.eof = False
        self._local_bytes = 0
        self._start_proc()

    def _start_proc(self):
        self._close_proc()
        cmd = [
            "curl",
            "-4",
            "--interface",
            "en0",
            "-sL",
            "--speed-limit",
            "10000",
            "--speed-time",
            "45",
        ]
        if self.offset > 0:
            cmd.extend(["--range", f"{self.offset}-"])
        cmd.append(self.url)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1024 * 1024,
        )

    def _close_proc(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def readable(self):
        return True

    def readinto(self, b):
        if self.eof:
            return 0
        view = memoryview(b)
        total_read = 0
        target_len = len(view)

        while total_read < target_len:
            if self.proc is None or self.proc.stdout is None:
                break
            chunk = self.proc.stdout.read(target_len - total_read)
            if chunk:
                n = len(chunk)
                view[total_read : total_read + n] = chunk
                total_read += n
                self.offset += n
                self._local_bytes += n
                if self._local_bytes >= 262144:  # batch update every 256 KB
                    with self.lock:
                        self.byte_counter.value += self._local_bytes
                    self._local_bytes = 0
            else:
                rc = self.proc.poll()
                if rc == 0:
                    self.eof = True
                    break
                # Connection dropped or stalled prematurely -> resume from self.offset
                self.resumes += 1
                if self.resumes > self.max_resumes:
                    self.eof = True
                    break
                time.sleep(1.0)
                self._start_proc()

        if self._local_bytes > 0 and self.eof:
            with self.lock:
                self.byte_counter.value += self._local_bytes
            self._local_bytes = 0

        return total_read

    def close(self):
        if self._local_bytes > 0:
            with self.lock:
                self.byte_counter.value += self._local_bytes
            self._local_bytes = 0
        self._close_proc()
        super().close()


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


GLOBAL_DEDUP_HASHES = None


def worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    next_shard_id: mp.Value,
    completed_shards: mp.Value,
    total_scanned_rows: mp.Value,
    total_matched_rows: mp.Value,
    total_dedup_skipped: mp.Value,
    total_bytes_downloaded: mp.Value,
    total_bytes_saved: mp.Value,
    start_time: float,
    lock: mp.Lock,
    output_dir_str: str,
):
    output_dir = Path(output_dir_str)
    buffers = {k: [] for k in ALL_OUTPUT_KEYS}
    buffered_rows = 0
    dedup_hashes = GLOBAL_DEDUP_HASHES

    while True:
        with lock:
            if completed_shards.value >= TARGET_SHARDS or next_shard_id.value >= TARGET_SHARDS:
                break

        try:
            archive_name = task_queue.get_nowait()
        except Exception:
            break

        url = ARCHIVE_BASE_URL + archive_name
        stream = ResumableCurlStream(url, total_bytes_downloaded, lock)

        try:
            tf = tarfile.open(fileobj=stream, mode="r|gz")
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
                if not m_chinese_area.any():
                    with lock:
                        total_scanned_rows.value += n
                    continue

                # 3. Deduplicate against existing 10M dataset
                cand_idx = np.flatnonzero(m_chinese_area)
                skipped = 0
                if dedup_hashes is not None and len(dedup_hashes) > 0:
                    cand_hashes = compute_row_hashes(glob_tgt[cand_idx])
                    pos = np.searchsorted(dedup_hashes, cand_hashes)
                    pos_clipped = np.minimum(pos, len(dedup_hashes) - 1)
                    is_dup = dedup_hashes[pos_clipped] == cand_hashes
                    skipped = int(is_dup.sum())
                    cand_idx = cand_idx[~is_dup]

                matched = len(cand_idx)
                with lock:
                    total_scanned_rows.value += n
                    total_matched_rows.value += matched
                    total_dedup_skipped.value += skipped

                if matched == 0:
                    continue

                buffers["binaryInputNCHWPacked"].append(bin_in[cand_idx])
                buffers["globalInputNC"].append(glob_in[cand_idx])
                buffers["policyTargetsNCMove"].append(pol_tgt[cand_idx])

                gt_sliced = glob_tgt[cand_idx]
                if gt_sliced.shape[1] == 64:
                    gt_sliced = np.pad(gt_sliced, ((0, 0), (0, 16)), mode="constant")
                buffers["globalTargetsNC"].append(gt_sliced)

                buffers["scoreDistrN"].append(score_d[cand_idx])
                buffers["valueTargetsNCHW"].append(val_tgt[cand_idx])

                if "qValueTargetsNCMove" in npz and npz["qValueTargetsNCMove"].shape[1:] == (3, 362):
                    buffers["qValueTargetsNCMove"].append(npz["qValueTargetsNCMove"][cand_idx])
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
                        dl_mb = total_bytes_downloaded.value / (1024 * 1024)
                        saved_mb = total_bytes_saved.value / (1024 * 1024)
                        elapsed = time.time() - start_time
                        dl_speed = dl_mb / max(elapsed, 1e-3)
                        rate = (done * ROWS_PER_SHARD) / max(elapsed, 1e-3)
                        eta_min = ((TARGET_SHARDS - done) * ROWS_PER_SHARD / max(rate, 1e-3)) / 60.0
                        print(
                            f"[Shard {done:04d}/{TARGET_SHARDS}] "
                            f"Rows: {done * ROWS_PER_SHARD:,}/{TARGET_ROWS:,} ({done / TARGET_SHARDS * 100:.1f}%) | "
                            f"Net: {dl_speed:.2f} MB/s ({dl_mb:,.0f} MB) | "
                            f"Speed: {rate:,.0f} rows/s | "
                            f"Saved: {saved_mb:,.1f} MB | "
                            f"DupSkip: {total_dedup_skipped.value:,} | "
                            f"ETA: {eta_min:.1f}m | "
                            f"W{worker_id:02d} ({archive_name}, r={stream.resumes})",
                            flush=True,
                        )
        except Exception as e:
            print(f"[Warn] W{worker_id:02d} ({archive_name}) exited at {stream.offset / 1e6:.1f} MB (r={stream.resumes}): {e}", flush=True)
        finally:
            stream.close()


def main():
    global GLOBAL_DEDUP_HASHES
    first_10m_dir = Path("/Users/user/Desktop/TurboGo/data/katago_19x19_chinese_10m")
    output_dir = Path("/Users/user/Desktop/TurboGo/data/katago_19x19_chinese_50m")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path("/Users/user/Desktop/TurboGo/data/existing_10m_hashes.npy")

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

    GLOBAL_DEDUP_HASHES = build_or_load_dedup_hashes(first_10m_dir, cache_path)

    print("Fetching archive list from katagoarchive.org via en0...", flush=True)
    archives = fetch_available_archives()
    print(f"Found {len(archives)} daily archives. Newest: {archives[0]}, Oldest: {archives[-1]}", flush=True)

    # Use fork context on macOS so workers share GLOBAL_DEDUP_HASHES zero-copy in memory
    ctx = mp.get_context("fork")
    task_queue = ctx.Queue()
    for a in archives:
        task_queue.put(a)

    num_workers = int(os.environ.get("NUM_WORKERS", "20"))
    next_shard_id = ctx.Value("i", start_shard_id)
    completed_shards = ctx.Value("i", len(existing_ids))
    total_scanned_rows = ctx.Value("q", 0)
    total_matched_rows = ctx.Value("q", len(existing_ids) * ROWS_PER_SHARD)
    total_dedup_skipped = ctx.Value("q", 0)
    total_bytes_downloaded = ctx.Value("q", 0)
    total_bytes_saved = ctx.Value("q", existing_bytes)
    lock = ctx.Lock()
    t0 = time.time()

    print(
        f"Starting {num_workers} parallel direct-en0 streaming workers to collect "
        f"{TARGET_ROWS:,} NEW rows ({TARGET_SHARDS} shards x {ROWS_PER_SHARD:,} rows)...",
        flush=True,
    )

    workers = []
    for wid in range(num_workers):
        p = ctx.Process(
            target=worker_loop,
            args=(
                wid,
                task_queue,
                next_shard_id,
                completed_shards,
                total_scanned_rows,
                total_matched_rows,
                total_dedup_skipped,
                total_bytes_downloaded,
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
        "dataset": "KataGo kata1 self-play training data (50M additional filtered & deduplicated)",
        "board_size": "19x19",
        "rules": "Chinese Area Scoring (SCORING_AREA + TAX_NONE)",
        "deduplicated_against": str(first_10m_dir),
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
    print(f"Total Disk Size: {final_bytes / (1024**3):.2f} GiB ({final_bytes / (1024**2):.1f} MiB)")
    print(f"Output Directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
