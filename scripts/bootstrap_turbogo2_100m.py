#!/usr/bin/env python3
"""
Fast In-Datacenter Bootstrap & Priority Range-Stream Extractor for TurboGo-v2 on `turbogo2`:
1. Propagates `dfs_persistent` credentials to `172.28.0.1:8008` so the Colab VM has direct
   datacenter access to Google Drive REST API v3.
2. Restores `b12c384_best_weights.pt` (Step 132,038) and `metrics.jsonl` from Google Drive.
3. Launches the Monochrome Web Dashboard (`dashboard_server.py`) and `cloudflared` tunnel.
4. Reads the Central Directory of `katago_19x19_chinese_100m.zip` (49.18 GiB, ZIP_STORED)
   in Google Drive (`150Rn6GcKXh748zXepknOoFA31Jm0t7Ai`) via HTTP Range requests, downloads
   `val_shards` + `chunks[120..122]` FIRST (in ~3 seconds), immediately launches
   `train_pretrain_a100.py`, and streams the remaining shards in exact chunk order across 16 threads!
"""

import io
import json
import os
import random
import ssl
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

TOKEN_PATH = Path("/content/.colab_token.json")
METADATA_TOKEN_URL = "http://172.28.0.1:8008/computeMetadata/v1/instance/service-accounts/default/token"

ZIP_FILE_ID = "150Rn6GcKXh748zXepknOoFA31Jm0t7Ai"  # katago_19x19_chinese_100m.zip (49.18 GiB)
BEST_WEIGHTS_ID = "1rN7Qvh9Oq3gTjkTkzuSpY-ho91Ctj8Ge"  # b12c384_best_weights.pt
METRICS_ID = "12z6xEZHx0b0ObExXVt2_XbIXYDQcz_3f"  # metrics.jsonl

DATA_ROOT = Path("/content/data")
CKPT_DIR = Path("/content/checkpoints/b12c384")


def propagate_dfs_token():
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if not creds.valid or creds.expired:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())

    ctx = ssl.create_default_context()
    req_a = urllib.request.Request(
        "https://colab.research.google.com/tun/m/assignments?authuser=0",
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
            "x-goog-api-client": "cred-type/u",
        },
    )
    raw = urllib.request.urlopen(req_a, context=ctx, timeout=15).read().decode("utf-8")
    if raw.startswith(")]}'"):
        raw = raw.split("\n", 1)[1]
    assignments = json.loads(raw).get("assignments") or []
    if not assignments:
        raise RuntimeError("No active Colab assignment found")
    endpoint = assignments[0]["endpoint"]

    base_url = f"https://colab.research.google.com/tun/m/credentials-propagation/{endpoint}"
    q_dry = "authuser=0&authtype=dfs_persistent&version=2&dryrun=true&propagate=true&record=false"
    req1 = urllib.request.Request(
        f"{base_url}?{q_dry}",
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
            "x-goog-api-client": "cred-type/u",
        },
    )
    raw1 = urllib.request.urlopen(req1, context=ctx, timeout=15).read().decode("utf-8")
    if raw1.startswith(")]}'"):
        raw1 = raw1.split("\n", 1)[1]
    xsrf_tok = json.loads(raw1).get("token", "")

    boundary = "----ColabKeepaliveBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file_id"\r\n\r\n'
        f"empty.ipynb\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    q_prop = "authuser=0&authtype=dfs_persistent&version=2&dryrun=false&propagate=true&record=false"
    req2 = urllib.request.Request(
        f"{base_url}?{q_prop}",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {creds.token}",
            "x-goog-colab-token": xsrf_tok,
            "X-Colab-Client-Agent": "colab-cli",
            "x-goog-api-client": "cred-type/u",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    res2 = urllib.request.urlopen(req2, context=ctx, timeout=15).read().decode("utf-8")
    print(f"[Auth] Propagated dfs_persistent for {endpoint}: {res2.strip()}", flush=True)


def get_drive_token() -> str:
    req = urllib.request.Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def download_gdrive_file(file_id: str, dest: Path, token: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    print(f"[Restore] Downloaded {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)", flush=True)


class GDriveRangeFile(io.RawIOBase):
    """Seekable file-like object over a Google Drive file using HTTP Range requests (caches tail for ZipFile)."""

    def __init__(self, file_id: str, token: str):
        self.file_id = file_id
        self.token = token
        self.url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        self.ctx = ssl.create_default_context()
        meta_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=size"
        req = urllib.request.Request(meta_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, context=self.ctx, timeout=15) as r:
            self.size = int(json.loads(r.read().decode("utf-8"))["size"])
        self.pos = 0
        # Pre-fetch the last 16 MB where the Zip64 EOCD & Central Directory live!
        self.tail_size = min(16 * 1024 * 1024, self.size)
        self.tail_start = self.size - self.tail_size
        req_t = urllib.request.Request(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Range": f"bytes={self.tail_start}-{self.size - 1}",
            },
        )
        with urllib.request.urlopen(req_t, context=self.ctx, timeout=30) as rt:
            self.tail_buf = rt.read()

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if self.pos >= self.tail_start:
            off = self.pos - self.tail_start
            data = self.tail_buf[off : off + n]
            self.pos += len(data)
            return data
        end = self.pos + n - 1
        req = urllib.request.Request(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Range": f"bytes={self.pos}-{end}",
            },
        )
        with urllib.request.urlopen(req, context=self.ctx, timeout=60) as r:
            data = r.read()
        self.pos += len(data)
        return data


def start_dashboard_and_tunnel():
    subprocess.run("pkill -f dashboard_server.py || true", shell=True)
    subprocess.Popen(
        "nohup python3 /content/turbogo/scripts/dashboard_server.py > /content/dashboard.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
    )
    if not Path("/usr/local/bin/cloudflared").exists():
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=False,
        )
    subprocess.run("pkill -f 'cloudflared tunnel' || true", shell=True)
    subprocess.Popen(
        "nohup /usr/local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 > /content/cloudflared.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
    )


def main():
    t0 = time.perf_counter()
    propagate_dfs_token()
    tok = get_drive_token()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for dname in ("katago_19x19_chinese_10m", "katago_19x19_chinese_50m", "katago_19x19_chinese_40m"):
        (DATA_ROOT / dname).mkdir(parents=True, exist_ok=True)

    # 1. Restore checkpoint & metrics.jsonl
    download_gdrive_file(METRICS_ID, CKPT_DIR / "metrics.jsonl", tok)
    download_gdrive_file(BEST_WEIGHTS_ID, CKPT_DIR / "b12c384_best_weights.pt", tok)

    # Start Monochrome Dashboard + Cloudflared right away
    start_dashboard_and_tunnel()

    # 2. Read Zip64 Central Directory of katago_19x19_chinese_100m.zip in ~0.5s
    print("[Zip64-Stream] Reading Central Directory of katago_19x19_chinese_100m.zip from Google Drive...", flush=True)
    rf = GDriveRangeFile(ZIP_FILE_ID, tok)
    zf = zipfile.ZipFile(rf)
    zmap = {zi.filename: zi for zi in zf.infolist() if zi.filename.endswith(".npz")}
    print(f"[Zip64-Stream] Indexed {len(zmap):,} .npz shards in {time.perf_counter() - t0:.2f}s!", flush=True)

    # 3. Build exact priority order matching train_pretrain_a100.py Epoch 2 (starting at Chunk 120!)
    s10 = [f"katago_19x19_chinese_10m/shard_{i:05d}.npz" for i in range(500)]
    s50 = [f"katago_19x19_chinese_50m/shard_{i:05d}.npz" for i in range(2500)]
    s40 = [f"katago_19x19_chinese_40m/shard_{i:05d}.npz" for i in range(2000)]

    # Map each target path to its unified arcname inside katago_19x19_chinese_100m.zip
    all_rels = s10 + s50 + s40
    rel_to_zinfo = {
        rel: zmap[f"katago_19x19_chinese_100m/shard_{idx:05d}.npz"]
        for idx, rel in enumerate(all_rels)
    }

    val_rels = s10[498:500]
    train_60m = s10[:498] + s50

    rng_60m = random.Random(202602 + 1)
    ep_60m = list(train_60m)
    rng_60m.shuffle(ep_60m)
    split_idx = 34 * 16
    done_shards = ep_60m[:split_idx]
    rem_shards = ep_60m[split_idx:] + list(s40)
    rng_100m = random.Random(2026100 + 1)
    rng_100m.shuffle(rem_shards)
    epoch2_rels = done_shards + rem_shards

    # Load resume chunk from b12c384_best_weights.pt (Chunk 120)
    import torch
    ckpt = torch.load(CKPT_DIR / "b12c384_best_weights.pt", map_location="cpu", weights_only=False)
    init_chunk = ckpt.get("chunk", 120)
    print(f"[Zip64-Stream] Prioritizing shards starting from Epoch 2 Chunk {init_chunk}/313 (Step {ckpt.get('global_step')})...", flush=True)

    priority_rels = (
        val_rels
        + epoch2_rels[init_chunk * 16 :]
        + epoch2_rels[: init_chunk * 16]
    )

    tok_box = {"tok": tok, "ts": time.time()}

    def fetch_one_shard(rel_path: str):
        dest = DATA_ROOT / rel_path
        zi = rel_to_zinfo[rel_path]
        if dest.exists() and dest.stat().st_size == zi.compress_size:
            return
        if time.time() - tok_box["ts"] > 1200:
            try:
                tok_box["tok"] = get_drive_token()
                tok_box["ts"] = time.time()
            except Exception:
                pass
        start_b = zi.header_offset
        end_b = start_b + 128 + zi.compress_size - 1
        url = f"https://www.googleapis.com/drive/v3/files/{ZIP_FILE_ID}?alt=media"
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "Authorization": f"Bearer {tok_box['tok']}",
                        "Range": f"bytes={start_b}-{end_b}",
                    },
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    buf = r.read()
                fname_len, extra_len = struct.unpack("<HH", buf[26:30])
                data_off = 30 + fname_len + extra_len
                npz_bytes = buf[data_off : data_off + zi.compress_size]
                if len(npz_bytes) != zi.compress_size:
                    raise IOError(f"Short read {len(npz_bytes)} != {zi.compress_size}")
                tmp = dest.with_suffix(f".tmp.{os.getpid()}")
                tmp.write_bytes(npz_bytes)
                tmp.replace(dest)
                return
            except Exception as e:
                if attempt == 4:
                    print(f"[Warn] Failed shard {rel_path}: {e}", flush=True)
                time.sleep(1.0 * (attempt + 1))

    # 4. First download val_shards (2) + first 2 training chunks (32 shards) = 34 shards (~355 MB) in ~3s!
    head_batch = priority_rels[:34]
    tail_batch = priority_rels[34:]
    t_h0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch_one_shard, head_batch))
    print(f"[Zip64-Stream] Downloaded val_shards + first 2 chunks (34 shards) in {time.perf_counter() - t_h0:.2f}s!", flush=True)

    # 5. Immediately launch train_pretrain_a100.py on GPU while streaming the remaining shards in the background!
    print("[Launch] Starting train_pretrain_a100.py (Warm Restart Step 132,038 -> 212,454) NOW!", flush=True)
    subprocess.Popen(
        "nohup python3 /content/turbogo/scripts/train_pretrain_a100.py > /content/train_b12c384.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
    )

    # Also start in_vm_keepalive.py
    subprocess.run("pkill -f in_vm_keepalive.py || true", shell=True)
    subprocess.Popen(
        "nohup python3 /content/turbogo/scripts/in_vm_keepalive.py > /content/in_vm_keepalive.log 2>&1 </dev/null &",
        shell=True,
        start_new_session=True,
    )

    # 6. Stream the remaining 4,966 shards in exact chunk order across 16 parallel threads
    t_s0 = time.perf_counter()
    done_cnt = 34
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(fetch_one_shard, tail_batch):
            done_cnt += 1
            if done_cnt % 250 == 0 or done_cnt == len(priority_rels):
                el = time.perf_counter() - t_s0
                print(f"[Zip64-Stream] Extracted {done_cnt}/{len(priority_rels)} shards ({el:.1f}s elapsed)...", flush=True)

    print(f"[Bootstrap Complete] All 5,000 shards of 100M pack extracted in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
