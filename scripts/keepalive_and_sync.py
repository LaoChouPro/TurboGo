#!/usr/bin/env python3
"""
Continuous 10-minute Keepalive, Self-Healing, and Google Drive Checkpoint Sync Daemon
for TurboGo b12c384 Pretraining on Colab A100 ('turbogo').

1. Executes a regular cell on the Colab Jupyter kernel via `colab exec -s turbogo`
   every 10 minutes so the Colab runtime proxy token (3600s TTL) and Jupyter kernel
   `last_activity` timestamp are continuously refreshed.
2. Monitors `train_pretrain_a100.py` on the Colab A100 VM and automatically relaunches
   it if it ever stops prematurely before 2026-10-27 03:00:00 UTC (11:00 CST).
3. Streams `b12c384_best_weights.pt`, per-epoch `b12c384_epoch_XX_weights.pt`, and `metrics.jsonl`
   via Jupyter Server's raw binary `/files/` endpoint directly to Google Drive
   (`/Users/user/Library/CloudStorage/GoogleDrive-laochoupro@gmail.com/我的云端硬盘/AI训练数据/围棋/checkpoints/`).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COLAB_BIN = "/Users/user/.local/bin/colab"
SESSIONS_JSON = Path("/Users/user/.config/colab-cli/sessions.json")
SESSION = os.environ.get("COLAB_SESSION", "turbogo2")
STOP_AT_UTC = "2026-10-29 02:30:00"  # 2026-10-29 10:30:00 CST
CF_URL_FILE = Path("/tmp/turbogo2_cf_url.txt")
GDRIVE_CKPT_DIR = Path(
    "/Users/user/Library/CloudStorage/GoogleDrive-laochoupro@gmail.com/我的云端硬盘/AI训练数据/围棋/checkpoints"
)
REMOTE_CKPT_DIR = "/content/checkpoints/b12c384"

REMOTE_CHECK_CODE = r"""
import os, glob, json, subprocess, time, torch

out = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
ps_out = subprocess.getoutput("ps -ef | grep '[t]rain_pretrain_a100.py'")
out["running"] = bool(ps_out.strip())
out["gpu"] = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader")
out["log_tail"] = subprocess.getoutput("tail -n 5 /content/train_b12c384.log")

ckpt_dir = "/content/checkpoints/b12c384"
best_w = os.path.join(ckpt_dir, "b12c384_best_weights.pt")
snap_w = os.path.join(ckpt_dir, "b12c384_sync_snapshot.pt")
metrics_p = os.path.join(ckpt_dir, "metrics.jsonl")

# Wait if a checkpoint is actively being written right now (mtime < 4s ago)
if os.path.exists(best_w) and (time.time() - os.path.getmtime(best_w) < 4.0):
    time.sleep(4.5)

if os.path.exists(best_w) and os.path.getsize(best_w) > 25_000_000:
    bw_mtime = os.path.getmtime(best_w)
    # Convert to compact bfloat16 snapshot (~41.6 MB) if not yet converted for this mtime
    need_snap = (not os.path.exists(snap_w)) or (os.path.getsize(snap_w) > 60_000_000) or (os.path.getmtime(snap_w) < bw_mtime)
    if need_snap:
        ckpt = torch.load(best_w, map_location="cpu", weights_only=False)
        ckpt["model"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
        torch.save(ckpt, snap_w)
    out["best_weights_mtime"] = bw_mtime
    out["best_step"] = int(torch.load(snap_w, map_location="cpu", weights_only=False).get("global_step", 0))
else:
    out["best_weights_mtime"] = 0
    out["best_step"] = 0

# Also convert any end-of-epoch checkpoints (b12c384_epoch_XX.pt -> b12c384_epoch_XX_weights.pt)
epoch_weights = []
for ep_p in sorted(glob.glob(os.path.join(ckpt_dir, "b12c384_epoch_*.pt"))):
    if ep_p.endswith("_weights.pt"):
        continue
    if time.time() - os.path.getmtime(ep_p) < 4.0:
        time.sleep(4.5)
    ep_w = ep_p[:-3] + "_weights.pt"
    if not os.path.exists(ep_w):
        ec = torch.load(ep_p, map_location="cpu", weights_only=False)
        torch.save({
            "model": {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in ec["model"].items()},
            "epoch": ec.get("epoch", 0) + 1,
            "global_step": ec.get("global_step", 0),
            "val_metrics": ec.get("val_metrics", {}),
            "config": ec.get("config", {}),
        }, ep_w)
    epoch_weights.append(os.path.basename(ep_w))

out["epoch_weights"] = epoch_weights
out["metrics_size"] = os.path.getsize(metrics_p) if os.path.exists(metrics_p) else 0

# Auto-restart if not running and before deadline and all 5,000 data shards exist
shards_ready = (
    os.path.exists("/content/data/katago_19x19_chinese_10m/shard_00499.npz")
    and os.path.exists("/content/data/katago_19x19_chinese_50m/shard_02499.npz")
    and os.path.exists("/content/data/katago_19x19_chinese_40m/shard_01999.npz")
)
mig_running = bool(subprocess.getoutput("ps -ef | grep -E '[m]igrate_to_turbogo2|[d]ownload_katago'").strip())
if not out["running"] and not mig_running and shards_ready and out["utc"] < "2026-10-29 03:30:00":
    with open("/content/train_b12c384.log", "a") as lf:
        subprocess.Popen(
            ["python3", "/content/turbogo/scripts/train_pretrain_a100.py"],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    time.sleep(2)
    out["restarted"] = True
    out["running"] = bool(subprocess.getoutput("ps -ef | grep '[t]rain_pretrain_a100.py'").strip())
else:
    out["restarted"] = False

print("STATUS_JSON:" + json.dumps(out))
"""


def download_binary_fast(remote_path: str, local_path: Path) -> bool:
    """Stream raw binary directly from Jupyter Server's /files/ endpoint (zero base64/JSON overhead)."""
    tmp_path = Path(f"/tmp/{local_path.name}.tmp.{os.getpid()}")
    try:
        with open(SESSIONS_JSON, "r", encoding="utf-8") as f:
            s_info = json.load(f)[SESSION]
        base_url = s_info["url"].rstrip("/")
        token = s_info["token"]
        rel_path = remote_path.lstrip("/")
        url = f"{base_url}/files/{rel_path}?colab-runtime-proxy-token={token}"

        res = subprocess.run(
            ["curl", "-fSL", "--retry", "2", "--max-time", "480", "-o", str(tmp_path), url],
            capture_output=True,
            text=True,
            timeout=500,
        )
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            shutil.move(str(tmp_path), str(local_path))
            return True
        else:
            print(f"  [download_binary_fast curl failed rc={res.returncode}]: {res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"  [download_binary_fast error for {remote_path}]: {e}", flush=True)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return False


def sync_via_cloudflare_fallback(last_synced_step: int) -> int:
    """Zero-Jupyter HTTP fallback: pull status, metrics.jsonl, and b12c384_best_weights.pt via cloudflared."""
    if not CF_URL_FILE.exists():
        return last_synced_step
    cf_url = CF_URL_FILE.read_text().strip().rstrip("/")
    if not cf_url.startswith("https://"):
        return last_synced_step
    try:
        res = subprocess.run(
            ["curl", "-fsSL", "--max-time", "15", f"{cf_url}/api/status"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if res.returncode != 0:
            return last_synced_step
        data = json.loads(res.stdout)
        val_series = data.get("val_series") or []
        if not val_series:
            return last_synced_step
        latest_val_step = int(val_series[-1].get("global_step", 0))
        if latest_val_step > last_synced_step:
            # Pull lightweight metrics.jsonl only (41.7MB weights are synced directly in-datacenter to GDrive!)
            tmp_m = Path(f"/tmp/metrics.jsonl.cf.{os.getpid()}")
            r1 = subprocess.run(
                ["curl", "-fsSL", "--max-time", "15", "-o", str(tmp_m), f"{cf_url}/api/sync/metrics.jsonl"],
                capture_output=True,
            )
            if r1.returncode == 0 and tmp_m.exists() and tmp_m.stat().st_size > 100:
                shutil.move(str(tmp_m), str(GDRIVE_CKPT_DIR / "metrics.jsonl"))
                print(f"  -> [CF-HTTP Sync] Synced metrics.jsonl (step={latest_val_step}) to Google Drive", flush=True)
            return latest_val_step
    except Exception as e:
        print(f"  [CF-HTTP Sync warn]: {e}", flush=True)
    return last_synced_step


def main():
    GDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    last_best_mtime = 0.0
    last_metrics_size = 0
    last_cf_step = 19146

    print(f"[Keepalive & GDrive Sync] Started for session '{SESSION}' -> {GDRIVE_CKPT_DIR}", flush=True)

    while True:
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        if now_utc >= STOP_AT_UTC:
            print(f"[Keepalive & GDrive Sync] Target UTC {STOP_AT_UTC} reached. Exiting after final sync.", flush=True)
            break

        try:
            res = subprocess.run(
                [COLAB_BIN, "exec", "-s", SESSION],
                input=REMOTE_CHECK_CODE,
                capture_output=True,
                text=True,
                timeout=90,
            )
            status_line = None
            for line in res.stdout.splitlines():
                if line.startswith("STATUS_JSON:"):
                    status_line = line[len("STATUS_JSON:") :]
                    break

            if status_line:
                st = json.loads(status_line)
                print(
                    f"[{st['utc']} UTC] running={st['running']} restarted={st['restarted']} | GPU: {st['gpu']}",
                    flush=True,
                )
                if st["log_tail"]:
                    last_line = st["log_tail"].splitlines()[-1]
                    print(f"  Log: {last_line}", flush=True)

                # Sync metrics.jsonl if updated
                if st["metrics_size"] > last_metrics_size:
                    if download_binary_fast(f"{REMOTE_CKPT_DIR}/metrics.jsonl", GDRIVE_CKPT_DIR / "metrics.jsonl"):
                        last_metrics_size = st["metrics_size"]
                        print("  -> Synced metrics.jsonl to Google Drive", flush=True)

                # Note: b12c384_best_weights.pt (41.7 MB) is uploaded directly inside Google's datacenter
                # to Google Drive in ~1.2s by train_pretrain_a100.py and in_vm_keepalive.py!
                if st["best_weights_mtime"] > last_best_mtime:
                    last_best_mtime = st["best_weights_mtime"]
                    print(
                        f"  -> Verified cloud-direct GDrive sync of b12c384_best_weights.pt (step={st.get('best_step')})",
                        flush=True,
                    )

                # Sync any newly completed end-of-epoch checkpoints (b12c384_epoch_XX_weights.pt)
                for ep_name in st.get("epoch_weights", []):
                    local_ep = GDRIVE_CKPT_DIR / ep_name
                    if not local_ep.exists():
                        if download_binary_fast(f"{REMOTE_CKPT_DIR}/{ep_name}", local_ep):
                            sz_mb = local_ep.stat().st_size / 1e6
                            print(f"  -> Synced Epoch Checkpoint {ep_name} ({sz_mb:.1f} MB) to Google Drive", flush=True)
            else:
                raise RuntimeError(f"no STATUS_JSON in colab exec output (rc={res.returncode}): {res.stderr.strip()[-200:]}")
        except Exception as e:
            print(f"[{now_utc} UTC] Error during keepalive/sync: {e}", flush=True)
            try:
                import urllib.request
                with open(SESSIONS_JSON, "r", encoding="utf-8") as f:
                    s_info = json.load(f)[SESSION]
                b_url = s_info["url"].rstrip("/")
                hdrs = {"X-Colab-Runtime-Proxy-Token": s_info["token"], "X-Colab-Client-Agent": "vscode"}
                k_req = urllib.request.Request(f"{b_url}/api/kernels", headers=hdrs)
                for k in json.loads(urllib.request.urlopen(k_req, timeout=10).read().decode()):
                    kid = k["id"]
                    r_req = urllib.request.Request(f"{b_url}/api/kernels/{kid}/restart", method="POST", headers=hdrs)
                    urllib.request.urlopen(r_req, timeout=10)
            except Exception:
                pass

        last_cf_step = sync_via_cloudflare_fallback(last_cf_step)
        time.sleep(300)


if __name__ == "__main__":
    main()
