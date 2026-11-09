#!/usr/bin/env python3
"""
24/7 External Keepalive & Self-Healing Daemon running on `laochou@aiserver.laochou.cloud`
via Clash proxy (`http://127.0.0.1:7890`).

Completely independent of the user's local MacBook:
1. Every 4 minutes (240s), runs `colab exec -s turbogo2` through `127.0.0.1:7890`, which:
   - Automatically refreshes the Google OAuth2 token (`~/.config/colab-cli/token.json`),
   - Automatically refreshes the 1-hour Colab Runtime Proxy token (`~/.config/colab-cli/sessions.json`),
   - Opens a real external WebSocket (`wss://.../api/kernels/{id}/channels`) and executes code on the kernel.
2. Verifies that `train_pretrain_a100.py`, `dashboard_server.py`, and `cloudflared` are running on `turbogo2`,
   and automatically restarts any stopped service.
"""

import json
import os
import subprocess
import time
from pathlib import Path

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

COLAB_BIN = str(Path.home() / "turbogo_keepalive/venv/bin/colab")
SESSION = "turbogo_mini"
STOP_AT_UTC = "2026-10-29 16:00:00"  # 2026-10-30 00:00:00 CST

REMOTE_CODE = r"""
import json, os, subprocess, time, urllib.request
from pathlib import Path

def is_running(pat):
    my_pid = os.getpid()
    for p in Path("/proc").iterdir():
        if p.name.isdigit() and int(p.name) != my_pid:
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
                if pat in cmd and "colab_kernel_launcher" not in cmd:
                    return True
            except Exception:
                pass
    return False

out = {
    "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    "train_running": is_running("train_turbogo_mini.py"),
    "bootstrap_running": is_running("bootstrap_turbogo_mini.py"),
    "dash_running": is_running("dashboard_server.py"),
    "cf_running": is_running("cloudflared tunnel"),
    "finished": os.path.exists("/content/checkpoints/turbogo_mini/turbogo_mini_final.pt"),
}
out["gpu"] = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader").strip()
out["log_tail"] = subprocess.getoutput("tail -n 2 /content/train_turbogo_mini.log 2>/dev/null || tail -n 2 /content/bootstrap.log 2>/dev/null").strip()

if not out["dash_running"] and os.path.exists("/content/turbogo/scripts/dashboard_server.py"):
    subprocess.Popen("nohup python3 /content/turbogo/scripts/dashboard_server.py > /content/dashboard.log 2>&1 </dev/null &", shell=True, start_new_session=True)

if not out["cf_running"] and os.path.exists("/usr/local/bin/cloudflared"):
    subprocess.Popen("nohup /usr/local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 > /content/cloudflared.log 2>&1 </dev/null &", shell=True, start_new_session=True)

val_ready = os.path.exists("/content/data/katago_19x19_chinese_10m/shard_00499.npz")
if not out["train_running"] and not out["bootstrap_running"] and not out["finished"] and val_ready and out["utc"] < "2026-10-29 16:00:00":
    subprocess.Popen("nohup python3 /content/turbogo/scripts/train_turbogo_mini.py >> /content/train_turbogo_mini.log 2>&1 </dev/null &", shell=True, start_new_session=True)
    out["restarted_train"] = True

print("AISERVER_HEARTBEAT:" + json.dumps(out))
"""


def main():
    print(f"[AIServer Keepalive] Started on {os.uname().nodename} via Clash 127.0.0.1:7890 for session '{SESSION}'", flush=True)
    while True:
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        if now_utc >= STOP_AT_UTC:
            print(f"[AIServer Keepalive] Reached {STOP_AT_UTC}. Stopping session '{SESSION}' and exiting.", flush=True)
            subprocess.run([COLAB_BIN, "stop", "-s", SESSION], check=False)
            break
        try:
            res = subprocess.run(
                [COLAB_BIN, "exec", "-s", SESSION],
                input=REMOTE_CODE,
                capture_output=True,
                text=True,
                timeout=90,
            )
            hb = None
            for line in res.stdout.splitlines():
                if line.startswith("AISERVER_HEARTBEAT:"):
                    hb = json.loads(line[len("AISERVER_HEARTBEAT:") :])
                    break
            if hb:
                print(
                    f"[{hb['utc']} UTC] OK | train={hb['train_running']} boot={hb['bootstrap_running']} fin={hb.get('finished')} | GPU: {hb['gpu']}\n"
                    f"  Log: {hb['log_tail'].splitlines()[-1] if hb['log_tail'] else '(starting)'}",
                    flush=True,
                )
                if hb.get("finished"):
                    print(f"[AIServer Keepalive] Training finished (turbogo_mini_final.pt found)! Stopping session '{SESSION}' to save CCU.", flush=True)
                    subprocess.run([COLAB_BIN, "stop", "-s", SESSION], check=False)
                    break
            else:
                print(f"[{now_utc} UTC] Warn: rc={res.returncode} stdout={res.stdout[-200:]} stderr={res.stderr[-200:]}", flush=True)
        except Exception as e:
            print(f"[{now_utc} UTC] Exception: {e}", flush=True)

        time.sleep(240)


if __name__ == "__main__":
    main()
