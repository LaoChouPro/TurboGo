#!/usr/bin/env python3
"""
In-VM Cloud Self-Keepalive Daemon for Google Colab (`turbogo2`).
Runs INSIDE the Colab VM itself so that even if the user's local MacBook sleeps,
closes its lid, or loses Wi-Fi, the Colab VM continuously:
  1. Refreshes its Google OAuth2 token against `oauth2.googleapis.com` every 3 minutes,
  2. Calls `https://colab.research.google.com/tun/m/assignments?authuser=0` to refresh
     the Colab assignment lease and obtain the latest `runtimeProxyInfo` token,
  3. Sends external HTTPS GET requests through its own public Colab Runtime Proxy
     (`https://8080-gpu-...prod.r.colab.dev/api/kernels` & `/api/sessions`) and hits
     local `kernel_manager_proxy` (`http://172.28.0.12:6000/api/kernels`),
  4. Monitors `train_pretrain_a100.py`, `dashboard_server.py`, and `cloudflared` and
     restarts them automatically if needed.
"""

import json
import os
import subprocess
import ssl
import time
import urllib.request
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

TOKEN_PATH = Path("/content/.colab_token.json")
STOP_AT_UTC = "2026-10-29 14:30:00"  # 2026-10-29 22:30:00 CST


def get_valid_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if not creds.valid or creds.expired:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def refresh_dfs_persistent_in_vm(creds: Credentials, endpoint: str):
    """Self-propagate dfs_persistent credentials from inside the Colab VM so 172.28.0.1:8008 never expires."""
    if not endpoint:
        return
    ctx = ssl.create_default_context()
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
    with urllib.request.urlopen(req1, context=ctx, timeout=15) as r1:
        raw1 = r1.read().decode("utf-8")
    if raw1.startswith(")]}'"):
        raw1 = raw1.split("\n", 1)[1]
    xsrf_tok = json.loads(raw1).get("token", "")
    if not xsrf_tok:
        return

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
    with urllib.request.urlopen(req2, context=ctx, timeout=15) as r2:
        r2.read()


def sync_ckpt_to_gdrive_in_vm(last_mtimes: dict):
    """Directly PATCH metrics.jsonl and turbogo_mini_best_weights.pt to Google Drive if updated."""
    ckpt_dir = Path("/content/checkpoints/turbogo_mini")
    metrics_p = ckpt_dir / "metrics.jsonl"
    best_w = ckpt_dir / "turbogo_mini_best_weights.pt"

    need_metrics = metrics_p.exists() and metrics_p.stat().st_mtime > last_mtimes.get("metrics", 0)
    need_best = (
        best_w.exists()
        and best_w.stat().st_size > 4_000_000
        and (time.time() - best_w.stat().st_mtime >= 2.0)
        and best_w.stat().st_mtime > last_mtimes.get("best_w", 0)
    )
    if not need_metrics and not need_best:
        return

    req = urllib.request.Request(
        "http://172.28.0.1:8008/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    tok = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())["access_token"]

    if need_metrics:
        mtime = metrics_p.stat().st_mtime
        m_req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files/1E3UBroUaFMY07ag8XQSbOlsqek72BiFi?uploadType=media",
            data=metrics_p.read_bytes(),
            method="PATCH",
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(m_req, timeout=15).read()
        last_mtimes["metrics"] = mtime

    if need_best:
        mtime = best_w.stat().st_mtime
        bw_bytes = best_w.read_bytes()
        b_req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files/1Uh11E6zGnYGJcLSTsdFWh82hYenpJRhd?uploadType=media",
            data=bw_bytes,
            method="PATCH",
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/octet-stream"},
        )
        urllib.request.urlopen(b_req, timeout=60).read()
        last_mtimes["best_w"] = mtime
        print(f"  -> [In-VM GDrive Sync] Uploaded turbogo_mini_best_weights.pt ({len(bw_bytes)/1e6:.2f} MB) to Google Drive!", flush=True)


def ping_colab_control_plane(creds: Credentials) -> dict:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        "https://colab.research.google.com/tun/m/assignments?authuser=0",
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
            "x-goog-api-client": "cred-type/u",
        },
    )
    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
    if raw.startswith(")]}'"):
        raw = raw.split("\n", 1)[1]
    data = json.loads(raw)
    assignments = data.get("assignments") or []
    if not assignments:
        return {}
    a = assignments[0]
    rpi = a.get("runtimeProxyInfo") or {}
    proxy_url = rpi.get("url", "").rstrip("/")
    proxy_token = rpi.get("token", "")
    endpoint = a.get("endpoint", "")

    # Send safe HTTP GET + POST /api/sessions keepalive requests through the public Colab Runtime Proxy URL
    # POST /api/sessions is pure HTTP (no WebSocket) and updates kernel `last_activity` to NOW()!
    if proxy_url and proxy_token:
        hdrs = {
            "X-Colab-Runtime-Proxy-Token": proxy_token,
            "X-Colab-Client-Agent": "vscode",
            "Content-Type": "application/json",
        }
        for path in ("/api/kernels", "/api/sessions"):
            try:
                p_req = urllib.request.Request(f"{proxy_url}{path}", headers=hdrs)
                with urllib.request.urlopen(p_req, context=ctx, timeout=15) as pr:
                    pr.read()
            except Exception:
                pass
        try:
            sess_body = json.dumps({
                "path": "keepalive_heartbeat.ipynb",
                "type": "notebook",
                "name": "",
                "kernel": {"id": None, "name": "python3"},
            }).encode("utf-8")
            s_req = urllib.request.Request(f"{proxy_url}/api/sessions", data=sess_body, method="POST", headers=hdrs)
            with urllib.request.urlopen(s_req, context=ctx, timeout=15) as sr:
                sess_info = json.loads(sr.read().decode("utf-8"))
                kid = (sess_info.get("kernel") or {}).get("id")
        except Exception:
            kid = None

        # Also execute a lightweight WebSocket cell ("1+1") through the public Colab Runtime Proxy
        # with _own_kernel=False so Colab's external proxy sees real WebSocket traffic & cell execution!
        try:
            import jupyter_kernel_client
            if hasattr(jupyter_kernel_client, "ColabKernelClient"):
                _Cls, _tkw = jupyter_kernel_client.ColabKernelClient, "proxy_token"
            elif hasattr(jupyter_kernel_client, "KernelClient"):
                _Cls, _tkw = jupyter_kernel_client.KernelClient, "token"
            else:
                _Cls, _tkw = jupyter_kernel_client.JupyterKernelClient, "token"
            kc = _Cls(
                server_url=proxy_url,
                **{_tkw: proxy_token},
                kernel_id=kid,
                client_kwargs={
                    "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                    "extra_params": {"colab-runtime-proxy-token": proxy_token},
                },
                headers={
                    "X-Colab-Client-Agent": "colab-cli",
                    "X-Colab-Runtime-Proxy-Token": proxy_token,
                },
            )
            kc._own_kernel = False
            kc.start()
            kc.execute("1+1", timeout=8)
            client_ws = kc._manager.client
            client_ws.stop_channels()
            if client_ws.kernel_socket:
                client_ws.kernel_socket.close()
        except Exception as ws_e:
            print(f"  [WS-Keepalive note]: {ws_e}", flush=True)

    # Query local kernel_manager_proxy over HTTP and garbage-collect old idle/starting kernels (keep newest 2)
    try:
        with urllib.request.urlopen("http://172.28.0.12:6000/api/kernels", timeout=5) as kr:
            ks = json.loads(kr.read().decode("utf-8"))
        reap_ks = [k for k in ks if k.get("execution_state") in ("idle", "starting")]
        reap_ks.sort(key=lambda x: x.get("last_activity", ""))
        for old_k in reap_ks[:-2]:
            try:
                d_req = urllib.request.Request(
                    f"http://172.28.0.12:6000/api/kernels/{old_k['id']}",
                    method="DELETE",
                )
                urllib.request.urlopen(d_req, timeout=5).read()
            except Exception:
                pass
    except Exception:
        pass

    return {"endpoint": endpoint, "proxy_url": proxy_url}


def is_proc_running(pattern: str) -> bool:
    """Safely check /proc/*/cmdline in pure Python without spawning any subshell."""
    my_pid = os.getpid()
    for p in Path("/proc").iterdir():
        if p.name.isdigit() and int(p.name) != my_pid:
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
                if pattern in cmd and "colab_kernel_launcher" not in cmd:
                    return True
            except Exception:
                pass
    return False


def main():
    print("[In-VM Cloud Self-Keepalive] Started inside Colab VM!", flush=True)
    last_mtimes = {}
    while True:
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        if now_utc >= STOP_AT_UTC:
            try:
                sync_ckpt_to_gdrive_in_vm(last_mtimes)
            except Exception:
                pass
            print(f"[In-VM Keepalive] Reached {STOP_AT_UTC}. Exiting.", flush=True)
            break
        try:
            creds = get_valid_creds()
            info = ping_colab_control_plane(creds)
            refresh_dfs_persistent_in_vm(creds, info.get("endpoint", ""))
            sync_ckpt_to_gdrive_in_vm(last_mtimes)
            print(f"[{now_utc} UTC] Self-keepalive & GDrive token OK: {info.get('proxy_url')}", flush=True)
        except Exception as e:
            print(f"[{now_utc} UTC] Self-keepalive error: {e}", flush=True)

        # Ensure dashboard_server.py and cloudflared stay running
        if not is_proc_running("dashboard_server.py"):
            subprocess.Popen(
                "nohup python3 /content/turbogo/scripts/dashboard_server.py > /content/dashboard.log 2>&1 </dev/null &",
                shell=True,
                start_new_session=True,
            )
        if not is_proc_running("cloudflared tunnel") and os.path.exists("/usr/local/bin/cloudflared"):
            subprocess.Popen(
                "nohup /usr/local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 > /content/cloudflared.log 2>&1 </dev/null &",
                shell=True,
                start_new_session=True,
            )

        # Ensure train_turbogo_mini.py stays running until turbogo_mini_final.pt is created
        shards_ready = os.path.exists("/content/data/katago_19x19_chinese_10m/shard_00499.npz")
        finished = os.path.exists("/content/checkpoints/turbogo_mini/turbogo_mini_final.pt")
        if shards_ready and not finished and not is_proc_running("train_turbogo_mini.py") and not is_proc_running("bootstrap_turbogo_mini.py"):
            subprocess.Popen(
                "nohup python3 /content/turbogo/scripts/train_turbogo_mini.py >> /content/train_turbogo_mini.log 2>&1 </dev/null &",
                shell=True,
                start_new_session=True,
            )
            print(f"[{now_utc} UTC] Auto-restarted train_turbogo_mini.py!", flush=True)

        time.sleep(180)


if __name__ == "__main__":
    main()
