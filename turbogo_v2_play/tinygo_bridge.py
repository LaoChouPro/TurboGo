#!/usr/bin/env python3
"""
TinyGo Large (10-Block ResNet c128, 3.22M params) Bridge for 19x19.com
Matches against 4段 AI (星灵狐) on 19x19.com.
Selects moves exclusively via TinyGo Large's intuition policy.
Streams live board, winrate, score lead, and ownership to http://aiserver.laochou.cloud:8798/mirror.
"""

import copy
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from tinygo_large.model import TinyGoNet
from remote_web_server import GoBoard19, move_to_str, PASS_MOVE

TINYGO_WEIGHTS = os.path.join(BASE_DIR, "weights", "best_model_large.pth")
REMOTE_MIRROR_URL = "http://aiserver.laochou.cloud:8798/api/sync_19x19"
REMOTE_GENMOVE_URL = "http://aiserver.laochou.cloud:8798/api/genmove_19x19"
LOG_PATH = os.path.join(BASE_DIR, "tinygo_vs_4dan.log")
STATUS_PATH = os.path.join(BASE_DIR, "tinygo_vs_4dan_status.json")

# Device for TinyGo Large
TINYGO_DEV = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"[TinyGo Bridge] Loading TinyGo Large onto {TINYGO_DEV}...", flush=True)

tinygo_model = TinyGoNet(num_blocks=10, channels=128, num_classes=361)
ckpt = torch.load(TINYGO_WEIGHTS, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    tinygo_model.load_state_dict(ckpt["model_state_dict"])
else:
    tinygo_model.load_state_dict(ckpt)
tinygo_model.to(TINYGO_DEV).eval()
print(f"[TinyGo Bridge] TinyGo Large loaded (3.22M params) successfully!", flush=True)

STATUS = {
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "running": True,
    "model_name": "TinyGo Large (3.22M ResNet)",
    "opponent": "星灵狐 (4段)",
    "current_game": None,
    "result": None,
}


def log_line(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def push_remote_sync(payload: dict):
    try:
        req = urllib.request.Request(
            REMOTE_MIRROR_URL,
            data=json.dumps({**payload, "session_id": "mirror_19x19"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1.5).read()
    except Exception:
        pass


def get_referee_eval(payload: dict) -> dict:
    """Query TurboGo on the remote server as an impartial referee for winrate and score lead."""
    try:
        req = urllib.request.Request(
            REMOTE_GENMOVE_URL,
            data=json.dumps({**payload, "session_id": "mirror_19x19"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {
            "black_winrate": 0.5,
            "white_winrate": 0.5,
            "black_score_lead": 0.0,
            "territory_ownership": [0.0] * 361,
            "gpu_nn_ms": 0.0,
        }


def eval_tinygo(payload: dict) -> dict:
    t0 = time.perf_counter()
    game = GoBoard19(komi=float(payload.get("komi", 7.5)))
    now = int(payload.get("now", 1))
    steps = payload.get("steps", [])
    stones = payload.get("stone", None)

    for s in steps:
        c_val = int(s.get("c", 0))
        coor = int(s.get("coor", -1))
        x_val = int(s.get("x", -1))
        y_val = int(s.get("y", -1))
        if c_val not in (1, -1):
            continue
        if coor < 0 and x_val < 0 and y_val < 0 and len(game.move_history) == 0:
            continue
        game.current_player = 1 if c_val == 1 else 2
        mv = coor if (0 <= coor < 361) else (y_val * 19 + x_val if (0 <= x_val < 19 and 0 <= y_val < 19) else PASS_MOVE)
        game.play(mv)

    if stones and len(stones) == 361:
        arr = np.array(stones, dtype=np.int8).reshape((19, 19))
        game.board = np.where(arr == 1, 1, np.where(arr == -1, 2, 0)).astype(np.int8)
    game.current_player = 1 if now == 1 else 2

    # 1. Prepare TinyGo 3-channel input
    pla = game.current_player
    opp = 3 - pla
    features = np.zeros((1, 3, 19, 19), dtype=np.float32)
    features[0, 0] = (game.board == pla).astype(np.float32)
    features[0, 1] = (game.board == opp).astype(np.float32)
    features[0, 2] = 1.0

    bx = torch.from_numpy(features).to(TINYGO_DEV)
    with torch.no_grad():
        logits_tensor = tinygo_model(bx)[0].float()
        if TINYGO_DEV == "mps":
            torch.mps.synchronize()
        dt_tinygo_ms = (time.perf_counter() - t0) * 1000.0
        logits = logits_tensor.cpu().numpy()

    # 2. Mask legal moves
    legal_mask = game.get_legal_mask()[:361]
    if not legal_mask.any():
        masked_logits = np.full(361, -1e9, dtype=np.float32)
    else:
        masked_logits = np.where(legal_mask, logits, -1e9)

    # 3. Probabilities and top candidates
    max_logit = float(masked_logits.max()) if legal_mask.any() else 0.0
    exp_logits = np.exp(masked_logits - max_logit)
    probs = exp_logits / max(float(exp_logits.sum()), 1e-9)

    sorted_indices = np.argsort(masked_logits)[::-1]
    top_candidates = []
    for m in sorted_indices[:12]:
        if not legal_mask[m]:
            break
        y, x = divmod(int(m), 19)
        top_candidates.append({
            "move": int(m),
            "x": int(x),
            "y": int(y),
            "coord": move_to_str(int(m)),
            "prob": float(probs[m]),
        })

    if not top_candidates:
        best_mv = 361
        best_coord = "PASS"
        best_prob = 1.0
        top_candidates.append({"move": 361, "x": -1, "y": -1, "coord": "PASS", "prob": 1.0})
    else:
        best_mv = top_candidates[0]["move"]
        best_coord = top_candidates[0]["coord"]
        best_prob = top_candidates[0]["prob"]

    # 4. Impartial referee evaluation (TurboGo) for live mirror
    referee = get_referee_eval(payload)

    return {
        "best_move": best_mv,
        "best_coord": best_coord,
        "best_prob": best_prob,
        "top_candidates": top_candidates,
        "black_winrate": referee.get("black_winrate", 0.5),
        "white_winrate": referee.get("white_winrate", 0.5),
        "black_score_lead": referee.get("black_score_lead", 0.0),
        "territory_ownership": referee.get("territory_ownership", [0.0] * 361),
        "gpu_nn_ms": round(dt_tinygo_ms, 2),
    }


class TinyGoBridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _send_json(self, data: dict, code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            self._send_json(STATUS)
            return
        self._send_json({"ok": True, "engine": "TinyGo Large"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/sync_board":
            meta = payload.get("meta", {})
            meta["model_name"] = "TinyGo Large (3.22M)"
            payload["meta"] = meta
            push_remote_sync(payload)
            self._send_json({"ok": True})
            return

        if self.path == "/genmove":
            meta = payload.get("meta", {})
            meta["model_name"] = "TinyGo Large (3.22M)"
            payload["meta"] = meta

            res = eval_tinygo(payload)

            step_num = meta.get("step", "?")
            user_color = "B" if payload.get("now", 1) == 1 else "W"
            opp_name = meta.get("ai_name", "星灵狐")
            opp_rank = meta.get("ai_rank", "4段")
            b_wr = res["black_winrate"] * 100.0
            w_wr = res["white_winrate"] * 100.0
            my_wr = b_wr if user_color == "B" else w_wr
            my_lead = res["black_score_lead"] if user_color == "B" else -res["black_score_lead"]

            STATUS["current_game"] = {
                "step": step_num,
                "user_color": user_color,
                "opponent": f"{opp_name}({opp_rank})",
                "last_move": res["best_coord"],
                "last_prob": round(res["best_prob"], 3),
                "my_winrate": round(my_wr, 1),
                "my_score_lead": round(my_lead, 1),
                "nn_ms": res["gpu_nn_ms"],
            }
            with open(STATUS_PATH, "w", encoding="utf-8") as f:
                json.dump(STATUS, f, ensure_ascii=False, indent=2)

            log_line(
                f"[TinyGo Large vs {opp_name}({opp_rank})] "
                f"#{step_num} ({user_color}) -> {res['best_coord']} (P={res['best_prob']*100:.1f}%) | "
                f"胜率(裁判)={my_wr:.1f}% | 目差={my_lead:+.1f} | 耗时={res['gpu_nn_ms']}ms (MPS)"
            )

            # Sync move to mirror
            push_remote_sync({
                **payload,
                "meta": {
                    **meta,
                    "model_name": "TinyGo Large (3.22M)",
                    "my_winrate": round(my_wr, 1),
                    "my_score_lead": round(my_lead, 1),
                    "step": step_num,
                    "last_move": res["best_coord"],
                    "status": f"TinyGo 落子 {res['best_coord']} (胜率 {my_wr:.1f}%)"
                }
            })

            self._send_json(res)
            return

        if self.path == "/log_event":
            ev = payload.get("event", "")
            if ev == "game_start":
                log_line(
                    f"=== 对局开始: TinyGo Large ({'执黑' if payload.get('user_color')==1 else '执白'}) "
                    f"vs {payload.get('ai_name')} ({payload.get('ai_rank')}), 贴目={payload.get('komi')} ==="
                )
                push_remote_sync({
                    "steps": [],
                    "now": 1,
                    "komi": payload.get("komi", 7.5),
                    "meta": {
                        "model_name": "TinyGo Large (3.22M)",
                        "ai_name": payload.get("ai_name"),
                        "ai_rank": payload.get("ai_rank"),
                        "user_color": payload.get("user_color", 1),
                        "status": "对局开始: TinyGo Large vs 星灵狐(4段)"
                    }
                })
            elif ev == "game_over":
                res_txt = payload.get("result_txt", "")
                is_win = bool(payload.get("is_win", False))
                moves_cnt = payload.get("moves", 0)
                log_line(
                    f"*** 对局结束: {'✅ TinyGo获胜' if is_win else '❌ TinyGo落败'} | "
                    f"{res_txt} | 总手数={moves_cnt} ***"
                )
                STATUS["running"] = False
                STATUS["result"] = {
                    "is_win": is_win,
                    "result_txt": res_txt,
                    "moves": moves_cnt,
                    "time": datetime.now().strftime("%H:%M:%S")
                }
                with open(STATUS_PATH, "w", encoding="utf-8") as f:
                    json.dump(STATUS, f, ensure_ascii=False, indent=2)

            self._send_json({"ok": True})
            return

        self._send_json({"ok": False, "error": "unknown endpoint"}, code=404)


if __name__ == "__main__":
    port = 8789
    server = ThreadingHTTPServer(("0.0.0.0", port), TinyGoBridgeHandler)
    log_line(f"TinyGo Large Bridge server started on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
