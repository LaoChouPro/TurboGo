#!/usr/bin/env python3
"""
Local 127.0.0.1:8789 Bridge for 19x19.com Ranked AI Climb (升降级对弈).
Forwards 1-Visit Intuition requests to http://aiserver.laochou.cloud:8798/api/genmove_19x19
(RTX 2070 SUPER FP16 + CUDA Graph) with local MPS fallback, and logs all moves & game results.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_web_server import FastGPUIntuitionEngine, GoBoard19, PASS_MOVE

REMOTE_URL = "http://aiserver.laochou.cloud:8798/api/genmove_19x19"
SYNC_URL = "http://aiserver.laochou.cloud:8798/api/sync_19x19"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climb_19x19.log")
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climb_19x19_status.json")
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "b12c384_best_weights.pt")

LOCAL_ENGINE = None
STATUS = {
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "running": True,
    "stopped_reason": None,
    "current_game": None,
    "games_played": 0,
    "wins": 0,
    "losses": 0,
    "initial_rank": None,
    "current_rank": None,
    "games": [],
}

if os.path.exists(STATUS_PATH):
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            old_st = json.load(f)
            if isinstance(old_st, dict) and old_st.get("games_played", 0) > 0:
                STATUS.update(old_st)
                STATUS["running"] = True
                STATUS["stopped_reason"] = None
    except Exception:
        pass


def push_remote_sync(payload: dict):
    try:
        req = urllib.request.Request(
            SYNC_URL,
            data=json.dumps({**payload, "session_id": "mirror_19x19"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2.0).read()
    except Exception:
        pass


def log_line(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_status():
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(STATUS, f, ensure_ascii=False, indent=2)


def eval_local_fallback(payload: dict) -> dict:
    global LOCAL_ENGINE
    if LOCAL_ENGINE is None:
        dev = "mps" if hasattr(__import__("torch").backends, "mps") and __import__("torch").backends.mps.is_available() else "cpu"
        LOCAL_ENGINE = FastGPUIntuitionEngine(CKPT_PATH, device_str=dev)
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
        import numpy as np
        arr = np.array(stones, dtype=np.int8).reshape((19, 19))
        game.board = np.where(arr == 1, 1, np.where(arr == -1, 2, 0)).astype(np.int8)
    game.current_player = 1 if now == 1 else 2
    return LOCAL_ENGINE.evaluate(game)


class BridgeHandler(BaseHTTPRequestHandler):
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
        self._send_json({"ok": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/sync_board":
            meta = payload.get("meta", {})
            meta["record"] = f"{STATUS['wins']}胜 {STATUS['losses']}负"
            payload["meta"] = meta
            push_remote_sync(payload)
            self._send_json({"ok": True})
            return

        if self.path == "/genmove":
            t0 = time.perf_counter()
            res = None
            source = "RTX2070S"
            meta = payload.get("meta", {})
            meta["record"] = f"{STATUS['wins']}胜 {STATUS['losses']}负"
            meta["user_color"] = payload.get("now", 1)
            meta["status"] = f"第{STATUS['games_played']+1}局 对局中"
            payload["meta"] = meta
            try:
                req = urllib.request.Request(
                    REMOTE_URL,
                    data=json.dumps({**payload, "session_id": "mirror_19x19"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                source = f"LocalMPS({e})"
                res = eval_local_fallback(payload)

            rtt_ms = (time.perf_counter() - t0) * 1000.0
            step_num = meta.get("step", "?")
            user_color = "B" if payload.get("now", 1) == 1 else "W"
            opp_name = meta.get("ai_name", "AI")
            opp_rank = meta.get("ai_rank", "?")
            user_rank = meta.get("user_rank", "?")
            b_wr = res["black_winrate"] * 100.0
            w_wr = res["white_winrate"] * 100.0
            my_wr = b_wr if user_color == "B" else w_wr
            my_lead = res["black_score_lead"] if user_color == "B" else -res["black_score_lead"]

            STATUS["current_rank"] = user_rank
            if STATUS["initial_rank"] is None and user_rank != "?":
                STATUS["initial_rank"] = user_rank
            STATUS["current_game"] = {
                "game_index": STATUS["games_played"] + 1,
                "user_rank": user_rank,
                "ai_name": opp_name,
                "ai_rank": opp_rank,
                "user_color": user_color,
                "komi": payload.get("komi", 7.5),
                "handicap": meta.get("handicap", 0),
                "step": step_num,
                "last_move": res["best_coord"],
                "last_prob": res["best_prob"],
                "my_winrate": round(my_wr, 1),
                "my_score_lead": round(my_lead, 1),
                "gpu_nn_ms": res["gpu_nn_ms"],
            }
            save_status()

            if isinstance(step_num, int) and (step_num <= 6 or step_num % 10 == 0 or my_wr < 20 or my_wr > 90):
                log_line(
                    f"[G#{STATUS['games_played']+1} {user_rank} vs {opp_name}({opp_rank})] "
                    f"#{step_num} ({user_color}) -> {res['best_coord']} (P={res['best_prob']*100:.1f}%) | "
                    f"胜率={my_wr:.1f}% | 目差={my_lead:+.1f} | GPU={res['gpu_nn_ms']:.2f}ms ({source})"
                )

            self._send_json(res)
            return

        if self.path == "/log_event":
            ev = payload.get("event", "")
            if ev == "game_start":
                log_line(
                    f"=== 开始第 {STATUS['games_played']+1} 局: 我方({payload.get('user_rank')}, "
                    f"{'执黑' if payload.get('user_color')==1 else '执白'}, 贴目={payload.get('komi')}, 让子={payload.get('handicap')}) "
                    f"vs {payload.get('ai_name')} ({payload.get('ai_rank')}) ==="
                )
                push_remote_sync({
                    "steps": [],
                    "now": 1,
                    "komi": payload.get("komi", 7.5),
                    "meta": {
                        "user_rank": payload.get("user_rank"),
                        "ai_name": payload.get("ai_name"),
                        "ai_rank": payload.get("ai_rank"),
                        "user_color": payload.get("user_color", 1),
                        "record": f"{STATUS['wins']}胜 {STATUS['losses']}负",
                        "status": f"第{STATUS['games_played']+1}局 开局"
                    }
                })
            elif ev == "game_over":
                moves_cnt = payload.get("moves", 0)
                res_txt = payload.get("result_txt", "")
                if STATUS["games"] and STATUS["games"][-1].get("moves") == moves_cnt and STATUS["games"][-1].get("result_txt") == res_txt:
                    self._send_json({"ok": True, "status": STATUS})
                    return
                STATUS["games_played"] = len(STATUS["games"]) + 1
                is_win = bool(payload.get("is_win", False))
                if is_win:
                    STATUS["wins"] = sum(1 for g in STATUS["games"] if g["is_win"]) + 1
                else:
                    STATUS["losses"] = sum(1 for g in STATUS["games"] if not g["is_win"]) + 1
                    STATUS["running"] = False
                    STATUS["stopped_reason"] = f"在 {payload.get('new_rank') or payload.get('user_rank')} 级别负于 {payload.get('ai_name')}({payload.get('ai_rank')}): {payload.get('result_txt')}"

                new_rank = payload.get("new_rank") or payload.get("user_rank")
                if new_rank:
                    STATUS["current_rank"] = new_rank

                game_rec = {
                    "game_num": STATUS["games_played"],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "user_rank_before": payload.get("user_rank"),
                    "user_rank_after": new_rank,
                    "ai_name": payload.get("ai_name"),
                    "ai_rank": payload.get("ai_rank"),
                    "user_color": "黑" if payload.get("user_color") == 1 else "白",
                    "handicap": payload.get("handicap", 0),
                    "komi": payload.get("komi", 7.5),
                    "moves": payload.get("moves", 0),
                    "is_win": is_win,
                    "result_txt": payload.get("result_txt", ""),
                    "promo_msg": payload.get("promo_msg", ""),
                    "goal_diff": payload.get("goal_diff"),
                }
                STATUS["games"].append(game_rec)
                save_status()

                outcome_str = "✅ 获胜" if is_win else "❌ 落败 (停止晋级)"
                log_line(
                    f"*** 第 {STATUS['games_played']} 局结束: {outcome_str} | {game_rec['result_txt']} | "
                    f"手数={game_rec['moves']} | 段位变化: {game_rec['user_rank_before']} -> {game_rec['user_rank_after']} "
                    f"({game_rec['promo_msg']}) | 累计战绩: {STATUS['wins']}胜 {STATUS['losses']}负 ***"
                )
                push_remote_sync({
                    "meta": {
                        "user_rank": new_rank,
                        "ai_name": payload.get("ai_name"),
                        "ai_rank": payload.get("ai_rank"),
                        "user_color": payload.get("user_color", 1),
                        "record": f"{STATUS['wins']}胜 {STATUS['losses']}负",
                        "status": f"第{STATUS['games_played']}局 {outcome_str}: {game_rec['result_txt']}"
                    }
                })
            elif ev == "info":
                log_line(f"[INFO] {payload.get('msg', '')}")

            self._send_json({"ok": True, "status": STATUS})
            return

        self._send_json({"error": "unknown"}, 404)


if __name__ == "__main__":
    save_status()
    log_line("Bridge server starting on http://127.0.0.1:8789 ...")
    srv = ThreadingHTTPServer(("127.0.0.1", 8789), BridgeHandler)
    srv.serve_forever()
