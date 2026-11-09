#!/usr/bin/env python3
"""
TurboGo-v2 Read-Only Local Inference & Game Engine Server
=========================================================
Loads the read-only TurboGo-v2 best checkpoint (`weights/b12c384_best_weights.pt`)
into `TurboGoHybridTransformer` (`b11c384-nbt22`, 22-Layer Global Self-Attention + SwiGLU + 2D-RoPE)
and serves a clean HTTP JSON API on http://127.0.0.1:8788 for the Swift Go client.
"""

import argparse
import copy
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PLAY_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLAY_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model import TurboGoHybridTransformer

COLS = "ABCDEFGHJKLMNOPQRST"


def move_to_coord(mv: int) -> str:
    if mv == 361:
        return "PASS"
    if mv < 0 or mv > 361:
        return "--"
    y, x = divmod(mv, 19)
    return f"{COLS[x]}{19 - y}"


def coord_to_move(coord: str) -> int:
    c = coord.strip().upper()
    if c in ("PASS", "P"):
        return 361
    if len(c) < 2:
        raise ValueError(f"Invalid coordinate: {coord}")
    col_ch = c[0]
    if col_ch not in COLS:
        raise ValueError(f"Invalid column letter: {col_ch}")
    x = COLS.index(col_ch)
    row_num = int(c[1:])
    if not (1 <= row_num <= 19):
        raise ValueError(f"Invalid row number: {row_num}")
    y = 19 - row_num
    return y * 19 + x


class GoBoard19:
    """19x19 Chinese-rules Go board with exact KataGo V7 22-plane + 19-global feature encoding."""

    def __init__(self, komi: float = 7.5):
        self.komi = komi
        self.board = np.zeros((19, 19), dtype=np.int8)  # 0=empty, 1=Black, 2=White
        self.to_play = 1  # 1=Black, 2=White
        self.ko_point: Optional[Tuple[int, int]] = None
        self.move_history: List[int] = []
        self.move_records: List[Dict] = []
        self.captures = {1: 0, 2: 0}  # stones captured by Black (1) and White (2)
        self.pos_hashes = {self.board.tobytes()}

    def clone(self) -> "GoBoard19":
        b = GoBoard19(self.komi)
        b.board = self.board.copy()
        b.to_play = self.to_play
        b.ko_point = self.ko_point
        b.move_history = list(self.move_history)
        b.move_records = copy.deepcopy(self.move_records)
        b.captures = dict(self.captures)
        b.pos_hashes = set(self.pos_hashes)
        return b

    def _find_group(self, start_y: int, start_x: int, board: np.ndarray):
        color = int(board[start_y, start_x])
        stones = [(start_y, start_x)]
        visited = {(start_y, start_x)}
        liberties = set()
        q = deque([(start_y, start_x)])
        while q:
            y, x = q.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 19 and 0 <= nx < 19:
                    c = int(board[ny, nx])
                    if c == 0:
                        liberties.add((ny, nx))
                    elif c == color and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        stones.append((ny, nx))
                        q.append((ny, nx))
        return stones, liberties

    def compute_groups_and_liberties(self) -> np.ndarray:
        visited = np.zeros((19, 19), dtype=bool)
        lib_count = np.zeros((19, 19), dtype=np.int16)
        for y in range(19):
            for x in range(19):
                if self.board[y, x] != 0 and not visited[y, x]:
                    stones, libs = self._find_group(y, x, self.board)
                    n_libs = len(libs)
                    for sy, sx in stones:
                        visited[sy, sx] = True
                        lib_count[sy, sx] = n_libs
        return lib_count

    def is_legal(self, move: int) -> bool:
        if move == 361:
            return True
        if move < 0 or move > 361:
            return False
        y, x = divmod(move, 19)
        if self.board[y, x] != 0:
            return False
        if self.ko_point is not None and (y, x) == self.ko_point:
            return False
        pla = self.to_play
        opp = 3 - pla

        tb = self.board.copy()
        tb[y, x] = pla
        captured_any = False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 19 and 0 <= nx < 19 and tb[ny, nx] == opp:
                stones, libs = self._find_group(ny, nx, tb)
                if len(libs) == 0:
                    captured_any = True
                    for sy, sx in stones:
                        tb[sy, sx] = 0

        if not captured_any:
            _, my_libs = self._find_group(y, x, tb)
            if len(my_libs) == 0:
                return False

        if tb.tobytes() in self.pos_hashes:
            return False
        return True

    def get_legal_mask(self) -> np.ndarray:
        mask = np.zeros(362, dtype=bool)
        mask[361] = True
        for m in range(361):
            y, x = divmod(m, 19)
            if self.board[y, x] == 0 and self.is_legal(m):
                mask[m] = True
        return mask

    def play(self, move: int) -> bool:
        if not self.is_legal(move):
            return False
        pla = self.to_play
        opp = 3 - pla
        if move == 361:
            self.ko_point = None
            self.move_history.append(361)
            self.to_play = opp
            return True

        y, x = divmod(move, 19)
        self.board[y, x] = pla
        captured = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 19 and 0 <= nx < 19 and self.board[ny, nx] == opp:
                stones, libs = self._find_group(ny, nx, self.board)
                if len(libs) == 0:
                    for sy, sx in stones:
                        self.board[sy, sx] = 0
                    captured.extend(stones)

        self.captures[pla] += len(captured)
        my_stones, my_libs = self._find_group(y, x, self.board)
        if len(captured) == 1 and len(my_stones) == 1 and len(my_libs) == 1:
            self.ko_point = captured[0]
        else:
            self.ko_point = None

        self.move_history.append(move)
        self.to_play = opp
        self.pos_hashes.add(self.board.tobytes())
        return True

    def chinese_area_score_black_minus_white(self) -> float:
        b_area = int((self.board == 1).sum())
        w_area = int((self.board == 2).sum())
        visited = np.zeros((19, 19), dtype=bool)
        for y in range(19):
            for x in range(19):
                if self.board[y, x] == 0 and not visited[y, x]:
                    region = [(y, x)]
                    visited[y, x] = True
                    q = deque([(y, x)])
                    borders = set()
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < 19 and 0 <= nx < 19:
                                c = int(self.board[ny, nx])
                                if c == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    region.append((ny, nx))
                                    q.append((ny, nx))
                                elif c != 0:
                                    borders.add(c)
                    if borders == {1}:
                        b_area += len(region)
                    elif borders == {2}:
                        w_area += len(region)
        return float(b_area - w_area) - self.komi

    def encode_katago_v7(self) -> Tuple[np.ndarray, np.ndarray]:
        spatial = np.zeros((22, 19, 19), dtype=np.float32)
        pla = self.to_play
        opp = 3 - pla

        spatial[0] = 1.0
        spatial[1] = (self.board == pla).astype(np.float32)
        spatial[2] = (self.board == opp).astype(np.float32)

        lib_count = self.compute_groups_and_liberties()
        spatial[3] = (lib_count == 1).astype(np.float32)
        spatial[4] = (lib_count == 2).astype(np.float32)
        spatial[5] = (lib_count == 3).astype(np.float32)

        if self.ko_point is not None:
            ky, kx = self.ko_point
            spatial[6, ky, kx] = 1.0

        n_hist = len(self.move_history)
        for k in range(1, 6):
            if n_hist >= k:
                mv = self.move_history[-k]
                if 0 <= mv < 361:
                    my, mx = divmod(mv, 19)
                    spatial[8 + k, my, mx] = 1.0

        spatial[14] = spatial[3]
        spatial[15] = spatial[3]
        spatial[16] = spatial[3]

        spatial[18] = spatial[1].copy()
        spatial[19] = spatial[2].copy()
        visited = np.zeros((19, 19), dtype=bool)
        for y in range(19):
            for x in range(19):
                if self.board[y, x] == 0 and not visited[y, x]:
                    region = [(y, x)]
                    visited[y, x] = True
                    q = deque([(y, x)])
                    borders = set()
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < 19 and 0 <= nx < 19:
                                c = int(self.board[ny, nx])
                                if c == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    region.append((ny, nx))
                                    q.append((ny, nx))
                                elif c != 0:
                                    borders.add(c)
                    if borders == {pla}:
                        for ry, rx in region:
                            spatial[18, ry, rx] = 1.0
                    elif borders == {opp}:
                        for ry, rx in region:
                            spatial[19, ry, rx] = 1.0

        global_in = np.zeros((19,), dtype=np.float32)
        for k in range(1, 6):
            if n_hist >= k and self.move_history[-k] == 361:
                global_in[k - 1] = 1.0

        self_komi = -self.komi if pla == 1 else self.komi
        global_in[5] = self_komi / 20.0
        global_in[6] = 1.0
        global_in[7] = 0.5
        global_in[18] = -0.5 if pla == 1 else 0.5
        return spatial, global_in


class TurboGoV2Engine:
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.lock = threading.Lock()

        t0 = time.perf_counter()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        sd = ckpt.get("model") or ckpt.get("ema_model") or ckpt

        self.model = TurboGoHybridTransformer(
            d_model=cfg.get("d_model", 384),
            c_mid=cfg.get("c_mid", 256),
            n_blocks=cfg.get("n_blocks", 11),
            internal_length=cfg.get("internal_length", 2),
            n_heads=cfg.get("n_heads", 8),
            ffn_dim=cfg.get("ffn_dim", 640),
            in_channels=cfg.get("in_channels", 32),
        )
        self.model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, strict=True)
        self.model.to(self.device).eval()

        num_params = sum(p.numel() for p in self.model.parameters())
        val_m = ckpt.get("val_metrics", {})
        self.model_info = {
            "name": "TurboGo-v2 Best (b11c384-nbt22)",
            "arch": cfg.get("arch", "turbogo_v2_nbt22"),
            "version": cfg.get("version", "v2_100m"),
            "params": num_params,
            "params_m": round(num_params / 1e6, 2),
            "epoch": ckpt.get("epoch", 0),
            "chunk": ckpt.get("chunk", 0),
            "global_step": ckpt.get("global_step", 0),
            "composite_score": ckpt.get("composite_score", 0.0),
            "val_top1_acc": val_m.get("val_top1_acc", 0.0),
            "val_top3_acc": val_m.get("val_top3_acc", 0.0),
            "val_wld_acc": val_m.get("val_wld_acc", 0.0),
            "val_score_mae_pts": val_m.get("val_score_mae_pts", 0.0),
            "val_p0_loss": val_m.get("val_p0_loss", 0.0),
            "val_loss": val_m.get("val_loss", 0.0),
            "device": str(self.device),
            "weights_path": str(weights_path),
            "load_time_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

        self.board = GoBoard19(komi=7.5)
        self.undo_stack: List[GoBoard19] = []
        self.top_k = 8  # Default: 8-candidate Policy + Value lookahead

        # Warmup inference on device
        self.evaluate_position(self.board)

    @torch.no_grad()
    def evaluate_position(self, board: GoBoard19) -> Dict:
        t0 = time.perf_counter()
        sx, gx = board.encode_katago_v7()
        bx = torch.from_numpy(sx).unsqueeze(0).to(self.device)
        bg = torch.from_numpy(gx).unsqueeze(0).to(self.device)

        pol_logits, wld_logits, score_pred, own_pred = self.model(bx, bg)
        p0 = pol_logits[0, 0].float().cpu().numpy()
        wld = F.softmax(wld_logits[0].float(), dim=-1).cpu().numpy()
        pla_wr = float(wld[0])
        pla_loss = float(wld[1])
        pla_lead = float(score_pred[0].item() * 20.0)
        own_map = own_pred[0].float().cpu().numpy()

        legal_mask = board.get_legal_mask()
        masked_logits = np.where(legal_mask, p0, -1e9)
        max_l = float(np.max(masked_logits))
        probs = np.where(legal_mask, np.exp(masked_logits - max_l), 0.0)
        prob_sum = float(probs.sum())
        if prob_sum > 0:
            probs /= prob_sum

        if board.to_play == 1:
            black_wr = pla_wr
            black_lead = pla_lead
            black_own = own_map
        else:
            black_wr = pla_loss
            black_lead = -pla_lead
            black_own = -own_map

        top_idx = np.argsort(probs)[::-1][:6]
        candidates = []
        for idx in top_idx:
            idx = int(idx)
            if not legal_mask[idx]:
                continue
            y, x = divmod(idx, 19) if idx < 361 else (-1, -1)
            candidates.append(
                {
                    "move": idx,
                    "coord": move_to_coord(idx),
                    "row": y,
                    "col": x,
                    "prob": round(float(probs[idx]) * 100.0, 2),
                }
            )

        infer_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "black_winrate": round(black_wr * 100.0, 2),
            "black_score_lead": round(black_lead, 2),
            "ownership_black": np.round(black_own, 3).tolist(),
            "candidates": candidates,
            "infer_ms": infer_ms,
        }

    @torch.no_grad()
    def select_ai_move(self, board: GoBoard19, top_k: int = 8) -> Tuple[int, Dict]:
        t0 = time.perf_counter()
        sx, gx = board.encode_katago_v7()
        bx = torch.from_numpy(sx).unsqueeze(0).to(self.device)
        bg = torch.from_numpy(gx).unsqueeze(0).to(self.device)

        pol_logits, wld_logits, score_pred, _ = self.model(bx, bg)
        p0 = pol_logits[0, 0].float().cpu().numpy()

        legal_mask = board.get_legal_mask()
        if len(board.move_history) < 120 and legal_mask[:361].any():
            legal_mask[361] = False

        masked_logits = np.where(legal_mask, p0, -1e9)
        if top_k <= 1:
            best_mv = int(np.argmax(masked_logits))
            return best_mv, {"search_ms": round((time.perf_counter() - t0) * 1000.0, 1), "top_k": 1}

        cand_indices = np.argsort(masked_logits)[::-1][:top_k]
        cand_indices = [int(m) for m in cand_indices if legal_mask[m]]
        if len(cand_indices) <= 1:
            best_mv = cand_indices[0] if cand_indices else 361
            return best_mv, {"search_ms": round((time.perf_counter() - t0) * 1000.0, 1), "top_k": top_k}

        prior_probs = np.exp(masked_logits[cand_indices] - masked_logits[cand_indices[0]])
        prior_probs /= prior_probs.sum()

        child_sx = []
        child_gx = []
        valid_cands = []
        valid_priors = []
        for m, pr in zip(cand_indices, prior_probs):
            b_next = board.clone()
            if b_next.play(m):
                csx, cgx = b_next.encode_katago_v7()
                child_sx.append(csx)
                child_gx.append(cgx)
                valid_cands.append(m)
                valid_priors.append(pr)

        if not valid_cands:
            return int(np.argmax(masked_logits)), {"search_ms": round((time.perf_counter() - t0) * 1000.0, 1), "top_k": top_k}

        cbx = torch.from_numpy(np.stack(child_sx, axis=0)).to(self.device)
        cbg = torch.from_numpy(np.stack(child_gx, axis=0)).to(self.device)
        _, c_wld_logits, c_score_pred, _ = self.model(cbx, cbg)
        c_wld = F.softmax(c_wld_logits.float(), dim=-1).cpu().numpy()

        my_wr = c_wld[:, 1] - c_wld[:, 0]
        my_lead = -c_score_pred.float().cpu().numpy() * 20.0
        my_util = my_wr + 0.05 * np.arctan(my_lead / 7.5)
        comb_score = my_util + 0.35 * np.log(np.array(valid_priors) + 1e-4)
        best_idx = int(np.argmax(comb_score))
        best_mv = valid_cands[best_idx]

        return best_mv, {
            "search_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "top_k": top_k,
        }

    def get_state_dict(self, extra: Optional[Dict] = None) -> Dict:
        eval_res = self.evaluate_position(self.board)
        last_mv = self.board.move_history[-1] if self.board.move_history else -1
        consec_passes = 0
        for mv in reversed(self.board.move_history):
            if mv == 361:
                consec_passes += 1
            else:
                break

        area_score = self.board.chinese_area_score_black_minus_white()
        state = {
            "ok": True,
            "model_info": self.model_info,
            "board": self.board.board.tolist(),
            "to_play": int(self.board.to_play),
            "ply": len(self.board.move_history),
            "last_move": int(last_mv),
            "last_coord": move_to_coord(last_mv) if last_mv >= 0 else "--",
            "captures": {"black": self.board.captures[1], "white": self.board.captures[2]},
            "komi": self.board.komi,
            "top_k": self.top_k,
            "game_over": consec_passes >= 2,
            "chinese_area_diff": round(area_score, 1),
            "move_records": self.board.move_records[-40:],
            **eval_res,
        }
        if extra:
            state.update(extra)
        return state


ENGINE: Optional[TurboGoV2Engine] = None


class EngineHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, payload: Dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert ENGINE is not None
        with ENGINE.lock:
            if self.path.startswith("/api/state") or self.path == "/":
                self._send_json(ENGINE.get_state_dict())
            else:
                self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self):
        assert ENGINE is not None
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            req = json.loads(raw) if raw.strip() else {}
        except Exception:
            req = {}

        with ENGINE.lock:
            if self.path.startswith("/api/new_game"):
                komi = float(req.get("komi", 7.5))
                if "top_k" in req:
                    ENGINE.top_k = max(1, min(24, int(req["top_k"])))
                ENGINE.board = GoBoard19(komi=komi)
                ENGINE.undo_stack.clear()
                self._send_json(ENGINE.get_state_dict())

            elif self.path.startswith("/api/set_config"):
                if "top_k" in req:
                    ENGINE.top_k = max(1, min(24, int(req["top_k"])))
                self._send_json(ENGINE.get_state_dict())

            elif self.path.startswith("/api/undo"):
                steps = int(req.get("steps", 1))
                for _ in range(steps):
                    if ENGINE.undo_stack:
                        ENGINE.board = ENGINE.undo_stack.pop()
                self._send_json(ENGINE.get_state_dict())

            elif self.path.startswith("/api/play"):
                if "move" in req:
                    mv = int(req["move"])
                elif "coord" in req:
                    try:
                        mv = coord_to_move(str(req["coord"]))
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, status=400)
                        return
                else:
                    self._send_json({"ok": False, "error": "Missing move or coord"}, status=400)
                    return

                snap = ENGINE.board.clone()
                color_played = ENGINE.board.to_play
                if not ENGINE.board.play(mv):
                    self._send_json({"ok": False, "error": f"Illegal move: {move_to_coord(mv)}"}, status=400)
                    return

                ENGINE.undo_stack.append(snap)
                st = ENGINE.get_state_dict()
                ENGINE.board.move_records.append(
                    {
                        "ply": len(ENGINE.board.move_history),
                        "color": color_played,
                        "move": mv,
                        "coord": move_to_coord(mv),
                        "black_winrate": st["black_winrate"],
                        "black_score_lead": st["black_score_lead"],
                    }
                )
                st["move_records"] = ENGINE.board.move_records[-40:]
                self._send_json(st)

            elif self.path.startswith("/api/ai_move"):
                top_k = int(req.get("top_k", ENGINE.top_k))
                snap = ENGINE.board.clone()
                color_played = ENGINE.board.to_play
                mv, search_meta = ENGINE.select_ai_move(ENGINE.board, top_k=top_k)
                ENGINE.board.play(mv)
                ENGINE.undo_stack.append(snap)
                st = ENGINE.get_state_dict(extra=search_meta)
                ENGINE.board.move_records.append(
                    {
                        "ply": len(ENGINE.board.move_history),
                        "color": color_played,
                        "move": mv,
                        "coord": move_to_coord(mv),
                        "black_winrate": st["black_winrate"],
                        "black_score_lead": st["black_score_lead"],
                    }
                )
                st["move_records"] = ENGINE.board.move_records[-40:]
                self._send_json(st)

            else:
                self._send_json({"ok": False, "error": "Unknown endpoint"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="TurboGo-v2 Local Inference Server")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PLAY_DIR / "weights" / "b12c384_best_weights.pt"),
        help="Path to read-only TurboGo-v2 weights file",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    global ENGINE
    w_path = Path(args.weights)
    if not w_path.exists():
        print(f"[Error] Weights file not found: {w_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[TurboGo-v2 Engine] Loading read-only checkpoint: {w_path} ...", flush=True)
    ENGINE = TurboGoV2Engine(w_path)
    info = ENGINE.model_info
    print("=" * 80, flush=True)
    print(f"  Model     : {info['name']} ({info['params_m']}M params, arch={info['arch']})", flush=True)
    print(f"  Progress  : Epoch {info['epoch']} | Chunk {info['chunk']} | Step {info['global_step']:,}", flush=True)
    print(
        f"  Val Stats : Top-1={info['val_top1_acc']:.2f}% | Top-3={info['val_top3_acc']:.2f}% | "
        f"WLD={info['val_wld_acc']:.2f}% | ScoreMAE={info['val_score_mae_pts']:.2f}pt",
        flush=True,
    )
    print(f"  Device    : {info['device']} (loaded in {info['load_time_ms']} ms)", flush=True)
    print(f"  Listening : http://{args.host}:{args.port}", flush=True)
    print("=" * 80, flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), EngineHTTPHandler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
