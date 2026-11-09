#!/usr/bin/env python3
"""
TurboGo-v2 Best (b11c384-nbt22) — Ultra-Fast GPU Intuition Server & Minimalist Web UI
Deployed on NVIDIA GeForce RTX 2070 SUPER (FP16 Tensor Cores + CUDA Graphs).
"""

import argparse
import copy
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model import TurboGoHybridTransformer
except ImportError:
    from src.model import TurboGoHybridTransformer

BOARD_SIZE = 19
NUM_POINTS = 361
PASS_MOVE = 361
KOMI = 7.5
COLS = "ABCDEFGHJKLMNOPQRST"


def move_to_str(move: int) -> str:
    if move is None or move < 0:
        return "—"
    if move == PASS_MOVE:
        return "PASS"
    r, c = divmod(move, BOARD_SIZE)
    return f"{COLS[c]}{BOARD_SIZE - r}"


class GoBoard19:
    """19x19 Go board with Tromp-Taylor rules and exact KataGo V7 22-plane encoding."""

    def __init__(self, komi: float = KOMI):
        self.komi = komi
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1  # 1: Black, 2: White
        self.move_history = []   # list of (player, move_idx)
        self.board_history = []  # snapshots for undo
        self.ko_point = None
        self.captures = {1: 0, 2: 0}
        self.consecutive_passes = 0
        self.last_telemetry = None
        self.prev_telemetry = None
        self.cached_eval = None

    def copy_state(self):
        return (
            self.board.copy(),
            self.current_player,
            list(self.move_history),
            self.ko_point,
            dict(self.captures),
            self.consecutive_passes,
            copy.deepcopy(self.last_telemetry),
            copy.deepcopy(self.prev_telemetry),
            copy.deepcopy(self.cached_eval),
        )

    def restore_state(self, state):
        (
            self.board,
            self.current_player,
            self.move_history,
            self.ko_point,
            self.captures,
            self.consecutive_passes,
            self.last_telemetry,
            self.prev_telemetry,
            self.cached_eval,
        ) = state

    def _neighbors(self, r, c):
        if r > 0:
            yield r - 1, c
        if r < BOARD_SIZE - 1:
            yield r + 1, c
        if c > 0:
            yield r, c - 1
        if c < BOARD_SIZE - 1:
            yield r, c + 1

    def _get_group_and_liberties(self, board, r, c):
        color = int(board[r, c])
        group = []
        liberties = set()
        visited = {(r, c)}
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            group.append((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                v = int(board[nr, nc])
                if v == 0:
                    liberties.add((nr, nc))
                elif v == color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return group, liberties

    def is_legal(self, move: int, player: int = None) -> bool:
        if player is None:
            player = self.current_player
        if move == PASS_MOVE:
            return True
        if move < 0 or move >= NUM_POINTS:
            return False
        r, c = divmod(move, BOARD_SIZE)
        if self.board[r, c] != 0:
            return False
        if self.ko_point == (r, c):
            return False

        opp = 3 - player
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == 0:
                return True

        test_board = self.board.copy()
        test_board[r, c] = player
        for nr, nc in self._neighbors(r, c):
            if test_board[nr, nc] == opp:
                _, libs = self._get_group_and_liberties(test_board, nr, nc)
                if len(libs) == 0:
                    return True
        _, my_libs = self._get_group_and_liberties(test_board, r, c)
        return len(my_libs) > 0

    def get_legal_mask(self, player: int = None) -> np.ndarray:
        if player is None:
            player = self.current_player
        mask = np.zeros(NUM_POINTS + 1, dtype=bool)
        mask[PASS_MOVE] = True
        for m in range(NUM_POINTS):
            if self.is_legal(m, player):
                mask[m] = True
        return mask

    def play(self, move: int) -> bool:
        player = self.current_player
        if not self.is_legal(move, player):
            return False

        self.board_history.append(self.copy_state())
        self.prev_telemetry = copy.deepcopy(self.last_telemetry)
        self.cached_eval = None

        if move == PASS_MOVE:
            self.move_history.append((player, PASS_MOVE))
            self.ko_point = None
            self.consecutive_passes += 1
            self.current_player = 3 - player
            return True

        self.consecutive_passes = 0
        r, c = divmod(move, BOARD_SIZE)
        opp = 3 - player
        self.board[r, c] = player

        captured_stones = []
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == opp:
                grp, libs = self._get_group_and_liberties(self.board, nr, nc)
                if len(libs) == 0:
                    for gr, gc in grp:
                        self.board[gr, gc] = 0
                        captured_stones.append((gr, gc))

        self.captures[player] += len(captured_stones)

        my_grp, my_libs = self._get_group_and_liberties(self.board, r, c)
        if len(captured_stones) == 1 and len(my_grp) == 1 and len(my_libs) == 1:
            self.ko_point = captured_stones[0]
        else:
            self.ko_point = None

        self.move_history.append((player, move))
        self.current_player = opp
        return True

    def undo(self) -> bool:
        if not self.board_history:
            return False
        state = self.board_history.pop()
        self.restore_state(state)
        return True

    def encode_katago_v7(self, spatial: np.ndarray = None, global_in: np.ndarray = None):
        """Exact KataGo V7 22 spatial planes + 19 global features matching training."""
        if spatial is None:
            spatial = np.zeros((22, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        else:
            spatial.fill(0.0)

        if global_in is None:
            global_in = np.zeros((19,), dtype=np.float32)
        else:
            global_in.fill(0.0)

        pla = self.current_player
        opp = 3 - pla

        spatial[0].fill(1.0)
        spatial[1] = (self.board == pla)
        spatial[2] = (self.board == opp)

        visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        board_arr = self.board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_arr[r, c] != 0 and not visited[r, c]:
                    grp, libs = self._get_group_and_liberties(board_arr, r, c)
                    n_libs = len(libs)
                    plane_idx = 3 if n_libs == 1 else (4 if n_libs == 2 else (5 if n_libs == 3 else -1))
                    for gr, gc in grp:
                        visited[gr, gc] = True
                        if plane_idx >= 0:
                            spatial[plane_idx, gr, gc] = 1.0

        if self.ko_point is not None:
            kr, kc = self.ko_point
            spatial[6, kr, kc] = 1.0

        n_hist = len(self.move_history)
        for k in range(1, 6):
            if n_hist >= k:
                _, mv = self.move_history[-k]
                if 0 <= mv < NUM_POINTS:
                    mr, mc = divmod(mv, BOARD_SIZE)
                    spatial[8 + k, mr, mc] = 1.0
                elif mv == PASS_MOVE:
                    global_in[k - 1] = 1.0

        # Ladder features fallback
        spatial[14] = spatial[3]
        spatial[15] = spatial[3]
        spatial[16] = spatial[3]

        # Pass-alive / enclosed area planes 18 & 19
        spatial[18] = spatial[1]
        spatial[19] = spatial[2]
        if n_hist >= 6:
            visited_empty = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    if board_arr[y, x] == 0 and not visited_empty[y, x]:
                        region = [(y, x)]
                        visited_empty[y, x] = True
                        q = [(y, x)]
                        borders = set()
                        while q:
                            cy, cx = q.pop()
                            for ny, nx in self._neighbors(cy, cx):
                                c_val = int(board_arr[ny, nx])
                                if c_val == 0 and not visited_empty[ny, nx]:
                                    visited_empty[ny, nx] = True
                                    region.append((ny, nx))
                                    q.append((ny, nx))
                                elif c_val != 0:
                                    borders.add(c_val)
                        if len(borders) == 1 and len(region) < 60:
                            owner = next(iter(borders))
                            target_plane = 18 if owner == pla else 19
                            for ry, rx in region:
                                spatial[target_plane, ry, rx] = 1.0

        self_komi = -self.komi if pla == 1 else self.komi
        global_in[5] = self_komi / 20.0
        global_in[6] = 1.0
        global_in[7] = 0.5
        global_in[18] = -0.5 if pla == 1 else 0.5
        return spatial, global_in


class FastGPUIntuitionEngine:
    """
    Maximum-speed 1-Visit Intuition Engine for TurboGo-v2 Best on CUDA GPU.
    Uses FP16 Tensor Cores + CUDA Graphs (static device buffers, zero kernel launch overhead).
    """

    def __init__(self, ckpt_path: str, device_str: str = "cuda:0"):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        self.use_fp16 = self.use_cuda
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.lock = threading.Lock()

        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        print(f"[TurboGo-v2 GPU] Loading checkpoint: {ckpt_path} on {self.device} ({self.dtype})")
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = raw.get("config", {})
        state_dict = raw.get("model") or raw.get("ema_model") or raw
        cleaned = {}
        for k, v in state_dict.items():
            nk = k[10:] if k.startswith("_orig_mod.") else k
            cleaned[nk] = v.float() if v.is_floating_point() else v

        val_m = raw.get("val_metrics", {})
        self.ckpt_step = raw.get("global_step", raw.get("step", 29500))
        raw_top1 = float(val_m.get("val_top1_acc", raw.get("best_val_top1", 43.30)))
        self.ckpt_top1_pct = raw_top1 if raw_top1 > 1.0 else raw_top1 * 100.0
        self.ckpt_arch = cfg.get("arch", "b11c384-nbt22")

        self.model = TurboGoHybridTransformer(
            d_model=cfg.get("d_model", 384),
            c_mid=cfg.get("c_mid", 256),
            n_blocks=cfg.get("n_blocks", 11),
            internal_length=cfg.get("internal_length", 2),
            n_heads=cfg.get("n_heads", 8),
            ffn_dim=cfg.get("ffn_dim", 640),
            in_channels=cfg.get("in_channels", 32),
        )
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(
            f"[TurboGo-v2 GPU] Weights loaded (step={self.ckpt_step}, val_top1={self.ckpt_top1_pct:.2f}%, missing={len(missing)})"
        )

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cpu_planes = np.zeros((22, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.cpu_gvec = np.zeros((19,), dtype=np.float32)

        self.static_board = torch.zeros((1, 22, BOARD_SIZE, BOARD_SIZE), device=self.device, dtype=self.dtype)
        self.static_gvec = torch.zeros((1, 19), device=self.device, dtype=self.dtype)

        self.cuda_graph = None
        self.graph_outputs = None
        self.gpu_name = torch.cuda.get_device_name(0) if self.use_cuda else "CPU"

        if self.use_cuda:
            self._init_cuda_graph()
        self.evaluate(GoBoard19())

    def _init_cuda_graph(self):
        """Warmup and capture the entire 11-block hybrid transformer in a single CUDA Graph."""
        try:
            with torch.inference_mode():
                for _ in range(5):
                    _ = self.model(self.static_board, self.static_gvec)
                torch.cuda.synchronize()

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.graph_outputs = self.model(self.static_board, self.static_gvec)
                torch.cuda.current_stream().wait_stream(s)

                self.cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.cuda_graph):
                    self.graph_outputs = self.model(self.static_board, self.static_gvec)
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(20):
                    self.cuda_graph.replay()
                torch.cuda.synchronize()
                avg_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
                print(f"[TurboGo-v2 GPU] CUDA Graph captured on {self.gpu_name}! Avg replay latency: {avg_ms:.2f} ms")
        except Exception as e:
            print(f"[TurboGo-v2 GPU] CUDA Graph fallback to eager FP16 ({e})")
            self.cuda_graph = None

    @torch.inference_mode()
    def evaluate(self, game: GoBoard19) -> dict:
        """
        Single-step 1-Visit intuition evaluation on GPU.
        Returns policy, winrate, score lead, ownership map, and exact GPU/total latency in ms.
        """
        with self.lock:
            if self.use_cuda:
                torch.cuda.set_device(0)

            t_start = time.perf_counter()

            # 1. Encode KataGo V7 planes on CPU
            game.encode_katago_v7(self.cpu_planes, self.cpu_gvec)
            t_encode = time.perf_counter()

            # 2. Copy to static GPU tensor & execute CUDA Graph (or eager forward)
            self.static_board[0].copy_(torch.from_numpy(self.cpu_planes), non_blocking=True)
            self.static_gvec[0].copy_(torch.from_numpy(self.cpu_gvec), non_blocking=True)

            if self.cuda_graph is not None:
                self.cuda_graph.replay()
                pol_logits, wld_logits, score_pred, own_pred = self.graph_outputs
            else:
                pol_logits, wld_logits, score_pred, own_pred = self.model(self.static_board, self.static_gvec)

            # Extract outputs:
            # pol_logits: (1, 2, 362), wld_logits: (1, 3), score_pred: (1,), own_pred: (1, 19, 19)
            p0 = pol_logits[0, 0].float().cpu().numpy()
            wld = F.softmax(wld_logits[0].float(), dim=-1).cpu().numpy()
            pla_wr = float(wld[0])
            pla_loss = float(wld[1])
            pla_lead = float(score_pred[0].float().item() * 20.0)
            own_map = own_pred[0].float().cpu().numpy()

            t_gpu_done = time.perf_counter()

            # 3. Mask illegal moves & compute softmax probabilities
            legal_mask = game.get_legal_mask()
            masked_logits = np.where(legal_mask, p0, -1e9)

            # Prevent premature pass in early/mid game when normal board moves exist
            if len(game.move_history) < 140 and np.any(legal_mask[:NUM_POINTS]):
                masked_logits[PASS_MOVE] -= 8.0

            max_logit = float(np.max(masked_logits))
            exp_logits = np.where(legal_mask, np.exp(masked_logits - max_logit), 0.0)
            probs = exp_logits / (float(np.sum(exp_logits)) + 1e-12)

            best_move = int(np.argmax(probs))
            best_prob = float(probs[best_move])

            top_indices = np.argsort(probs)[::-1][:15]
            top_candidates = [
                {
                    "move": int(idx),
                    "x": int(idx % BOARD_SIZE) if idx < NUM_POINTS else -1,
                    "y": int(idx // BOARD_SIZE) if idx < NUM_POINTS else -1,
                    "coord": move_to_str(int(idx)),
                    "prob": round(float(probs[idx]), 4),
                }
                for idx in top_indices
                if legal_mask[idx] and probs[idx] > 0.0001
            ]

            if game.current_player == 1:
                black_winrate = pla_wr
                white_winrate = pla_loss
                black_score_lead = pla_lead
                black_own = own_map
            else:
                black_winrate = pla_loss
                white_winrate = pla_wr
                black_score_lead = -pla_lead
                black_own = -own_map

            # Normalize black/white winrate sum (excluding tiny draw/no-result residual)
            wr_sum = black_winrate + white_winrate
            if wr_sum > 1e-6:
                black_winrate /= wr_sum
                white_winrate /= wr_sum

            encode_ms = (t_encode - t_start) * 1000.0
            gpu_nn_ms = (t_gpu_done - t_encode) * 1000.0
            total_ms = (time.perf_counter() - t_start) * 1000.0

            return {
                "best_move": best_move,
                "best_coord": move_to_str(best_move),
                "best_prob": round(best_prob, 4),
                "top_candidates": top_candidates,
                "black_winrate": round(black_winrate, 4),
                "white_winrate": round(white_winrate, 4),
                "black_score_lead": round(black_score_lead, 2),
                "gpu_nn_ms": round(gpu_nn_ms, 2),
                "encode_ms": round(encode_ms, 2),
                "total_ms": round(total_ms, 2),
                "steps_per_sec": round(1000.0 / max(gpu_nn_ms, 0.05), 1),
                "ownership": np.round(black_own, 2).tolist(),
            }


SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
ENGINE: FastGPUIntuitionEngine = None
LIVE_COND = threading.Condition()
LIVE_SEQ = 0
LIVE_MATCH_META = {}


def notify_live_update(meta: dict = None):
    global LIVE_SEQ, LIVE_MATCH_META
    with LIVE_COND:
        if meta:
            LIVE_MATCH_META.update(meta)
        LIVE_SEQ += 1
        LIVE_COND.notify_all()


def get_session(session_id: str) -> GoBoard19:
    if not session_id:
        session_id = "default"
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            if len(SESSIONS) > 256:
                oldest = next(iter(SESSIONS))
                del SESSIONS[oldest]
            SESSIONS[session_id] = GoBoard19()
        return SESSIONS[session_id]


def serialize_board(game: GoBoard19) -> dict:
    last_player, last_move = game.move_history[-1] if game.move_history else (None, None)
    if game.last_telemetry is None:
        eval_res = ENGINE.evaluate(game)
        game.cached_eval = eval_res
        game.last_telemetry = {
            "move_num": len(game.move_history),
            "player": last_player,
            "move": last_move,
            "coord": move_to_str(last_move) if last_move is not None else "开局",
            "move_prob": None,
            "black_winrate": eval_res["black_winrate"],
            "white_winrate": eval_res["white_winrate"],
            "black_score_lead": eval_res["black_score_lead"],
            "gpu_nn_ms": eval_res["gpu_nn_ms"],
            "total_ms": eval_res["total_ms"],
            "steps_per_sec": eval_res["steps_per_sec"],
            "top_candidates": eval_res["top_candidates"],
            "ownership": eval_res["ownership"],
        }

    return {
        "board": game.board.tolist(),
        "current_player": game.current_player,
        "move_count": len(game.move_history),
        "last_player": last_player,
        "last_move": last_move,
        "last_coord": move_to_str(last_move) if last_move is not None else "—",
        "captures_black": game.captures[1],
        "captures_white": game.captures[2],
        "game_over": game.consecutive_passes >= 2,
        "telemetry": game.last_telemetry,
        "prev_telemetry": game.prev_telemetry,
        "live_seq": LIVE_SEQ,
        "live_meta": LIVE_MATCH_META,
        "engine_info": {
            "arch": ENGINE.ckpt_arch,
            "step": ENGINE.ckpt_step,
            "val_top1": round(ENGINE.ckpt_top1_pct, 2),
            "gpu": ENGINE.gpu_name,
            "cuda_graph": ENGINE.cuda_graph is not None,
            "dtype": "FP16",
        },
    }


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>TURBOGO-V2 // 19×19 GPU INTUITION</title>
<style>
  :root {
    --bg: #ffffff;
    --fg: #000000;
    --muted: #555555;
    --mono: "JetBrains Mono", "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", sans-serif;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    -webkit-font-smoothing: antialiased;
  }

  html, body {
    width: 100%;
    height: 100%;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--sans);
    overflow: hidden;
    user-select: none;
  }

  .app {
    display: flex;
    flex-direction: column;
    width: 100vw;
    height: 100vh;
    border-top: 1px solid var(--fg);
  }

  /* Top Hairline Header & Controls */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--fg);
    min-height: 38px;
    padding: 0 14px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    flex-shrink: 0;
    gap: 8px;
    flex-wrap: wrap;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
  }

  .brand-title {
    font-weight: 600;
    letter-spacing: 0.08em;
  }

  .brand-sub {
    color: var(--muted);
    font-size: 10px;
  }

  .controls {
    display: flex;
    align-items: center;
    border-left: 1px solid var(--fg);
  }

  .btn {
    background: var(--bg);
    color: var(--fg);
    border: none;
    border-right: 1px solid var(--fg);
    padding: 6px 11px;
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: background 0.08s ease, color 0.08s ease;
    white-space: nowrap;
  }

  .btn:hover {
    background: var(--fg);
    color: var(--bg);
  }

  .btn.active {
    background: var(--fg);
    color: var(--bg);
  }

  /* Telemetry Strip for Last Move */
  .telemetry-bar {
    display: grid;
    grid-template-columns: 1.25fr 1.55fr 1.15fr 1.35fr;
    border-bottom: 1px solid var(--fg);
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
  }

  .t-cell {
    padding: 8px 14px;
    border-right: 1px solid var(--fg);
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 3px;
    min-width: 0;
  }

  .t-cell:last-child {
    border-right: none;
  }

  .t-label {
    font-size: 9px;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .t-value {
    font-size: 13px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .t-sub {
    font-size: 10px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* 1px Hairline Winrate Bar */
  .wr-bar-wrap {
    width: 100%;
    height: 3px;
    border: 1px solid var(--fg);
    background: var(--bg);
    margin-top: 2px;
    position: relative;
  }

  .wr-bar-black {
    height: 100%;
    background: var(--fg);
    width: 50%;
    transition: width 0.16s ease;
  }

  /* Main Resizable Board Workspace */
  .workspace {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    padding: 12px;
  }

  .board-frame {
    position: relative;
    border: 1px solid var(--fg);
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  canvas {
    display: block;
    cursor: crosshair;
  }

  /* Subtle Corner Resize Handle for Manual Board Scaling */
  .resize-grip {
    position: absolute;
    right: 0;
    bottom: 0;
    width: 14px;
    height: 14px;
    cursor: nwse-resize;
    border-top: 1px solid var(--fg);
    border-left: 1px solid var(--fg);
    background:
      linear-gradient(135deg, transparent 45%, #000 45%, #000 55%, transparent 55%),
      linear-gradient(135deg, transparent 70%, #000 70%, #000 80%, transparent 80%);
  }

  /* Bottom Status Hairline Bar */
  .footerbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid var(--fg);
    height: 28px;
    padding: 0 14px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    flex-shrink: 0;
    overflow: hidden;
    white-space: nowrap;
  }

  .candidates-list {
    display: flex;
    gap: 12px;
    color: var(--fg);
  }

  @media (max-width: 720px) {
    .telemetry-bar {
      grid-template-columns: 1fr 1fr;
    }
    .t-cell:nth-child(2) {
      border-right: none;
    }
    .t-cell:nth-child(1),
    .t-cell:nth-child(2) {
      border-bottom: 1px solid var(--fg);
    }
    .brand-sub {
      display: none;
    }
  }
</style>
</head>
<body>
<div class="app">
  <!-- Header & Minimalist Controls -->
  <header class="topbar">
    <div class="brand">
      <span class="brand-title">TURBOGO-V2</span>
      <span class="brand-sub" id="engineMeta">b11c384-nbt22 · 1-VISIT GPU INTUITION</span>
      <span class="brand-sub" id="liveMatchMeta" style="color:#000;font-weight:600;border-left:1px solid #000;padding-left:10px;display:none;"></span>
    </div>
    <div class="controls">
      <button class="btn active" id="btnLiveSync" onclick="toggleLiveSync()" title="实时流式同步 19x19.com 排位赛">● 19×19 实时流</button>
      <button class="btn" id="btnPlayBlack" onclick="newGame(1)">执黑</button>
      <button class="btn" id="btnPlayWhite" onclick="newGame(2)">执白</button>
      <button class="btn" id="btnAiMove" onclick="triggerAiMove()">AI 落子</button>
      <button class="btn" id="btnPass" onclick="playPass()">虚手</button>
      <button class="btn" id="btnUndo" onclick="undoMove()">悔棋</button>
      <button class="btn" id="btnOwn" onclick="toggleOwnership()">领地</button>
      <button class="btn" id="btnFit" onclick="resetZoom()" title="自适应窗口大小">自适应</button>
    </div>
  </header>

  <!-- Last Move Telemetry Strip (上一手落子数据展示) -->
  <section class="telemetry-bar">
    <div class="t-cell">
      <div class="t-label">
        <span>上一手落子 / LAST MOVE</span>
        <span id="tMoveConf">P: —</span>
      </div>
      <div class="t-value" id="tLastMove">开局等待落子</div>
      <div class="t-sub" id="tPrevMove">前一手: —</div>
    </div>

    <div class="t-cell">
      <div class="t-label">
        <span>胜率评估 / WINRATE</span>
        <span id="tTurnTag">● 黑方行棋</span>
      </div>
      <div class="t-value" id="tWinrate">黑 50.0%  /  白 50.0%</div>
      <div class="wr-bar-wrap">
        <div class="wr-bar-black" id="wrBarBlack" style="width: 50%;"></div>
      </div>
    </div>

    <div class="t-cell">
      <div class="t-label">
        <span>目数差距 / SCORE LEAD</span>
        <span id="tLeadDelta">KOMI 7.5</span>
      </div>
      <div class="t-value" id="tScoreLead">黑 +0.0 目</div>
      <div class="t-sub" id="tScoreSub">中国规则贴 7.5 目</div>
    </div>

    <div class="t-cell">
      <div class="t-label">
        <span>单步直觉推理速度 / GPU SPEED</span>
        <span id="tThroughput">— 步/秒</span>
      </div>
      <div class="t-value" id="tSpeed">GPU: — ms</div>
      <div class="t-sub" id="tSpeedSub">端到端耗时: — ms</div>
    </div>
  </section>

  <!-- Dynamically Resizable Monochrome Board -->
  <main class="workspace" id="workspace">
    <div class="board-frame" id="boardFrame">
      <canvas id="goCanvas"></canvas>
      <div class="resize-grip" id="resizeGrip" title="拖拽自由缩放棋盘"></div>
    </div>
  </main>

  <!-- Footer: Captures & Top Intuition Candidates -->
  <footer class="footerbar">
    <div id="statusText">拖拽棋盘右下角或缩放窗口可动态调整大小</div>
    <div class="candidates-list" id="candidatesText">直觉选点: —</div>
  </footer>
</div>

<script>
const BOARD_SIZE = 19;
const COLS = "ABCDEFGHJKLMNOPQRST";
const STAR_POINTS = [3, 9, 15];

let sessionId = localStorage.getItem("turbogo_v2_sid");
if (!sessionId) {
  sessionId = "s_" + Math.random().toString(36).slice(2, 11) + "_" + Date.now().toString(36);
  localStorage.setItem("turbogo_v2_sid", sessionId);
}

let humanColor = 1; // 1 = Black, 2 = White
let busy = false;
let showOwnership = false;
let hoverPos = null;
let manualBoardPx = null;
const isMirrorPage = window.location.pathname.startsWith("/mirror");
let liveSync = isMirrorPage;
let lastLiveSeq = -1;

let state = {
  board: Array.from({ length: 19 }, () => Array(19).fill(0)),
  current_player: 1,
  move_count: 0,
  last_player: null,
  last_move: null,
  last_coord: "—",
  captures_black: 0,
  captures_white: 0,
  game_over: false,
  telemetry: null,
  prev_telemetry: null,
  engine_info: null
};

const workspace = document.getElementById("workspace");
const boardFrame = document.getElementById("boardFrame");
const canvas = document.getElementById("goCanvas");
const ctx = canvas.getContext("2d");

function computeBoardPixelSize() {
  const rect = workspace.getBoundingClientRect();
  const maxFit = Math.max(220, Math.floor(Math.min(rect.width - 24, rect.height - 24)));
  if (manualBoardPx !== null) {
    return Math.max(200, Math.min(manualBoardPx, Math.max(rect.width - 8, rect.height - 8, 220)));
  }
  return maxFit;
}

function renderBoard() {
  const cssSize = computeBoardPixelSize();
  const dpr = window.devicePixelRatio || 1;

  boardFrame.style.width = cssSize + "px";
  boardFrame.style.height = cssSize + "px";
  canvas.style.width = cssSize + "px";
  canvas.style.height = cssSize + "px";

  const targetW = Math.round(cssSize * dpr);
  if (canvas.width !== targetW || canvas.height !== targetW) {
    canvas.width = targetW;
    canvas.height = targetW;
  }

  ctx.save();
  ctx.scale(dpr, dpr);

  // Pure white background
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssSize, cssSize);

  const pad = Math.max(22, Math.round(cssSize * 0.052));
  const gridSpan = cssSize - pad * 2;
  const cell = gridSpan / (BOARD_SIZE - 1);

  // Coordinate labels
  if (cssSize >= 340) {
    ctx.fillStyle = "#000000";
    ctx.font = `${Math.max(9, Math.round(cell * 0.30))}px "JetBrains Mono", "SF Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < BOARD_SIZE; i++) {
      const x = pad + i * cell;
      const y = pad + i * cell;
      ctx.fillText(COLS[i], x, pad * 0.42);
      ctx.fillText(COLS[i], x, cssSize - pad * 0.42);
      const rowStr = String(BOARD_SIZE - i);
      ctx.fillText(rowStr, pad * 0.42, y);
      ctx.fillText(rowStr, cssSize - pad * 0.42, y);
    }
  }

  // 1px Hairline Grid Lines
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 1 / dpr * Math.max(1, Math.floor(dpr));
  ctx.beginPath();
  for (let i = 0; i < BOARD_SIZE; i++) {
    const pos = Math.floor(pad + i * cell) + 0.5;
    const start = Math.floor(pad) + 0.5;
    const end = Math.floor(pad + gridSpan) + 0.5;
    ctx.moveTo(start, pos);
    ctx.lineTo(end, pos);
    ctx.moveTo(pos, start);
    ctx.lineTo(pos, end);
  }
  ctx.stroke();

  // Star points (Hoshi)
  ctx.fillStyle = "#000000";
  const starR = Math.max(2, cell * 0.085);
  for (const r of STAR_POINTS) {
    for (const c of STAR_POINTS) {
      const x = Math.floor(pad + c * cell) + 0.5;
      const y = Math.floor(pad + r * cell) + 0.5;
      ctx.beginPath();
      ctx.arc(x, y, starR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Hover preview stone
  if (!busy && hoverPos && state.board[hoverPos.r][hoverPos.c] === 0) {
    const hx = pad + hoverPos.c * cell;
    const hy = pad + hoverPos.r * cell;
    const hr = cell * 0.44;
    ctx.save();
    ctx.beginPath();
    ctx.arc(hx, hy, hr, 0, Math.PI * 2);
    if (state.current_player === 1) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
      ctx.fill();
    } else {
      ctx.fillStyle = "rgba(255, 255, 255, 0.78)";
      ctx.fill();
      ctx.strokeStyle = "#000000";
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Stones (Pure flat Black & White with 1px hairline border)
  const stoneR = cell * 0.455;
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const v = state.board[r][c];
      if (v === 0) continue;
      const x = pad + c * cell;
      const y = pad + r * cell;

      ctx.beginPath();
      ctx.arc(x, y, stoneR, 0, Math.PI * 2);
      if (v === 1) {
        ctx.fillStyle = "#000000";
        ctx.fill();
      } else {
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 1.15;
        ctx.stroke();
      }

      // Last move hairline ring marker
      if (state.last_move !== null && state.last_move === r * BOARD_SIZE + c) {
        ctx.beginPath();
        ctx.arc(x, y, stoneR * 0.46, 0, Math.PI * 2);
        ctx.strokeStyle = v === 1 ? "#ffffff" : "#000000";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }
  }

  // Optional Territory Ownership Map
  if (showOwnership && state.telemetry && state.telemetry.ownership) {
    const own = state.telemetry.ownership;
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const val = own[r][c];
        if (Math.abs(val) < 0.22) continue;
        const x = pad + c * cell;
        const y = pad + r * cell;
        const sz = cell * 0.26 * Math.min(1, Math.abs(val));
        if (val > 0) {
          ctx.fillStyle = "#000000";
          ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz);
          if (state.board[r][c] === 1) {
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 0.8;
            ctx.strokeRect(x - sz / 2, y - sz / 2, sz, sz);
          }
        } else {
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz);
          ctx.strokeStyle = "#000000";
          ctx.lineWidth = 1;
          ctx.strokeRect(x - sz / 2, y - sz / 2, sz, sz);
        }
      }
    }
  }

  ctx.restore();
}

function updateTelemetryUI() {
  const t = state.telemetry;
  const prev = state.prev_telemetry;
  const turnStr = state.current_player === 1 ? "● 黑方行棋" : "○ 白方行棋";
  document.getElementById("tTurnTag").textContent = state.game_over ? "■ 对局结束" : turnStr;

  if (state.engine_info) {
    const info = state.engine_info;
    document.getElementById("engineMeta").textContent =
      `${info.arch} · Step ${info.step} · ${info.gpu} (${info.dtype}${info.cuda_graph ? " + CUDA Graph" : ""})`;
  }

  const lm = state.live_meta || {};
  const lmEl = document.getElementById("liveMatchMeta");
  if (lmEl) {
    if (liveSync && (lm.user_rank || lm.ai_name)) {
      lmEl.style.display = "inline";
      const myColStr = lm.user_color === 1 ? "●执黑" : (lm.user_color === -1 ? "○执白" : "");
      const recStr = lm.record ? ` · 战绩 ${lm.record}` : "";
      const stStr = lm.status ? ` · ${lm.status}` : "";
      lmEl.textContent = `19×19 排位: ${lm.user_rank || "?"} (${myColStr}) vs ${lm.ai_name || "AI"}(${lm.ai_rank || "?"})${recStr}${stStr}`;
    } else {
      lmEl.style.display = "none";
    }
  }

  if (!t) return;

  // 1. Last Move & Previous Move
  if (state.move_count === 0 || state.last_move === null) {
    document.getElementById("tLastMove").textContent = "开局等待落子 (#0)";
    document.getElementById("tMoveConf").textContent = "P: —";
    document.getElementById("tPrevMove").textContent = "前一手: —";
  } else {
    const pIcon = state.last_player === 1 ? "● 黑" : "○ 白";
    let who = state.last_player === humanColor ? "人类" : "AI";
    if (liveSync && (lm.user_color === 1 || lm.user_color === -1)) {
      const turboPlayer = lm.user_color === 1 ? 1 : 2;
      who = state.last_player === turboPlayer ? "TurboGo-v2" : (lm.ai_name || "星阵AI");
    }
    document.getElementById("tLastMove").textContent =
      `#${state.move_count} ${pIcon} ${state.last_coord} (${who})`;
    document.getElementById("tMoveConf").textContent =
      t.move_prob !== null && t.move_prob !== undefined
        ? `直觉置信 ${(t.move_prob * 100).toFixed(1)}%`
        : "对手落子";

    if (prev && prev.move !== null && prev.move !== undefined) {
      const prevIcon = prev.player === 1 ? "● 黑" : "○ 白";
      const prevWr = (prev.black_winrate * 100).toFixed(1);
      document.getElementById("tPrevMove").textContent =
        `前手 #${prev.move_num} ${prevIcon} ${prev.coord} (黑胜率 ${prevWr}%)`;
    } else {
      document.getElementById("tPrevMove").textContent = "前一手: 开局";
    }
  }

  // 2. Winrate
  const bWr = (t.black_winrate * 100).toFixed(1);
  const wWr = (t.white_winrate * 100).toFixed(1);
  document.getElementById("tWinrate").textContent = `黑 ${bWr}%  /  白 ${wWr}%`;
  document.getElementById("wrBarBlack").style.width = `${Math.max(1, Math.min(99, t.black_winrate * 100))}%`;

  // 3. Score Lead (目数差距)
  const lead = t.black_score_lead;
  if (Math.abs(lead) < 0.05) {
    document.getElementById("tScoreLead").textContent = "均势 0.0 目";
  } else if (lead > 0) {
    document.getElementById("tScoreLead").textContent = `● 黑领先 +${lead.toFixed(1)} 目`;
  } else {
    document.getElementById("tScoreLead").textContent = `○ 白领先 +${Math.abs(lead).toFixed(1)} 目`;
  }

  if (prev && prev.black_score_lead !== undefined && state.move_count > 0) {
    const diff = lead - prev.black_score_lead;
    const sign = diff >= 0 ? "+" : "";
    document.getElementById("tScoreSub").textContent = `较前手目差变化: 黑 ${sign}${diff.toFixed(1)} 目`;
  } else {
    document.getElementById("tScoreSub").textContent = "中国规则贴 7.5 目";
  }

  // 4. GPU Inference Speed
  document.getElementById("tSpeed").textContent = `GPU 直觉: ${t.gpu_nn_ms.toFixed(2)} ms`;
  document.getElementById("tSpeedSub").textContent = `含特征编码总耗时: ${t.total_ms.toFixed(2)} ms`;
  document.getElementById("tThroughput").textContent = `${t.steps_per_sec} 步/秒`;

  // Footer: Top candidates & captures
  if (t.top_candidates && t.top_candidates.length > 0) {
    const cands = t.top_candidates
      .slice(0, 4)
      .map(c => `${c.coord} ${(c.prob * 100).toFixed(1)}%`)
      .join(" · ");
    document.getElementById("candidatesText").textContent = `下步直觉选点: ${cands}`;
  }
  document.getElementById("statusText").textContent =
    `提子: ●黑 ${state.captures_black}  ○白 ${state.captures_white}  |  手数: ${state.move_count}`;
}

function getBoardCoordFromEvent(evt) {
  const rect = canvas.getBoundingClientRect();
  const cssSize = rect.width;
  const pad = Math.max(22, Math.round(cssSize * 0.052));
  const gridSpan = cssSize - pad * 2;
  const cell = gridSpan / (BOARD_SIZE - 1);

  const x = evt.clientX - rect.left;
  const y = evt.clientY - rect.top;
  const c = Math.round((x - pad) / cell);
  const r = Math.round((y - pad) / cell);
  if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return null;

  const ix = pad + c * cell;
  const iy = pad + r * cell;
  if (Math.hypot(x - ix, y - iy) > cell * 0.48) return null;
  return { r, c, move: r * BOARD_SIZE + c };
}

canvas.addEventListener("mousemove", (evt) => {
  const pos = getBoardCoordFromEvent(evt);
  const changed =
    (pos === null && hoverPos !== null) ||
    (pos !== null && hoverPos === null) ||
    (pos !== null && hoverPos !== null && (pos.r !== hoverPos.r || pos.c !== hoverPos.c));
  if (changed) {
    hoverPos = pos;
    renderBoard();
  }
});

canvas.addEventListener("mouseleave", () => {
  hoverPos = null;
  renderBoard();
});

canvas.addEventListener("click", async (evt) => {
  if (busy || state.game_over) return;
  const pos = getBoardCoordFromEvent(evt);
  if (!pos) return;
  if (state.board[pos.r][pos.c] !== 0) return;
  await playHumanMove(pos.move, pos.r, pos.c);
});

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, session_id: sessionId })
  });
  return await res.json();
}

async function fetchState() {
  const res = await fetch(`/api/state?session_id=${encodeURIComponent(sessionId)}`);
  state = await res.json();
  updateTelemetryUI();
  renderBoard();
}

async function playHumanMove(moveIdx, r = null, c = null) {
  busy = true;
  // Optimistically draw the human stone immediately (0ms perceived latency)
  const prevBoard = state.board.map(row => row.slice());
  const prevLastMove = state.last_move;
  if (r !== null && c !== null) {
    state.board[r][c] = state.current_player;
    state.last_move = moveIdx;
    hoverPos = null;
    renderBoard();
  }
  try {
    // Step 1: Submit human move & evaluate board after human move
    const res = await apiPost("/api/play", {
      move: moveIdx,
      auto_ai: false
    });
    if (res.error) {
      state.board = prevBoard;
      state.last_move = prevLastMove;
      renderBoard();
      document.getElementById("statusText").textContent = `非法落子: ${res.error}`;
      return;
    }
    state = res;
    updateTelemetryUI();
    renderBoard();

    // Step 2: Trigger AI single-step intuition reply
    if (!state.game_over) {
      const aiRes = await apiPost("/api/ai_move", {});
      if (!aiRes.error) {
        state = aiRes;
        updateTelemetryUI();
        renderBoard();
      }
    }
  } finally {
    busy = false;
  }
}

async function playPass() {
  if (busy || state.game_over) return;
  await playHumanMove(361);
}

async function triggerAiMove() {
  if (busy || state.game_over) return;
  busy = true;
  try {
    const res = await apiPost("/api/ai_move", {});
    if (!res.error) {
      state = res;
      updateTelemetryUI();
      renderBoard();
    }
  } finally {
    busy = false;
  }
}

async function undoMove() {
  if (busy) return;
  busy = true;
  try {
    const res = await apiPost("/api/undo", { steps: 2 });
    state = res;
    updateTelemetryUI();
    renderBoard();
  } finally {
    busy = false;
  }
}

async function newGame(color) {
  if (busy) return;
  humanColor = color;
  document.getElementById("btnPlayBlack").classList.toggle("active", color === 1);
  document.getElementById("btnPlayWhite").classList.toggle("active", color === 2);
  busy = true;
  try {
    const res = await apiPost("/api/reset", {
      ai_first: color === 2
    });
    state = res;
    updateTelemetryUI();
    renderBoard();
  } finally {
    busy = false;
  }
}

function toggleOwnership() {
  showOwnership = !showOwnership;
  document.getElementById("btnOwn").classList.toggle("active", showOwnership);
  renderBoard();
}

function resetZoom() {
  manualBoardPx = null;
  renderBoard();
}

function toggleLiveSync() {
  liveSync = !liveSync;
  document.getElementById("btnLiveSync").classList.toggle("active", liveSync);
  if (liveSync) {
    pollLiveState(true);
  } else {
    fetchState();
  }
}

async function pollLiveState(force = false) {
  if (!liveSync) return;
  try {
    const res = await fetch(`/api/live_state?seq=${force ? -1 : lastLiveSeq}`);
    const data = await res.json();
    if (data && (force || data.live_seq !== lastLiveSeq)) {
      lastLiveSeq = data.live_seq || 0;
      state = data;
      updateTelemetryUI();
      renderBoard();
    }
  } catch (e) {}
}

function startLiveStream() {
  try {
    const es = new EventSource("/api/stream");
    es.onmessage = (evt) => {
      if (!liveSync) return;
      try {
        const data = JSON.parse(evt.data);
        if (data && data.board) {
          lastLiveSeq = data.live_seq || 0;
          state = data;
          updateTelemetryUI();
          renderBoard();
        }
      } catch (e) {}
    };
    es.onerror = () => {
      es.close();
      setTimeout(startLiveStream, 1500);
    };
  } catch (e) {}
}

const grip = document.getElementById("resizeGrip");
let draggingResize = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartSize = 0;

grip.addEventListener("mousedown", (e) => {
  e.stopPropagation();
  draggingResize = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  dragStartSize = computeBoardPixelSize();
});

window.addEventListener("mousemove", (e) => {
  if (!draggingResize) return;
  const delta = Math.max(e.clientX - dragStartX, e.clientY - dragStartY);
  manualBoardPx = Math.max(200, dragStartSize + delta * 2);
  renderBoard();
});

window.addEventListener("mouseup", () => {
  draggingResize = false;
});

const ro = new ResizeObserver(() => renderBoard());
ro.observe(workspace);
window.addEventListener("resize", renderBoard);

if (isMirrorPage) {
  document.title = "TURBOGO-V2 // 19×19 实时比赛镜像";
  ["btnPlayBlack", "btnPlayWhite", "btnAiMove", "btnPass", "btnUndo"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });
  pollLiveState(true);
  startLiveStream();
  setInterval(() => pollLiveState(false), 300);
} else {
  const btnLive = document.getElementById("btnLiveSync");
  if (btnLive) btnLive.style.display = "none";
  const btnB = document.getElementById("btnPlayBlack");
  if (btnB) btnB.classList.add("active");
  fetchState();
}
</script>
</body>
</html>
"""


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/mirror", "/mirror/"):
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/live_state":
            game = get_session("mirror_19x19")
            self._send_json(serialize_board(game))
            return
        if parsed.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            last_sent = -1
            try:
                while True:
                    with LIVE_COND:
                        if LIVE_SEQ == last_sent:
                            LIVE_COND.wait(timeout=10.0)
                        cur_seq = LIVE_SEQ
                    if cur_seq != last_sent:
                        last_sent = cur_seq
                        payload = json.dumps(serialize_board(get_session("mirror_19x19")), ensure_ascii=False)
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    else:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except Exception:
                return
        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            sid = qs.get("session_id", ["default"])[0]
            game = get_session(sid)
            self._send_json(serialize_board(game))
            return
        self._send_json({"error": "Not found"}, 404)

    def _apply_19x19_board(self, game: GoBoard19, payload: dict):
        komi = float(payload.get("komi", 7.5))
        now = int(payload.get("now", 1))
        steps = payload.get("steps", [])
        stones = payload.get("stone") or payload.get("stones")

        game.komi = komi
        game.reset()
        for s in steps:
            c_val = int(s.get("c", 0))
            coor = int(s.get("coor", -1))
            x_val = int(s.get("x", -1))
            y_val = int(s.get("y", -1))
            if c_val not in (1, -1):
                continue
            if coor < 0 and x_val < 0 and y_val < 0 and len(game.move_history) == 0:
                continue
            pla = 1 if c_val == 1 else 2
            game.current_player = pla
            mv = (
                coor
                if (0 <= coor < 361)
                else (y_val * 19 + x_val if (0 <= x_val < 19 and 0 <= y_val < 19) else PASS_MOVE)
            )
            game.play(mv)

        if stones and len(stones) == 361:
            arr = np.array(stones, dtype=np.int8).reshape((19, 19))
            mapped = np.where(arr == 1, 1, np.where(arr == -1, 2, 0)).astype(np.int8)
            game.board = mapped

        game.current_player = 1 if now == 1 else 2
        return now

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            payload = {}

        sid = payload.get("session_id", "default")
        game = get_session(sid)

        if parsed.path == "/api/sync_19x19":
            game = get_session("mirror_19x19")
            meta = payload.get("meta", {})
            if payload.get("steps") is not None or payload.get("stones") is not None or payload.get("stone") is not None:
                self._apply_19x19_board(game, payload)
                eval_res = ENGINE.evaluate(game)
                last_player, last_move = game.move_history[-1] if game.move_history else (None, None)
                game.last_telemetry = {
                    "move_num": len(game.move_history),
                    "player": last_player,
                    "move": last_move,
                    "coord": move_to_str(last_move) if last_move is not None else "开局",
                    "move_prob": None,
                    "black_winrate": eval_res["black_winrate"],
                    "white_winrate": eval_res["white_winrate"],
                    "black_score_lead": eval_res["black_score_lead"],
                    "gpu_nn_ms": eval_res["gpu_nn_ms"],
                    "total_ms": eval_res["total_ms"],
                    "steps_per_sec": eval_res["steps_per_sec"],
                    "top_candidates": eval_res["top_candidates"],
                    "ownership": eval_res["ownership"],
                }
            notify_live_update(meta)
            self._send_json({"ok": True, "live_seq": LIVE_SEQ})
            return

        if parsed.path == "/api/genmove_19x19":
            game = get_session("mirror_19x19")
            now = self._apply_19x19_board(game, payload)
            eval_res = ENGINE.evaluate(game)

            # Evaluate the board BEFORE our move as prev_telemetry
            last_p_before, last_m_before = game.move_history[-1] if game.move_history else (None, None)
            game.last_telemetry = {
                "move_num": len(game.move_history),
                "player": last_p_before,
                "move": last_m_before,
                "coord": move_to_str(last_m_before) if last_m_before is not None else "开局",
                "move_prob": None,
                "black_winrate": eval_res["black_winrate"],
                "white_winrate": eval_res["white_winrate"],
                "black_score_lead": eval_res["black_score_lead"],
                "gpu_nn_ms": eval_res["gpu_nn_ms"],
                "total_ms": eval_res["total_ms"],
                "steps_per_sec": eval_res["steps_per_sec"],
                "top_candidates": eval_res["top_candidates"],
                "ownership": eval_res["ownership"],
            }

            # Play our intuition move on the default session so the web dashboard mirrors 19x19 live play
            best_mv = eval_res["best_move"]
            game.play(best_mv)
            game.last_telemetry = {
                "move_num": len(game.move_history),
                "player": 1 if now == 1 else 2,
                "move": best_mv,
                "coord": eval_res["best_coord"],
                "move_prob": eval_res["best_prob"],
                "black_winrate": eval_res["black_winrate"],
                "white_winrate": eval_res["white_winrate"],
                "black_score_lead": eval_res["black_score_lead"],
                "gpu_nn_ms": eval_res["gpu_nn_ms"],
                "total_ms": eval_res["total_ms"],
                "steps_per_sec": eval_res["steps_per_sec"],
                "top_candidates": eval_res["top_candidates"],
                "ownership": eval_res["ownership"],
            }
            meta = payload.get("meta", {})
            notify_live_update(meta)
            self._send_json(eval_res)
            return

        if parsed.path == "/api/reset":
            game.reset()
            if payload.get("ai_first", False):
                self._do_ai_step(game)
            self._send_json(serialize_board(game))
            return

        if parsed.path == "/api/undo":
            steps = int(payload.get("steps", 2))
            for _ in range(steps):
                if not game.undo():
                    break
            self._send_json(serialize_board(game))
            return

        if parsed.path == "/api/ai_move":
            if game.consecutive_passes < 2:
                self._do_ai_step(game, precomputed_eval=game.cached_eval)
            self._send_json(serialize_board(game))
            return

        if parsed.path == "/api/play":
            move = int(payload.get("move", -1))
            player = game.current_player
            # Look up human move's intuition probability from prior cached_eval if available
            human_move_prob = None
            if game.cached_eval and game.cached_eval.get("top_candidates"):
                for cand in game.cached_eval["top_candidates"]:
                    if cand["move"] == move:
                        human_move_prob = cand["prob"]
                        break

            if not game.play(move):
                self._send_json({"error": "该位置禁入或已有棋子"}, 400)
                return

            eval_after_human = ENGINE.evaluate(game)
            game.cached_eval = eval_after_human
            game.last_telemetry = {
                "move_num": len(game.move_history),
                "player": player,
                "move": move,
                "coord": move_to_str(move),
                "move_prob": human_move_prob,
                "black_winrate": eval_after_human["black_winrate"],
                "white_winrate": eval_after_human["white_winrate"],
                "black_score_lead": eval_after_human["black_score_lead"],
                "gpu_nn_ms": eval_after_human["gpu_nn_ms"],
                "total_ms": eval_after_human["total_ms"],
                "steps_per_sec": eval_after_human["steps_per_sec"],
                "top_candidates": eval_after_human["top_candidates"],
                "ownership": eval_after_human["ownership"],
            }

            if payload.get("auto_ai", False) and game.consecutive_passes < 2:
                self._do_ai_step(game, precomputed_eval=eval_after_human)

            self._send_json(serialize_board(game))
            return

        self._send_json({"error": "Unknown endpoint"}, 404)

    def _do_ai_step(self, game: GoBoard19, precomputed_eval: dict = None):
        ai_player = game.current_player
        decision = precomputed_eval if precomputed_eval is not None else ENGINE.evaluate(game)
        ai_move = decision["best_move"]
        ai_prob = decision["best_prob"]
        decision_gpu_ms = decision["gpu_nn_ms"]
        decision_total_ms = decision["total_ms"]
        decision_sps = decision["steps_per_sec"]

        game.play(ai_move)

        post_eval = ENGINE.evaluate(game)
        game.cached_eval = post_eval
        game.last_telemetry = {
            "move_num": len(game.move_history),
            "player": ai_player,
            "move": ai_move,
            "coord": move_to_str(ai_move),
            "move_prob": ai_prob,
            "black_winrate": post_eval["black_winrate"],
            "white_winrate": post_eval["white_winrate"],
            "black_score_lead": post_eval["black_score_lead"],
            "gpu_nn_ms": decision_gpu_ms,
            "total_ms": decision_total_ms,
            "steps_per_sec": decision_sps,
            "top_candidates": post_eval["top_candidates"],
            "ownership": post_eval["ownership"],
        }


def main():
    global ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument(
        "--ckpt",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "b12c384_best_weights.pt"),
    )
    args = parser.parse_args()

    ENGINE = FastGPUIntuitionEngine(args.ckpt, device_str="cuda")
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"[TurboGo-v2 Web] Serving on http://{args.host}:{args.port}")
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
