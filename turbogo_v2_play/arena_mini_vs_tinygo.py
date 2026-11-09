#!/usr/bin/env python3
"""
30-Game Paired-Opening Statistical Arena Match (15 Unique Fuseki Pairs x 2 Colors = 30 Games):
  TurboGo-Mini (`turbogo_mini_best_weights.pt`, 12-Layer Global Self-Attention + SwiGLU + 2D-RoPE, 3.055M params)
  vs.
  TinyGo Large (`best_model_large.pth`, 10-Block ResNet c128, 3.219M params)
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PLAY_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLAY_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PLAY_DIR))

from arena_vs_tinygo_large import (
    adjudicate_position,
    load_tinygo_large,
    load_turbogo_v2,
    select_tinygo_move,
    select_turbogo_move,
)
from engine_server import GoBoard19, move_to_coord


def generate_unique_openings(
    mini_model: torch.nn.Module,
    device: torch.device,
    num_openings: int = 15,
    opening_plies: int = 8,
) -> List[List[int]]:
    """
    Generate `num_openings` guaranteed-distinct 8-ply balanced Fuseki openings
    (ensuring the board state after 8 plies is unique across all pairs) so paired
    color-swapped games never repeat a transposition.
    """
    openings: List[List[int]] = []
    seen_board_hashes = set()
    seed = 1000

    while len(openings) < num_openings:
        rng = np.random.default_rng(seed)
        seed += 1
        board = GoBoard19(komi=7.5)
        for _ in range(opening_plies):
            sx, gx = board.encode_katago_v7()
            bx = torch.from_numpy(sx).unsqueeze(0).to(device)
            bg = torch.from_numpy(gx).unsqueeze(0).to(device)
            with torch.no_grad():
                pol_logits, _, _, _ = mini_model(bx, bg)
            p0 = pol_logits[0, 0].float().cpu().numpy()
            legal_mask = board.get_legal_mask()
            legal_mask[361] = False
            masked_logits = np.where(legal_mask, p0, -1e9)
            max_l = masked_logits[:361].max()
            # Allow natural Fuseki variety (star point, komoku, takamoku, mokuhazushi, kakari, shimari)
            cand_mask = (masked_logits[:361] >= max_l - 1.6) & legal_mask[:361]
            cand_indices = np.nonzero(cand_mask)[0]
            probs = np.exp((masked_logits[cand_indices] - max_l) / 0.85)
            probs /= probs.sum()
            mv = int(rng.choice(cand_indices, p=probs))
            board.play(mv)

        b_hash = board.board.tobytes()
        if b_hash not in seen_board_hashes:
            seen_board_hashes.add(b_hash)
            openings.append(list(board.move_history))

    return openings


def binomial_two_sided_p_value(wins: int, n: int) -> float:
    """Exact two-sided binomial test p-value against H0: p = 0.5."""
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def wilson_95_ci(wins: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = 1.95996
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def winrate_to_elo(wr: float) -> float:
    wr_c = min(0.99, max(0.01, wr))
    return -400.0 * math.log10(1.0 / wr_c - 1.0)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mini_path = PLAY_DIR / "weights" / "turbogo_mini_best_weights.pt"
    tinygo_path = PLAY_DIR / "weights" / "best_model_large.pth"

    mini_model, mini_ckpt = load_turbogo_v2(mini_path, device)
    tinygo_model, tg_meta = load_tinygo_large(tinygo_path, device)

    num_pairs = 15  # 15 unique openings x 2 colors (Black & White) = 30 games
    max_moves = 200

    print("=" * 98, flush=True)
    print("  30-Game Paired-Opening Arena: TurboGo-Mini (3.06M, 1-Visit) vs TinyGo Large (3.22M, 1-Visit)", flush=True)
    print("  15 Distinct 8-Ply Fuseki Openings × 2 Colors (15 Black + 15 White = 30 Games, Komi=7.5)", flush=True)
    print("=" * 98, flush=True)

    openings = generate_unique_openings(mini_model, device, num_openings=num_pairs, opening_plies=8)

    turbo_wins = 0
    turbo_wins_black = 0
    turbo_wins_white = 0
    games = []
    t0 = time.perf_counter()

    for pair_idx, init_moves in enumerate(openings):
        for turbo_color in (1, 2):  # 1=Black, 2=White
            board = GoBoard19(komi=7.5)
            for m in init_moves:
                board.play(m)

            decisive_streak = 0
            resigned_winner = None
            result_str = ""

            for ply in range(len(init_moves), max_moves):
                if board.to_play == turbo_color:
                    mv, _, _ = select_turbogo_move(
                        mini_model, board, device, top_k=1, opening_temp_moves=0, rng=None
                    )
                else:
                    mv = select_tinygo_move(
                        tinygo_model, board, device, opening_temp_moves=0, rng=None
                    )
                board.play(mv)

                if ply >= 70 and ply % 10 == 0:
                    b_wr, b_lead, b_own_sc = adjudicate_position(mini_model, board, device)
                    area_sc = board.chinese_area_score_black_minus_white()
                    margin = 0.5 * (b_lead + b_own_sc)
                    if b_wr > 0.975 and margin > 12.0:
                        decisive_streak += 1
                        if decisive_streak >= 2:
                            resigned_winner = 1
                            result_str = f"B+{margin:.1f} (目数差 {margin:+.1f} / 盘面 {area_sc:+.1f} @ {ply+1}手)"
                            break
                    elif b_wr < 0.025 and margin < -12.0:
                        decisive_streak += 1
                        if decisive_streak >= 2:
                            resigned_winner = 2
                            result_str = f"W+{-margin:.1f} (目数差 {margin:+.1f} / 盘面 {area_sc:+.1f} @ {ply+1}手)"
                            break
                    else:
                        decisive_streak = 0

            if resigned_winner is not None:
                winner = resigned_winner
            else:
                b_wr, b_lead, b_own_sc = adjudicate_position(mini_model, board, device)
                area_sc = board.chinese_area_score_black_minus_white()
                margin = 0.5 * (b_lead + b_own_sc)
                winner = 1 if margin > 0 else 2
                result_str = (
                    (f"B+{margin:.1f}" if winner == 1 else f"W+{-margin:.1f}")
                    + f" (盘面 {area_sc:+.1f}, {len(board.move_history)}手)"
                )

            turbo_won = winner == turbo_color
            if turbo_won:
                turbo_wins += 1
                if turbo_color == 1:
                    turbo_wins_black += 1
                else:
                    turbo_wins_white += 1

            opening_str = " ".join(move_to_coord(m) for m in init_moves)
            matchup = "TurboGo-Mini(B) vs TinyGo-L(W)" if turbo_color == 1 else "TinyGo-L(B) vs TurboGo-Mini(W)"
            tag = "TurboGo-Mini WIN" if turbo_won else "TinyGo-Large WIN"
            g_no = len(games) + 1
            print(
                f"  Game {g_no:2d} (Pair {pair_idx+1:2d}) [{matchup}] -> {tag} | {result_str:<40} | Open: {opening_str}",
                flush=True,
            )
            games.append(
                {
                    "game": g_no,
                    "pair": pair_idx + 1,
                    "turbo_color": "B" if turbo_color == 1 else "W",
                    "turbo_won": turbo_won,
                    "result": result_str,
                    "moves": len(board.move_history),
                    "opening": opening_str,
                }
            )

    dt = time.perf_counter() - t0
    total = num_pairs * 2
    wr_30 = turbo_wins / total
    ci_lo, ci_hi = wilson_95_ci(turbo_wins, total)
    p_val_30 = binomial_two_sided_p_value(turbo_wins, total)
    elo_30 = winrate_to_elo(wr_30)

    # Combined 40-game statistics (previous 10 games: 8/10 [B: 5/5, W: 3/5] + these 30 games)
    tot_40_wins = 8 + turbo_wins
    tot_40_b = 5 + turbo_wins_black
    tot_40_w = 3 + turbo_wins_white
    wr_40 = tot_40_wins / 40.0
    ci40_lo, ci40_hi = wilson_95_ci(tot_40_wins, 40)
    p_val_40 = binomial_two_sided_p_value(tot_40_wins, 40)
    elo_40 = winrate_to_elo(wr_40)

    print("=" * 98, flush=True)
    print(
        f"  [30-Game Batch ({dt:.1f}s)] TurboGo-Mini: {turbo_wins}/{total} ({wr_30*100:.1f}%) "
        f"| Black: {turbo_wins_black}/{num_pairs} ({turbo_wins_black/num_pairs*100:.1f}%) "
        f"| White: {turbo_wins_white}/{num_pairs} ({turbo_wins_white/num_pairs*100:.1f}%)",
        flush=True,
    )
    print(
        f"  [30-Game Stats] Elo: {elo_30:+.0f} | 95% Wilson CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] | Two-sided p-value: {p_val_30:.6f}",
        flush=True,
    )
    print(
        f"  [Cumulative 40 Games] TurboGo-Mini: {tot_40_wins}/40 ({wr_40*100:.1f}%) "
        f"| Black: {tot_40_b}/20 ({tot_40_b/20*100:.1f}%) | White: {tot_40_w}/20 ({tot_40_w/20*100:.1f}%)",
        flush=True,
    )
    print(
        f"  [40-Game Stats] Elo: {elo_40:+.0f} | 95% Wilson CI: [{ci40_lo*100:.1f}%, {ci40_hi*100:.1f}%] | Two-sided p-value: {p_val_40:.6f}",
        flush=True,
    )
    print("=" * 98, flush=True)

    out_file = PLAY_DIR / "arena_mini_vs_tinygo_30games.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_30_games": {
                    "total_games": total,
                    "turbo_wins": turbo_wins,
                    "tinygo_wins": total - turbo_wins,
                    "win_rate_pct": round(wr_30 * 100.0, 2),
                    "black_wins": turbo_wins_black,
                    "white_wins": turbo_wins_white,
                    "elo_diff": round(elo_30, 1),
                    "wilson_95_ci_pct": [round(ci_lo * 100.0, 1), round(ci_hi * 100.0, 1)],
                    "p_value": p_val_30,
                    "games": games,
                },
                "cumulative_40_games": {
                    "total_games": 40,
                    "turbo_wins": tot_40_wins,
                    "tinygo_wins": 40 - tot_40_wins,
                    "win_rate_pct": round(wr_40 * 100.0, 2),
                    "black_wins": tot_40_b,
                    "white_wins": tot_40_w,
                    "elo_diff": round(elo_40, 1),
                    "wilson_95_ci_pct": [round(ci40_lo * 100.0, 1), round(ci40_hi * 100.0, 1)],
                    "p_value": p_val_40,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
