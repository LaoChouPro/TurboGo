#!/usr/bin/env python3
"""
Head-to-Head Benchmark & 19x19 Arena Match:
  TurboGo-v2 Best (`b11c384-nbt22`, 22-Layer Global Self-Attention + SwiGLU + 2D-RoPE, 20.82M params)
  vs.
  TinyGo Large (`best_model_large.pth`, 10-Block ResNet c128, 3.22M params)
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PLAY_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLAY_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PLAY_DIR))

from src.model import TurboGoHybridTransformer
from engine_server import GoBoard19, move_to_coord
from tinygo_large.model import TinyGoNet


def load_turbogo_v2(weights_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    sd = ckpt.get("model") or ckpt.get("ema_model") or ckpt
    model = TurboGoHybridTransformer(
        d_model=cfg.get("d_model", 384),
        c_mid=cfg.get("c_mid", 256),
        n_blocks=cfg.get("n_blocks", 11),
        internal_length=cfg.get("internal_length", 2),
        n_heads=cfg.get("n_heads", 8),
        ffn_dim=cfg.get("ffn_dim", 640),
        in_channels=cfg.get("in_channels", 32),
    )
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, strict=True)
    model.to(device).eval()
    return model, ckpt


def load_tinygo_large(weights_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = TinyGoNet(num_blocks=10, channels=128, num_classes=361)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        meta = {"epoch": ckpt.get("epoch"), "acc": ckpt.get("acc")}
    else:
        model.load_state_dict(ckpt, strict=True)
        meta = {}
    model.to(device).eval()
    return model, meta


@torch.no_grad()
def run_val_benchmark(
    turbogo_model: nn.Module,
    tinygo_model: nn.Module,
    device: torch.device,
    num_samples: int = 2048,
) -> dict:
    val_shard = ROOT_DIR / "data/katago_19x19_chinese_10m/shard_00498.npz"
    if not val_shard.exists():
        print("[Benchmark] Validation shard not found, skipping validation benchmark.")
        return {}

    print("\n" + "=" * 90, flush=True)
    print(f"[1/2] Held-Out Validation Set Benchmark ({val_shard.name}, N={num_samples:,} positions)", flush=True)
    print("=" * 90, flush=True)

    shard = np.load(val_shard)
    bits = (
        np.unpackbits(shard["binaryInputNCHWPacked"][:num_samples], axis=2)[:, :, :361]
        .reshape(num_samples, 22, 19, 19)
        .astype(np.float32)
    )
    g = shard["globalInputNC"][:num_samples].astype(np.float32)
    pol_raw = shard["policyTargetsNCMove"][:num_samples, 0, :].astype(np.float32)
    gt_moves = np.argmax(pol_raw, axis=1)

    results = {}
    bs = 128

    # 1. Evaluate TinyGo Large
    # Plane 1 of KataGo V7 is current player stones, Plane 2 is opponent stones, Plane 0 is all 1s
    t0 = time.perf_counter()
    tg_top1 = tg_top3 = tg_top5 = 0
    tg_ce_sum = 0.0
    valid_board_samples = 0

    for i in range(0, num_samples, bs):
        b_slice = bits[i : i + bs]
        # Construct TinyGo's exact 3-channel input: [my_stones, opp_stones, ones]
        tg_in = np.stack([b_slice[:, 1], b_slice[:, 2], np.ones_like(b_slice[:, 0])], axis=1)
        bx = torch.from_numpy(tg_in).to(device)
        logits_361 = tinygo_model(bx).float()

        # Mask occupied & ko intersections for fair evaluation
        illegal = (b_slice[:, 1] > 0.5) | (b_slice[:, 2] > 0.5) | (b_slice[:, 6] > 0.5)
        illegal_t = torch.from_numpy(illegal.reshape(-1, 361)).to(device)
        logits_361 = logits_361.masked_fill(illegal_t, -1e4)

        # Pad pass logit with -1e4 since TinyGo only outputs 361 board moves
        pass_pad = torch.full((logits_361.shape[0], 1), -10.0, device=device)
        logits_362 = torch.cat([logits_361, pass_pad], dim=-1)

        tp = torch.from_numpy(np.maximum(pol_raw[i : i + bs], 0.0)).to(device)
        tp = tp / tp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        lp = F.log_softmax(logits_362, dim=-1)
        tg_ce_sum += (-(tp * lp).sum(dim=-1)).sum().item()

        preds = logits_361.topk(5, dim=-1).indices.cpu().numpy()
        gt_m = gt_moves[i : i + bs]
        tg_top1 += int((preds[:, 0] == gt_m).sum())
        tg_top3 += int(np.any(preds[:, :3] == gt_m[:, None], axis=1).sum())
        tg_top5 += int(np.any(preds[:, :5] == gt_m[:, None], axis=1).sum())

    dt_tg = time.perf_counter() - t0
    results["TinyGo Large (10b ResNet, 3.22M)"] = {
        "top1": round(tg_top1 / num_samples * 100.0, 2),
        "top3": round(tg_top3 / num_samples * 100.0, 2),
        "top5": round(tg_top5 / num_samples * 100.0, 2),
        "pol_ce": round(tg_ce_sum / num_samples, 4),
        "pos_per_s": round(num_samples / dt_tg, 1),
    }

    # 2. Evaluate TurboGo-v2 Best
    t0 = time.perf_counter()
    tb_top1 = tb_top3 = tb_top5 = 0
    tb_ce_sum = 0.0
    for i in range(0, num_samples, bs):
        bx = torch.from_numpy(bits[i : i + bs]).to(device)
        bg = torch.from_numpy(g[i : i + bs]).to(device)
        tp = torch.from_numpy(np.maximum(pol_raw[i : i + bs], 0.0)).to(device)
        tp = tp / tp.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        pol_logits, _, _, _ = turbogo_model(bx, bg)
        p0 = pol_logits[:, 0].float()
        lp0 = F.log_softmax(p0, dim=-1)
        tb_ce_sum += (-(tp * lp0).sum(dim=-1)).sum().item()

        preds = p0.topk(5, dim=-1).indices.cpu().numpy()
        gt_m = gt_moves[i : i + bs]
        tb_top1 += int((preds[:, 0] == gt_m).sum())
        tb_top3 += int(np.any(preds[:, :3] == gt_m[:, None], axis=1).sum())
        tb_top5 += int(np.any(preds[:, :5] == gt_m[:, None], axis=1).sum())

    dt_tb = time.perf_counter() - t0
    results["TurboGo-v2 Best (22-Layer Attn, 20.82M)"] = {
        "top1": round(tb_top1 / num_samples * 100.0, 2),
        "top3": round(tb_top3 / num_samples * 100.0, 2),
        "top5": round(tb_top5 / num_samples * 100.0, 2),
        "pol_ce": round(tb_ce_sum / num_samples, 4),
        "pos_per_s": round(num_samples / dt_tb, 1),
    }

    for name, r in results.items():
        print(
            f"  {name:<42} | Top-1: {r['top1']:5.2f}% | Top-3: {r['top3']:5.2f}% | "
            f"Top-5: {r['top5']:5.2f}% | PolCE: {r['pol_ce']:.4f} | Speed: {r['pos_per_s']:6.1f} pos/s",
            flush=True,
        )
    return results


@torch.no_grad()
def select_tinygo_move(
    model: nn.Module,
    board: GoBoard19,
    device: torch.device,
    opening_temp_moves: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> int:
    pla = board.to_play
    opp = 3 - pla
    features = np.zeros((1, 3, 19, 19), dtype=np.float32)
    features[0, 0] = (board.board == pla).astype(np.float32)
    features[0, 1] = (board.board == opp).astype(np.float32)
    features[0, 2] = 1.0

    bx = torch.from_numpy(features).to(device)
    logits = model(bx)[0].float().cpu().numpy()  # (361,)

    legal_mask = board.get_legal_mask()[:361]
    if not legal_mask.any():
        return 361

    masked_logits = np.where(legal_mask, logits, -1e9)
    if rng is not None and len(board.move_history) < opening_temp_moves:
        max_l = masked_logits.max()
        cand_mask = (masked_logits >= max_l - 1.0) & legal_mask
        cand_indices = np.nonzero(cand_mask)[0]
        if len(cand_indices) > 1:
            probs = np.exp((masked_logits[cand_indices] - max_l) / 0.6)
            probs /= probs.sum()
            return int(rng.choice(cand_indices, p=probs))

    return int(np.argmax(masked_logits))


@torch.no_grad()
def select_turbogo_move(
    model: nn.Module,
    board: GoBoard19,
    device: torch.device,
    top_k: int = 1,
    opening_temp_moves: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float, float]:
    sx, gx = board.encode_katago_v7()
    bx = torch.from_numpy(sx).unsqueeze(0).to(device)
    bg = torch.from_numpy(gx).unsqueeze(0).to(device)

    pol_logits, wld_logits, score_pred, _ = model(bx, bg)
    p0 = pol_logits[0, 0].float().cpu().numpy()
    wld = F.softmax(wld_logits[0].float(), dim=-1).cpu().numpy()
    pla_wr = float(wld[0])
    pla_lead = float(score_pred[0].item() * 20.0)
    black_wr = pla_wr if board.to_play == 1 else float(wld[1])
    black_lead = pla_lead if board.to_play == 1 else -pla_lead

    legal_mask = board.get_legal_mask()
    if len(board.move_history) < 120 and legal_mask[:361].any():
        legal_mask[361] = False

    masked_logits = np.where(legal_mask, p0, -1e9)
    if rng is not None and len(board.move_history) < opening_temp_moves:
        max_l = masked_logits[:361].max()
        cand_mask = (masked_logits[:361] >= max_l - 1.0) & legal_mask[:361]
        cand_indices = np.nonzero(cand_mask)[0]
        if len(cand_indices) > 1:
            probs = np.exp((masked_logits[cand_indices] - max_l) / 0.6)
            probs /= probs.sum()
            return int(rng.choice(cand_indices, p=probs)), black_wr, black_lead

    if top_k <= 1:
        return int(np.argmax(masked_logits)), black_wr, black_lead

    cand_indices = np.argsort(masked_logits)[::-1][:top_k]
    cand_indices = [int(m) for m in cand_indices if legal_mask[m]]
    if len(cand_indices) <= 1:
        return (cand_indices[0] if cand_indices else 361), black_wr, black_lead

    prior_probs = np.exp(masked_logits[cand_indices] - masked_logits[cand_indices[0]])
    prior_probs /= prior_probs.sum()

    child_sx, child_gx, valid_cands, valid_priors = [], [], [], []
    for m, pr in zip(cand_indices, prior_probs):
        b_next = board.clone()
        if b_next.play(m):
            csx, cgx = b_next.encode_katago_v7()
            child_sx.append(csx)
            child_gx.append(cgx)
            valid_cands.append(m)
            valid_priors.append(pr)

    if not valid_cands:
        return int(np.argmax(masked_logits)), black_wr, black_lead

    cbx = torch.from_numpy(np.stack(child_sx, axis=0)).to(device)
    cbg = torch.from_numpy(np.stack(child_gx, axis=0)).to(device)
    _, c_wld_logits, c_score_pred, _ = model(cbx, cbg)
    c_wld = F.softmax(c_wld_logits.float(), dim=-1).cpu().numpy()
    my_wr = c_wld[:, 1] - c_wld[:, 0]
    my_lead = -c_score_pred.float().cpu().numpy() * 20.0
    my_util = my_wr + 0.05 * np.arctan(my_lead / 7.5)
    comb_score = my_util + 0.35 * np.log(np.array(valid_priors) + 1e-4)
    return valid_cands[int(np.argmax(comb_score))], black_wr, black_lead


@torch.no_grad()
def adjudicate_position(turbogo_model: nn.Module, board: GoBoard19, device: torch.device) -> Tuple[float, float, float]:
    """Return (black_winrate, black_score_lead, black_ownership_score_minus_komi)."""
    sx, gx = board.encode_katago_v7()
    bx = torch.from_numpy(sx).unsqueeze(0).to(device)
    bg = torch.from_numpy(gx).unsqueeze(0).to(device)
    _, wld_logits, score_pred, own_pred = turbogo_model(bx, bg)
    wld = F.softmax(wld_logits[0].float(), dim=-1).cpu().numpy()
    pla_wr = float(wld[0])
    pla_lead = float(score_pred[0].item() * 20.0)
    own_map = own_pred[0].float().cpu().numpy()
    if board.to_play == 1:
        b_wr = pla_wr
        b_lead = pla_lead
        b_own = own_map
    else:
        b_wr = float(wld[1])
        b_lead = -pla_lead
        b_own = -own_map
    b_own_score = float(b_own.sum()) - board.komi
    return b_wr, b_lead, b_own_score


def run_arena_series(
    turbogo_model: nn.Module,
    tinygo_model: nn.Module,
    device: torch.device,
    num_pairs: int = 5,
    turbo_top_k: int = 1,
    max_moves: int = 220,
    mode_title: str = "Raw Policy Network (1-Visit vs 1-Visit)",
) -> dict:
    print("\n" + "=" * 90, flush=True)
    print(f"[2/2] 19x19 Head-to-Head Arena: {mode_title} ({num_pairs * 2} games, Color-Swapped)", flush=True)
    print("=" * 90, flush=True)

    turbo_wins = 0
    turbo_wins_black = 0
    turbo_wins_white = 0
    games = []

    for pair_idx in range(num_pairs):
        for turbo_color in (1, 2):  # 1=Black, 2=White
            rng = np.random.default_rng(2026 + pair_idx)
            board = GoBoard19(komi=7.5)
            decisive_streak = 0
            resigned_winner = None
            result_str = ""

            for ply in range(max_moves):
                if board.to_play == turbo_color:
                    mv, b_wr, b_lead = select_turbogo_move(
                        turbogo_model, board, device, top_k=turbo_top_k, opening_temp_moves=6, rng=rng
                    )
                else:
                    mv = select_tinygo_move(tinygo_model, board, device, opening_temp_moves=6, rng=rng)
                board.play(mv)

                if ply >= 70 and ply % 10 == 0:
                    b_wr, b_lead, b_own_sc = adjudicate_position(turbogo_model, board, device)
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
                b_wr, b_lead, b_own_sc = adjudicate_position(turbogo_model, board, device)
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

            opening = " ".join(move_to_coord(m) for m in board.move_history[:8])
            matchup = "TurboGo-v2(B) vs TinyGo-L(W)" if turbo_color == 1 else "TinyGo-L(B) vs TurboGo-v2(W)"
            tag = "TurboGo WIN " if turbo_won else "TinyGo  WIN "
            print(f"  Game {len(games)+1:2d} [{matchup}] -> {tag} | {result_str:<42} | Open: {opening}", flush=True)
            games.append(
                {
                    "game": len(games) + 1,
                    "turbo_color": "B" if turbo_color == 1 else "W",
                    "turbo_won": turbo_won,
                    "result": result_str,
                    "moves": len(board.move_history),
                    "opening": opening,
                }
            )

    total = num_pairs * 2
    print("-" * 90, flush=True)
    print(
        f"  [{mode_title}] TurboGo-v2 Score: {turbo_wins}/{total} ({turbo_wins/total*100:.1f}%) "
        f"| As Black: {turbo_wins_black}/{num_pairs} | As White: {turbo_wins_white}/{num_pairs}",
        flush=True,
    )
    return {
        "mode": mode_title,
        "total_games": total,
        "turbo_wins": turbo_wins,
        "tinygo_wins": total - turbo_wins,
        "turbo_win_rate": round(turbo_wins / total * 100.0, 1),
        "turbo_wins_as_black": turbo_wins_black,
        "turbo_wins_as_white": turbo_wins_white,
        "games": games,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    turbogo_path = PLAY_DIR / "weights" / "b12c384_best_weights.pt"
    tinygo_path = PLAY_DIR / "weights" / "best_model_large.pth"

    turbogo_model, tb_ckpt = load_turbogo_v2(turbogo_path, device)
    tinygo_model, tg_meta = load_tinygo_large(tinygo_path, device)

    tb_params = sum(p.numel() for p in turbogo_model.parameters())
    tg_params = sum(p.numel() for p in tinygo_model.parameters())

    print("=" * 90, flush=True)
    print(f"  Device        : {device}", flush=True)
    print(
        f"  Model A       : TurboGo-v2 Best ({tb_params/1e6:.2f}M params, Step {tb_ckpt.get('global_step')}, "
        f"22-Layer Global Self-Attention + SwiGLU + 2D-RoPE)",
        flush=True,
    )
    print(
        f"  Model B       : TinyGo Large ({tg_params/1e6:.2f}M params, Epoch {tg_meta.get('epoch')}, "
        f"Val Acc {tg_meta.get('acc', 0.0):.2f}%, 10-Block ResNet c128)",
        flush=True,
    )
    print("=" * 90, flush=True)

    val_res = run_val_benchmark(turbogo_model, tinygo_model, device, num_samples=2048)
    arena_1v1 = run_arena_series(
        turbogo_model,
        tinygo_model,
        device,
        num_pairs=5,
        turbo_top_k=1,
        max_moves=200,
        mode_title="Pure Policy Network (1-Visit vs 1-Visit)",
    )
    arena_search = run_arena_series(
        turbogo_model,
        tinygo_model,
        device,
        num_pairs=3,
        turbo_top_k=6,
        max_moves=180,
        mode_title="TurboGo-v2 (Top-6 Search) vs TinyGo Large (1-Visit)",
    )

    out_file = PLAY_DIR / "arena_vs_tinygo_large_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "val_benchmark": val_res,
                "arena_pure_policy": arena_1v1,
                "arena_search": arena_search,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved full comparison results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
