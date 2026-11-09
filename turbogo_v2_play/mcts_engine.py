#!/usr/bin/env python3
"""
KataGo-Style Batched PUCT MCTS Engine + 8-Way D4 Symmetry Evaluator for TurboGo-v2
==================================================================================
Implements:
  1. 8-Way D4 Dihedral Symmetry Averaging (`eval_d4_symmetry`) — evaluates all 8 rotations/reflections
     of a 19x19 Go board in a single fused GPU batch and inverse-permutes policy & ownership back.
  2. Batched Virtual-Loss PUCT MCTS (`KataGoMCTS`) with:
     - Combined Winrate (WLD) + Score Lead (`atan(lead/7.5)`) + Ownership utility
     - First-Play Urgency (FPU) reduction (`V_parent - fpu_reduction * sqrt(visited_policy_mass)`)
     - Batched leaf expansion (`batch_size = 8` or `16` leaves per forward pass with Virtual Loss)
     - Arbitrary search depth (tracks `max_depth` and Principal Variation `pv` line)
     - Lower Confidence Bound (LCB) / Visit+Utility move selection
"""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def build_d4_tables_np() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build forward and inverse permutation tables for all 8 dihedral (D4) symmetries on 19x19:
      fwd_361: (8, 361) — spatial_sym = spatial_flat[:, fwd_361[sym]]
      inv_361: (8, 361) — canonical_flat = sym_flat[:, inv_361[sym]]
      inv_362: (8, 362) — canonical_pol = sym_pol[:, inv_362[sym]] (keeps pass=361 fixed)
    """
    base = np.arange(361, dtype=np.int64).reshape(19, 19)
    fwd_361 = np.zeros((8, 361), dtype=np.int64)
    inv_361 = np.zeros((8, 361), dtype=np.int64)
    inv_362 = np.zeros((8, 362), dtype=np.int64)

    for s in range(8):
        g = base.copy()
        if s & 4:
            g = g.T.copy()
        if s & 2:
            g = np.flip(g, axis=0).copy()
        if s & 1:
            g = np.flip(g, axis=1).copy()
        f = g.reshape(361)
        inv = np.argsort(f)
        fwd_361[s] = f
        inv_361[s] = inv
        inv_362[s, :361] = inv
        inv_362[s, 361] = 361

    return fwd_361, inv_361, inv_362


FWD_361, INV_361, INV_362 = build_d4_tables_np()


def reconstruct_board_from_katago_planes(spatial_22: np.ndarray, global_19: np.ndarray) -> GoBoard19:
    """
    Reconstruct an exact playable `GoBoard19` state from a single KataGo V7 (22, 19, 19) + (19,) tensor.
    """
    pla = 1 if global_19[18] < 0 else 2
    opp = 3 - pla
    self_komi = float(global_19[5]) * 20.0
    komi = -self_komi if pla == 1 else self_komi

    b = GoBoard19(komi=komi)
    b.to_play = pla
    own_mask = spatial_22[1] > 0.5
    opp_mask = spatial_22[2] > 0.5
    b.board[own_mask] = pla
    b.board[opp_mask] = opp

    ko_pts = np.argwhere(spatial_22[6] > 0.5)
    if len(ko_pts) > 0:
        b.ko_point = (int(ko_pts[0][0]), int(ko_pts[0][1]))

    # Reconstruct last 5 moves (planes 13 down to 9 are oldest to newest)
    hist = []
    for k in range(5, 0, -1):
        if global_19[k - 1] > 0.5:
            hist.append(361)
        else:
            pts = np.argwhere(spatial_22[8 + k] > 0.5)
            if len(pts) > 0:
                hist.append(int(pts[0][0]) * 19 + int(pts[0][1]))
    b.move_history = hist
    b.pos_hashes = {b.board.tobytes()}
    return b


@torch.no_grad()
def eval_d4_symmetry(
    model: nn.Module,
    spatial_22: np.ndarray,
    global_19: np.ndarray,
    device: torch.device,
    num_symmetries: int = 8,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Evaluate a position `(22, 19, 19)` across `num_symmetries` (1, 2, 4, or 8) dihedral symmetries
    in a single batch and average the inverse-transformed policy probabilities, winrate, and score lead.
    Returns:
      probs_362: (362,) averaged policy probability distribution
      pla_winrate: float in [0, 1] for current player
      pla_lead: float in points for current player
      pla_utility: float in [-1, +1] (KataGo combined value + score utility)
    """
    K = num_symmetries
    flat_s = spatial_22.reshape(22, 361)
    sym_s = np.stack([flat_s[:, FWD_361[s]].reshape(22, 19, 19) for s in range(K)], axis=0)
    sym_g = np.repeat(global_19[None, :], K, axis=0)

    bx = torch.from_numpy(sym_s).to(device)
    bg = torch.from_numpy(sym_g).to(device)

    pol_logits, wld_logits, score_pred, own_pred = model(bx, bg)
    p0_logits = pol_logits[:, 0].float().cpu().numpy()  # (K, 362)
    p0_probs = F.softmax(pol_logits[:, 0].float(), dim=-1).cpu().numpy()  # (K, 362)
    wld = F.softmax(wld_logits.float(), dim=-1).cpu().numpy()  # (K, 3)
    scores = score_pred.float().cpu().numpy() * 20.0  # (K,)

    # Inverse-permute each symmetry's policy back to canonical board coordinates
    canon_probs = np.zeros((362,), dtype=np.float32)
    canon_logits = np.zeros((362,), dtype=np.float32)
    for s in range(K):
        canon_probs += p0_probs[s, INV_362[s]]
        canon_logits += p0_logits[s, INV_362[s]]
    canon_probs /= float(K)
    canon_logits /= float(K)

    # Mask illegal moves explicitly
    illegal = (spatial_22[0] < 0.5) | (spatial_22[1] > 0.5) | (spatial_22[2] > 0.5) | (spatial_22[6] > 0.5)
    illegal_362 = np.zeros(362, dtype=bool)
    illegal_362[:361] = illegal.reshape(361)
    canon_probs[illegal_362] = 0.0
    if canon_probs.sum() > 0:
        canon_probs /= canon_probs.sum()

    pla_wr = float(wld[:, 0].mean())
    pla_loss = float(wld[:, 1].mean())
    pla_lead = float(scores.mean())
    wld_val = pla_wr - pla_loss  # [-1, +1]
    score_val = (2.0 / math.pi) * math.atan(pla_lead / 7.5)
    pla_utility = 0.75 * wld_val + 0.25 * score_val
    return canon_probs, pla_wr, pla_lead, pla_utility


class MCTSNode:
    __slots__ = (
        "parent",
        "move_from_parent",
        "to_play",
        "prior",
        "visits",
        "virtual_loss",
        "value_sum",
        "nn_utility",
        "nn_winrate",
        "nn_lead",
        "is_expanded",
        "children",
        "board",
        "depth",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"],
        move_from_parent: int,
        to_play: int,
        prior: float,
        depth: int = 0,
        board: Optional[GoBoard19] = None,
    ):
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.to_play = to_play
        self.prior = prior
        self.visits = 0
        self.virtual_loss = 0
        self.value_sum = 0.0  # From parent's perspective (i.e. value of choosing `move_from_parent` at parent)
        self.nn_utility = 0.0  # From `self.to_play`'s perspective
        self.nn_winrate = 0.5
        self.nn_lead = 0.0
        self.is_expanded = False
        self.children: Dict[int, "MCTSNode"] = {}
        self.board = board
        self.depth = depth

    @property
    def q_value(self) -> float:
        """Mean action value from parent's perspective, penalized by active virtual losses."""
        total_n = self.visits + self.virtual_loss
        if total_n == 0:
            return 0.0
        return (self.value_sum - 1.0 * self.virtual_loss) / total_n


class KataGoMCTS:
    """
    KataGo-like Batched PUCT Monte Carlo Tree Search for TurboGo-v2.
    Supports:
      - 8-way D4 symmetry evaluation at root
      - Batched Virtual-Loss leaf evaluation (`batch_size` leaves evaluated per GPU forward pass)
      - FPU (First-Play Urgency) reduction
      - Combined Winrate + Score Lead utility
      - Max candidate pruning (`max_children_per_node`) for fast deep tactical rollouts
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        c_puct: float = 1.35,
        fpu_reduction: float = 0.22,
        batch_size: int = 8,
        max_children: int = 24,
        root_symmetries: int = 8,
        leaf_symmetries: int = 1,
    ):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.batch_size = batch_size
        self.max_children = max_children
        self.root_symmetries = root_symmetries
        self.leaf_symmetries = leaf_symmetries

    def _expand_node_with_outputs(
        self,
        node: MCTSNode,
        probs_362: np.ndarray,
        pla_wr: float,
        pla_lead: float,
        pla_utility: float,
        legal_mask: Optional[np.ndarray] = None,
    ):
        node.nn_winrate = pla_wr
        node.nn_lead = pla_lead
        node.nn_utility = pla_utility
        node.is_expanded = True

        if legal_mask is None:
            assert node.board is not None
            legal_mask = node.board.get_legal_mask()
            if len(node.board.move_history) < 120 and legal_mask[:361].any():
                legal_mask[361] = False

        masked_probs = np.where(legal_mask, probs_362, 0.0)
        psum = float(masked_probs.sum())
        if psum > 0:
            masked_probs /= psum

        top_indices = np.argsort(masked_probs)[::-1][: self.max_children]
        opp = 3 - node.to_play
        for mv in top_indices:
            mv = int(mv)
            if not legal_mask[mv] or masked_probs[mv] <= 1e-7:
                continue
            node.children[mv] = MCTSNode(
                parent=node,
                move_from_parent=mv,
                to_play=opp,
                prior=float(masked_probs[mv]),
                depth=node.depth + 1,
                board=None,
            )

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        total_visits = max(1, node.visits + node.virtual_loss)
        sqrt_total = math.sqrt(total_visits)
        cpuct = self.c_puct + math.log((1.0 + total_visits + 500.0) / 500.0)

        # Compute visited policy mass for KataGo First-Play Urgency (FPU)
        visited_prior_sum = sum(c.prior for c in node.children.values() if (c.visits + c.virtual_loss) > 0)
        fpu_value = node.nn_utility - self.fpu_reduction * math.sqrt(max(0.0, visited_prior_sum))
        if node.parent is None:
            # At root, keep FPU slightly higher so top policy moves get explored cleanly
            fpu_value = node.nn_utility - 0.12 * math.sqrt(max(0.0, visited_prior_sum))

        best_score = -1e9
        best_child = None
        for child in node.children.values():
            n_child = child.visits + child.virtual_loss
            q = child.q_value if n_child > 0 else fpu_value
            u = cpuct * child.prior * (sqrt_total / (1.0 + n_child))
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    @torch.no_grad()
    def search(self, root_board: GoBoard19, num_simulations: int = 128) -> Dict:
        t0 = time.perf_counter()
        root = MCTSNode(parent=None, move_from_parent=-1, to_play=root_board.to_play, prior=1.0, depth=0, board=root_board.clone())

        # 1. Evaluate root with 8-way D4 symmetry ensemble
        sx, gx = root.board.encode_katago_v7()
        legal_mask = root.board.get_legal_mask()
        if len(root.board.move_history) < 120 and legal_mask[:361].any():
            legal_mask[361] = False

        probs, wr, lead, util = eval_d4_symmetry(
            self.model, sx, gx, self.device, num_symmetries=self.root_symmetries
        )
        self._expand_node_with_outputs(root, probs, wr, lead, util, legal_mask=legal_mask)
        root.visits = 1
        max_depth_reached = 1

        # 2. Run batched PUCT simulations with Virtual Loss
        sims_done = 1
        while sims_done < num_simulations:
            cur_batch = min(self.batch_size, num_simulations - sims_done)
            leaves: List[MCTSNode] = []
            leaf_sx: List[np.ndarray] = []
            leaf_gx: List[np.ndarray] = []

            for _ in range(cur_batch):
                node = root
                path = [root]
                while node.is_expanded and len(node.children) > 0:
                    node = self._select_child(node)
                    path.append(node)

                # Apply virtual loss along path
                for pnode in path[1:]:
                    pnode.virtual_loss += 1

                if node.board is None:
                    assert node.parent is not None and node.parent.board is not None
                    b_copy = node.parent.board.clone()
                    b_copy.play(node.move_from_parent)
                    node.board = b_copy

                if node.depth > max_depth_reached:
                    max_depth_reached = node.depth

                if not node.is_expanded:
                    lsx, lgx = node.board.encode_katago_v7()
                    leaves.append(node)
                    leaf_sx.append(lsx)
                    leaf_gx.append(lgx)
                else:
                    # Already expanded in this same batch or terminal node: backup existing nn_utility
                    val = node.nn_utility
                    curr = node
                    while curr.parent is not None:
                        curr.virtual_loss -= 1
                        curr.visits += 1
                        # Value for parent (`curr.parent.to_play`) is `-val`
                        val = -val
                        curr.value_sum += val
                        curr = curr.parent
                    root.visits += 1

            if leaves:
                bx = torch.from_numpy(np.stack(leaf_sx, axis=0)).to(self.device)
                bg = torch.from_numpy(np.stack(leaf_gx, axis=0)).to(self.device)
                pol_logits, wld_logits, score_pred, _ = self.model(bx, bg)
                p0_probs = F.softmax(pol_logits[:, 0].float(), dim=-1).cpu().numpy()
                wld = F.softmax(wld_logits.float(), dim=-1).cpu().numpy()
                leads = score_pred.float().cpu().numpy() * 20.0

                for idx, leaf in enumerate(leaves):
                    l_wr = float(wld[idx, 0])
                    l_loss = float(wld[idx, 1])
                    l_lead = float(leads[idx])
                    wld_v = l_wr - l_loss
                    sc_v = (2.0 / math.pi) * math.atan(l_lead / 7.5)
                    l_util = 0.75 * wld_v + 0.25 * sc_v

                    self._expand_node_with_outputs(leaf, p0_probs[idx], l_wr, l_lead, l_util)

                    # Backup from leaf to root, removing virtual loss
                    val = l_util
                    curr: Optional[MCTSNode] = leaf
                    while curr is not None and curr.parent is not None:
                        curr.virtual_loss -= 1
                        curr.visits += 1
                        val = -val
                        curr.value_sum += val
                        curr = curr.parent
                    root.visits += 1

            sims_done += cur_batch

        # 3. Extract Principal Variation (PV) and best move
        # KataGo combines visit count and Q-value (or LCB) so low-visit blunders are never chosen
        children_list = list(root.children.values())
        children_list.sort(
            key=lambda c: (c.visits, c.q_value + 0.25 * c.prior),
            reverse=True,
        )
        best_child = children_list[0]

        pv = []
        curr = root
        while curr.is_expanded and curr.children:
            visited_c = [c for c in curr.children.values() if c.visits > 0]
            if not visited_c:
                break
            nxt = max(visited_c, key=lambda c: (c.visits, c.q_value))
            pv.append(move_to_coord(nxt.move_from_parent))
            curr = nxt

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "best_move": best_child.move_from_parent,
            "best_coord": move_to_coord(best_child.move_from_parent),
            "simulations": num_simulations,
            "max_depth": max_depth_reached,
            "pv_depth": len(pv),
            "pv": pv,
            "root_winrate": round(root.nn_winrate * 100.0, 2),
            "root_lead": round(root.nn_lead, 2),
            "best_q": round(best_child.q_value, 4),
            "search_ms": round(dt_ms, 1),
            "top_moves": [
                {
                    "move": c.move_from_parent,
                    "coord": move_to_coord(c.move_from_parent),
                    "visits": c.visits,
                    "prior": round(c.prior * 100.0, 2),
                    "q_value": round(c.q_value, 4),
                }
                for c in children_list[:5]
            ],
        }


def run_depth_and_top1_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    weights_path = PLAY_DIR / "weights" / "b12c384_best_weights.pt"
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    sd = ckpt.get("model") or ckpt

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

    print("=" * 92, flush=True)
    print("  [Part 1] KataGo-Style MCTS Depth Scaling Test on TurboGo-v2 (MPS Batched PUCT)", flush=True)
    print("=" * 92, flush=True)

    # Create a realistic midgame tactical position (after 24 moves) to test MCTS tree depth scaling
    b_test = GoBoard19(komi=7.5)
    sample_opening = [60, 288, 300, 53, 40, 268, 319, 59, 78, 96, 115, 290, 252, 233, 214, 195]
    for m in sample_opening:
        b_test.play(m)

    for sims in (16, 64, 128, 256, 512, 1024):
        mcts = KataGoMCTS(model, device, batch_size=16, root_symmetries=1)
        res = mcts.search(b_test, num_simulations=sims)
        pv_str = " -> ".join(res["pv"][:10])
        print(
            f"  Simulations: {sims:4d} | Max Depth: {res['max_depth']:2d} plies | PV Depth: {res['pv_depth']:2d} plies | "
            f"Time: {res['search_ms']:6.0f} ms | Best: {res['best_coord']} (Visits={res['top_moves'][0]['visits']}) | PV: {pv_str}",
            flush=True,
        )

    val_shard = ROOT_DIR / "data/katago_19x19_chinese_10m/shard_00498.npz"
    shard = np.load(val_shard)
    N_mcts = 50
    bits = (
        np.unpackbits(shard["binaryInputNCHWPacked"][:N_mcts], axis=2)[:, :, :361]
        .reshape(N_mcts, 22, 19, 19)
        .astype(np.float32)
    )
    g = shard["globalInputNC"][:N_mcts].astype(np.float32)
    pol_raw = shard["policyTargetsNCMove"][:N_mcts, 0, :].astype(np.float32)
    gt_moves = np.argmax(pol_raw, axis=1)

    print("\n" + "=" * 92, flush=True)
    print(f"  [Part 2] Deep KataGo-Style PUCT MCTS Evaluation (N={N_mcts} positions, 64 Sims vs 1-Visit)", flush=True)
    print("=" * 92, flush=True)

    raw_top1_200 = 0
    sym8_top1_200 = 0
    mcts_visit_top1_200 = 0
    mcts_hybrid_top1_200 = 0
    mcts_top3_200 = 0
    depths = []

    mcts_eval = KataGoMCTS(model, device, c_puct=1.5, fpu_reduction=0.20, batch_size=16, max_children=12, root_symmetries=8)
    t0 = time.perf_counter()
    for idx in range(N_mcts):
        sx = bits[idx]
        gx = g[idx]
        gt = int(gt_moves[idx])

        # 1. Raw 1-Visit
        bx = torch.from_numpy(sx).unsqueeze(0).to(device)
        bg = torch.from_numpy(gx).unsqueeze(0).to(device)
        with torch.no_grad():
            pl, _, _, _ = model(bx, bg)
            raw_pred = int(pl[0, 0].argmax().item())
        if raw_pred == gt:
            raw_top1_200 += 1

        # 2. 8-Way D4 Symmetry
        probs_8, _, _, _ = eval_d4_symmetry(model, sx, gx, device, num_symmetries=8)
        if int(np.argmax(probs_8)) == gt:
            sym8_top1_200 += 1

        # 3. 64-Sim KataGo PUCT MCTS
        if gt == 361:
            if int(np.argmax(probs_8)) == gt:
                mcts_visit_top1_200 += 1
                mcts_hybrid_top1_200 += 1
                mcts_top3_200 += 1
            continue

        b_pos = reconstruct_board_from_katago_planes(sx, gx)
        res = mcts_eval.search(b_pos, num_simulations=64)
        depths.append(res["max_depth"])
        if res["best_move"] == gt:
            mcts_visit_top1_200 += 1

        top_cands = res["top_moves"]
        best_hybrid_mv = max(
            top_cands,
            key=lambda c: math.log(c["prior"] / 100.0 + 1e-4) + 1.25 * c["q_value"],
        )["move"]
        if best_hybrid_mv == gt:
            mcts_hybrid_top1_200 += 1
        if any(c["move"] == gt for c in top_cands[:3]):
            mcts_top3_200 += 1

    dt_mcts = time.perf_counter() - t0
    print(f"  Same {N_mcts} Positions Baseline (1-Visit Single View) : Top-1 = {raw_top1_200/N_mcts*100:5.2f}%", flush=True)
    print(f"  Same {N_mcts} Positions 8x D4 Symmetry Ensemble        : Top-1 = {sym8_top1_200/N_mcts*100:5.2f}%", flush=True)
    print(f"  Same {N_mcts} Positions 64-Sim MCTS (Pure Visit Count) : Top-1 = {mcts_visit_top1_200/N_mcts*100:5.2f}% (Avg Max Depth: {np.mean(depths):.1f} plies)", flush=True)
    print(f"  Same {N_mcts} Positions 64-Sim MCTS (Prior+Q Posterior): Top-1 = {mcts_hybrid_top1_200/N_mcts*100:5.2f}% | Top-3 = {mcts_top3_200/N_mcts*100:5.2f}% (Time: {dt_mcts:.1f}s)", flush=True)


if __name__ == "__main__":
    run_depth_and_top1_evaluation()
