# TurboGo: High-Efficiency Nested Bottleneck Transformer Go AI (`TurboGo-Mini` 3M & `TurboGo-Large` 20M)

**TurboGo** is a pure Pre-RMSNorm Nested Bottleneck Transformer (NBT) Go engine with **Learnable Multi-Scale 2D Rotary Position Embeddings (`rope_freqs`)**, **SwiGLU FFNs**, and a **32-Channel Explicit Tactical Input Stem**, trained on 100,000,000 KataGo 19×19 Chinese-rules positions.

---

## 1. Model Variants & Checkpoints

| Variant | Architecture | Parameters | Checkpoint in Repo / Release |
| :--- | :--- | ---: | :--- |
| **TurboGo-Mini** (`b6c192-nbt12`) | `d_model=192, c_mid=128, n_blocks=6, internal_length=2, n_heads=4, ffn_dim=256` (12 Global 2D-RoPE Self-Attention + 12 SwiGLU FFN layers) | **3.06M** | [`checkpoints/turbogo_mini_best_weights.pt`](checkpoints/turbogo_mini_best_weights.pt) (`5.9 MB`) |
| **TurboGo-Large (v2)** (`b11c384-nbt22`) | `d_model=384, c_mid=256, n_blocks=11, internal_length=2, n_heads=8, ffn_dim=640` (22 Global 2D-RoPE Self-Attention + 22 SwiGLU FFN layers) | **20.15M** | `src/model.py` (`TurboGoHybridTransformer`) |
| **TurboGo-v1** (`b12c384`) | 4 Stages × (2 `KataGPoolConvBlocks` + 1 `FlashAttn2DRoPEBlock`), `d_model=384, n_heads=8` | **30.87M** | [`checkpoints/b12c384_best_weights.pt`](checkpoints/b12c384_best_weights.pt) (`40 MB`, BF16) & Release `v2.0.0` (`b12c384_best_fp32_weights.pt`, `118 MB`) |

---

## 2. Key Architectural Highlights (`src/model.py`)

1. **Nested Bottleneck Transformer (`TurboNBTBlock`)**:
   - Down-projects `c_main -> c_mid`, applies residual 3×3 Depthwise Convolution + KataGo Global Pooling (`[mean, mean*0.5, max]`), runs `internal_length=2` pairs of `(Learnable2DRoPEAttention + SwiGLUFFNBlock)` over all 361 board intersections, and up-projects back to the pure identity residual stream (`x = x + sub(x)`).
2. **Learnable Multi-Scale 2D-RoPE (`rope_freqs`)**:
   - Every attention head learns its own 2D `(x, y)` spatial frequency bank initialized across 8 compass/diagonal orientations and log-spaced wavelengths from `2.0` to `38.0` intersections.
3. **32-Channel Explicit Tactical Input Stem (`augment_tactical_planes_32`)**:
   - Expands the 22 KataGo V7 binary input planes to 32 channels on GPU by appending explicit multiplicative tactical planes (`own/opp × 1/2/3-liberty`, `own/opp × ladder`, and 4-connected `immediate capture / immediate escape` intersections).
4. **Dual Policy, Value (WLD), Score Lead & Ownership Heads**:
   - `KataGPoolPolicyHeadV2`: Outputs `(B, 2, 362)` logits (`[current_player_policy, opponent_next_move_policy]`) with built-in illegal move masking.
   - `KataGPoolValueHeadV2`: Outputs `(wld_logits, score_pred, own_pred)`.

---

## 3. Repository Structure

```text
TurboGo/
├── src/
│   ├── model.py                           # TurboGo-v2 (20.15M) & TurboGo-Mini (3.06M) NBT Transformer
│   └── model_v1.py                        # TurboGo-v1 (30.87M) Hybrid Conv-Transformer
├── checkpoints/
│   ├── turbogo_mini_best_weights.pt       # Pretrained TurboGo-Mini (3.06M) weights
│   ├── turbogo_mini_metrics.jsonl         # Training & validation metrics log for TurboGo-Mini
│   ├── b12c384_best_weights.pt            # Pretrained b12c384 weights
│   └── arena_vs_katago10b_results.json    # Benchmark & arena evaluation results
├── scripts/
│   ├── train_turbogo_mini.py              # High-throughput BF16 + D4 symmetry training pipeline for TurboGo-Mini
│   ├── train_pretrain_a100.py             # Distributed / single-GPU pretraining script for TurboGo-Large
│   ├── deploy_and_arena_vs_katago10b.py   # Exact PyTorch loader for KataGo tf2-b10c384 & head-to-head arena
│   └── dashboard_server.py                # Real-time training telemetry web dashboard
└── turbogo_v2_play/
    ├── mcts_engine.py                     # Batched PUCT MCTS engine for TurboGo
    ├── engine_server.py                   # HTTP / GTP inference server
    ├── TurboGoPlay.swift                  # Native macOS GUI launcher
    └── arena_mini_vs_tinygo.py            # Head-to-head match runner against TinyGo
```
