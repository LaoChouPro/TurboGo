#!/usr/bin/env python3
import json
import os
import sys
import time

STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climb_19x19_status.json")

last_games_len = 0
last_step_sig = None
last_change_ts = time.time()

while True:
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        time.sleep(1.5)
        continue

    games = st.get("games", [])
    if len(games) > last_games_len:
        for g in games[last_games_len:]:
            res_icon = "WIN" if g["is_win"] else "LOSS"
            print(
                f"[Game #{g['game_num']} {res_icon}] {g['user_rank_before']} ({g['user_color']}) vs {g['ai_name']}({g['ai_rank']}) "
                f"-> {g['result_txt']} ({g['moves']} moves) | Rank: {g['user_rank_after']} ({g['promo_msg']})",
                flush=True,
            )
        last_games_len = len(games)
        last_change_ts = time.time()

    if not st.get("running", True):
        print(f"[CLIMB FINISHED] Stopped reason: {st.get('stopped_reason')}", flush=True)
        print(f"Final Rank: {st.get('current_rank')} | Record: {st.get('wins')}W - {st.get('losses')}L", flush=True)
        sys.exit(0)

    cg = st.get("current_game") or {}
    step_sig = (cg.get("game_index"), cg.get("step"))
    if step_sig != last_step_sig:
        last_step_sig = step_sig
        last_change_ts = time.time()
    elif time.time() - last_change_ts > 45.0:
        print(f"[WATCHDOG] No move progress for 45s at {step_sig}. Exiting to inspect browser.", flush=True)
        sys.exit(0)

    time.sleep(2.0)
