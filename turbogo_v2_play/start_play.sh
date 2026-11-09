#!/usr/bin/env zsh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# 1. Start the read-only TurboGo-v2 inference server if not already listening on 127.0.0.1:8788
if ! curl -fsSL --max-time 2 http://127.0.0.1:8788/api/state >/dev/null 2>&1; then
  echo "[TurboGo-v2] Starting local read-only inference engine on http://127.0.0.1:8788 ..."
  nohup "$DIR/.venv/bin/python" "$DIR/engine_server.py" > "$DIR/engine_server.log" 2>&1 </dev/null &
  for i in {1..30}; do
    if curl -fsSL --max-time 1 http://127.0.0.1:8788/api/state >/dev/null 2>&1; then
      echo "[TurboGo-v2] Engine ready!"
      break
    fi
    sleep 0.5
  done
else
  echo "[TurboGo-v2] Engine is already running on http://127.0.0.1:8788"
fi

# 2. Launch the Swift Go client (pass --cli for terminal mode, or default to macOS SwiftUI GUI)
exec "$DIR/TurboGoV2Play" "$@"
