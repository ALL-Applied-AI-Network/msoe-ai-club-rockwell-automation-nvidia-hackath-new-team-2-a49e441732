#!/bin/bash
# Start the Isaac cell server on this box, with its window on the desktop (display :1). Run from cell/:
#     isaac/run_server.sh [--headless]
# Logs to isaac/server.log; the Mac reaches it through an SSH tunnel (lab/isaaccell.py opens one).
cd "$(dirname "$0")/.."
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export TWIN="${TWIN:-$HOME/fanuc-er4ia-sim}"
exec "${ISAAC:-$HOME/isaacsim}/python.sh" isaac/server.py "$@" 2>&1 | tee isaac/server.log
