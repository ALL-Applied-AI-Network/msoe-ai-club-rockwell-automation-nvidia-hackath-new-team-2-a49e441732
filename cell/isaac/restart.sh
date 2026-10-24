#!/bin/zsh
# From the Mac: sync cell/isaac + cell/calibration to the GPU box, restart the Isaac cell server, wait until it's ready.
#     isaac/restart.sh            (ISAAC_SSH=user@host to use another box)
# A start right after a kill sometimes hangs inside Kit's startup: wait for the old process to exit and free the GPU,
# and if the new one prints nothing for 40 s, kill it and start again (the second start has always worked).
H=${ISAAC_SSH:-joel@shepherd-traffic-base-server}
cd "$(dirname "$0")/.."
rsync -a --exclude __pycache__ calibration isaac "$H":cell-isaac/ || exit 1
for attempt in 1 2 3; do
  ssh -o ConnectTimeout=15 "$H" '
    pkill -f "[k]it/python/bin/python3 isaac/server.py"
    for i in $(seq 1 30); do pgrep -f "[k]it/python/bin/python3 isaac/server.py" >/dev/null || break; sleep 1; done
    for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && break; sleep 1; done
    rm -f ~/cell-isaac/isaac/server.log
    cd ~/cell-isaac && (setsid nohup isaac/run_server.sh >/dev/null 2>&1 </dev/null &)'
  for i in $(seq 1 8); do
    sleep 5
    n=$(ssh "$H" 'wc -l < ~/cell-isaac/isaac/server.log 2>/dev/null || echo 0')
    [ "${n:-0}" -gt 5 ] && break
  done
  if [ "${n:-0}" -le 5 ]; then echo "Kit hung at startup (attempt $attempt); restarting"; continue; fi
  for i in $(seq 1 60); do
    r=$(ssh "$H" 'grep -c "ready on\|Traceback" ~/cell-isaac/isaac/server.log 2>/dev/null')
    [ "${r:-0}" -gt 0 ] && break
    sleep 5
  done
  ssh "$H" 'grep -m3 "ready on\|Traceback\|Error:" ~/cell-isaac/isaac/server.log | grep -v mat_print'
  exit 0
done
echo "the Isaac server did not start"; exit 1
