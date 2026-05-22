#!/usr/bin/env bash
# Oversample regions the baseline canary showed as model-weak.
# Currently: left-side (x ∈ [0.05, 0.25]) and bottom (y ∈ [0.75, 0.95]).
# Fires ~30 click_at across those regions, with no-slam-cache
# invalidation between each call so each click is a full slam +
# detect (which is what produces useful per-step history rows).
#
# Usage: scripts/collect_targeted.sh [--base http://127.0.0.1:8765]
set -u
BASE=http://127.0.0.1:8765
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

probe() {
  local x=$1 y=$2 idx=$3 total=$4
  printf "  [%2d/%d] targeted_click_at (%5s, %5s) ... " "$idx" "$total" "$x" "$y"
  resp=$(curl -s --max-time 120 -X POST "$BASE/api/mouse/click_at" \
    -H 'Content-Type: application/json' \
    -d "{\"x_pct\": $x, \"y_pct\": $y, \"button\": \"left\"}")
  ok=$(echo "$resp" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('ok', False))
except: print('?')")
  steps=$(echo "$resp" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('steps', '?'))
except: print('?')")
  echo "ok=$ok steps=$steps"
  curl -s --max-time 5 -X POST "$BASE/api/mouse/move" \
    -H 'Content-Type: application/json' \
    -d '{"dx": 1, "dy": 0}' >/dev/null
  sleep 1
}

# Left strip — 5 x-positions x 6 y-positions, biased low x.
xs="0.05 0.08 0.12 0.18 0.25"
ys="0.10 0.25 0.45 0.65 0.80 0.92"

# Build the click list
total=0
for x in $xs; do for y in $ys; do total=$((total+1)); done; done
# Bottom strip — 6 x-positions x 4 y-positions, biased high y.
xs2="0.10 0.25 0.40 0.55 0.75 0.90"
ys2="0.78 0.85 0.90 0.94"
for x in $xs2; do for y in $ys2; do total=$((total+1)); done; done

echo "targeted probe: $total click_at positions across model-weak regions"

i=0
for x in $xs; do
  for y in $ys; do
    i=$((i+1))
    probe "$x" "$y" "$i" "$total"
  done
done
for x in $xs2; do
  for y in $ys2; do
    i=$((i+1))
    probe "$x" "$y" "$i" "$total"
  done
done

echo
echo "=== summary ==="
find ~/.local/share/handsneyes/runs -name "history.jsonl" -newer /tmp/handsneyes_cc.log 2>/dev/null | xargs wc -l 2>/dev/null | tail -1
