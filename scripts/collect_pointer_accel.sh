#!/usr/bin/env bash
# Collect pointer-acceleration training data by firing /api/mouse/click_at
# at a varied grid of pixel positions. Each invocation runs the visual-
# servo homer end-to-end — which (after this commit) persists every
# step as a JSONL row under <run>/homer/<id>/history.jsonl. The dataset
# builder consumes those rows.
#
# Choose --grid 4 for a 4×3 = 12-point probe (~3 min). Each click_at
# does ~5–15 homer iterations, so a single 12-point probe produces
# ~60–180 training samples covering the full screen.
#
# Usage:
#   scripts/collect_pointer_accel.sh                # default 4×3 grid
#   scripts/collect_pointer_accel.sh --grid 6       # 6×4 grid (24 pts)
#   scripts/collect_pointer_accel.sh --base http://127.0.0.1:8765
set -u
BASE=http://127.0.0.1:8765
GRID=4
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2 ;;
    --grid) GRID=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
COLS=$GRID
ROWS=$(( GRID * 3 / 4 ))
[ "$ROWS" -lt 1 ] && ROWS=1
TOTAL=$(( COLS * ROWS ))
echo "probing $COLS × $ROWS = $TOTAL click_at positions"

# Avoid corners (homer slams there before each run anyway) and the
# very-edge regions which can be off-camera depending on bezel.
# Sample at 0.10..0.90 in both axes.
i=0
for r in $(seq 1 $ROWS); do
  for c in $(seq 1 $COLS); do
    i=$(( i + 1 ))
    x_pct=$(python3 -c "print(round(0.10 + ($c-1)*(0.80/($COLS-1 if $COLS>1 else 1)), 4))")
    y_pct=$(python3 -c "print(round(0.10 + ($r-1)*(0.80/($ROWS-1 if $ROWS>1 else 1)), 4))")
    printf "  [%2d/%d] click_at (%5s, %5s) ... " "$i" "$TOTAL" "$x_pct" "$y_pct"
    resp=$(curl -s --max-time 120 -X POST "$BASE/api/mouse/click_at" \
      -H 'Content-Type: application/json' \
      -d "{\"x_pct\": $x_pct, \"y_pct\": $y_pct, \"button\": \"left\"}")
    ok=$(echo "$resp" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('ok', False))
except: print('?')")
    echo "ok=$ok"
    # Force the next click_at to run with full slam + detect:
    # invalidate cc's no-slam click cache by sending a manual move.
    # Without this, consecutive probes within the cache TTL would
    # skip the slam phase and the long-jump model would never fire,
    # which is exactly the data we're trying to collect.
    curl -s --max-time 5 -X POST "$BASE/api/mouse/move" \
      -H 'Content-Type: application/json' \
      -d '{"dx": 1, "dy": 0}' >/dev/null
    sleep 1
  done
done

echo
echo "=== summary ==="
total=$(find ~/.local/share/handsneyes/runs -path "*/homer/*" -name "history.jsonl" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1)
echo "$total"
