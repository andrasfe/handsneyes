#!/usr/bin/env bash
# Variant of collect_pointer_accel.sh tuned for macOS self-capture
# training. Avoids the screen regions that would fire destructive
# actions on a busy Mac desktop:
#
#   - y < 0.18: macOS menu bar + window controls + browser tab strip
#   - y > 0.78: macOS dock + most app status bars
#
# That confines clicks to the centre 60% of the y-axis, which on a
# typical desktop lands inside browser content / IDE editors / Notes /
# wallpaper. Clicks at these positions move the cursor (and on text
# they reposition the caret) but don't fire app launches.
#
# Usage: scripts/collect_pointer_accel_safe.sh [--base URL] [--grid N]
set -u
BASE=http://127.0.0.1:8765
GRID=8
MIN_X=0.10
MAX_X=0.90
MIN_Y=0.20
MAX_Y=0.76
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2 ;;
    --grid) GRID=$2; shift 2 ;;
    --min-y) MIN_Y=$2; shift 2 ;;
    --max-y) MAX_Y=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
COLS=$GRID
ROWS=$(( GRID * 3 / 4 ))
[ "$ROWS" -lt 1 ] && ROWS=1
TOTAL=$(( COLS * ROWS ))
echo "safe-grid probe: $COLS × $ROWS = $TOTAL points"
echo "  x in [$MIN_X, $MAX_X],  y in [$MIN_Y, $MAX_Y]"
i=0
for r in $(seq 1 $ROWS); do
  for c in $(seq 1 $COLS); do
    i=$(( i + 1 ))
    x_pct=$(python3 -c "print(round($MIN_X + ($c-1)*(($MAX_X - $MIN_X) / ($COLS-1 if $COLS>1 else 1)), 4))")
    y_pct=$(python3 -c "print(round($MIN_Y + ($r-1)*(($MAX_Y - $MIN_Y) / ($ROWS-1 if $ROWS>1 else 1)), 4))")
    printf "  [%2d/%d] click_at (%5s, %5s) ... " "$i" "$TOTAL" "$x_pct" "$y_pct"
    resp=$(curl -s --max-time 120 -X POST "$BASE/api/mouse/click_at" \
      -H 'Content-Type: application/json' \
      -d "{\"x_pct\": $x_pct, \"y_pct\": $y_pct, \"button\": \"left\"}")
    ok=$(echo "$resp" | python3 -c "import json,sys
try: d=json.load(sys.stdin); print(f\"ok={d.get('ok')} steps={d.get('steps','?')} reason={d.get('reason','?')}\")
except: print('?')")
    echo "$ok"
    curl -s --max-time 5 -X POST "$BASE/api/mouse/move" \
      -H 'Content-Type: application/json' \
      -d '{"dx": 1, "dy": 0}' >/dev/null
    sleep 1
  done
done

echo
echo "=== summary ==="
find ~/.local/share/handsneyes/runs -name "history.jsonl" -mmin -30 \
  | xargs wc -l 2>/dev/null | tail -1
