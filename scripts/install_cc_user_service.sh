#!/usr/bin/env bash
# install_cc_user_service.sh — run the Command Center as a systemd *user*
# service so it starts on login and restarts if it dies.
#
# No root required (unlike the gateway, which needs it to bind L2CAP).
#
#   bash scripts/install_cc_user_service.sh              # install + start
#   bash scripts/install_cc_user_service.sh --uninstall
#
# Status / logs:
#   systemctl --user status handsneyes-cc
#   journalctl --user -u handsneyes-cc -f
#
# Note: user services normally start at login. To have it come up on boot
# without logging in, an admin must run: sudo loginctl enable-linger $USER
set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$SELF")/.." && pwd)}"
VENV="${VENV:-$REPO_DIR/.venv}"
CC_BIN="$VENV/bin/handsneyes"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/handsneyes-cc.service"

if [ "${1:-}" = "--uninstall" ]; then
    systemctl --user disable --now handsneyes-cc 2>/dev/null
    rm -f "$UNIT"
    systemctl --user daemon-reload
    echo "removed handsneyes-cc user service"
    exit 0
fi

[ -x "$CC_BIN" ] || { echo "FATAL: $CC_BIN not found (pip install -e .)" >&2; exit 1; }

mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=handsneyes Command Center
After=network.target

[Service]
Type=simple
# WorkingDirectory is load-bearing: the target registry is read from
# ./config/targets.toml, and without it cc falls back to the built-in
# headless target (pointing at the old Pi address) with no error.
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$CC_BIN cc
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now handsneyes-cc 2>&1 | tail -2
sleep 6

printf '%-20s %s\n' handsneyes-cc "$(systemctl --user is-active handsneyes-cc)"
if ss -ltn 2>/dev/null | grep -q ':8765'; then
    echo "Command Center listening on :8765"
    echo "target: $(journalctl --user -u handsneyes-cc -n 20 --no-pager 2>/dev/null | grep -o "target=.*" | tail -1)"
else
    echo "WARN not listening — journalctl --user -u handsneyes-cc -n 30"
fi
