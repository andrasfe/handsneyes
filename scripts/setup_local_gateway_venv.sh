#!/usr/bin/env bash
# setup_local_gateway_venv.sh — build the Python venv that runs the
# afferent BT-HID gateway on the local Linux host (a regular machine
# acting as the Bluetooth mouse, replacing a separate Pi). Run as the
# normal user, NOT root.
#
#   bash scripts/setup_local_gateway_venv.sh
#
# Why a dedicated venv (and why it's fiddly):
#   * The gateway needs python-dbus, which is a SYSTEM package
#     (python3-dbus) that can't be pip-installed cleanly — so the venv is
#     created with --system-site-packages to see it.
#   * BUT --system-site-packages also lets the venv borrow packages from
#     the invoking user's ~/.local. The gateway is later launched as ROOT
#     (it must, to bind L2CAP PSM 17/19), and root can't see your
#     ~/.local — so anything that resolved from there is missing at
#     runtime (the classic "ModuleNotFoundError: anyio" when run as root).
#   * Fix: install the gateway's web stack INTO the venv with
#     PYTHONNOUSERSITE=1 so pip ignores ~/.local and everything lands in
#     the venv itself, visible to root.
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "$(dirname "$SELF")/.." && pwd)"
GW_VENV="${GW_VENV:-$REPO_DIR/.venv-gw}"

echo "[1/3] Ensuring system python3-dbus is present (needs sudo once)..."
if ! python3 -c "import dbus" >/dev/null 2>&1; then
    sudo apt-get install -y -qq python3-dbus python3-gi python3-evdev
fi
python3 -c "import dbus" >/dev/null 2>&1 \
    && echo "  system dbus OK" \
    || { echo "  FATAL: python3-dbus still missing" >&2; exit 1; }

echo "[2/3] Creating venv at $GW_VENV (--system-site-packages, for dbus)..."
python3 -m venv --system-site-packages "$GW_VENV"
"$GW_VENV/bin/python" -m pip install -q -U pip

echo "[3/3] Installing the gateway INTO the venv (PYTHONNOUSERSITE=1)..."
# PYTHONNOUSERSITE=1 + --ignore-installed forces every dep into the venv
# rather than being satisfied from ~/.local (invisible to root at runtime).
PYTHONNOUSERSITE=1 "$GW_VENV/bin/pip" install -q --ignore-installed \
    "afferent[gateway]" evdev

echo "  verifying imports as root would see them (user-site disabled)..."
PYTHONNOUSERSITE=1 "$GW_VENV/bin/python" -c \
    "import dbus, anyio, fastapi, uvicorn, evdev, afferent.gateway.server; print('  gateway venv OK')"

echo
echo "Done. Gateway binary: $GW_VENV/bin/afferent-gateway"
echo "Next: sudo bash scripts/setup_local_hid.sh"
