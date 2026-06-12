#!/usr/bin/env bash
# pi_offline_install.sh — install handsneyes on the Pi WITHOUT internet.
#
# Runs ON THE PI. The Pi normally has no route to PyPI: radio_mode
# disables WiFi so the BCM43436s shared radio is free for BT HID, and
# the USB ECM link only reaches the dev mac. So instead of pip:
#
#   1. Create ~/handsneyes/.venv (--system-site-packages for the apt
#      dbus/evdev bindings).
#   2. Copy the dependency tree from the legacy terminaleyes venv —
#      it carries the exact piwheels builds (fastapi, uvicorn,
#      pydantic, numpy, opencv, ...) this hardware already runs.
#   3. Point the venv at ~/handsneyes/src via a .pth (editable-install
#      equivalent, no build backend needed).
#   4. Write the handsneyes-pi console script by hand.
#
# Idempotent — safe to re-run after every rsync.

set -uo pipefail

HN_DIR="$HOME/handsneyes"
LEGACY_VENV="$HOME/terminaleyes/.venv"
VENV="$HN_DIR/.venv"

die() { echo "ERROR: $1" >&2; exit 1; }

[ -d "$HN_DIR/src/handsneyes" ] || die "$HN_DIR/src/handsneyes missing — rsync the repo first"

PYVER=$(python3 -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')

if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV" || die "venv creation failed"
fi
SITE="$VENV/lib/$PYVER/site-packages"
[ -d "$SITE" ] || die "venv site-packages not found at $SITE"

# 2. Dependency tree from the legacy venv (skip if it's not there —
#    a future online Pi can pip install instead).
LEGACY_SITE="$LEGACY_VENV/lib/$PYVER/site-packages"
if [ -d "$LEGACY_SITE" ]; then
    echo "Copying dependency tree from $LEGACY_SITE ..."
    rsync -a \
        --exclude='terminaleyes*' \
        --exclude='__editable__*' \
        --exclude='pip*' \
        "$LEGACY_SITE/" "$SITE/"
else
    echo "WARNING: legacy venv not found — assuming deps are already in $SITE"
fi

# 3. Editable-install equivalent.
echo "$HN_DIR/src" > "$SITE/handsneyes_src.pth"

# 4. Console script.
cat > "$VENV/bin/handsneyes-pi" <<EOF
#!$VENV/bin/python3
from handsneyes.pi.server import main

if __name__ == "__main__":
    main()
EOF
chmod +x "$VENV/bin/handsneyes-pi"

# 5. Smoke import.
"$VENV/bin/python3" -c "from handsneyes.pi.server import main; print('INSTALL_OK')" \
    || die "handsneyes.pi.server import failed"

echo "offline install complete: $VENV/bin/handsneyes-pi"
