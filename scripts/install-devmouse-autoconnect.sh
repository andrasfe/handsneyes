#!/usr/bin/env bash
# One-shot installer for the devmouse BT auto-reconnect agent on a macOS
# TARGET being driven by handsneyes. Run it ON the target Mac:
#
#   curl -fsSL https://raw.githubusercontent.com/andrasfe/handsneyes/main/scripts/install-devmouse-autoconnect.sh | bash
#
# It fetches the agent, ensures blueutil (via Homebrew), and installs the
# LaunchAgent so the Mac auto-reconnects to the Pi HID gateway ("devmouse")
# whenever it's awake and in range — like a normal Bluetooth mouse. After
# this the target never needs a manual Connect again.
set -uo pipefail

AGENT_URL="https://raw.githubusercontent.com/andrasfe/handsneyes/main/scripts/macos-devmouse-autoconnect.sh"
DEST="$HOME/.local/bin/macos-devmouse-autoconnect.sh"

echo "[devmouse-install] fetching agent..."
mkdir -p "$(dirname "$DEST")"
curl -fsSL "$AGENT_URL" -o "$DEST" || { echo "DEVMOUSE_INSTALL: FETCH_FAILED"; exit 1; }
chmod +x "$DEST"

# Ensure blueutil (the agent uses it to connect).
have_blueutil() {
    command -v blueutil >/dev/null 2>&1 \
        || [ -x /opt/homebrew/bin/blueutil ] || [ -x /usr/local/bin/blueutil ]
}
if ! have_blueutil; then
    if command -v brew >/dev/null 2>&1; then
        echo "[devmouse-install] installing blueutil via Homebrew..."
        brew install blueutil || echo "[devmouse-install] brew install blueutil failed"
    else
        echo "DEVMOUSE_INSTALL: NO_BREW — install Homebrew (brew.sh), then: brew install blueutil"
    fi
fi

bash "$DEST" --install && echo "DEVMOUSE_INSTALL: DONE (agent loaded)"
if have_blueutil; then
    echo "DEVMOUSE_INSTALL: blueutil OK — auto-reconnect active"
else
    echo "DEVMOUSE_INSTALL: WARNING blueutil missing — agent installed but can't connect until blueutil is present"
fi
