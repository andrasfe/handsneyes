"""Declarative cursor-theme advice for GNOME targets.

The HSV cursor finder is much faster + cheaper when the target uses a
high-contrast saturated-red cursor. The ``xcursor-themes`` package
ships exactly such a theme (``redglass``); this module just returns
the apt/gsettings commands the operator runs ON THE TARGET MACHINE.
We never execute them via HID — that's brittle and not the adapter's
job.
"""

from __future__ import annotations

from handsneyes.platforms.base import CursorThemeAdvice


def advice() -> CursorThemeAdvice:
    return CursorThemeAdvice(
        summary=(
            "Install xcursor-themes and switch to the saturated-red "
            "redglass cursor at size 96. HSV detection works "
            "instantly on this cursor; the motion-variance fallback "
            "still works on the default cursor but is slower."
        ),
        shell_commands=(
            "sudo apt install -y xcursor-themes",
            "gsettings set org.gnome.desktop.interface "
            "cursor-theme 'redglass'",
            "gsettings set org.gnome.desktop.interface cursor-size 96",
        ),
        manual_steps=(
            "Log out and log back in (or open a fresh app window) for "
            "the theme change to take effect.",
        ),
        verify_question=(
            "Look at the cursor on screen. Is it a saturated red, "
            "much larger than a default cursor, easy to spot at a "
            "glance?"
        ),
    )
