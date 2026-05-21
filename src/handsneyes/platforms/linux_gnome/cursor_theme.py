"""Declarative cursor-theme advice for GNOME targets.

Default Yaru white cursor at size 24 works out of the box —
oscillation-variance (wiggle) + frame-diff (shape) detection have
been validated end-to-end against the shipped pointer_accel model
trained on Yaru-cursor data. The HSV finder is a fast-path
optimisation that lights up only if the operator chooses to install
the saturated-red ``redglass`` theme; it cuts per-click steps from
~9 to ~3 but is not required.
"""

from __future__ import annotations

from handsneyes.platforms.base import CursorThemeAdvice


def advice() -> CursorThemeAdvice:
    return CursorThemeAdvice(
        summary=(
            "Default Yaru cursor at size 24 works as-is — wiggle + "
            "shape detection succeed on it. Optionally install the "
            "saturated-red redglass theme at size 96 if you want HSV "
            "fast-path detection (~3x fewer per-click iterations)."
        ),
        shell_commands=(
            "# Optional fast-path (HSV detection on red cursor):",
            "sudo apt install -y xcursor-themes",
            "gsettings set org.gnome.desktop.interface "
            "cursor-theme 'redglass'",
            "gsettings set org.gnome.desktop.interface cursor-size 96",
            "",
            "# Revert to default Yaru:",
            "gsettings reset org.gnome.desktop.interface cursor-theme",
            "gsettings reset org.gnome.desktop.interface cursor-size",
        ),
        manual_steps=(
            "Log out and log back in (or open a fresh app window) for "
            "the theme change to take effect on running apps.",
        ),
        verify_question=(
            "Look at the cursor on screen. If aiming for the HSV "
            "fast-path: is it a saturated red, much larger than a "
            "default cursor, easy to spot at a glance? For Yaru "
            "default: a small white arrow is fine."
        ),
    )
