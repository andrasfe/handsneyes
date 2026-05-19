"""APP_ALIASES — short names → AppHint for the linux_gnome target.

Ported verbatim from terminaleyes/agents/launch.py. Maps free-form
user names ("shell", "the terminal", "browser") to a canonical
``AppHint`` carrying the string we type into the GNOME Activities
launcher plus the OCR substrings to expect in the top bar / window
title once the app is focused.
"""

from __future__ import annotations

import re

from handsneyes.platforms.base import AppHint

APP_ALIASES: dict[str, AppHint] = {
    "terminal":               AppHint("terminal", ("terminal",)),
    "the terminal":           AppHint("terminal", ("terminal",)),
    "a terminal":             AppHint("terminal", ("terminal",)),
    "shell":                  AppHint("terminal", ("terminal",)),
    "gnome-terminal":         AppHint("gnome-terminal", ("terminal",)),
    "files":                  AppHint("files", ("files", "nautilus")),
    "the files":              AppHint("files", ("files", "nautilus")),
    "file manager":           AppHint("files", ("files", "nautilus")),
    "nautilus":               AppHint("nautilus", ("files", "nautilus")),
    "calculator":             AppHint("calculator", ("calculator",)),
    "calc":                   AppHint("calculator", ("calculator",)),
    "settings":               AppHint("settings", ("settings",)),
    "system settings":        AppHint("settings", ("settings",)),
    "text editor":            AppHint("text editor", ("text editor", "gedit")),
    "gedit":                  AppHint("gedit", ("gedit", "text editor")),
    "firefox":                AppHint("firefox", ("firefox",)),
    "the firefox":            AppHint("firefox", ("firefox",)),
    "chrome":                 AppHint("google-chrome", ("chrome", "chromium")),
    "google chrome":          AppHint("google-chrome", ("chrome", "chromium")),
    "chromium":               AppHint("chromium", ("chromium", "chrome")),
    "browser":                AppHint("firefox", ("firefox", "chrome")),
    "web browser":            AppHint("firefox", ("firefox", "chrome")),
    "libreoffice":            AppHint("libreoffice", ("libreoffice",)),
    "libreoffice writer":     AppHint("libreoffice writer",
                                      ("writer", "libreoffice")),
    "libreoffice-writer":     AppHint("libreoffice writer",
                                      ("writer", "libreoffice")),
    "writer":                 AppHint("libreoffice writer",
                                      ("writer", "libreoffice")),
    "libreoffice calc":       AppHint("libreoffice calc",
                                      ("calc", "libreoffice")),
    "libreoffice-calc":       AppHint("libreoffice calc",
                                      ("calc", "libreoffice")),
    "libreoffice impress":    AppHint("libreoffice impress",
                                      ("impress", "libreoffice")),
    "libreoffice-impress":    AppHint("libreoffice impress",
                                      ("impress", "libreoffice")),
}


def canonicalise(alias: str) -> AppHint:
    """Look up an alias; fall back to a passthrough hint."""
    key = alias.strip().lower()
    if key in APP_ALIASES:
        return APP_ALIASES[key]
    # Unknown name: type it as-is, expect the same word in the top bar.
    # Strip leading articles ("the ", "a ", "an ") for the expected match.
    base = re.sub(r"^(?:the|a|an)\s+", "", key)
    return AppHint(canonical=base, expect_substrings=(base,))
