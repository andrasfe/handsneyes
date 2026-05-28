"""Target registry — multi-host support.

A *target* is a named bundle of ``(camera_index, pi_url, transport,
platform, screen_size)``. The CLI and the future Command Center UI
resolve a user-friendly name (``--target couch-ubuntu``) to a
:class:`Target` via :class:`TargetRegistry`, which in turn picks the
right platform adapter, output dir, and HID/capture backends.

Phase B scope: declarative TOML loader + dataclass. Phase C wires
the registry into runtime construction of AgentContext for the UI.

Configuration sources (priority):

  1. ``HANDSNEYES_TARGETS_FILE`` env var (absolute path to a TOML file)
  2. ``./config/targets.toml`` in the current working directory
  3. ``~/.config/handsneyes/targets.toml``
  4. A built-in default with a single "headless" target so the CLI
     works out of the box without any configuration.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Transport = Literal["bt", "usb"]


_DEFAULT_CONFIG_PATHS = (
    Path("config/targets.toml"),
    Path.home() / ".config" / "handsneyes" / "targets.toml",
)


@dataclass(frozen=True)
class Target:
    """One configured target host."""

    name: str
    platform: str = "headless"
    camera_index: int = 0
    pi_url: str = "http://10.0.0.2:8080"
    transport: Transport = "bt"
    screen_size: tuple[int, int] = (1920, 1080)
    description: str = ""
    # "webcam" (default): cv2.VideoCapture(camera_index) — for a
    # remote machine the dev mac watches.
    # "screen": grab the local display directly via Pillow's
    #   ImageGrab. Used when the target IS the same machine running
    #   the cc (self-driving setup). camera_index is then the
    #   display index (0 = primary).
    capture_source: str = "webcam"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target name must be non-empty")
        if self.transport not in ("bt", "usb"):
            raise ValueError(
                f"Target {self.name!r}: transport must be 'bt' or "
                f"'usb', got {self.transport!r}"
            )
        if self.capture_source not in ("webcam", "screen"):
            raise ValueError(
                f"Target {self.name!r}: capture_source must be "
                f"'webcam' or 'screen', got {self.capture_source!r}"
            )


_DEFAULT_TARGET = Target(
    name="headless",
    platform="headless",
    description=(
        "Default fallback. No HID, no webcam — used by --dry-run and "
        "by tests so the CLI works out of the box without a targets "
        "file."
    ),
)


@dataclass
class TargetRegistry:
    """In-memory registry of configured targets."""

    targets: dict[str, Target] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> TargetRegistry:
        if not path.exists():
            raise FileNotFoundError(f"Targets file not found: {path}")
        data = tomllib.loads(path.read_text("utf-8"))
        rows = data.get("target") or []
        if not isinstance(rows, list):
            raise ValueError(
                f"{path}: top-level [[target]] must be an array of "
                f"tables, got {type(rows).__name__}"
            )
        targets: dict[str, Target] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}: each [[target]] entry must be a table"
                )
            name = str(row.get("name", "")).strip()
            if not name:
                raise ValueError(
                    f"{path}: [[target]] entry missing 'name'"
                )
            if name in targets:
                raise ValueError(
                    f"{path}: duplicate target name {name!r}"
                )
            screen = row.get("screen_size", [1920, 1080])
            if not (isinstance(screen, list) and len(screen) == 2):
                raise ValueError(
                    f"{path}: target {name!r}: screen_size must be "
                    f"[width, height]"
                )
            targets[name] = Target(
                name=name,
                platform=str(row.get("platform", "headless")),
                camera_index=int(row.get("camera_index", 0)),
                pi_url=str(row.get("pi_url", "http://10.0.0.2:8080")),
                transport=str(row.get("transport", "bt")),  # type: ignore[arg-type]
                screen_size=(int(screen[0]), int(screen[1])),
                description=str(row.get("description", "")),
                capture_source=str(row.get("capture_source", "webcam")),
            )
        return cls(targets=targets, source=path)

    @classmethod
    def load_default(cls) -> TargetRegistry:
        """Try the standard config paths, then fall back to the
        single built-in headless target.
        """
        env_path = os.environ.get("HANDSNEYES_TARGETS_FILE")
        candidates: list[Path] = []
        if env_path:
            candidates.append(Path(env_path).expanduser())
        candidates.extend(_DEFAULT_CONFIG_PATHS)
        for p in candidates:
            if p.exists():
                return cls.from_file(p)
        return cls(targets={_DEFAULT_TARGET.name: _DEFAULT_TARGET})

    def names(self) -> list[str]:
        return sorted(self.targets.keys())

    def get(self, name: str) -> Target:
        if name not in self.targets:
            available = ", ".join(self.names()) or "(none)"
            raise KeyError(
                f"Unknown target {name!r}. Available: {available}"
            )
        return self.targets[name]


__all__ = ["Target", "TargetRegistry", "Transport"]
