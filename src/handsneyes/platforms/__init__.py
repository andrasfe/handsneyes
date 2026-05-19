"""Platform adapter registry.

Resolves the short name attached to a ``Target`` (e.g. ``"linux_gnome"``,
``"macos"``, ``"headless"``) to a concrete :class:`PlatformAdapter`
subclass via the ``handsneyes.platforms`` entry-point group.

Built-in adapters are registered in this project's ``pyproject.toml``.
Third-party packages may register their own adapters under the same
group with no core changes.

Lookup precedence:

1. ``HANDSNEYES_PLATFORM`` env var (debug override — overrides whatever
   the caller asked for).
2. The ``name`` argument.

If no adapter is registered under the resolved name,
:class:`UnknownPlatformError` is raised — caught early at controller
startup, *before* the webcam opens, so the user is not staring at a
black frame waiting for a HID error.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from handsneyes.platforms.base import PlatformAdapter

_GROUP = "handsneyes.platforms"
_ENV_OVERRIDE = "HANDSNEYES_PLATFORM"

logger = logging.getLogger(__name__)


class UnknownPlatformError(LookupError):
    """Raised when a requested platform name has no registered adapter."""


def available_platforms() -> list[str]:
    """Return sorted names of every adapter registered under the group."""
    return sorted({ep.name for ep in entry_points(group=_GROUP)})


def load_adapter(name: str) -> PlatformAdapter:
    """Construct and return the :class:`PlatformAdapter` for ``name``.

    Each call instantiates a fresh adapter — adapters are stateless w.r.t.
    a run, so this is cheap, and it avoids action-at-a-distance from
    cached instances when targets switch.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    resolved = override or name
    if override and override != name:
        logger.warning(
            "Platform override via %s: requested %r, using %r",
            _ENV_OVERRIDE,
            name,
            override,
        )

    eps = [ep for ep in entry_points(group=_GROUP) if ep.name == resolved]
    if not eps:
        available = ", ".join(available_platforms()) or "(none registered)"
        raise UnknownPlatformError(
            f"No platform adapter registered under name {resolved!r}. "
            f"Available: {available}. "
            f"See docs/porting-to-new-os.md to add one."
        )
    cls = eps[0].load()
    return cls()


__all__ = [
    "UnknownPlatformError",
    "available_platforms",
    "load_adapter",
]
