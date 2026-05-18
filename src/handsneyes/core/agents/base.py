"""Agent base classes and the typed Outcome return value."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from handsneyes.core.agents.context import AgentContext


@dataclass
class Outcome:
    """Typed return value for any agent run.

    ``success`` is the only required field. ``reason`` is a short human
    string for logs. ``data`` is a free-form dict for agent-specific
    structured detail (image paths, measurements, sub-outcomes).
    """

    success: bool
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class Agent(ABC):
    """Base class for all agents.

    Subclasses implement :meth:`run`. They access shared resources
    (capture, mouse, keyboard, vision client, vault) via ``self.ctx``.
    Higher-tier agents may instantiate lower-tier ones with the same
    ``ctx`` to compose behaviour.
    """

    name: str = ""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    @abstractmethod
    async def run(self, **kwargs: Any) -> Outcome:  # noqa: ANN401
        """Perform the agent's job. Must return an :class:`Outcome`."""
