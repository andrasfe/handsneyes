"""Shared resources injected into every agent."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from handsneyes.core.capture.webcam import WebcamCapture
    from handsneyes.core.vault import Vault
    from handsneyes.io.keyboard.base import KeyboardOutput
    from handsneyes.io.mouse.base import MouseOutput
    from handsneyes.platforms.base import PlatformAdapter

logger = logging.getLogger(__name__)


_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class AgentContext:
    """Bag of shared infrastructure passed to every agent.

    Most fields are optional so agents that don't need them (e.g. the
    Vault used standalone in a CLI) can construct a minimal context.
    Construct with whatever you have; agents document which fields
    they require.
    """

    # I/O — Pi-side HID
    mouse: MouseOutput | None = None
    keyboard: KeyboardOutput | None = None

    # Per-target OS adapter. Holds OS-specific knowledge (shortcut
    # remap, app launcher hotkeys, browser-focus recovery, per-OS
    # model checkpoint locations). Optional so tests can build bare
    # contexts; agents that need it document the requirement.
    platform: PlatformAdapter | None = None

    # Vision input
    capture: WebcamCapture | None = None

    # LLM clients — OpenAI-compatible (LM Studio) for multimodal calls
    vision_client: Any = None
    vision_model: str = ""
    # OCR-specialised model served by the same vision_client. When
    # set, OcrAgent uses this model name instead of vision_model so
    # general-purpose Q&A and pure OCR can use different specialists.
    ocr_model: str = ""

    # ShowUI grounding helper (callable: b64, prompt -> (x, y) | None)
    showui_query: Any = None

    # Storage
    vault: Vault | None = None

    # Per-session output directory. When set, every agent that
    # captures a frame should save the raw image into this folder
    # via :meth:`record_frame`. Sequential filename prefixes make the
    # capture order easy to inspect after a run.
    output_dir: Path | None = None

    # Internal counter for sequential frame filenames. Mutated by
    # :meth:`record_frame`; not part of the public API.
    _frame_counter: int = 0

    # Free-form scratchpad for cross-agent state during a run.
    scratch: dict[str, Any] = field(default_factory=dict)

    # Optional run identifier used in step-log rows. Set per-run by
    # the controller so each row in ``steps.jsonl`` is joinable to
    # the cc RunRecord and to the journal entries.
    run_id: str | None = None

    # Internal step counter for the JSONL step log. Mutated by
    # :meth:`record_step`; not part of the public API.
    _step_counter: int = 0

    # ───────────────────── frame recording ─────────────────────

    def record_frame(
        self,
        image: np.ndarray,
        *,
        label: str = "frame",
        ext: str = "png",
    ) -> Path | None:
        """Save a captured frame to ``output_dir`` if one is configured.

        Returns the path written, or ``None`` when no ``output_dir``
        was set (a no-op so agents can record unconditionally).

        Filename shape: ``{seq:04d}_{HHMMSS}_{label}.{ext}``. Labels
        are sanitised to safe filename chars.
        """
        if self.output_dir is None or image is None:
            return None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Could not create output_dir %s: %s",
                self.output_dir,
                e,
            )
            return None
        self._frame_counter += 1
        seq = f"{self._frame_counter:04d}"
        ts = datetime.now().strftime("%H%M%S")
        safe = _LABEL_SAFE.sub("_", label)[:80] or "frame"
        path = self.output_dir / f"{seq}_{ts}_{safe}.{ext}"
        try:
            ok = cv2.imwrite(str(path), image)
            if not ok:
                logger.warning("cv2.imwrite returned False for %s", path)
                return None
        except Exception as e:
            logger.warning("Failed to write frame to %s: %s", path, e)
            return None
        return path

    async def capture_and_record(
        self,
        *,
        label: str = "frame",
    ) -> np.ndarray | None:
        """Convenience: capture a frame and record it in one call."""
        if self.capture is None:
            return None
        try:
            frame = await self.capture.capture_frame()
        except Exception as e:
            logger.warning("capture failed for %r: %s", label, e)
            return None
        self.record_frame(frame.image, label=label)
        return frame.image

    # ───────────────────── step recording (ML dataset) ─────────────────────

    def _latest_frame_seq(self) -> int | None:
        """Return the current frame counter, or ``None`` if no frames yet."""
        return self._frame_counter if self._frame_counter > 0 else None

    def record_step(
        self,
        *,
        intent: str,
        agent_name: str,
        kwargs: dict[str, Any] | None,
        outcome_success: bool,
        outcome_reason: str,
        history: list[dict[str, Any]] | None = None,
        frame_before_seq: int | None = None,
        frame_after_seq: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """Append one row to ``<output_dir>/steps.jsonl``.

        Describes the decision the controller just made: what intent
        was being served, which agent was called with which kwargs,
        what frame sequence numbers were on disk before and after,
        and the outcome.

        Returns the file path written, or ``None`` when no
        ``output_dir`` is configured. Errors are swallowed (logged at
        debug) — step recording must never break a live run.
        """
        if self.output_dir is None:
            return None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.debug(
                "steps.jsonl: cannot mkdir %s: %s",
                self.output_dir,
                e,
            )
            return None
        self._step_counter += 1
        row = {
            "run_id": self.run_id,
            "step_idx": self._step_counter,
            "ts": time.time(),
            "intent": intent,
            "agent": agent_name,
            "kwargs": kwargs or {},
            "history": history or [],
            "frame_before_seq": frame_before_seq,
            "frame_after_seq": frame_after_seq,
            "outcome": {
                "success": bool(outcome_success),
                "reason": outcome_reason,
            },
        }
        if extra:
            row["extra"] = extra
        path = self.output_dir / "steps.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.debug("steps.jsonl append failed: %s", e)
            return None
        return path

    def subdir(self, name: str) -> Path | None:
        """Return a subdirectory of ``output_dir`` (creating it lazily)."""
        if self.output_dir is None:
            return None
        safe = _LABEL_SAFE.sub("_", name) or "sub"
        path = self.output_dir / safe
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create subdir %s: %s", path, e)
            return None
        return path
