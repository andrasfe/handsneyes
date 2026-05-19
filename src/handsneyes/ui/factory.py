# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commandcenter; lint cleanup deferred.
"""Default :class:`ContextFactory` for the command-center runner.

A factory is invoked at the start of every run to build a fresh
:class:`AgentContext` (mouse + keyboard + capture + vision client +
output dir). The runner closes those resources when the run ends, so
the webcam is held only while a run is in flight — exactly matching
the lifecycle of the ``handsneyes do`` CLI.

Usage from server bootstrap::

    from handsneyes.ui.factory import (
        make_default_context_factory,
    )
    factory = make_default_context_factory(
        settings, base_dir=store.watch_dir, bus=bus,
    )
    app = create_app(factory, frame_store=store, bus=bus)

The per-run output dir is named after the runner's ``run_id`` (read
from the bus's ``current_run_id()``) so the UI can correlate frames
to run records: ``<watch_dir>/<run_id>/0001_..._navigate_check.png``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def make_target_context_factory(
    target,
    adapter,
    *,
    base_dir: Path,
    bus=None,
    vision_base_url: str = "http://localhost:1234/v1",
    vision_model: str = "nemotron-3-nano-omni",
) -> Callable[[], Awaitable[tuple[Any, Any, Any, Any]]]:
    """Build a runner-compatible factory from a :class:`Target` + adapter.

    This is the handsneyes-native entry point — the older
    ``make_default_context_factory`` (kept below for back-compat with
    the verbatim cc port) consumes a terminaleyes-style settings
    object. Use this one in new code.
    """
    base_dir = Path(base_dir).expanduser().resolve()

    async def factory():
        from openai import AsyncOpenAI

        from handsneyes.core.agents.context import AgentContext
        from handsneyes.core.capture.webcam import WebcamCapture
        from handsneyes.io.keyboard import PlatformKeyboard
        from handsneyes.io.keyboard.backends.http import HttpKeyboardOutput
        from handsneyes.io.mouse.backends.http import HttpMouseOutput

        run_id = None
        if bus is not None:
            try:
                run_id = bus.current_run_id()
            except Exception:
                run_id = None
        sub = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = base_dir / sub
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Could not create per-run output dir %s: %s",
                output_dir, e,
            )
            output_dir = None

        raw_kb = HttpKeyboardOutput(
            base_url=target.pi_url, timeout=10.0, transport=target.transport,
        )
        keyboard = PlatformKeyboard(raw_kb, adapter)
        mouse = HttpMouseOutput(
            base_url=target.pi_url, timeout=10.0, transport=target.transport,
        )
        try:
            await raw_kb.connect()
        except Exception as e:
            logger.warning("keyboard connect failed (%s) — running offline", e)
        try:
            await mouse.connect()
        except Exception as e:
            logger.warning("mouse connect failed (%s) — running offline", e)

        capture = None
        try:
            capture = WebcamCapture(device_index=target.camera_index)
            await capture.open()
        except Exception as e:
            logger.warning("webcam open failed (%s) — running blind", e)
            capture = None

        client = AsyncOpenAI(
            base_url=vision_base_url, api_key="not-needed",
        )

        ctx = AgentContext(
            mouse=mouse,
            keyboard=keyboard,
            capture=capture,
            vision_client=client,
            vision_model=vision_model,
            ocr_model=vision_model,
            output_dir=output_dir,
            platform=adapter,
        )
        return ctx, raw_kb, mouse, capture

    return factory


def make_default_context_factory(
    settings,
    *,
    base_dir: Path,
    bus=None,
) -> Callable[[], Awaitable[tuple[Any, Any, Any, Any]]]:
    """Build a runner-compatible ``ContextFactory``.

    ``base_dir`` is the directory under which per-run subdirectories
    are created (same as :class:`FrameStore.watch_dir` so the UI sees
    everything). ``bus`` is the optional :class:`LogBus`; when set,
    the factory uses ``bus.current_run_id()`` to name the per-run
    subdir for clean frame ↔ run correlation in the UI.
    """
    base_dir = Path(base_dir).expanduser().resolve()

    async def factory():
        from openai import AsyncOpenAI
        from handsneyes.core.agents.context import AgentContext
        from handsneyes.core.capture.webcam import WebcamCapture
        from handsneyes.core.vision.session_adapter import ConditionEvaluator
        from handsneyes.io.keyboard.backends.http import HttpKeyboardOutput
        from handsneyes.io.mouse.backends.http import HttpMouseOutput

        cfg = settings.commander

        # Per-run output subdir. Use the bus's current run id when
        # available so the UI can map frame.run_id → runner record.
        run_id = None
        if bus is not None:
            try:
                run_id = bus.current_run_id()
            except Exception:
                run_id = None
        sub = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = base_dir / sub
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Could not create per-run output dir %s: %s",
                output_dir, e,
            )
            output_dir = None

        keyboard = HttpKeyboardOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        mouse = HttpMouseOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        await keyboard.connect()
        await mouse.connect()

        resolution = None
        if (settings.capture.resolution_width
                and settings.capture.resolution_height):
            resolution = (
                settings.capture.resolution_width,
                settings.capture.resolution_height,
            )
        capture = WebcamCapture(
            device_index=settings.capture.device_index,
            resolution=resolution,
        )
        await capture.open()

        client = AsyncOpenAI(
            base_url=cfg.lmstudio_base_url, api_key="not-needed",
        )
        evaluator = ConditionEvaluator(
            model=cfg.lmstudio_model,
            base_url=cfg.lmstudio_base_url,
            max_tokens=cfg.lmstudio_max_tokens,
        )

        ctx = AgentContext(
            mouse=mouse,
            keyboard=keyboard,
            capture=capture,
            vision_client=client,
            vision_model=cfg.lmstudio_model,
            ocr_model=cfg.lmstudio_ocr_model,
            evaluator=evaluator,
            output_dir=output_dir,
        )
        return ctx, keyboard, mouse, capture

    return factory
