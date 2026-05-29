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
    vision_model: str = "nvidia/nemotron-3-nano-omni",
    runtime_state: dict | None = None,
) -> Callable[[], Awaitable[tuple[Any, Any, Any, Any]]]:
    """Build a runner-compatible factory from a :class:`Target` + adapter.

    This is the handsneyes-native entry point — the older
    ``make_default_context_factory`` (kept below for back-compat with
    the verbatim cc port) consumes a terminaleyes-style settings
    object. Use this one in new code.

    ``runtime_state`` is a dict shared with create_app for cross-cutting
    toggles (e.g. ``use_self_capture``) that the UI can flip without
    a cc restart. The factory consults it at each invocation, so a
    checkbox flip takes effect on the next /api/run or /api/mouse/*.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    # Default to an empty dict so call sites that don't supply one
    # still work. Falsy values mean "use the target's configured source".
    runtime_state = runtime_state if runtime_state is not None else {}

    async def factory():
        from openai import AsyncOpenAI

        from handsneyes.core.agents.context import AgentContext
        from handsneyes.core.capture.screen import ScreenCapture
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
        # UI override beats targets.toml: if the operator toggled
        # "self capture" in the cc, force ScreenCapture regardless of
        # what the active target says.
        effective_source = target.capture_source
        if runtime_state.get("use_self_capture"):
            effective_source = "screen"
        try:
            import asyncio as _asyncio
            if effective_source == "screen":
                # Self-driving setup: cc + target are the same box.
                # Pillow grabs the local display directly — no webcam,
                # no calibration. camera_index doubles as the display
                # index here (0 = primary).
                capture = ScreenCapture(display_index=target.camera_index)
            else:
                capture = WebcamCapture(device_index=target.camera_index)
            # Hard cap: cv2.VideoCapture(N) on macOS can block forever
            # when N points at a busy / phantom device. 10s is generous
            # for a real USB webcam (the warmup loop also runs here).
            # ScreenCapture.open() is a single grab — well under 10s.
            await _asyncio.wait_for(capture.open(), timeout=10.0)
        except _asyncio.TimeoutError:
            logger.warning(
                "capture open exceeded 10s (source=%s, index=%d) — running blind",
                effective_source, target.camera_index,
            )
            capture = None
        except Exception as e:
            logger.warning("capture open failed (%s) — running blind", e)
            capture = None

        client = AsyncOpenAI(
            base_url=vision_base_url, api_key="not-needed",
        )

        # Pre-build the Vault when a passphrase is available in the
        # environment. The cc runs in a background asyncio task with
        # no TTY, so getpass.getpass would block forever if LoginAgent
        # tried to resolve a vault name without a pre-instantiated
        # vault on ctx.
        import os as _os
        vault = None
        if _os.environ.get("HANDSNEYES_VAULT_PASSPHRASE") is not None:
            try:
                from handsneyes.core.vault import Vault, get_passphrase
                vault = Vault(get_passphrase())
            except Exception as e:
                logger.warning(
                    "vault init failed (%s) — Unlock will fail "
                    "until HANDSNEYES_VAULT_PASSPHRASE is set + "
                    "`handsneyes vault add <name>` has been run",
                    e,
                )

        # On macOS self-capture the cursor isn't in the framebuffer
        # the capture reads. The visual-servo loop would never converge.
        # Attach a Quartz-based cursor oracle so the homer has an
        # authoritative source.
        cursor_reader = None
        if effective_source == "screen" and adapter.name == "macos":
            try:
                from handsneyes.platforms.macos.cursor_reader import (
                    QuartzCursorReader,
                )
                cursor_reader = QuartzCursorReader()
            except Exception as e:
                logger.warning(
                    "QuartzCursorReader unavailable (%s) — homer will "
                    "fail on macOS self-capture until pyobjc-framework-"
                    "Quartz is installed",
                    e,
                )

        ctx = AgentContext(
            mouse=mouse,
            keyboard=keyboard,
            capture=capture,
            cursor_reader=cursor_reader,
            vision_client=client,
            vision_model=vision_model,
            ocr_model=vision_model,
            vault=vault,
            output_dir=output_dir,
            platform=adapter,
        )
        # Snapshot UI-controlled tuning knobs into ctx.scratch so the
        # homer can pick them up at run time without coupling to the
        # FastAPI app state. Live changes to these from the UI take
        # effect on the NEXT run, not in-flight.
        ctx.scratch["pointer_accel_scale_x"] = float(
            runtime_state.get("pointer_accel_scale_x", 1.0),
        )
        ctx.scratch["pointer_accel_scale_y"] = float(
            runtime_state.get("pointer_accel_scale_y", 1.0),
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
