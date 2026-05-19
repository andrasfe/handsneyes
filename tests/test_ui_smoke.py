"""Smoke tests for the handsneyes.ui Command Center.

Phase C scope: confirm the FastAPI app boots, serves the SPA, and
exposes the documented endpoints. The deep behaviour of the
commandcenter port (frame store polling, log bus SSE, runner
lifecycle) is empirically validated against the live UI — these
tests pin down the wiring.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used as fixture annotation

import pytest
from fastapi.testclient import TestClient

from handsneyes.platforms import load_adapter
from handsneyes.targets import Target
from handsneyes.ui.factory import make_target_context_factory
from handsneyes.ui.frame_store import FrameStore
from handsneyes.ui.log_bus import LogBus
from handsneyes.ui.server import create_app


@pytest.fixture
def app_client(tmp_path: Path):  # noqa: ANN001, ANN201
    """Build a fresh app + TestClient over a tmp watch dir."""
    target = Target(name="test-headless", platform="headless")
    adapter = load_adapter("headless")
    store = FrameStore(watch_dir=tmp_path, max_frames=50)
    bus = LogBus()
    factory = make_target_context_factory(
        target, adapter, base_dir=tmp_path, bus=bus,
    )
    app = create_app(factory, frame_store=store, bus=bus)
    with TestClient(app) as client:
        yield client


class TestAppBoot:
    def test_app_serves_spa(self, app_client) -> None:  # noqa: ANN001
        r = app_client.get("/")
        assert r.status_code == 200
        # SPA shell: HTML with a "Command Center" or similar identifier
        assert "html" in r.text.lower() or "<!doctype" in r.text.lower()

    def test_state_endpoint_returns_json(self, app_client) -> None:  # noqa: ANN001
        r = app_client.get("/api/state")
        assert r.status_code == 200
        payload = r.json()
        assert "busy" in payload or "frame_count" in payload

    def test_frames_index_empty(self, app_client) -> None:  # noqa: ANN001
        r = app_client.get("/api/frames")
        assert r.status_code == 200
        # The empty store returns an empty list or a wrapped object.
        body = r.json()
        assert isinstance(body, (list, dict))

    def test_runs_index_empty(self, app_client) -> None:  # noqa: ANN001
        r = app_client.get("/api/runs")
        assert r.status_code == 200


class TestFactoryShapes:
    @pytest.mark.asyncio
    async def test_make_target_factory_returns_callable(
        self, tmp_path: Path,
    ) -> None:
        target = Target(name="t", platform="headless")
        adapter = load_adapter("headless")
        bus = LogBus()
        factory = make_target_context_factory(
            target, adapter, base_dir=tmp_path, bus=bus,
        )
        assert callable(factory)

    @pytest.mark.asyncio
    async def test_factory_builds_context(self, tmp_path: Path) -> None:
        target = Target(name="t", platform="headless")
        adapter = load_adapter("headless")
        bus = LogBus()
        factory = make_target_context_factory(
            target, adapter, base_dir=tmp_path, bus=bus,
        )
        # Use AsyncMock-style real-ish components by patching the
        # WebcamCapture so we don't actually try to open a camera.
        # Easiest: just call the factory and accept the webcam may
        # fail open (it'll log a warning and set capture=None on the
        # ctx).
        ctx, kb, mouse, capture = await factory()
        assert ctx.platform is adapter
        assert ctx.output_dir is not None
        # Tear down whatever resources opened. capture may be None on
        # CI without a webcam — that's the documented behaviour.
        if hasattr(kb, "disconnect"):
            await kb.disconnect()
        if hasattr(mouse, "disconnect"):
            await mouse.disconnect()
        if capture is not None:
            await capture.close()
