# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/raspi; lint cleanup deferred.
"""REST API server that runs on the Raspberry Pi.

Accepts keyboard and mouse commands over HTTP and translates them into
USB HID reports via HidWriter/MouseHidWriter. This is the bridge between
the handsneyes agent (running on another machine) and the target machine
the Pi is plugged into.

USB HID endpoints (keyboard via /dev/hidg0, mouse via /dev/hidg1):

    GET  /health          -> {"status": "ok", ...}
    POST /keystroke       <- {"key": "Enter"}
    POST /key-combo       <- {"modifiers": ["ctrl"], "key": "c"}
    POST /text            <- {"text": "ls -la"}
    POST /mouse/move      <- {"x": 10, "y": -5}
    POST /mouse/click     <- {"button": "left"}
    POST /mouse/scroll    <- {"amount": -3}

Bluetooth HID endpoints (keyboard + mouse, optional):

    POST /bt/keystroke  <- {"key": "Enter"}
    POST /bt/key-combo  <- {"modifiers": ["ctrl"], "key": "c"}
    POST /bt/text       <- {"text": "hello"}
    POST /bt/mouse/move   <- {"x": 10, "y": -5}
    POST /bt/mouse/click  <- {"button": "left"}
    POST /bt/mouse/scroll <- {"amount": -3}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from handsneyes.pi.hid_writer import HidWriteError, HidWriter, MouseHidWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class KeystrokeRequest(BaseModel):
    key: str = Field(description="Key name (e.g., 'Enter', 'Tab', 'a')")


class KeyComboRequest(BaseModel):
    modifiers: list[str] = Field(description="Modifier keys (e.g., ['ctrl'])")
    key: str = Field(description="Main key in the combination")


class TextInputRequest(BaseModel):
    text: str = Field(description="Text to type")
    warmup: bool = Field(
        default=True,
        description=(
            "If True (default), send_text uses a double-tap-with-"
            "backspace warmup for the first character to overcome "
            "the receiver dropping the first keypress after idle. "
            "Set False for input contexts where Backspace is bound "
            "to non-deletion (e.g. some browser URL bars where it "
            "triggers back-navigation, producing a doubled first "
            "character)."
        ),
    )


class MouseMoveRequest(BaseModel):
    x: int = Field(description="Relative X movement (-127 to 127)")
    y: int = Field(description="Relative Y movement (-127 to 127)")


class MouseClickRequest(BaseModel):
    button: str = Field(default="left", description="Button: left, right, middle")


class MouseScrollRequest(BaseModel):
    amount: int = Field(description="Scroll amount (-127 to 127, positive=up)")


class HealthResponse(BaseModel):
    status: str = "ok"
    hid_device: str = "/dev/hidg0"
    hid_open: bool = False
    mouse_hid_device: str = "/dev/hidg1"
    mouse_hid_open: bool = False
    bt_hid_connected: bool = False


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    hid_device: str = "/dev/hidg0",
    mouse_hid_device: str = "/dev/hidg1",
    keypress_delay: float = 0.02,
    inter_char_delay: float = 0.01,
    writer: HidWriter | None = None,
    mouse_writer: MouseHidWriter | None = None,
    bt_hid: object | None = None,
    enable_bt_hid: bool = True,
) -> FastAPI:
    """Create the Pi REST API application.

    Args:
        hid_device: Path to the keyboard HID gadget device.
        mouse_hid_device: Path to the mouse HID gadget device.
        keypress_delay: Seconds between key press and release.
        inter_char_delay: Seconds between characters when typing text.
        writer: Optional pre-configured HidWriter (for testing).
        mouse_writer: Optional pre-configured MouseHidWriter (for testing).
        bt_hid: Optional pre-configured BluetoothHidServer (for testing).
        enable_bt_hid: Whether to try initializing BT HID on startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Keyboard
        w = app.state.writer
        if w is None:
            w = HidWriter(
                device_path=hid_device,
                keypress_delay=keypress_delay,
                inter_char_delay=inter_char_delay,
            )
            app.state.writer = w
        try:
            await w.open()
            logger.info("Pi keyboard server started (device=%s)", hid_device)
        except HidWriteError:
            logger.warning(
                "USB HID device %s not available — USB keyboard endpoints will return errors. "
                "Connect USB cable and restart, or use Bluetooth endpoints.",
                hid_device,
            )

        # Mouse
        mw = app.state.mouse_writer
        if mw is None:
            mw = MouseHidWriter(device_path=mouse_hid_device)
            app.state.mouse_writer = mw
        try:
            await mw.open()
            logger.info("Pi mouse server started (device=%s)", mouse_hid_device)
        except HidWriteError:
            logger.warning(
                "USB mouse HID device %s not available — USB mouse endpoints will return errors.",
                mouse_hid_device,
            )

        # Try to initialize Bluetooth HID (non-fatal if it fails)
        bt_accept_task = None
        if app.state.bt_hid is None and enable_bt_hid:
            try:
                from handsneyes.pi.bt_hid import (
                    BluetoothHidServer,
                    configure_bluetooth_adapter,
                    register_sdp_profile,
                )
                configure_bluetooth_adapter()
                register_sdp_profile()
                bt = BluetoothHidServer()
                await bt.start()
                app.state.bt_hid = bt
                logger.info("Bluetooth HID server started (keyboard + mouse)")
                logger.info("Waiting for Bluetooth pairing in background...")

                # Accept connections in background so the REST API stays responsive
                async def _accept_bt_loop() -> None:
                    while True:
                        try:
                            addr = await bt.wait_for_connection()
                            logger.info("Bluetooth device connected: %s", addr)
                        except Exception as e:
                            logger.warning("Bluetooth accept error: %s", e)
                            break

                import asyncio
                bt_accept_task = asyncio.create_task(_accept_bt_loop())

            except Exception as exc:
                logger.info("Bluetooth HID not available: %s", exc)
                app.state.bt_hid = None

        yield

        # Cancel BT accept loop
        if bt_accept_task is not None:
            bt_accept_task.cancel()

        # Cleanup
        if app.state.bt_hid is not None:
            try:
                await app.state.bt_hid.stop()
            except Exception:
                pass
        await mw.close()
        await w.close()
        logger.info("Pi keyboard+mouse server stopped")

    app = FastAPI(
        title="handsneyes Pi HID",
        description="Raspberry Pi USB HID keyboard + Bluetooth keyboard/mouse REST API",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.state.writer = writer
    app.state.mouse_writer = mouse_writer
    app.state.bt_hid = bt_hid

    @app.get("/health")
    async def health_check() -> HealthResponse:
        w: HidWriter = app.state.writer
        mw: MouseHidWriter = app.state.mouse_writer
        bt = app.state.bt_hid
        return HealthResponse(
            status="ok",
            hid_device=hid_device,
            hid_open=w.is_open if w else False,
            mouse_hid_device=mouse_hid_device,
            mouse_hid_open=mw.is_open if mw else False,
            bt_hid_connected=bt.is_connected if bt else False,
        )

    # -------------------------------------------------------------------
    # USB HID keyboard endpoints
    # -------------------------------------------------------------------

    @app.post("/keystroke")
    async def receive_keystroke(request: KeystrokeRequest) -> dict[str, str]:
        w: HidWriter = app.state.writer
        try:
            await w.send_keystroke(request.key)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "key": request.key}

    @app.post("/key-combo")
    async def receive_key_combo(request: KeyComboRequest) -> dict[str, str]:
        w: HidWriter = app.state.writer
        try:
            await w.send_key_combo(request.modifiers, request.key)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "combo": f"{'+'.join(request.modifiers)}+{request.key}"}

    @app.post("/text")
    async def receive_text(request: TextInputRequest) -> dict[str, str]:
        w: HidWriter = app.state.writer
        try:
            await w.send_text(request.text)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "length": str(len(request.text))}

    # -------------------------------------------------------------------
    # USB HID mouse endpoints
    # -------------------------------------------------------------------

    @app.post("/mouse/move")
    async def mouse_move(request: MouseMoveRequest) -> dict[str, str]:
        mw: MouseHidWriter = app.state.mouse_writer
        try:
            await mw.move(request.x, request.y)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "x": str(request.x), "y": str(request.y)}

    @app.post("/mouse/click")
    async def mouse_click(request: MouseClickRequest) -> dict[str, str]:
        mw: MouseHidWriter = app.state.mouse_writer
        try:
            await mw.click(request.button)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "button": request.button}

    @app.post("/mouse/scroll")
    async def mouse_scroll(request: MouseScrollRequest) -> dict[str, str]:
        mw: MouseHidWriter = app.state.mouse_writer
        try:
            await mw.scroll(request.amount)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "amount": str(request.amount)}

    # -------------------------------------------------------------------
    # Bluetooth HID helper
    # -------------------------------------------------------------------

    def _get_bt():  # type: ignore[no-untyped-def]
        bt = app.state.bt_hid
        if bt is None:
            raise HTTPException(
                status_code=503,
                detail="Bluetooth HID not initialized. Run setup_bt_hid.sh first.",
            )
        return bt

    # -------------------------------------------------------------------
    # Bluetooth keyboard endpoints
    # -------------------------------------------------------------------

    @app.post("/bt/keystroke")
    async def bt_keystroke(request: KeystrokeRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.send_keystroke(request.key)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "key": request.key, "transport": "bluetooth"}

    @app.post("/bt/key-combo")
    async def bt_key_combo(request: KeyComboRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.send_key_combo(request.modifiers, request.key)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "status": "ok",
            "combo": f"{'+'.join(request.modifiers)}+{request.key}",
            "transport": "bluetooth",
        }

    @app.post("/bt/text")
    async def bt_text(request: TextInputRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.send_text(request.text, warmup=request.warmup)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "length": str(len(request.text)), "transport": "bluetooth"}

    # -------------------------------------------------------------------
    # Bluetooth mouse endpoints
    # -------------------------------------------------------------------

    @app.post("/bt/mouse/move")
    async def bt_mouse_move(request: MouseMoveRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.move(request.x, request.y)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "x": str(request.x), "y": str(request.y)}

    @app.post("/bt/mouse/click")
    async def bt_mouse_click(request: MouseClickRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.click(request.button)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "button": request.button}

    @app.post("/bt/mouse/press")
    async def bt_mouse_press(request: MouseClickRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.press(request.button)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "button": request.button, "state": "pressed"}

    @app.post("/bt/mouse/release")
    async def bt_mouse_release(request: MouseClickRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.release(request.button)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "button": request.button, "state": "released"}

    @app.post("/bt/mouse/scroll")
    async def bt_mouse_scroll(request: MouseScrollRequest) -> dict[str, str]:
        bt = _get_bt()
        try:
            await bt.scroll(request.amount)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "amount": str(request.amount)}

    return app


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    hid_device: str = "/dev/hidg0",
    mouse_hid_device: str = "/dev/hidg1",
) -> None:
    """Run the Pi keyboard+mouse server."""
    app = create_app(hid_device=hid_device, mouse_hid_device=mouse_hid_device)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
