# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commander; lint cleanup deferred.
"""DEPRECATED — static mouse calibration.

Superseded by the closed-loop, model-driven homer
(``commander.closed_loop_homer``). This module is kept temporarily so that
``MOVE_STEP_SIZE`` and ``MOVE_DELAY`` (still imported by
``InteractiveSession._send_hid_moves``) remain available. Slated for
deletion once the closed-loop path is proven across several targets.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from handsneyes.io.mouse.base import MouseOutput

logger = logging.getLogger(__name__)

# Movement step size — 3 units/report balances speed with staying
# in macOS's linear acceleration zone. At 3ms delay = ~1000 units/sec.
MOVE_STEP_SIZE = 3
MOVE_DELAY = 0.003

# Coarse movement for calibration edge-finding
COARSE_STEP = 3
COARSE_DELAY = 0.003

CALIBRATION_FILE = Path.home() / ".handsneyes" / "calibration.json"


@dataclass
class CalibrationResult:
    """Stores the calibrated HID-to-screen mapping."""

    hid_units_per_full_x: float
    hid_units_per_full_y: float
    step_size: int = MOVE_STEP_SIZE
    move_delay: float = MOVE_DELAY
    calibrated: bool = True

    def hid_units_for_pct(self, dx_pct: float, dy_pct: float) -> tuple[int, int]:
        """Convert a screen-percentage delta to HID units."""
        return (
            int(dx_pct * self.hid_units_per_full_x),
            int(dy_pct * self.hid_units_per_full_y),
        )

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Calibration saved to %s", path)

    @staticmethod
    def load(path: Path = CALIBRATION_FILE) -> CalibrationResult | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return CalibrationResult(
                hid_units_per_full_x=float(data["hid_units_per_full_x"]),
                hid_units_per_full_y=float(data["hid_units_per_full_y"]),
                step_size=int(data.get("step_size", MOVE_STEP_SIZE)),
                move_delay=float(data.get("move_delay", MOVE_DELAY)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load calibration: %s", e)
            return None


DEFAULT_CALIBRATION = CalibrationResult(
    hid_units_per_full_x=1920.0,
    hid_units_per_full_y=1080.0,
    calibrated=False,
)


class MouseCalibrator:
    """Interactive calibration — user positions cursor, we measure HID travel."""

    def __init__(self, mouse: MouseOutput) -> None:
        self._mouse = mouse

    async def calibrate_or_load(self) -> CalibrationResult:
        """Load saved calibration if available, otherwise run full calibration."""
        saved = CalibrationResult.load()

        if saved is not None:
            print(f"\n  Saved calibration found ({saved.hid_units_per_full_x:.0f} x {saved.hid_units_per_full_y:.0f})")
            answer = await self._ask("  Use saved calibration? [Y/n]: ")
            if answer.lower() != "n":
                print("  Using saved calibration.\n")
                return saved
            print()

        cal = await self._full_calibration()
        cal.save()
        print(f"  Saved to {CALIBRATION_FILE}\n")
        return cal

    async def force_calibrate(self) -> CalibrationResult:
        """Force a full re-calibration."""
        cal = await self._full_calibration()
        cal.save()
        print(f"  Saved to {CALIBRATION_FILE}\n")
        return cal

    async def _full_calibration(self) -> CalibrationResult:
        """Run interactive calibration."""
        print("  Mouse Calibration")
        print("  You will position the cursor on the TARGET screen.\n")

        # Horizontal
        await self._ask(
            "  On the TARGET screen, move the cursor to the LEFT edge.\n"
            "  Press Enter here when ready: "
        )

        print("  Moving cursor RIGHT... press Enter when it reaches the right edge.")
        hid_x = await self._measure_edge(dx=1, dy=0)
        if hid_x is None or hid_x <= 0:
            print("  Skipped — using defaults.")
            return DEFAULT_CALIBRATION
        print(f"  → {hid_x} HID units = screen width\n")

        # Vertical
        await self._ask(
            "  Now move the cursor to the TOP edge on the TARGET screen.\n"
            "  Press Enter here when ready: "
        )

        print("  Moving cursor DOWN... press Enter when it reaches the bottom edge.")
        hid_y = await self._measure_edge(dx=0, dy=1)
        if hid_y is None or hid_y <= 0:
            hid_y = hid_x
        print(f"  → {hid_y} HID units = screen height\n")

        cal = CalibrationResult(
            hid_units_per_full_x=float(hid_x),
            hid_units_per_full_y=float(hid_y),
        )

        print(f"  Calibration complete:")
        print(f"    {cal.hid_units_per_full_x:.0f} HID units = full width")
        print(f"    {cal.hid_units_per_full_y:.0f} HID units = full height")
        return cal

    async def _measure_edge(self, dx: int, dy: int) -> int | None:
        """Send HID units in one direction until user presses Enter."""
        total_sent = 0
        stop_event = asyncio.Event()

        async def _move_loop():
            nonlocal total_sent
            while not stop_event.is_set():
                await self._mouse.move(dx * COARSE_STEP, dy * COARSE_STEP)
                total_sent += COARSE_STEP
                await asyncio.sleep(COARSE_DELAY)

        async def _wait_for_enter():
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None, lambda: input("  → Press Enter when cursor hits the edge: ")
            )
            stop_event.set()
            if answer.strip().lower() == "skip":
                return "skip"
            return "ok"

        move_task = asyncio.create_task(_move_loop())
        result = await _wait_for_enter()
        stop_event.set()
        await move_task

        if result == "skip":
            return None
        return total_sent

    async def _precise_move(self, dx_total: int, dy_total: int) -> None:
        """Send HID moves one unit at a time with slow delay."""
        rem_x = abs(dx_total)
        rem_y = abs(dy_total)
        sign_x = 1 if dx_total >= 0 else -1
        sign_y = 1 if dy_total >= 0 else -1

        while rem_x > 0 or rem_y > 0:
            sx = min(rem_x, MOVE_STEP_SIZE) * sign_x if rem_x > 0 else 0
            sy = min(rem_y, MOVE_STEP_SIZE) * sign_y if rem_y > 0 else 0
            if sx != 0 or sy != 0:
                await self._mouse.move(sx, sy)
            rem_x -= abs(sx)
            rem_y -= abs(sy)
            await asyncio.sleep(MOVE_DELAY)

    @staticmethod
    async def _ask(prompt: str) -> str:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, lambda: input(prompt))
        return answer.strip() or "y"
