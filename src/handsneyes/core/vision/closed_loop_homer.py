# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commander; lint cleanup deferred.
"""Closed-loop, model-driven cursor homing.

No static calibration, no fragile frame-diff cursor detection. The
algorithm trusts two facts:

1. Slamming many ``(-20, -20)`` HID moves places the cursor at the
   top-left of the screen. After the slam, the cursor's screen-fraction
   position is ``(0, 0)``.
2. ShowUI reliably grounds the target every step. The target is a
   fixed UI element; its screen-fraction is approximately constant
   across iterations (small jitter due to camera noise).

Each step we:

* Ask ShowUI for the target's current ``(tx, ty)`` in screen fractions.
* Compute the delta ``(tx - cx, ty - cy)`` from the open-loop cursor
  estimate.
* Pick a magnitude (COARSE/MEDIUM/FINE/MICRO) from the distance.
* Convert direction × magnitude to HID units using a learned
  ``pct_per_hid`` ratio (online EMA, refined whenever the observed
  target shift gives us evidence of how far the cursor moved relative
  to the screen).
* Send the move and update the open-loop cursor estimate.

When the cursor estimate is within ``GATE_THRESHOLD_PCT`` of the target
we ask gemma a strict-JSON yes/no "is the cursor on the named element"
gate. We click ONLY on YES_CLICK.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from handsneyes.core.vision.homer_prompts import FINAL_GATE_PROMPT
from handsneyes.core.vision.imaging import (
    enhance_for_screen,
    numpy_to_base64_png,
    resize_for_mllm,
)

if TYPE_CHECKING:
    # InteractiveSession typing-only import dropped — Phase A AgentContext
    # plus session-adapter shims provide the interface the homer uses.
    InteractiveSession = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# HID-units per move-step at each magnitude tier. Tunable.
MAGNITUDE_HID: dict[str, int] = {
    "COARSE": 220,
    "MEDIUM": 80,
    "FINE":   24,
    "MICRO":  6,
}


def magnitude_for_distance(d: float) -> str:
    if d > 0.30:
        return "COARSE"
    if d > 0.10:
        return "MEDIUM"
    if d > 0.03:
        return "FINE"
    return "MICRO"


# Once cursor estimate is this close to the target we hand off to the gate.
GATE_THRESHOLD_PCT = 0.04

# Default pct/HID guess. macOS with cursor acceleration enabled empirically
# moves the cursor about 2.5× faster than the linear (1 HID ≈ 1 px on
# 1920px screen) baseline. We start above the linear value and refine.
DEFAULT_PCT_PER_HID = 2.5 / 1920.0

# Treat this fraction of the remaining distance as the per-step cap so we
# never overshoot in one go.
STEP_DISTANCE_FRACTION = 0.55

# Reject a re-queried ShowUI target hit if it's farther than this from the
# cached anchor — assume ShowUI flipped to a different element.
TARGET_ANCHOR_TOLERANCE = 0.15

MAX_STEPS = 30
SETTLE_SEC = 0.20
PROOF_DIR = Path("/tmp/handsneyes_homer")


@dataclass
class StepRecord:
    cursor_pct: tuple[float, float] | None
    target_pct: tuple[float, float] | None
    distance_pct: float | None
    direction: str
    magnitude: str
    hid_dx: int
    hid_dy: int
    note: str = ""


@dataclass
class ClickOutcome:
    clicked: bool
    steps: int
    reason: str
    proof_path: str | None = None
    history: list[StepRecord] = field(default_factory=list)


class ClosedLoopHomer:
    def __init__(self, *, session: "InteractiveSession") -> None:
        self._session = session
        self._pct_per_hid_x: float = DEFAULT_PCT_PER_HID
        self._pct_per_hid_y: float = DEFAULT_PCT_PER_HID

    async def run(self, target_desc: str, button: str = "left") -> ClickOutcome:
        run_dir = PROOF_DIR / datetime.now().strftime("%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        history: list[StepRecord] = []
        last_proof: str | None = None
        showui_prompts = self._showui_prompt_variants(target_desc)

        print(f"  Homing (closed-loop): {target_desc}")
        print(f"  Step log: {run_dir}/")

        # 1) Slam: cursor goes to top-left corner. Trust this absolutely.
        await self._slam_to_corner()
        cursor_pct: tuple[float, float] = (0.005, 0.005)

        # 2) Locate the target ONCE up front. ShowUI is unstable across
        # frames — we cache the first hit and ignore re-readings unless
        # they agree with the anchor. The target is a fixed UI element,
        # so its true position does not change.
        first_frame = await self._session._capture.capture_frame()
        b64 = await self._encode(first_frame.image)

        # Phase 0: scene map. Gemma enumerates clickable elements with
        # labels + approximate positions. Then we pick the entry that
        # matches the user's target. This avoids ShowUI's habit of
        # locking onto any element whose label happens to contain a
        # keyword (e.g. a "Run" tab, when the user wanted the "Run"
        # button on the right).
        scene = await self._scene_map(b64, run_dir)
        match = self._best_scene_match(scene, target_desc)
        anchor_pct: tuple[float, float] | None = None
        anchor_source: str = "none"
        if match is not None:
            print(
                f"  Scene-map matched {match['label']!r} "
                f"({match['description'][:60]}, region={match['region']}) "
                f"— grounding via ShowUI."
            )
            ground_prompts = [
                f"Click on {match['label']}",
                f"Click on the {match['label']} button",
                f"Click on {match['label']} button",
            ]
            for p in ground_prompts:
                pos = await self._session._showui_query(b64, p)
                if pos is not None:
                    anchor_pct = pos
                    anchor_source = f"scene_map[{match['label']!r}] grounded by ShowUI"
                    break
        if anchor_pct is None:
            anchor_pct = await self._locate_target(b64, target_desc, showui_prompts)
            anchor_source = "showui"
            if anchor_pct is None:
                logger.warning("ShowUI missed initial anchor; trying gemma.")
                anchor_pct = await self._session._ask_gemma_location(b64, target_desc)
                anchor_source = "gemma"
        if anchor_pct is None:
            print("  Could not locate target. Abort, no click.")
            return ClickOutcome(
                clicked=False, steps=0, reason="target_lost",
                proof_path=None, history=history,
            )
        print(
            f"  Anchor target ≈ ({anchor_pct[0]:.2%}, {anchor_pct[1]:.2%}) "
            f"via {anchor_source}"
        )

        # When the anchor came from gemma (which sometimes guesses near
        # any text matching the keyword), verify what's actually there
        # before we commit to homing. The algorithm should refuse to
        # click rather than land on the wrong thing.
        if anchor_source == "gemma":
            verified = await self._verify_anchor(
                first_frame.image, anchor_pct, target_desc,
            )
            if not verified:
                print("  Anchor verification FAILED — aborting without click.")
                return ClickOutcome(
                    clicked=False, steps=0, reason="anchor_unverified",
                    proof_path=None, history=history,
                )

        # 3) Iterate: move toward anchor, refine using ShowUI re-reads
        # only if they're consistent with the anchor.
        anchor_seen_count = 1  # we already saw the anchor once
        absent_streak = 0
        validator_holds = 0
        MAX_VALIDATOR_HOLDS = 3
        for step in range(1, MAX_STEPS + 1):
            t_step = time.monotonic()
            frame = await self._session._capture.capture_frame()
            b64 = await self._encode(frame.image)

            # Re-query ShowUI; reject outliers far from anchor.
            new_target = await self._locate_target(b64, target_desc, showui_prompts)
            if new_target is None:
                target_pct: tuple[float, float] | None = None
                absent_streak += 1
            else:
                drift = math.hypot(
                    new_target[0] - anchor_pct[0],
                    new_target[1] - anchor_pct[1],
                )
                if drift <= TARGET_ANCHOR_TOLERANCE:
                    target_pct = new_target
                    anchor_seen_count += 1
                    absent_streak = 0
                else:
                    logger.debug(
                        "Rejecting ShowUI drift: anchor=%s, new=%s, drift=%.2f",
                        anchor_pct, new_target, drift,
                    )
                    target_pct = None
                    absent_streak += 1

            # If ShowUI failed or gave an outlier, treat the target as
            # potentially hidden by the cursor — but only if the cursor
            # estimate is already close to the anchor.
            cx, cy = cursor_pct
            anchor_dx = anchor_pct[0] - cx
            anchor_dy = anchor_pct[1] - cy
            anchor_distance = math.hypot(anchor_dx, anchor_dy)

            if target_pct is None:
                cursor_in_zone = anchor_distance <= GATE_THRESHOLD_PCT * 1.5
                if cursor_in_zone and anchor_seen_count >= 2:
                    # ShowUI saw the anchor reliably, now can't see it,
                    # and the cursor estimate sits on the anchor. Before
                    # clicking, validate visually that the cursor really
                    # is on the target.
                    print(
                        f"  [{step:02d}] anchor_seen={anchor_seen_count} "
                        f"absent_streak={absent_streak} cursor_in_zone — "
                        f"validating click."
                    )
                    if await self._validate_click(frame.image, cursor_pct, target_desc, run_dir=run_dir, step=step):
                        await self._session._executor._mouse.click(button)
                        proof = await self._capture_proof(run_dir, step)
                        history.append(StepRecord(
                            cursor_pct=cursor_pct, target_pct=None,
                            distance_pct=anchor_distance, direction="NONE",
                            magnitude="MICRO", hid_dx=0, hid_dy=0,
                            note="validated_cursor_in_zone_target_absent",
                        ))
                        return ClickOutcome(
                            clicked=True, steps=step,
                            reason="validated_cursor_in_zone_target_absent",
                            proof_path=proof, history=history,
                        )
                    validator_holds += 1
                    if validator_holds >= MAX_VALIDATOR_HOLDS:
                        print(
                            f"  [{step:02d}] validator held {validator_holds} "
                            f"times — anchor likely wrong, aborting."
                        )
                        return ClickOutcome(
                            clicked=False, steps=step,
                            reason="validator_held_repeatedly",
                            proof_path=last_proof, history=history,
                        )
                    print(f"  [{step:02d}] click validator HOLD — micro-nudge SE.")
                    await self._send_hid(MAGNITUDE_HID["MICRO"], MAGNITUDE_HID["MICRO"])
                    cursor_pct = (
                        cx + MAGNITUDE_HID["MICRO"] * self._pct_per_hid_x,
                        cy + MAGNITUDE_HID["MICRO"] * self._pct_per_hid_y,
                    )
                    history.append(StepRecord(
                        cursor_pct=cursor_pct, target_pct=None,
                        distance_pct=anchor_distance, direction="SE",
                        magnitude="MICRO",
                        hid_dx=MAGNITUDE_HID["MICRO"],
                        hid_dy=MAGNITUDE_HID["MICRO"],
                        note="validator_hold_after_target_absent",
                    ))
                    last_proof = self._dump_step(
                        run_dir, step, frame.image, cursor_pct, None, history[-1],
                    )
                    await asyncio.sleep(SETTLE_SEC)
                    continue
                # Cursor still far from target; treat anchor as ground
                # truth and keep moving.
                target_pct = anchor_pct

            tx, ty = target_pct
            dx_pct = tx - cx
            dy_pct = ty - cy
            distance = math.hypot(dx_pct, dy_pct)

            # If cursor estimate is within the gate threshold of the
            # anchor target, click — no further confirmation needed.
            # ShowUI gave us the target, open-loop gave us the cursor;
            # both are model-driven signals we trust. The gemma gate is
            # too unreliable for sub-pixel cursor judgment.
            anchor_dist_now = math.hypot(
                anchor_pct[0] - cx, anchor_pct[1] - cy,
            )
            if anchor_dist_now <= GATE_THRESHOLD_PCT:
                print(
                    f"  [{step:02d}] cursor≈({cx:.2%},{cy:.2%}) within "
                    f"{GATE_THRESHOLD_PCT:.0%} of anchor "
                    f"({anchor_pct[0]:.2%},{anchor_pct[1]:.2%}) — validating click."
                )
                if await self._validate_click(frame.image, cursor_pct, target_desc, run_dir=run_dir, step=step):
                    await self._session._executor._mouse.click(button)
                    proof = await self._capture_proof(run_dir, step)
                    history.append(StepRecord(
                        cursor_pct=cursor_pct, target_pct=target_pct,
                        distance_pct=anchor_dist_now, direction="NONE",
                        magnitude="MICRO", hid_dx=0, hid_dy=0,
                        note="validated_cursor_in_anchor_zone",
                    ))
                    return ClickOutcome(
                        clicked=True, steps=step,
                        reason="validated_cursor_in_anchor_zone",
                        proof_path=proof, history=history,
                    )
                validator_holds += 1
                if validator_holds >= MAX_VALIDATOR_HOLDS:
                    print(
                        f"  [{step:02d}] validator held {validator_holds} "
                        f"times — anchor likely wrong, aborting."
                    )
                    return ClickOutcome(
                        clicked=False, steps=step,
                        reason="validator_held_repeatedly",
                        proof_path=last_proof, history=history,
                    )
                print(f"  [{step:02d}] click validator HOLD — nudging toward fresh target.")
                hid_dx_n, hid_dy_n = self._hid_for_pct(dx_pct, dy_pct, "MICRO")
                await self._send_hid(hid_dx_n, hid_dy_n)
                cursor_pct = (
                    cx + hid_dx_n * self._pct_per_hid_x,
                    cy + hid_dy_n * self._pct_per_hid_y,
                )
                history.append(StepRecord(
                    cursor_pct=cursor_pct, target_pct=target_pct,
                    distance_pct=anchor_dist_now,
                    direction=self._compass(dx_pct, dy_pct),
                    magnitude="MICRO", hid_dx=hid_dx_n, hid_dy=hid_dy_n,
                    note="validator_hold_in_zone",
                ))
                last_proof = self._dump_step(
                    run_dir, step, frame.image, cursor_pct, target_pct, history[-1],
                )
                await asyncio.sleep(SETTLE_SEC)
                continue

            # (The "within threshold of anchor" early-return above
            # handles the click case using anchor_dist_now. Otherwise
            # we keep moving toward the freshly-located target.)

            # Pick magnitude from distance and convert to HID. Cap by a
            # fraction of the predicted-pixel distance so we never blast
            # past the target in one step.
            magnitude = magnitude_for_distance(distance)
            hid_dx, hid_dy = self._hid_for_pct(dx_pct, dy_pct, magnitude)
            hid_cap_x = max(
                MAGNITUDE_HID["MICRO"],
                int(abs(dx_pct) / max(self._pct_per_hid_x, 1e-6) * STEP_DISTANCE_FRACTION),
            )
            hid_cap_y = max(
                MAGNITUDE_HID["MICRO"],
                int(abs(dy_pct) / max(self._pct_per_hid_y, 1e-6) * STEP_DISTANCE_FRACTION),
            )
            if abs(hid_dx) > hid_cap_x:
                hid_dx = int(math.copysign(hid_cap_x, hid_dx))
            if abs(hid_dy) > hid_cap_y:
                hid_dy = int(math.copysign(hid_cap_y, hid_dy))
            direction = self._compass(dx_pct, dy_pct)

            await self._send_hid(hid_dx, hid_dy)
            await asyncio.sleep(SETTLE_SEC)

            # Open-loop cursor update.
            cursor_pct = (
                cx + hid_dx * self._pct_per_hid_x,
                cy + hid_dy * self._pct_per_hid_y,
            )

            elapsed = time.monotonic() - t_step
            print(
                f"  [{step:02d}] dir={direction:<4} mag={magnitude:<6} "
                f"hid=({hid_dx:+4d},{hid_dy:+4d}) "
                f"cursor=({cx:.2f},{cy:.2f})→({cursor_pct[0]:.2f},{cursor_pct[1]:.2f}) "
                f"target=({tx:.2f},{ty:.2f}) dist={distance:.2%} ratio=({self._pct_per_hid_x:.5f},{self._pct_per_hid_y:.5f}) {elapsed:.1f}s"
            )
            history.append(StepRecord(
                cursor_pct=cursor_pct, target_pct=target_pct,
                distance_pct=distance, direction=direction, magnitude=magnitude,
                hid_dx=hid_dx, hid_dy=hid_dy,
            ))
            last_proof = self._dump_step(
                run_dir, step, frame.image, cursor_pct, target_pct, history[-1],
            )

        print(f"  Reached MAX_STEPS={MAX_STEPS} without on-target gate. NOT clicking.")
        return ClickOutcome(
            clicked=False, steps=MAX_STEPS, reason="max_steps",
            proof_path=last_proof, history=history,
        )

    # ────────────────────── helpers ──────────────────────

    @staticmethod
    def _showui_prompt_variants(target_desc: str) -> list[str]:
        import re as _re
        variants: list[str] = []
        variants.append(f"Click on {target_desc}")
        for q in _re.findall(r"['\"]([^'\"]+)['\"]", target_desc):
            variants.append(f"Click on {q}")
            variants.append(f"Click on the {q} button")
        for cap in _re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", target_desc):
            variants.append(f"Click on {cap}")
            variants.append(f"Click on the {cap} button")
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            k = v.lower().strip()
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
        return out

    async def _locate_target(
        self, b64: str, target_desc: str, prompts: list[str],
    ) -> tuple[float, float] | None:
        for p in prompts:
            pos = await self._session._showui_query(b64, p)
            if pos is not None:
                logger.debug("ShowUI hit '%s' → %s", p, pos)
                return pos
        return None

    @staticmethod
    def _compass(dx: float, dy: float) -> str:
        ax, ay = abs(dx), abs(dy)
        if ax < 0.005 and ay < 0.005:
            return "NONE"
        sx = "E" if dx > 0 else ("W" if dx < 0 else "")
        sy = "S" if dy > 0 else ("N" if dy < 0 else "")
        if ax > 2 * ay:
            return sx or "NONE"
        if ay > 2 * ax:
            return sy or "NONE"
        return f"{sy}{sx}" or "NONE"

    def _hid_for_pct(self, dx_pct: float, dy_pct: float, magnitude: str) -> tuple[int, int]:
        budget = MAGNITUDE_HID[magnitude]
        norm = math.hypot(dx_pct, dy_pct) or 1.0
        ux, uy = dx_pct / norm, dy_pct / norm
        return int(round(ux * budget)), int(round(uy * budget))

    async def _slam_to_corner(self) -> None:
        print("  Slamming to top-left corner...")
        for _ in range(200):
            try:
                await self._session._executor._mouse.move(-20, -20)
            except Exception:
                pass
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.3)

    async def _send_hid(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        await self._session._send_hid_moves(dx, dy)

    async def _scene_map(
        self, b64: str, run_dir: Path | None,
    ) -> list[dict]:
        """Ask gemma to enumerate clickable elements on the screen.

        Returns a list of dicts with keys: label, description, region.
        We do NOT ask gemma for coordinates — gemma estimates them
        unreliably. ShowUI grounds the winning entry's label later.
        """
        await self._session._ensure_client()
        prompt = (
            "You are a JSON API. Look at the screen image and "
            "enumerate up to 12 *clickable* UI elements (buttons, "
            "tabs, icons, links). Skip plain text, code, and decorative "
            "imagery.\n\n"
            "Respond with ONLY a JSON object — no preamble, no markdown.\n\n"
            'Schema: {"elements": [\n'
            '  {"label": "<exact text shown on the element, or short '
            'name if no text>",\n'
            '   "description": "<short: type, colour, role>",\n'
            '   "region": "<one of: top-left, top-center, top-right, '
            'middle-left, middle-center, middle-right, bottom-left, '
            'bottom-center, bottom-right>"}\n'
            ']}\n\n'
            "Be concrete: prefer real text shown on the element "
            "(\"Run\", \"Sign In\") over guessed roles. Pick the "
            "region that contains the element's centre."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": "Enumerate clickable elements. Reply JSON only."},
                ],
            },
        ]
        try:
            resp = await self._session._client.chat.completions.create(
                model=self._session._model,
                max_tokens=2000,
                temperature=0.0,
                messages=messages,
            )
            raw = self._session._evaluator._best_text_from_response(resp) or ""
            if run_dir is not None:
                try:
                    (run_dir / "scene_map_raw.txt").write_text(raw)
                except Exception:
                    pass
            data = self._session._evaluator._extract_json(raw) or {}
            elements = data.get("elements") or []
            cleaned: list[dict] = []
            for e in elements:
                if not isinstance(e, dict):
                    continue
                label = str(e.get("label", "")).strip()
                desc = str(e.get("description", "")).strip()
                region = str(e.get("region", "")).strip().lower()
                if not label and not desc:
                    continue
                cleaned.append({
                    "label": label, "description": desc, "region": region,
                })
            print(f"  Scene-map: {len(cleaned)} clickable element(s)")
            for i, e in enumerate(cleaned):
                print(
                    f"    {i:2d}. {e['label']!r:25s} "
                    f"[{e['region']:15s}] — {e['description'][:80]}"
                )
            if run_dir is not None:
                try:
                    import json as _json
                    (run_dir / "scene_map.json").write_text(
                        _json.dumps(cleaned, indent=2)
                    )
                except Exception:
                    pass
            return cleaned
        except Exception as e:
            logger.error("Scene map query failed: %s", e)
            return []

    @classmethod
    def _best_scene_match(
        cls, scene: list[dict], target_desc: str,
    ) -> dict | None:
        """Pick the scene element that best matches the user's target.

        Scoring combines:
        - 2 points per primary-target keyword found in label
        - 1 point per primary-target keyword in description or region
        - 1 bonus point if the user mentioned a positional region
          (e.g. "right side") that overlaps the element's region.

        Returns the sole top-scorer when its score strictly exceeds the
        runner-up. Otherwise returns None.
        """
        if not scene:
            return None
        keywords = cls._target_keywords(target_desc)
        position_words = cls._target_position_words(target_desc)
        if not keywords:
            return None
        import re as _re
        def _word_in(needle: str, haystack: str) -> bool:
            return _re.search(rf"\b{_re.escape(needle)}\b", haystack) is not None

        scored: list[tuple[float, dict]] = []
        for elem in scene:
            label = elem.get("label", "").lower()
            desc = elem.get("description", "").lower()
            region = elem.get("region", "").lower()
            score = 0.0
            for k in keywords:
                if _word_in(k, label):
                    score += 2.0
                elif _word_in(k, desc) or _word_in(k, region):
                    score += 1.0
            for pw in position_words:
                if _word_in(pw, region) or _word_in(pw, desc):
                    score += 1.0
            scored.append((score, elem))
        scored.sort(key=lambda t: t[0], reverse=True)
        top_score = scored[0][0]
        if top_score == 0:
            return None
        runner_up = scored[1][0] if len(scored) > 1 else 0
        if top_score <= runner_up:
            print(
                f"  Scene-map ambiguous: top={top_score} runner_up={runner_up}; "
                f"falling back to ShowUI."
            )
            return None
        return scored[0][1]

    @staticmethod
    def _target_position_words(target_desc: str) -> list[str]:
        """Extract positional cues from the user's description."""
        out: list[str] = []
        lower = target_desc.lower()
        for word in (
            "top", "bottom", "left", "right", "center", "centre",
            "upper", "lower",
        ):
            if word in lower:
                out.append(word)
        return out

    async def _validate_click(
        self,
        image: np.ndarray,
        cursor_pct: tuple[float, float],
        target_desc: str,
        run_dir: Path | None = None,
        step: int | None = None,
    ) -> bool:
        """Blind visual check before any click.

        Two-step protocol — gemma is NOT told what the expected target is:
        1) Ask gemma to describe what is under the crosshair on a fresh
           annotated frame. No target hint = no leading.
        2) Programmatically check whether the description mentions the
           target's distinguishing keywords. Only then do we click.

        Also dumps the pre-click annotated image to ``run_dir`` for
        post-hoc audit when arguments are provided.
        """
        await self._session._ensure_client()
        h, w = image.shape[:2]
        cx = int(cursor_pct[0] * w)
        cy = int(cursor_pct[1] * h)
        # Crop a small region centred on the click position (no
        # annotation overlaid — earlier attempts had gemma describing
        # the crosshair itself). Then mark the absolute centre with a
        # subtle dot only on the SAVED audit image.
        crop_w = int(w * 0.18)
        crop_h = int(h * 0.18)
        x0 = max(0, cx - crop_w // 2)
        y0 = max(0, cy - crop_h // 2)
        x1 = min(w, x0 + crop_w)
        y1 = min(h, y0 + crop_h)
        crop = image[y0:y1, x0:x1].copy()
        if run_dir is not None and step is not None:
            audit = crop.copy()
            ax = (cx - x0)
            ay = (cy - y0)
            cv2.circle(audit, (ax, ay), 4, (0, 255, 255), -1)
            try:
                cv2.imwrite(
                    str(run_dir / f"step_{step:02d}_preclick.png"), audit,
                )
            except Exception:
                pass
        b64 = numpy_to_base64_png(
            resize_for_mllm(enhance_for_screen(crop),
                            max_dimension=900, min_dimension=512)
        )

        prompt = (
            "You are a JSON API. This image is a CROP of a screen "
            "centred on the spot where a mouse click is about to "
            "happen. The click will land at the GEOMETRIC CENTRE of "
            "this image. Describe ONLY the UI element at that centre.\n\n"
            "Respond with ONLY a JSON object — no preamble, no "
            "markdown.\n\n"
            'Schema: {"label": "<exact text shown on the element at '
            'the centre, or empty string if none>", '
            '"description": "<short description of element type and '
            'colour, e.g. \'small blue rounded button\'>"}\n\n'
            "If the centre is on whitespace or unclear background, set "
            'label="" and description="empty/background". Do not '
            "describe markers, crosshairs, or annotations — there are "
            "none in this image."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": "Describe what is under the crosshair. Reply JSON only."},
                ],
            },
        ]
        try:
            resp = await self._session._client.chat.completions.create(
                model=self._session._model,
                max_tokens=300,
                temperature=0.0,
                messages=messages,
            )
            raw = self._session._evaluator._best_text_from_response(resp) or ""
            data = self._session._evaluator._extract_json(raw) or {}
            import re as _re
            label = str(data.get("label", "")).strip()
            description = str(data.get("description", "")).strip()
            combined = f"{label} {description}".lower()
            keywords = self._target_keywords(target_desc)
            matched = [
                k for k in keywords
                if _re.search(rf"\b{_re.escape(k)}\b", combined)
            ]
            verdict = "CLICK" if matched else "HOLD"
            print(
                f"  Validate click: under-label={label!r} desc={description!r} "
                f"keywords={keywords} matched={matched} → {verdict}"
            )
            return bool(matched)
        except Exception as e:
            logger.error("Click validation failed: %s", e)
            return False

    @staticmethod
    def _target_keywords(target_desc: str) -> list[str]:
        """Extract distinguishing keywords for the PRIMARY target only.

        Cuts the description at the first positional clause ("on the",
        "above ...", "below ...", etc.) so that positional context like
        "below the Run button" doesn't introduce competing keywords. The
        remaining head-of-noun-phrase tokens are filtered through a
        stopword list and returned in order.
        """
        import re as _re
        stopwords = {
            "click", "on", "the", "a", "an", "button", "icon", "tab",
            "of", "in", "at", "side", "with", "and", "or",
            "labeled", "label", "rectangular", "square", "round",
            "rounded", "small", "big", "large", "tiny", "this",
            "written", "it", "that", "says",
        }
        # Anything after one of these phrases is positional context, not
        # the target itself.
        cut_phrases = [
            " on the ", " on top ", " in the ", " at the ", " above ",
            " below ", " next to ", " to the ", " near the ", " near ",
            " left of ", " right of ", " inside the ",
        ]
        head = target_desc.lower()
        for phrase in cut_phrases:
            idx = head.find(phrase)
            if idx >= 0:
                head = head[:idx]
        # Use the original-case head segment so we keep capitalisation
        # information for token extraction.
        head_orig = target_desc[: len(head)]
        keywords: list[str] = []
        for quoted, token in _re.findall(
            r"['\"]([^'\"]+)['\"]|([A-Za-z][A-Za-z0-9]{1,})", head_orig
        ):
            piece = quoted or token
            if not piece:
                continue
            lower = piece.lower()
            if lower in stopwords:
                continue
            keywords.append(lower)
        # Always include any quoted strings (regardless of position) —
        # quoting is a strong signal of literal element text.
        for q in _re.findall(r"['\"]([^'\"]+)['\"]", target_desc):
            if q.lower() not in keywords:
                keywords.append(q.lower())
        seen: set[str] = set()
        out: list[str] = []
        for k in keywords:
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    async def _verify_anchor(
        self, image: np.ndarray, anchor_pct: tuple[float, float],
        target_desc: str,
    ) -> bool:
        """Ask gemma if a cropped region around the anchor matches the target.

        Returns True only on a clear positive. Defaults to False on any
        ambiguity to prevent clicking the wrong element.
        """
        await self._session._ensure_client()
        h, w = image.shape[:2]
        # The webcam sees the monitor PLUS a frame of desk/wall around
        # it. When the anchor is near the image edge, a verification
        # crop will include off-screen background and gemma will
        # reasonably reject "yellow desk surface" as the target. Skip
        # verification at edges and trust the anchor.
        margin = 0.08
        if (
            anchor_pct[0] < margin or anchor_pct[0] > 1 - margin
            or anchor_pct[1] < margin or anchor_pct[1] > 1 - margin
        ):
            print("  Anchor near image edge — skipping verification.")
            return True
        cx = int(anchor_pct[0] * w)
        cy = int(anchor_pct[1] * h)
        win_w = int(w * 0.14)
        win_h = int(h * 0.14)
        x0 = max(0, cx - win_w // 2)
        y0 = max(0, cy - win_h // 2)
        x1 = min(w, x0 + win_w)
        y1 = min(h, y0 + win_h)
        crop = image[y0:y1, x0:x1].copy()
        # Mark the anchor with a small crosshair so gemma knows where
        # we'd click.
        cx_local = cx - x0
        cy_local = cy - y0
        cv2.drawMarker(
            crop, (cx_local, cy_local), (0, 255, 255),
            markerType=cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=2,
        )
        cv2.circle(crop, (cx_local, cy_local), 14, (0, 255, 255), 2)
        b64_crop = numpy_to_base64_png(
            resize_for_mllm(enhance_for_screen(crop), max_dimension=900, min_dimension=512)
        )

        prompt = (
            "You are a JSON API verifying a click target. The yellow "
            "crosshair marks where a click is about to happen. Look at "
            "what is under the crosshair.\n\n"
            f"Expected target: {target_desc}\n\n"
            "Respond with ONLY a JSON object — no preamble, no markdown.\n\n"
            'Schema: {"matches": true|false, '
            '"what_is_there": "<short description of element under '
            'the crosshair>"}\n\n'
            "Be conservative: only return matches=true if the element "
            "under the crosshair is unambiguously the expected target. "
            "Adjacent elements, similar buttons, or empty space => false."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_crop}", "detail": "high"},
                    },
                    {"type": "text", "text": "Verify the click target. Reply JSON only."},
                ],
            },
        ]
        try:
            resp = await self._session._client.chat.completions.create(
                model=self._session._model,
                max_tokens=300,
                temperature=0.0,
                messages=messages,
            )
            raw = self._session._evaluator._best_text_from_response(resp) or ""
            data = self._session._evaluator._extract_json(raw) or {}
            matches = bool(data.get("matches", False))
            what = str(data.get("what_is_there", ""))[:140]
            print(f"  Verify anchor: matches={matches} what={what!r}")
            return matches
        except Exception as e:
            logger.error("Anchor verification failed: %s", e)
            return False

    async def _ask_final_gate(self, b64: str, target_desc: str) -> bool:
        """Returns True if cursor likely covers the target → safe to click.

        Asks gemma whether the *target element* is still clearly visible.
        If gemma says target_visible=false, we infer the cursor is on top
        of it (the only reason a fixed UI element would suddenly be
        obscured during a homing run is the cursor moving over it).
        """
        await self._session._ensure_client()
        prompt = FINAL_GATE_PROMPT.format(target_description=target_desc)
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": "Is the target clearly visible? Reply JSON only."},
                ],
            },
        ]
        try:
            resp = await self._session._client.chat.completions.create(
                model=self._session._model,
                max_tokens=300,
                temperature=0.0,
                messages=messages,
            )
            raw = self._session._evaluator._best_text_from_response(resp) or ""
            data = self._session._evaluator._extract_json(raw) or {}
            visible = bool(data.get("target_visible", True))
            reason = str(data.get("reason", ""))[:140]
            print(f"       gate target_visible={visible} reason={reason}")
            return not visible
        except Exception as e:
            logger.error("Final-gate query failed: %s", e)
            return False

    @staticmethod
    async def _encode(image: np.ndarray) -> str:
        resized = resize_for_mllm(enhance_for_screen(image), max_dimension=1280, min_dimension=768)
        return numpy_to_base64_png(resized)

    def _dump_step(
        self,
        run_dir: Path,
        step: int,
        image: np.ndarray,
        cursor_pct: tuple[float, float] | None,
        target_pct: tuple[float, float] | None,
        rec: StepRecord,
    ) -> str | None:
        try:
            out = image.copy()
            h, w = out.shape[:2]
            if target_pct is not None:
                tx = int(target_pct[0] * w)
                ty = int(target_pct[1] * h)
                cv2.rectangle(out, (tx - 30, ty - 18), (tx + 30, ty + 18), (0, 0, 255), 2)
                cv2.putText(out, "TARGET", (tx + 32, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if cursor_pct is not None:
                cx = int(cursor_pct[0] * w)
                cy = int(cursor_pct[1] * h)
                cv2.circle(out, (cx, cy), 18, (255, 200, 0), 2)
                cv2.putText(out, "CURSOR(est)", (cx + 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
            label = (
                f"step {step:02d} dir={rec.direction} mag={rec.magnitude} "
                f"hid=({rec.hid_dx:+d},{rec.hid_dy:+d})"
            )
            if rec.distance_pct is not None:
                label += f" dist={rec.distance_pct:.2%}"
            cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(out, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            path = run_dir / f"step_{step:02d}.png"
            cv2.imwrite(str(path), out)
            return str(path)
        except Exception as e:
            logger.debug("dump_step failed: %s", e)
            return None

    async def _capture_proof(self, run_dir: Path, step: int) -> str | None:
        await asyncio.sleep(0.2)
        try:
            frame = await self._session._capture.capture_frame()
            path = run_dir / f"step_{step:02d}_after_click.png"
            cv2.imwrite(str(path), frame.image)
            return str(path)
        except Exception:
            return None
