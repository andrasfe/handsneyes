# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commander; lint cleanup deferred.
"""Prompts for the closed-loop homer.

Two prompts:
- NAVIGATOR_PROMPT: per-step "which way and how much" decision.
- FINAL_GATE_PROMPT: last-mile "is the cursor on the right element" check.

Both expect strict JSON. The image is a webcam photo of a monitor; the
target is described in natural language. Coordinates are NOT used —
only categorical direction + magnitude.
"""

NAVIGATOR_PROMPT = """\
You are a JSON API that outputs a single mouse-move decision per call.
The image is a webcam photo of a real monitor. Your job is to move the
cursor toward this target:

    {target_description}

You MUST respond with ONLY a JSON object — no preamble, no markdown,
no narration. Start your response with `{{` and end with `}}`.

JSON schema (with example values):

{{
  "target_present": true,
  "cursor_visible": true,
  "on_target": false,
  "direction": "SE",
  "magnitude": "COARSE",
  "reasoning": "cursor in top-left, target in bottom-right corner"
}}

Field meanings:
- target_present: can you see the target element in the image?
- cursor_visible: can you see the mouse cursor (arrow / pointer)?
- on_target: is the cursor's tip currently sitting on the target element?
- direction: compass direction the cursor must travel.
  N=up, S=down, E=right, W=left, NE=up-right, NW=up-left,
  SE=down-right, SW=down-left, NONE=do not move.
- magnitude: how far to travel in this single step.
  COARSE = roughly a quarter of the screen.
  MEDIUM = roughly an eighth of the screen.
  FINE   = a small icon or button width.
  MICRO  = a hair-width nudge.
- reasoning: ONE short sentence (under 25 words).

Decision rules — follow exactly:
1. If on_target is true, set direction="NONE" and magnitude="MICRO".
2. If cursor_visible is false but you know roughly where the cursor is
   from context, COMMIT TO A DIRECTION ANYWAY — never return NONE
   unless you are confident the cursor is already on target.
3. When the cursor is more than half a screen from the target, prefer
   COARSE.
4. When the cursor is within a button's width of the target, prefer
   FINE or MICRO.
5. Never use COARSE when the cursor is already close to the target.

Respond with the JSON object only.
"""


FINAL_GATE_PROMPT = """\
You are a JSON API that decides whether a target UI element is still
clearly visible on screen, or whether something is now covering it.

Target element:

    {target_description}

You MUST respond with ONLY a JSON object — no preamble, no markdown.
Start your response with `{{` and end with `}}`.

JSON schema (with example values):

{{
  "target_visible": false,
  "reason": "the time text is no longer readable; something small is over it"
}}

Field meanings:
- target_visible: true if you can clearly read / see the target element
  in the image; false if it appears partly or fully obscured (e.g., the
  mouse cursor is covering it).
- reason: ONE short sentence (under 25 words).

Important:
- The mouse cursor is a tiny arrow — easy to miss. Focus on whether the
  TARGET ELEMENT itself is clearly visible.
- If the target is fully readable and unobstructed, target_visible=true.
- If the target's centre/text is partly hidden by anything small that
  looks like a cursor, target_visible=false.

Respond with the JSON object only.
"""
