"""Text-target keyword + prompt-variant helpers.

These two pure-string functions are factored out of the legacy
``closed_loop_homer`` so the tier-1 :class:`TargetAgent` can use them
without pulling in the entire 990-line homer module — which itself
isn't required for handsneyes Phase A (the visual servo homer is a
tier-3 concern landing in a later phase).

The behaviour is byte-identical to terminaleyes'
``ClosedLoopHomer._target_keywords`` and
``ClosedLoopHomer._showui_prompt_variants``.
"""

from __future__ import annotations

import re

_STOPWORDS = frozenset(
    {
        "click", "on", "the", "a", "an", "button", "icon", "tab",
        "of", "in", "at", "side", "with", "and", "or",
        "labeled", "label", "rectangular", "square", "round",
        "rounded", "small", "big", "large", "tiny", "this",
        "written", "it", "that", "says",
    }
)

# Anything after one of these phrases is positional context relative
# to the primary target, not the target itself.
_CUT_PHRASES = (
    " on the ", " on top ", " in the ", " at the ", " above ",
    " below ", " next to ", " to the ", " near the ", " near ",
    " left of ", " right of ", " inside the ",
)


def target_keywords(target_desc: str) -> list[str]:
    """Distinguishing keywords for the PRIMARY target only.

    Cuts the description at the first positional clause ("on the",
    "above ...", "below ...") so positional context like "below the
    Run button" doesn't introduce competing keywords. Remaining
    head-of-noun-phrase tokens are filtered through a stopword list
    and returned in order. Quoted strings always survive, regardless
    of position, because quoting is a strong signal of literal element
    text.
    """
    head = target_desc.lower()
    for phrase in _CUT_PHRASES:
        idx = head.find(phrase)
        if idx >= 0:
            head = head[:idx]
    head_orig = target_desc[: len(head)]
    keywords: list[str] = []
    for quoted, token in re.findall(
        r"['\"]([^'\"]+)['\"]|([A-Za-z][A-Za-z0-9]{1,})", head_orig
    ):
        piece = quoted or token
        if not piece:
            continue
        lower = piece.lower()
        if lower in _STOPWORDS:
            continue
        keywords.append(lower)
    # Always include quoted strings regardless of position.
    for q in re.findall(r"['\"]([^'\"]+)['\"]", target_desc):
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


def showui_prompt_variants(target_desc: str) -> list[str]:
    """Generate ShowUI click-prompt variants for a target description.

    Pads the literal description with quoted-substring and
    Capitalised-Token variants. De-duplicates by case-insensitive key.
    """
    variants: list[str] = [f"Click on {target_desc}"]
    for q in re.findall(r"['\"]([^'\"]+)['\"]", target_desc):
        variants.append(f"Click on {q}")
        variants.append(f"Click on the {q} button")
    for cap in re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", target_desc):
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
