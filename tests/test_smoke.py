"""Smoke test: every package imports cleanly."""

import handsneyes
import handsneyes.core  # noqa: F401
import handsneyes.core.agents  # noqa: F401
import handsneyes.core.capture  # noqa: F401
import handsneyes.core.vault  # noqa: F401
import handsneyes.core.vision  # noqa: F401
import handsneyes.io  # noqa: F401
import handsneyes.io.keyboard  # noqa: F401
import handsneyes.io.mouse  # noqa: F401
import handsneyes.platforms  # noqa: F401
import handsneyes.targets  # noqa: F401
import handsneyes.ui  # noqa: F401


def test_version() -> None:
    assert handsneyes.__version__ == "0.1.0"
