"""End-to-end CLI smoke tests.

Phase A success criterion: ``handsneyes do --target headless --dry-run
"<intent>"`` produces a sensible plan that names a real agent class.

We invoke ``cli.main`` directly with argv lists rather than spawning
subprocesses — same code path, faster, and captures stdout via
``capsys``.
"""

from __future__ import annotations

import json
import re

import pytest  # noqa: TC002 — fixtures (capsys, MonkeyPatch) need runtime import

from handsneyes.cli import main

# ─── basic plumbing ────────────────────────────────────────────────


def test_version_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "handsneyes" in out
    assert re.search(r"\d+\.\d+\.\d+", out)


def test_platforms_lists_headless(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["platforms"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "headless" in out


def test_unknown_target_returns_exit_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        ["do", "--target", "no_such_platform", "--dry-run", "scroll down 6"]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "no_such_platform" in err
    assert "Available" in err


# ─── do --dry-run plans ────────────────────────────────────────────


def _plan_for(intent: str, target: str = "headless") -> dict:
    """Return the parsed JSON plan from a dry-run invocation."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["do", "--target", target, "--dry-run", intent])
    payload = json.loads(buf.getvalue())
    payload["__rc__"] = rc
    return payload


def test_dry_run_scroll_down_emits_plan() -> None:
    p = _plan_for("scroll down 6")
    assert p["__rc__"] == 0
    assert p["intent"] == "scroll down 6"
    assert p["target"] == "headless"
    assert p["platform"] == "headless"
    assert p["executable"] is False
    assert p["plan"][0]["agent"] == "scroll"
    assert p["plan"][0]["kwargs"]["direction"] == "down"
    assert p["plan"][0]["kwargs"]["amount"] == 6


def test_dry_run_scroll_up_default_amount() -> None:
    p = _plan_for("scroll up")
    assert p["plan"][0]["kwargs"]["direction"] == "up"
    # Default is 4 when no number in the intent.
    assert p["plan"][0]["kwargs"]["amount"] == 4


def test_dry_run_lock() -> None:
    p = _plan_for("lock the screen now")
    assert p["plan"][0]["agent"] == "key_combo"
    assert p["plan"][0]["kwargs"]["modifiers"] == ["super"]
    assert p["plan"][0]["kwargs"]["key"] == "l"


def test_dry_run_login_with_vault_arg() -> None:
    p = _plan_for("login --vault myhost")
    assert p["plan"][0]["agent"] == "login"
    assert p["plan"][0]["kwargs"]["vault_name"] == "myhost"


def test_dry_run_type_strips_quotes() -> None:
    p = _plan_for('type "hello world"')
    step = p["plan"][0]
    assert step["agent"] == "type"
    assert step["kwargs"]["text"] == "hello world"
    assert step["kwargs"]["submit"] is False


def test_dry_run_navigate_url() -> None:
    p = _plan_for("go to reddit.com/r/LocalLLaMA")
    assert p["__rc__"] == 0
    step = p["plan"][0]
    assert step["agent"] == "navigate"
    assert step["kwargs"]["url"] == "https://reddit.com/r/LocalLLaMA"


def test_dry_run_navigate_https_already_present() -> None:
    p = _plan_for("open https://example.com/path")
    assert p["plan"][0]["kwargs"]["url"] == "https://example.com/path"


def test_dry_run_bare_url() -> None:
    p = _plan_for("https://example.com")
    assert p["plan"][0]["agent"] == "navigate"


def test_dry_run_click_target() -> None:
    p = _plan_for("click the Run button")
    step = p["plan"][0]
    assert step["agent"] == "click"
    assert step["kwargs"]["target"] == "the Run button"


def test_dry_run_focus() -> None:
    p = _plan_for("focus")
    assert p["plan"][0]["agent"] == "focus"


def test_dry_run_unmatched_intent_returns_exit_1() -> None:
    p = _plan_for("eat a sandwich")
    assert p["__rc__"] == 1
    assert p["plan"] == []
    assert "no rule matched" in p["error"]


# ─── vault status round-trip via CLI (uses tmp file via env) ──────


def test_vault_status_runs_when_no_file(
    capsys: pytest.CaptureFixture[str],
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point the default vault path at tmp_path so we don't touch the
    # user's real vault. The vault module's DEFAULT_PATH is computed
    # at import time; patch the module attribute directly.
    monkeypatch.setattr(
        "handsneyes.core.vault.DEFAULT_PATH", tmp_path / "vault.enc"
    )
    monkeypatch.setenv("HANDSNEYES_VAULT_PASSPHRASE", "test-passphrase")
    rc = main(["vault", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "exists" in out
    assert "False" in out
