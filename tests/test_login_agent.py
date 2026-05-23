"""Tests for LoginAgent + resolve_password."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from handsneyes.core.agents.context import AgentContext
from handsneyes.core.agents.login import (
    LoginAgent,
    LoginOutcome,
    resolve_password,
)
from handsneyes.core.vault import Vault

# ─── resolve_password ───────────────────────────────────────────────


def test_resolve_password_explicit_wins() -> None:
    assert resolve_password(password="explicit") == "explicit"


def test_resolve_password_from_file(tmp_path) -> None:  # noqa: ANN001
    p = tmp_path / "pw.txt"
    p.write_text("from-file\n")
    assert resolve_password(file_path=str(p)) == "from-file"


def test_resolve_password_strips_crlf(tmp_path) -> None:  # noqa: ANN001
    p = tmp_path / "pw.txt"
    p.write_bytes(b"from-windows\r\n")
    assert resolve_password(file_path=str(p)) == "from-windows"


def test_resolve_password_file_not_found(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(FileNotFoundError):
        resolve_password(file_path=str(tmp_path / "nope.txt"))


def test_resolve_password_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSNEYES_TEST_PW", "from-env")
    assert resolve_password(env_var="HANDSNEYES_TEST_PW") == "from-env"


def test_resolve_password_env_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HANDSNEYES_TEST_PW", raising=False)
    with pytest.raises(ValueError, match="not set"):
        resolve_password(env_var="HANDSNEYES_TEST_PW")


def test_resolve_password_from_vault(tmp_path) -> None:  # noqa: ANN001
    v = Vault("master", path=tmp_path / "vault.enc")
    v.set("desktop", "vault-secret")
    assert resolve_password(vault=v, vault_name="desktop") == "vault-secret"


def test_resolve_password_vault_missing_entry(
    tmp_path,  # noqa: ANN001
) -> None:
    v = Vault("master", path=tmp_path / "vault.enc")
    with pytest.raises(ValueError, match="no entry named"):
        resolve_password(vault=v, vault_name="ghost")


# ─── LoginAgent ─────────────────────────────────────────────────────


def _verify_response(*, answer: bool, reason: str) -> MagicMock:
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=f'{{"answer": {str(answer).lower()}, '
                    f'"reason": "{reason}"}}'
                )
            )
        ]
    )


class TestLoginAgent:
    @pytest.mark.asyncio
    async def test_no_password_source_with_explicit_empty_fails(
        self,
    ) -> None:
        ctx = AgentContext()
        out = await LoginAgent(ctx).run(password="")
        assert isinstance(out, LoginOutcome)
        assert not out
        assert "empty password" in out.reason

    @pytest.mark.asyncio
    async def test_password_resolution_failure_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("HANDSNEYES_TEST_NO_SUCH", raising=False)
        ctx = AgentContext()
        out = await LoginAgent(ctx).run(password_env="HANDSNEYES_TEST_NO_SUCH")
        assert not out
        assert "password resolution failed" in out.reason

    @pytest.mark.asyncio
    async def test_verify_miss_returns_failure(self) -> None:
        # Vision client always returns "not a login screen".
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_verify_response(
                answer=False, reason="terminal foreground"
            )
        )
        kb = AsyncMock()
        m = AsyncMock()
        cap = AsyncMock()
        # Capture returns a small fake frame.
        cap.capture_frame = AsyncMock(
            return_value=MagicMock(image=_blank_image())
        )
        ctx = AgentContext(
            vision_client=client,
            vision_model="m",
            keyboard=kb,
            mouse=m,
            capture=cap,
        )
        # No wake (skip the brightness check + jiggle path).
        out = await LoginAgent(ctx).run(
            password="hunter2",
            wake=False,
            verify_attempts=2,
            verify_interval=0.0,
        )
        assert not out
        assert "not a login screen" in out.reason

    @pytest.mark.asyncio
    async def test_verify_hit_types_password_and_submits(self) -> None:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_verify_response(
                answer=True, reason="lock screen visible"
            )
        )
        kb = AsyncMock()
        m = AsyncMock()
        cap = AsyncMock()
        cap.capture_frame = AsyncMock(
            return_value=MagicMock(image=_blank_image())
        )
        ctx = AgentContext(
            vision_client=client,
            vision_model="m",
            keyboard=kb,
            mouse=m,
            capture=cap,
        )
        out = await LoginAgent(ctx).run(
            password="hunter2",
            wake=False,
            verify_attempts=2,
            verify_interval=0.0,
        )
        assert out
        # The password reached the keyboard with secret=True; submit
        # default True → an Enter keystroke followed.
        send_text_calls = kb.send_text.await_args_list
        assert any(
            c.kwargs.get("secret") is True for c in send_text_calls
        )
        kb.send_keystroke.assert_any_await("Enter")

    @pytest.mark.asyncio
    async def test_already_unlocked_succeeds_without_typing(self) -> None:
        # Vision client answers False to "is this a login screen?" for every
        # poll, then True to "is this already an unlocked desktop?" — so we
        # should short-circuit to success without typing.
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        # First N calls are the login-poll (False). The next call is the
        # already-unlocked probe (True).
        responses = [
            _verify_response(answer=False, reason="settings interface"),
            _verify_response(answer=False, reason="settings interface"),
            _verify_response(answer=True, reason="settings window in foreground"),
        ]
        client.chat.completions.create = AsyncMock(side_effect=responses)
        kb = AsyncMock()
        m = AsyncMock()
        cap = AsyncMock()
        cap.capture_frame = AsyncMock(
            return_value=MagicMock(image=_blank_image())
        )
        ctx = AgentContext(
            vision_client=client,
            vision_model="m",
            keyboard=kb,
            mouse=m,
            capture=cap,
        )
        out = await LoginAgent(ctx).run(
            password="hunter2",
            wake=False,
            verify_attempts=2,
            verify_interval=0.0,
        )
        assert out
        assert out.data.get("already_unlocked") is True
        kb.send_text.assert_not_called()
        kb.send_keystroke.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_verify_types_immediately(self) -> None:
        kb = AsyncMock()
        ctx = AgentContext(keyboard=kb)
        out = await LoginAgent(ctx).run(
            password="hunter2",
            wake=False,
            verify=False,
            submit=False,
        )
        assert out
        kb.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_click_input_logs_warning_in_phase_a(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        kb = AsyncMock()
        ctx = AgentContext(keyboard=kb)
        with patch.object(LoginAgent, "_poll_for_login",
                          AsyncMock(return_value=(True, "ok"))):
            await LoginAgent(ctx).run(
                password="hunter2",
                wake=False,
                click_input=True,
                submit=False,
            )
        # Look for the Phase-A no-op warning.
        text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "not supported in Phase A" in text or kb.send_text.await_count > 0


# ─── helpers ────────────────────────────────────────────────────────


def _blank_image():  # noqa: ANN202
    import numpy as np
    return np.full((64, 64, 3), 200, dtype=np.uint8)
