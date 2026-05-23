"""LoginAgent — wake screen, verify it's a login prompt, type password.

Compose-only agent. Builds on:

  - :class:`WakeAgent` — mouse jiggle / arrow / click
  - :class:`VerifyAgent` — visual yes/no oracle (polled until login
    screen is showing)
  - :class:`TypeAgent` — secret-mode text entry + optional Enter

Password sources, in priority order: explicit ``password=`` arg >
:class:`Vault` lookup by ``vault_name`` > file path > env var >
interactive ``getpass``. The agent never echoes or logs the value.

NOTE: terminaleyes additionally supports ``click_input=True``, which
uses the VisualServoHomer to click an input field visually before
typing. That path requires the full homer (tier-3 click engine),
which is deferred to Phase B. The argument is preserved on the run
signature for API stability and silently behaves as no-op for Phase A.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from handsneyes.core.agents.base import Agent, Outcome
from handsneyes.core.agents.type_text import TypeAgent
from handsneyes.core.agents.verify import VerifyAgent
from handsneyes.core.agents.wake import WakeAgent

logger = logging.getLogger(__name__)


LOGIN_QUESTION = (
    "Look at the screen. Decide whether it shows a LOGIN or PASSWORD "
    "entry screen, judging by visual structure ONLY — NOT by whether "
    "the literal word 'password' appears.\n\n"
    "Visual cues that count:\n"
    "  * a single prominent text input, often centred\n"
    "  * the input may show hidden-character dots or circles\n"
    "  * a user avatar / username\n"
    "  * a 'Sign in', 'Unlock', or 'Log in' button\n"
    "  * a system lock screen with a large clock\n"
    "  * dark or blurred background\n\n"
    "If the screen is a normal application, terminal, file manager, "
    "browser, etc., return false even if the word 'password' happens "
    "to be visible somewhere."
)


ALREADY_UNLOCKED_QUESTION = (
    "Look at the screen. Is this a normal UNLOCKED desktop or application "
    "interface — i.e. the user is already logged in and past any lock/login "
    "screen?\n\n"
    "Return TRUE for: a regular desktop with a taskbar/dock, an open "
    "application window (settings, browser, terminal, file manager, editor, "
    "etc.), a populated home screen.\n\n"
    "Return FALSE for: a lock screen, a login/password prompt, a screensaver, "
    "a black/asleep screen, a boot screen, or anything that obscures the "
    "logged-in session."
)


@dataclass
class LoginOutcome(Outcome):
    pass


def resolve_password(
    *,
    password: str | None = None,
    vault: object | None = None,
    vault_name: str | None = None,
    file_path: str | None = None,
    env_var: str | None = None,
) -> str:
    """Return the password from the chosen source.

    Priority: ``password`` (explicit) > ``vault_name`` (via ``vault``)
    > ``file_path`` > ``env_var`` > interactive ``getpass`` prompt.

    Never logs or prints the value.
    """
    if password is not None:
        return password
    if vault_name:
        if vault is None:
            from handsneyes.core.vault import Vault, get_passphrase
            passphrase = get_passphrase(prompt="Vault passphrase: ")
            vault = Vault(passphrase)
        try:
            return str(vault.get(vault_name))  # type: ignore[attr-defined]
        except KeyError as e:
            raise ValueError(
                f"Vault has no entry named {vault_name!r}. "
                "Use `handsneyes vault add` first."
            ) from e
    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Password file not found: {path}")
        pw = path.read_text()
        if pw.endswith("\r\n"):
            pw = pw[:-2]
        elif pw.endswith("\n"):
            pw = pw[:-1]
        return pw
    if env_var:
        val = os.environ.get(env_var)
        if val is None:
            raise ValueError(
                f"Environment variable {env_var!r} is not set"
            )
        return val
    return getpass.getpass("Remote password: ")


class LoginAgent(Agent):
    """Wake → poll-verify login screen → type password → submit."""

    name = "login"

    async def run(  # type: ignore[override]
        self,
        *,
        password: str | None = None,
        vault_name: str | None = None,
        password_file: str | None = None,
        password_env: str | None = None,
        wake: bool = True,
        verify: bool = True,
        verify_attempts: int = 6,
        verify_interval: float = 1.0,
        click_input: bool = False,  # accepted for API parity; no-op in Phase A
        submit: bool = True,
    ) -> LoginOutcome:
        # 0. Resolve password BEFORE any I/O so a bad source fails fast.
        try:
            pw = resolve_password(
                password=password,
                vault=self.ctx.vault,
                vault_name=vault_name,
                file_path=password_file,
                env_var=password_env,
            )
        except Exception as e:
            return LoginOutcome(
                success=False,
                reason=f"password resolution failed: {e}",
            )
        if not pw:
            return LoginOutcome(
                success=False,
                reason="empty password — refusing to send",
            )

        # 1. Wake.
        if wake:
            wake_outcome = await WakeAgent(self.ctx).run()
            if not wake_outcome:
                logger.warning(
                    "Wake step reported failure: %s", wake_outcome.reason
                )

        # 2. Polled visual verification.
        if verify:
            ok, reason = await self._poll_for_login(
                attempts=verify_attempts, interval=verify_interval,
            )
            if not ok:
                # Polling never saw a login screen. Before failing, check
                # whether the target is already unlocked — if so, the unlock
                # task is a no-op success.
                already_unlocked = await VerifyAgent(self.ctx).run(
                    question=ALREADY_UNLOCKED_QUESTION, visual_only=True,
                )
                if already_unlocked:
                    logger.info(
                        "LoginAgent: target already unlocked — skipping "
                        "password entry (%s)", already_unlocked.reason,
                    )
                    return LoginOutcome(
                        success=True,
                        reason="already unlocked — no password sent",
                        data={
                            "verified": True,
                            "already_unlocked": True,
                            "submit": False,
                        },
                    )
                return LoginOutcome(
                    success=False,
                    reason=(
                        f"visual verification: not a login screen "
                        f"after {verify_attempts} polls ({reason})"
                    ),
                    data={"verified": False},
                )

        if click_input:
            logger.warning(
                "click_input=True is not supported in Phase A "
                "(requires VisualServoHomer, deferred to Phase B). "
                "Relying on the lock screen's auto-focus instead."
            )

        # 4. Type and submit.
        type_outcome = await TypeAgent(self.ctx).run(
            text=pw, secret=True, submit=submit,
        )
        # Drop the password reference promptly.
        del pw
        if not type_outcome:
            return LoginOutcome(
                success=False,
                reason=f"type step failed: {type_outcome.reason}",
            )
        return LoginOutcome(
            success=True,
            reason="login submitted",
            data={"submit": submit, "verified": verify},
        )

    # ───────────────────── helpers ─────────────────────

    async def _poll_for_login(
        self, *, attempts: int, interval: float,
    ) -> tuple[bool, str]:
        verifier = VerifyAgent(self.ctx)
        last_reason = ""
        for i in range(1, attempts + 1):
            logger.info("LoginAgent: verification attempt %d/%d", i, attempts)
            v = await verifier.run(
                question=LOGIN_QUESTION, visual_only=True,
            )
            last_reason = v.reason
            if v:
                return True, v.reason
            if i == attempts:
                break
            # Nudge between polls. Alternate mouse + arrow.
            try:
                if i % 2 == 1 and self.ctx.mouse is not None:
                    await self.ctx.mouse.move(20, 0)
                    await asyncio.sleep(0.05)
                    await self.ctx.mouse.move(-20, 0)
                elif self.ctx.keyboard is not None:
                    await self.ctx.keyboard.send_keystroke("Down")
            except Exception as e:
                logger.warning("Nudge between polls failed: %s", e)
            await asyncio.sleep(interval)
        return False, last_reason
