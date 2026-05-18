"""Tests for handsneyes.core.vault."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from handsneyes.core.vault import Vault, VaultError, VaultPassphraseError

if TYPE_CHECKING:
    from pathlib import Path


def _new_vault(path: Path, passphrase: str = "correct-horse-battery") -> Vault:
    return Vault(passphrase, path=path / "vault.enc")


def test_passphrase_required() -> None:
    with pytest.raises(VaultError, match="must not be empty"):
        Vault("")


def test_set_then_get_round_trip(tmp_path: Path) -> None:
    v = _new_vault(tmp_path)
    v.set("desktop", "hunter2")
    fresh = _new_vault(tmp_path)
    assert fresh.get("desktop") == "hunter2"


def test_get_missing_raises_keyerror(tmp_path: Path) -> None:
    v = _new_vault(tmp_path)
    v.set("desktop", "hunter2")
    with pytest.raises(KeyError, match="ghost"):
        v.get("ghost")


def test_wrong_passphrase_raises(tmp_path: Path) -> None:
    v = _new_vault(tmp_path, passphrase="alpha")
    v.set("desktop", "hunter2")
    wrong = Vault("beta", path=tmp_path / "vault.enc")
    with pytest.raises(VaultPassphraseError):
        wrong.get("desktop")


def test_remove_returns_true_when_present(tmp_path: Path) -> None:
    v = _new_vault(tmp_path)
    v.set("desktop", "hunter2")
    assert v.remove("desktop") is True
    assert v.remove("desktop") is False


def test_names_returns_sorted_entry_names_only(tmp_path: Path) -> None:
    v = _new_vault(tmp_path)
    v.set("zeta", "z")
    v.set("alpha", "a")
    v.set("mu", "m")
    assert v.names() == ["alpha", "mu", "zeta"]


def test_set_rejects_empty_name(tmp_path: Path) -> None:
    v = _new_vault(tmp_path)
    with pytest.raises(VaultError, match="non-empty"):
        v.set("", "value")


def test_status_when_no_file(tmp_path: Path) -> None:
    v = _new_vault(tmp_path)
    status = v.status()
    assert status.exists is False
    assert status.entry_count == 0


def test_file_mode_is_0600(tmp_path: Path) -> None:
    import stat

    v = _new_vault(tmp_path)
    v.set("k", "v")
    mode = stat.S_IMODE((tmp_path / "vault.enc").stat().st_mode)
    assert mode == 0o600
