"""Tests for the multi-host targets registry."""

from __future__ import annotations

import pytest

from handsneyes.targets import Target, TargetRegistry


def test_target_requires_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Target(name="")


def test_target_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError, match="transport"):
        Target(name="x", transport="serial")  # type: ignore[arg-type]


def test_target_defaults() -> None:
    t = Target(name="x")
    assert t.platform == "headless"
    assert t.transport == "bt"
    assert t.screen_size == (1920, 1080)


# ─── TOML loader ────────────────────────────────────────────────────


def _write(path: object, body: str) -> object:  # noqa: ANN001
    path.write_text(body)
    return path


def test_load_from_file_minimal(tmp_path) -> None:  # noqa: ANN001
    f = _write(tmp_path / "t.toml", """
[[target]]
name = "couch"
platform = "linux_gnome"
camera_index = 0
pi_url = "http://10.0.0.42:8080"
transport = "bt"
screen_size = [1920, 1080]
""")
    reg = TargetRegistry.from_file(f)  # type: ignore[arg-type]
    assert reg.names() == ["couch"]
    t = reg.get("couch")
    assert t.platform == "linux_gnome"
    assert t.pi_url == "http://10.0.0.42:8080"


def test_load_multiple(tmp_path) -> None:  # noqa: ANN001
    f = _write(tmp_path / "t.toml", """
[[target]]
name = "couch"
platform = "linux_gnome"

[[target]]
name = "studio-mac"
platform = "macos"
transport = "usb"
screen_size = [2560, 1664]
description = "M2 mac mini"
""")
    reg = TargetRegistry.from_file(f)  # type: ignore[arg-type]
    assert reg.names() == ["couch", "studio-mac"]
    mac = reg.get("studio-mac")
    assert mac.platform == "macos"
    assert mac.transport == "usb"
    assert mac.screen_size == (2560, 1664)
    assert mac.description == "M2 mac mini"


def test_load_missing_name_raises(tmp_path) -> None:  # noqa: ANN001
    f = _write(tmp_path / "t.toml", """
[[target]]
platform = "linux_gnome"
""")
    with pytest.raises(ValueError, match="name"):
        TargetRegistry.from_file(f)  # type: ignore[arg-type]


def test_load_duplicate_name_raises(tmp_path) -> None:  # noqa: ANN001
    f = _write(tmp_path / "t.toml", """
[[target]]
name = "couch"

[[target]]
name = "couch"
""")
    with pytest.raises(ValueError, match="duplicate"):
        TargetRegistry.from_file(f)  # type: ignore[arg-type]


def test_load_default_fallback_to_headless(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,  # noqa: ANN001
) -> None:
    # Make sure no local config interferes by pointing at a non-existent
    # env override + monkeypatching the default search paths.
    monkeypatch.delenv("HANDSNEYES_TARGETS_FILE", raising=False)
    monkeypatch.setattr(
        "handsneyes.targets._DEFAULT_CONFIG_PATHS",
        (tmp_path / "nope1.toml", tmp_path / "nope2.toml"),
    )
    reg = TargetRegistry.load_default()
    assert reg.names() == ["headless"]
    assert reg.get("headless").platform == "headless"


def test_load_default_env_override(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = _write(tmp_path / "envt.toml", """
[[target]]
name = "via-env"
platform = "macos"
""")
    monkeypatch.setenv("HANDSNEYES_TARGETS_FILE", str(f))
    reg = TargetRegistry.load_default()
    assert "via-env" in reg.names()


def test_get_unknown_raises_helpful(tmp_path) -> None:  # noqa: ANN001
    f = _write(tmp_path / "t.toml", """
[[target]]
name = "couch"
""")
    reg = TargetRegistry.from_file(f)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="Available"):
        reg.get("nope")
