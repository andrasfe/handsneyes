"""handsneyes command-line interface.

Phase A subcommands:

  - ``handsneyes do <intent> [--target NAME] [--dry-run]`` — Plan an
    intent and (when --dry-run) print the plan as JSON. Non-dry-run
    is not yet wired to executors — Phase B brings that up.
  - ``handsneyes platforms`` — list registered adapters.
  - ``handsneyes vault {add,get,list,remove,status}`` — credential
    store passthrough to :class:`Vault`.
  - ``handsneyes version`` — print package version.

The CLI deliberately stays thin: it shells out to ``core/agents/*``
and ``platforms.load_adapter`` for the real work, so most surface
testing happens via the library tests in ``tests/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import TYPE_CHECKING

import handsneyes
from handsneyes.core.agents.controller import plan_intent
from handsneyes.platforms import (
    UnknownPlatformError,
    available_platforms,
    load_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="handsneyes",
        description=(
            "Vision-based agentic terminal controller. Phase A scope: "
            "rule-planner, dry-run, vault, and platform adapter listing."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"handsneyes {handsneyes.__version__}",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── do ─────────────────────────────────────────────────────────
    do = sub.add_parser(
        "do",
        help=(
            "Plan (and, in later phases, execute) a free-form intent."
        ),
    )
    do.add_argument(
        "intent",
        help="Free-form intent, e.g. 'scroll down 6'",
    )
    do.add_argument(
        "--target",
        default="headless",
        help=(
            "Target name (looks up the platform adapter). Default: "
            "headless."
        ),
    )
    do.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan as JSON and exit without executing.",
    )

    # ── platforms ──────────────────────────────────────────────────
    sub.add_parser(
        "platforms",
        help="List registered platform adapters.",
    )

    # ── targets ────────────────────────────────────────────────────
    sub.add_parser(
        "targets",
        help="List configured targets (from targets.toml).",
    )

    # ── vault ──────────────────────────────────────────────────────
    v = sub.add_parser("vault", help="Credential vault operations.")
    vsub = v.add_subparsers(dest="vault_command", required=True)
    vadd = vsub.add_parser("add", help="Add or update an entry.")
    vadd.add_argument("name")
    vget = vsub.add_parser("get", help="Read an entry.")
    vget.add_argument("name")
    vsub.add_parser("list", help="List entry names.")
    vrm = vsub.add_parser("remove", help="Remove an entry.")
    vrm.add_argument("name")
    vsub.add_parser("status", help="Show vault path + entry count.")

    # ── commandcenter / cc ────────────────────────────────────────
    cc = sub.add_parser(
        "commandcenter",
        aliases=["cc"],
        help="Start the Command Center web UI (FastAPI on LAN).",
    )
    cc.add_argument("--host", default="0.0.0.0")
    cc.add_argument("--port", type=int, default=8765)
    cc.add_argument(
        "--frames-dir",
        default=None,
        help=(
            "Watch directory for captured frames. Default: "
            "$HANDSNEYES_OUTPUT_DIR or "
            "~/.local/share/handsneyes/runs/"
        ),
    )
    cc.add_argument("--max-frames", type=int, default=500)

    # ── version (also via --version) ──────────────────────────────
    sub.add_parser("version", help="Print handsneyes version.")
    return p


# ───────────────────── command handlers ─────────────────────


def _cmd_do(args: argparse.Namespace) -> int:
    # Resolve --target through the targets registry first. The argument
    # may name either a configured target (in which case we use its
    # platform) or a bare platform adapter name (back-compat with
    # Phase A — "headless" is both a default target and a platform).
    from handsneyes.targets import TargetRegistry

    registry = TargetRegistry.load_default()
    target_name = args.target
    if target_name in registry.targets:
        target = registry.get(target_name)
        platform_name = target.platform
    else:
        target = None
        platform_name = target_name

    try:
        adapter = load_adapter(platform_name)
    except UnknownPlatformError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    plan = plan_intent(args.intent)
    payload: dict[str, object] = {
        "intent": args.intent,
        "target": target_name,
        "platform": adapter.name,
        "platform_display": adapter.display_name,
        "plan": [step.as_dict() for step in plan],
        "executable": False,
    }
    if target is not None:
        payload["target_config"] = {
            "pi_url": target.pi_url,
            "transport": target.transport,
            "camera_index": target.camera_index,
            "screen_size": list(target.screen_size),
        }
    if not plan:
        payload["error"] = (
            "no rule matched this intent — Phase B will add LLM "
            "fallback. For now, try: 'scroll down 6', 'type hello', "
            "'login', or 'lock'."
        )
    if args.dry_run or not plan:
        print(json.dumps(payload, indent=2))
        return 0 if plan else 1

    # ── Non-dry-run execution ──────────────────────────────────────
    # Build an AgentContext from the resolved target + adapter, then
    # hand the plan to PlanExecutor. Headless targets have no HID /
    # capture / vision client, so the executor will report graceful
    # per-step "no X in context" outcomes — the path runs without
    # raising.
    import asyncio

    from handsneyes.core.agents.context import AgentContext
    from handsneyes.core.agents.executor import PlanExecutor
    from handsneyes.io.keyboard import HttpKeyboardOutput, PlatformKeyboard
    from handsneyes.io.mouse import HttpMouseOutput

    keyboard: object | None = None
    mouse: object | None = None
    if target is not None and target.platform != "headless":
        raw_kb = HttpKeyboardOutput(
            base_url=target.pi_url,
            transport=target.transport,
        )
        keyboard = PlatformKeyboard(raw_kb, adapter)
        mouse = HttpMouseOutput(
            base_url=target.pi_url,
            transport=target.transport,
        )
    ctx = AgentContext(
        keyboard=keyboard,  # type: ignore[arg-type]
        mouse=mouse,  # type: ignore[arg-type]
        platform=adapter,
    )

    async def _go() -> int:
        if keyboard is not None and hasattr(keyboard, "connect"):
            try:
                await keyboard.connect()
            except Exception as e:  # noqa: BLE001
                print(
                    f"warning: keyboard connect failed: {e}",
                    file=sys.stderr,
                )
        if mouse is not None and hasattr(mouse, "connect"):
            try:
                await mouse.connect()
            except Exception as e:  # noqa: BLE001
                print(
                    f"warning: mouse connect failed: {e}",
                    file=sys.stderr,
                )
        executor = PlanExecutor(ctx)
        results = await executor.run(plan)
        if keyboard is not None and hasattr(keyboard, "disconnect"):
            with contextlib.suppress(Exception):
                await keyboard.disconnect()
        if mouse is not None and hasattr(mouse, "disconnect"):
            with contextlib.suppress(Exception):
                await mouse.disconnect()
        payload["results"] = [r.as_dict() for r in results]
        payload["executable"] = True
        all_ok = all(r.outcome.success for r in results)
        payload["success"] = all_ok
        print(json.dumps(payload, indent=2))
        return 0 if all_ok else 1

    return asyncio.run(_go())


def _cmd_platforms(_: argparse.Namespace) -> int:
    names = available_platforms()
    if not names:
        print("(no platforms registered)")
        return 0
    for n in names:
        try:
            ad = load_adapter(n)
            print(f"{n}\t{ad.display_name}")
        except UnknownPlatformError:
            print(f"{n}\t(unloadable)")
    return 0


def _cmd_vault(args: argparse.Namespace) -> int:
    from handsneyes.core.vault import (
        Vault,
        VaultError,
        VaultPassphraseError,
        get_passphrase,
    )

    try:
        passphrase = get_passphrase()
        vault = Vault(passphrase)
    except VaultError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cmd = args.vault_command
    try:
        if cmd == "add":
            import getpass as gp
            value = gp.getpass(f"Value for {args.name!r}: ")
            vault.set(args.name, value)
            print(f"stored {args.name!r}")
            return 0
        if cmd == "get":
            print(vault.get(args.name))
            return 0
        if cmd == "list":
            for name in vault.names():
                print(name)
            return 0
        if cmd == "remove":
            existed = vault.remove(args.name)
            print(f"{args.name!r}: {'removed' if existed else 'not found'}")
            return 0 if existed else 1
        if cmd == "status":
            s = vault.status()
            print(f"backend     {s.backend}")
            print(f"path        {s.path}")
            print(f"exists      {s.exists}")
            print(
                f"entries     "
                f"{'?' if s.entry_count is None else s.entry_count}"
            )
            return 0
    except VaultPassphraseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"handsneyes {handsneyes.__version__}")
    return 0


def _cmd_targets(_: argparse.Namespace) -> int:
    from handsneyes.targets import TargetRegistry

    reg = TargetRegistry.load_default()
    if not reg.targets:
        print("(no targets configured)")
        return 0
    source = str(reg.source) if reg.source else "(built-in default)"
    print(f"# source: {source}")
    for name in reg.names():
        t = reg.get(name)
        suffix = f"  ({t.description})" if t.description else ""
        print(
            f"{name}\tplatform={t.platform}\tpi={t.pi_url}\t"
            f"transport={t.transport}\tcam={t.camera_index}{suffix}"
        )
    return 0


def _cmd_commandcenter(args: argparse.Namespace) -> int:
    """Start the Command Center FastAPI server.

    Resolves the active target from the registry (first configured
    target, or the headless default) and builds a per-run factory
    that constructs an AgentContext with that target's HID + camera.
    """
    from pathlib import Path

    import uvicorn

    from handsneyes.targets import TargetRegistry
    from handsneyes.ui.factory import make_target_context_factory
    from handsneyes.ui.frame_store import DEFAULT_WATCH_DIR, FrameStore
    from handsneyes.ui.log_bus import LogBus
    from handsneyes.ui.server import create_app

    registry = TargetRegistry.load_default()
    # Pick the first non-headless target if one exists; otherwise
    # headless. For multi-target switching we'll need a UI control;
    # Phase C ships single-target.
    names = registry.names()
    chosen = next(
        (n for n in names if registry.get(n).platform != "headless"),
        names[0] if names else "headless",
    )
    target = registry.get(chosen)
    try:
        adapter = load_adapter(target.platform)
    except UnknownPlatformError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(
        f"handsneyes cc: target={chosen!r}, platform={target.platform}, "
        f"pi={target.pi_url}",
        file=sys.stderr,
    )

    watch_dir = (
        Path(args.frames_dir).expanduser().resolve()
        if args.frames_dir else DEFAULT_WATCH_DIR
    )
    watch_dir.mkdir(parents=True, exist_ok=True)

    store = FrameStore(watch_dir=watch_dir, max_frames=args.max_frames)
    bus = LogBus()
    context_factory = make_target_context_factory(
        target, adapter, base_dir=watch_dir, bus=bus,
    )
    app = create_app(
        context_factory, frame_store=store, bus=bus,
    )

    print(
        f"handsneyes Command Center → http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


_HANDLERS = {
    "do": _cmd_do,
    "platforms": _cmd_platforms,
    "targets": _cmd_targets,
    "vault": _cmd_vault,
    "version": _cmd_version,
    "commandcenter": _cmd_commandcenter,
    "cc": _cmd_commandcenter,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
