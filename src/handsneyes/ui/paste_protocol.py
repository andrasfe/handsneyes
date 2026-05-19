# mypy: ignore-errors
# ruff: noqa
# Ported verbatim from terminaleyes/commandcenter; lint cleanup deferred.
"""Wire protocol for paste-file integrity verification.

Pure-logic helpers — no I/O, no OCR. The endpoint code in
``server.py`` drives the keyboard / capture; this module builds the
host-side commands, parses the OCR'd output, and computes the
local digest set so the diff can be done in-process.

## Protocol

After the file body has been written via ``cat > path``, the endpoint
asks the host to print integrity tokens framed by unique sentinels so
that a single OCR pass can locate them in arbitrary terminal noise.

### Whole-file SHA-256

Host command::

    echo "===SHA:"; shasum -a 256 PATH 2>/dev/null | awk '{print $1}'; \\
        echo ":SHA==="

OCR target — one short ASCII line of 64 lowercase hex chars between
the framing tokens. Lowercase hex has no homoglyphs (no ``OoIlSs``
characters), and tesseract is reliable on this charset with a
whitelist + single-line PSM.

### Per-chunk MD5 (used only for repair)

If the whole-file SHA doesn't match, the endpoint runs a chunk-level
diff to identify exactly which 2 KB blocks differ, and only those
need to be retransmitted::

    echo "===CHUNKS:"; N=2048; n=NN
    for i in $(seq 0 $((n-1))); do
      printf "%d " $i
      dd if=PATH bs=$N skip=$i count=1 2>/dev/null \\
        | openssl md5 2>/dev/null | awk '{print $NF}'
    done
    echo ":CHUNKS==="

Each line is ``IDX HEX32`` between framing tokens. We parse them
into ``{idx: hash}`` and compare to the local ``chunk_hashes``.

### Repair (per-chunk overwrite)

For each chunk index that differs, retype the chunk into place::

    printf '%s' '<BASE64>' | base64 -d \\
      | dd of=PATH bs=2048 seek=IDX conv=notrunc 2>/dev/null

Base64 is the safe carrier: charset ``[A-Za-z0-9+/=]`` requires no
shell escaping, and ``base64 -d`` exists on macOS, Linux, BSD.

Loop until whole-file SHA matches, or max repair rounds.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass


# Default chunk size. 2048 keeps the per-chunk-hash output to a small
# number of lines (max 25 for a 50 KB file) so a single OCR pass on a
# maximised terminal captures the whole list. Larger chunks = fewer
# lines but bigger retransmits per bad chunk.
CHUNK_SIZE = 2048

# Framing markers chosen to be unlikely to collide with terminal noise
# (typed file content, shell prompts, ANSI escapes).
SHA_OPEN = "===SHA:"
SHA_CLOSE = ":SHA==="
CHUNKS_OPEN = "===CHUNKS:"
CHUNKS_CLOSE = ":CHUNKS==="

# OCR-tolerant regex — allows uppercase even though the command emits
# lowercase, because tesseract may misclassify some chars. We lower()
# the captured group before comparing.
_SHA_RE = re.compile(
    rf"{re.escape(SHA_OPEN)}\s*([0-9a-fA-F]{{64}})\s*{re.escape(SHA_CLOSE)}",
    re.DOTALL,
)

_CHUNKS_BLOCK_RE = re.compile(
    rf"{re.escape(CHUNKS_OPEN)}(.*?){re.escape(CHUNKS_CLOSE)}",
    re.DOTALL,
)

# A chunk line: "IDX HEX32". We allow a bit of OCR slop (extra
# whitespace, stray chars at the edges).
_CHUNK_LINE_RE = re.compile(
    r"(?:^|\s)(\d{1,4})\s+([0-9a-fA-F]{32})(?=\s|$)",
    re.MULTILINE,
)


# ────────────────────────── digests ──────────────────────────

def file_sha256(content: bytes) -> str:
    """Lowercase hex SHA-256 of the file payload."""
    return hashlib.sha256(content).hexdigest()


def chunk_hashes(content: bytes, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Per-chunk MD5 hex (32 lowercase chars). MD5 is fine here — we
    only need a low-collision check to identify *which* chunks differ;
    the overall file is still SHA-256-verified."""
    out: list[str] = []
    n = len(content)
    if n == 0:
        return out
    for i in range(0, n, chunk_size):
        chunk = content[i : i + chunk_size]
        out.append(hashlib.md5(chunk).hexdigest())
    return out


def n_chunks(content_len: int, chunk_size: int = CHUNK_SIZE) -> int:
    if content_len <= 0:
        return 0
    return (content_len + chunk_size - 1) // chunk_size


# ────────────────────── host-side commands ─────────────────

def cmd_sha_print(path: str) -> str:
    """Shell command that prints the whole-file SHA-256 between
    SHA_OPEN/SHA_CLOSE framing markers, each on its own line.

    Cross-platform: ``shasum -a 256`` works on macOS (perl shasum)
    and most Linux distros; ``sha256sum`` is Linux-only and would
    have to be auto-detected, which we don't bother with for v1.
    """
    return (
        f'echo "{SHA_OPEN}"; '
        f"shasum -a 256 {path} 2>/dev/null | awk '{{print $1}}'; "
        f'echo "{SHA_CLOSE}"'
    )


def cmd_chunks_print(
    path: str, n: int, chunk_size: int = CHUNK_SIZE,
) -> str:
    """Shell loop that prints per-chunk MD5 hashes framed."""
    return (
        f'echo "{CHUNKS_OPEN}"; '
        f"N={chunk_size}; n={n}; "
        f"for i in $(seq 0 $((n-1))); do "
        f'printf "%d " $i; '
        f"dd if={path} bs=$N skip=$i count=1 2>/dev/null "
        f"| openssl md5 2>/dev/null | awk '{{print $NF}}'; "
        f"done; "
        f'echo "{CHUNKS_CLOSE}"'
    )


def cmd_overwrite_chunk(
    path: str,
    idx: int,
    payload: bytes,
    chunk_size: int = CHUNK_SIZE,
) -> str:
    """Shell command that overwrites chunk ``idx`` in ``path`` with
    ``payload``. Uses base64 + ``dd seek=`` to avoid shell-escaping
    binary content and to scribble in place rather than rewriting
    the whole file.

    ``conv=notrunc`` is critical — without it, ``dd of=PATH`` would
    truncate the file to the end of the written block.
    """
    b64 = base64.b64encode(payload).decode("ascii")
    return (
        f"printf '%s' '{b64}' | base64 -d "
        f"| dd of={path} bs={chunk_size} seek={idx} "
        f"conv=notrunc 2>/dev/null"
    )


# ───────────────────────── OCR parsing ─────────────────────

def parse_sha_from_ocr(ocr_text: str) -> str | None:
    """Pull the framed SHA-256 hex out of an OCR pass. Returns
    lowercase hex or ``None`` if the framing/length didn't match.

    ``findall`` + ``[-1]`` rather than ``search`` because the host's
    shell echoes the typed command line, which itself contains both
    framing tokens but no inner hex — the *real* output framing comes
    after, and we want the last well-formed match.
    """
    if not ocr_text:
        return None
    hits = _SHA_RE.findall(ocr_text)
    if not hits:
        return None
    return hits[-1].lower()


def parse_chunks_from_ocr(ocr_text: str) -> dict[int, str]:
    """Extract ``{idx: md5_hex_lowercase}`` from the chunks block of
    an OCR pass. Tolerates extra whitespace and stray characters
    between lines; rejects malformed lines silently.

    The host's shell echoes the typed command line (which contains
    both framing tokens with no chunk lines between them) before the
    real output, so we gather ALL framed blocks and parse over the
    concatenation — the empty command-echo block contributes nothing
    and the real block contributes everything.
    """
    out: dict[int, str] = {}
    if not ocr_text:
        return out
    blocks = _CHUNKS_BLOCK_RE.findall(ocr_text)
    text = "\n".join(blocks) if blocks else ocr_text
    for m in _CHUNK_LINE_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        out[idx] = m.group(2).lower()
    return out


# ─────────────────────── diff ──────────────────────────────

@dataclass
class ChunkDiff:
    bad_indices: list[int]      # chunks whose host hash != local
    unknown_indices: list[int]  # chunks the OCR didn't return at all


def diff_chunks(
    local_hashes: list[str], host_hashes: dict[int, str],
) -> ChunkDiff:
    """Compare per-chunk local hashes against what we OCR'd from the
    host. ``unknown_indices`` are chunks the OCR couldn't read at all
    (typically off-screen rows on a too-tall output block). Those
    should be retransmitted defensively in a repair round, since
    they may or may not actually be wrong."""
    bad: list[int] = []
    unknown: list[int] = []
    for i, local in enumerate(local_hashes):
        host = host_hashes.get(i)
        if host is None:
            unknown.append(i)
        elif host != local:
            bad.append(i)
    return ChunkDiff(bad_indices=bad, unknown_indices=unknown)
