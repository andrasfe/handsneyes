"""Encrypted local vault for credentials and other small secrets.

Format on disk (``~/.config/handsneyes/vault.enc``, mode 0600):

    +-------------------+----------+----------+----------------+
    | magic 8 b "HEVAULT" | salt 16 | nonce 12 | AES-GCM blob |
    +-------------------+----------+----------+----------------+

Crypto: scrypt KDF (N=2**15, r=8, p=1, length=32) → AES-256-GCM. The
plaintext is JSON ``{"name": "value", ...}``. The 16-byte GCM tag is
appended to the ciphertext by the AES-GCM implementation, so an
attacker who tampers with the file gets ``InvalidTag`` on decryption.

Master passphrase sources (priority): ``HANDSNEYES_VAULT_PASSPHRASE``
env var (intended for scripting; warn the user) > ``getpass.getpass``
prompt. The passphrase is held in memory for the lifetime of the
:class:`Vault` instance and never written to disk.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

logger = logging.getLogger(__name__)


MAGIC = b"HEVAULT1"  # 8 bytes — handsneyes magic
# terminaleyes used "TEVAULT1" with otherwise-identical format. Read
# both so an existing terminaleyes vault works without migration; new
# writes always use HEVAULT1.
_LEGACY_MAGICS = (b"TEVAULT1",)
SALT_LEN = 16
NONCE_LEN = 12
KEY_LEN = 32
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1


def _default_path() -> Path:
    """Resolve the vault file. Prefer the handsneyes path; fall back
    to a pre-existing terminaleyes vault if the handsneyes file isn't
    there yet (one-time migration: first write at the handsneyes path
    leaves the terminaleyes file untouched as a backup)."""
    primary = Path.home() / ".config" / "handsneyes" / "vault.enc"
    if primary.exists():
        return primary
    legacy = Path.home() / ".config" / "terminaleyes" / "vault.enc"
    if legacy.exists():
        return legacy
    return primary


DEFAULT_PATH = _default_path()
DEFAULT_DIR_MODE = 0o700
DEFAULT_FILE_MODE = 0o600


class VaultError(Exception):
    """Raised for any vault failure (bad passphrase, corrupt file, etc.)."""


class VaultPassphraseError(VaultError):
    """Raised when the master passphrase is wrong (decryption fails)."""


def get_passphrase(*, prompt: str = "Vault passphrase: ") -> str:
    """Resolve the master passphrase from env or interactive prompt.

    Order:
      1. ``HANDSNEYES_VAULT_PASSPHRASE`` env var (warn — leaks via
         the process env to anyone with ``ps -e ww`` or /proc access).
      2. ``TERMINALEYES_VAULT_PASSPHRASE`` env var (back-compat with
         the legacy terminaleyes setup so an existing operator env
         keeps working without renaming the variable).
      3. ``getpass.getpass`` prompt.

    Empty strings are treated as "not set" — pass an empty value
    explicitly to :class:`Vault` if that's what you want.
    """
    for var in ("HANDSNEYES_VAULT_PASSPHRASE", "TERMINALEYES_VAULT_PASSPHRASE"):
        env = os.environ.get(var)
        if env:
            logger.warning(
                "Using %s from environment — fine for scripting but "
                "visible to other processes on this host.", var,
            )
            return env
    return getpass.getpass(prompt)


@dataclass
class VaultStatus:
    """Lightweight description of the vault for ``vault status``."""

    backend: str
    path: Path
    exists: bool
    entry_count: int | None  # None = couldn't decrypt (no passphrase)


class Vault:
    """File-backed AES-GCM vault.

    Use :meth:`get`, :meth:`set`, :meth:`remove`, :meth:`names` for
    typical operations. The first call after construction loads and
    decrypts the file; the plaintext is cached for the lifetime of the
    instance. Always treats the passphrase as opaque — never logs it.
    """

    def __init__(
        self,
        passphrase: str,
        *,
        path: Path | None = None,
    ) -> None:
        if not passphrase:
            raise VaultError("Vault passphrase must not be empty")
        self._passphrase = passphrase
        self._path = path or DEFAULT_PATH
        self._cache: dict[str, str] | None = None

    # ───────────────────── public API ─────────────────────

    def get(self, name: str) -> str:
        """Return the value stored under ``name``. Raises ``KeyError``."""
        data = self._load()
        if name not in data:
            raise KeyError(f"Vault has no entry named {name!r}")
        return data[name]

    def set(self, name: str, value: str) -> None:
        """Store/overwrite the value under ``name``."""
        if not isinstance(name, str) or not name:
            raise VaultError("Vault entry name must be a non-empty string")
        data = self._load()
        data[name] = value
        self._save(data)

    def remove(self, name: str) -> bool:
        """Delete ``name``. Returns True if it existed, False otherwise."""
        data = self._load()
        if name in data:
            del data[name]
            self._save(data)
            return True
        return False

    def names(self) -> list[str]:
        """Return sorted list of entry names. Never returns values."""
        return sorted(self._load().keys())

    def status(self) -> VaultStatus:
        try:
            count = len(self._load())
        except Exception:
            count = None
        return VaultStatus(
            backend="file",
            path=self._path,
            exists=self._path.exists(),
            entry_count=count,
        )

    # ───────────────────── internals ─────────────────────

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = Scrypt(
            salt=salt,
            length=KEY_LEN,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
        )
        return kdf.derive(self._passphrase.encode("utf-8"))

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        blob = self._path.read_bytes()
        if len(blob) < len(MAGIC) + SALT_LEN + NONCE_LEN + 16:
            raise VaultError(
                f"Vault file {self._path} is too small to be valid"
            )
        head = blob[: len(MAGIC)]
        if head != MAGIC and head not in _LEGACY_MAGICS:
            raise VaultError(
                f"Vault file {self._path} has wrong magic header"
            )
        offset = len(MAGIC)
        salt = blob[offset : offset + SALT_LEN]
        offset += SALT_LEN
        nonce = blob[offset : offset + NONCE_LEN]
        offset += NONCE_LEN
        ciphertext = blob[offset:]
        try:
            key = self._derive_key(salt)
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        except InvalidTag as e:
            raise VaultPassphraseError(
                "Vault decryption failed — wrong passphrase or "
                "corrupted file"
            ) from e
        try:
            data = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise VaultError(
                f"Vault payload is not valid JSON: {e}"
            ) from e
        if not isinstance(data, dict):
            raise VaultError(
                f"Vault payload is {type(data).__name__}, expected dict"
            )
        self._cache = {str(k): str(v) for k, v in data.items()}
        return self._cache

    def _save(self, data: dict[str, str]) -> None:
        salt = secrets.token_bytes(SALT_LEN)
        nonce = secrets.token_bytes(NONCE_LEN)
        key = self._derive_key(salt)
        plaintext = json.dumps(data, ensure_ascii=False).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
        blob = MAGIC + salt + nonce + ciphertext

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=DEFAULT_DIR_MODE,
        )
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            DEFAULT_FILE_MODE,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise
        os.chmod(tmp_path, DEFAULT_FILE_MODE)
        os.replace(tmp_path, self._path)
        os.chmod(self._path, DEFAULT_FILE_MODE)
        self._cache = data


__all__ = [
    "Vault",
    "VaultError",
    "VaultPassphraseError",
    "VaultStatus",
    "get_passphrase",
]
