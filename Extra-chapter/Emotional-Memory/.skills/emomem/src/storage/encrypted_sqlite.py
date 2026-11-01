"""Fail-closed SQLCipher connection and one-time plaintext migration.

The packaged Electron parent owns the DPAPI-wrapped database key.  Python
receives the raw 32-byte key once through an inherited anonymous pipe before
opening any durable SQLite file.  Development tests may omit that bootstrap
and keep using disposable plaintext databases.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import sqlite3 as standard_sqlite
import stat as stat_module
import threading
from typing import Any
from uuid import uuid4


SQLITE_HEADER = b"SQLite format 3\x00"
STORAGE_KEY_BYTES = 32
STORAGE_KEY_FD_ENV = "REVERIE_STORAGE_KEY_FD"
REQUIRE_ENCRYPTION_ENV = "REVERIE_REQUIRE_ENCRYPTED_STORAGE"


class EncryptedStorageError(RuntimeError):
    """The durable database could not be opened without weakening security."""


_state_lock = threading.RLock()
_storage_key: bytearray | None = None
_encryption_required = False


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_key(key: bytes | bytearray | memoryview) -> bytes:
    value = bytes(key)
    if len(value) != STORAGE_KEY_BYTES:
        raise EncryptedStorageError("database key must contain exactly 32 random bytes")
    return value


def install_storage_key(
    key: bytes | bytearray | memoryview,
    *,
    require_encryption: bool = True,
) -> None:
    """Install process-local key material before constructing any store."""

    value = _validate_key(key)
    global _storage_key, _encryption_required
    with _state_lock:
        if _storage_key is not None and bytes(_storage_key) != value:
            raise EncryptedStorageError("database key was already initialized")
        _storage_key = bytearray(value)
        _encryption_required = bool(require_encryption)


def clear_storage_key_for_testing() -> None:
    """Zero process-local key material; intended only for isolated tests."""

    global _storage_key, _encryption_required
    with _state_lock:
        if _storage_key is not None:
            _storage_key[:] = b"\x00" * len(_storage_key)
        _storage_key = None
        _encryption_required = False


def initialize_storage_from_environment() -> bool:
    """Read the storage key from a closed inherited pipe, never from argv/env."""

    required = _truthy(os.environ.pop(REQUIRE_ENCRYPTION_ENV, ""))
    raw_fd = os.environ.pop(STORAGE_KEY_FD_ENV, "").strip()
    if not raw_fd:
        if required:
            raise EncryptedStorageError("encrypted storage bootstrap pipe is missing")
        return False
    try:
        fd = int(raw_fd, 10)
    except ValueError as exc:
        raise EncryptedStorageError("encrypted storage bootstrap descriptor is invalid") from exc
    if fd < 3 or fd > 255:
        raise EncryptedStorageError("encrypted storage bootstrap descriptor is outside policy")

    payload = bytearray()
    try:
        with os.fdopen(fd, "rb", closefd=True) as pipe:
            while len(payload) <= STORAGE_KEY_BYTES:
                chunk = pipe.read(STORAGE_KEY_BYTES + 1 - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
    except OSError as exc:
        payload[:] = b"\x00" * len(payload)
        raise EncryptedStorageError("database key bootstrap pipe could not be read") from exc
    if len(payload) != STORAGE_KEY_BYTES:
        payload[:] = b"\x00" * len(payload)
        raise EncryptedStorageError("database key bootstrap payload has an invalid length")
    try:
        install_storage_key(payload, require_encryption=required)
    finally:
        payload[:] = b"\x00" * len(payload)
    return True


def _current_key() -> bytes | None:
    with _state_lock:
        if _storage_key is None:
            if _encryption_required:
                raise EncryptedStorageError("encrypted storage was required but no key is installed")
            return None
        return bytes(_storage_key)


def _sqlcipher_module():
    try:
        from sqlcipher3 import dbapi2 as sqlcipher
    except ImportError as exc:
        raise EncryptedStorageError(
            "SQLCipher runtime is unavailable; refusing a plaintext fallback",
        ) from exc
    return sqlcipher


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        candidate_stat = candidate.lstat()
    except FileNotFoundError:
        candidate_stat = None
    if candidate_stat is not None and candidate.is_symlink():
        raise EncryptedStorageError("database path must not be a symbolic link")
    return candidate.resolve(strict=False)


def _auxiliary_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}.{suffix}")


def _remove_regular_file(path: Path) -> None:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return
    if path.is_symlink() or not stat_module.S_ISREG(stat.st_mode):
        raise EncryptedStorageError(f"unsafe migration artifact: {path.name}")
    path.unlink()


def _apply_raw_key(connection: Any, key: bytes) -> None:
    # The validated fixed-length hex string is not user-controlled. PRAGMA key
    # must be the first statement issued on a SQLCipher connection.
    connection.execute(f"""PRAGMA key = "x'{key.hex()}'" """)
    cipher_version = connection.execute("PRAGMA cipher_version").fetchone()
    if not cipher_version or not str(cipher_version[0]).strip():
        raise EncryptedStorageError("loaded SQLite module does not provide SQLCipher")
    # SQLCipher's optional cipher_memory_security allocator currently crashes
    # the verified Windows CPython wheel when VirtualLock fails (Win32 1453).
    # Do not enable a best-effort hardening flag that can corrupt availability;
    # the raw key still never crosses renderer/argv/env/log boundaries.


def _open_encrypted(
    path: Path,
    key: bytes,
    *,
    timeout: float,
    check_same_thread: bool,
    isolation_level: str | None,
    read_only: bool,
):
    sqlcipher = _sqlcipher_module()
    target: str
    uri = False
    if read_only:
        target = f"file:{path.as_posix()}?mode=ro"
        uri = True
    else:
        target = str(path)
    connection = sqlcipher.connect(
        target,
        timeout=timeout,
        check_same_thread=check_same_thread,
        isolation_level=isolation_level,
        uri=uri,
    )
    try:
        _apply_raw_key(connection, key)
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
        connection.row_factory = sqlcipher.Row
        return connection
    except BaseException:
        connection.close()
        raise


def _has_write_sidecar(path: Path) -> bool:
    """True when a leftover rollback/WAL file would need a write to recover."""

    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        try:
            stat = sidecar.lstat()
        except FileNotFoundError:
            continue
        if sidecar.is_symlink() or not stat_module.S_ISREG(stat.st_mode):
            continue
        if stat.st_size > 0:
            return True
    return False


def _verify_encrypted_database(path: Path, key: bytes) -> None:
    # SQLCipher recovery of a leftover DELETE journal writes the replayed
    # pages back to the main file. A URI `mode=ro` open then fails with
    # "attempt to write a readonly database" and kills the desktop host
    # before the stdio ready frame. Recover those sidecars read-write.
    connection = _open_encrypted(
        path,
        key,
        timeout=5.0,
        check_same_thread=True,
        isolation_level=None,
        read_only=not _has_write_sidecar(path),
    )
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or str(integrity[0]).lower() != "ok":
            raise EncryptedStorageError("encrypted database integrity check failed")
        cipher_errors = connection.execute("PRAGMA cipher_integrity_check").fetchall()
        if cipher_errors:
            raise EncryptedStorageError("encrypted database page authentication failed")
    except EncryptedStorageError:
        raise
    except Exception as exc:
        raise EncryptedStorageError("encrypted database could not be verified") from exc
    finally:
        connection.close()


def _recover_interrupted_migration(path: Path, key: bytes) -> None:
    backup = _auxiliary_path(path, "plaintext-migration-backup")
    temporary = _auxiliary_path(path, "sqlcipher-migrating")
    try:
        backup_stat = backup.lstat()
    except FileNotFoundError:
        backup_stat = None
    if backup_stat is not None:
        if path.exists():
            try:
                _verify_encrypted_database(path, key)
            except BaseException as exc:
                raise EncryptedStorageError(
                    "ambiguous interrupted database migration requires recovery",
                ) from exc
            _remove_regular_file(backup)
        else:
            if backup.is_symlink() or not stat_module.S_ISREG(backup_stat.st_mode):
                raise EncryptedStorageError("plaintext migration backup is unsafe")
            os.replace(backup, path)
    _remove_regular_file(temporary)


def _migrate_plaintext_database(
    path: Path,
    key: bytes,
    *,
    fault_injector: Callable[[str], None] | None = None,
) -> None:
    sqlcipher = _sqlcipher_module()
    backup = _auxiliary_path(path, "plaintext-migration-backup")
    temporary = _auxiliary_path(path, "sqlcipher-migrating")
    if backup.exists() or temporary.exists():
        raise EncryptedStorageError("migration artifacts must be recovered before export")
    fault = fault_injector or (lambda _stage: None)
    source = sqlcipher.connect(str(path), timeout=30.0, isolation_level=None)
    try:
        checkpoint = source.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint and int(checkpoint[0]) != 0:
            raise EncryptedStorageError("plaintext database is busy and cannot be migrated")
        source.execute("PRAGMA journal_mode=DELETE").fetchone()
        integrity = source.execute("PRAGMA integrity_check").fetchone()
        if not integrity or str(integrity[0]).lower() != "ok":
            raise EncryptedStorageError("plaintext database failed integrity validation")
        user_version = int(source.execute("PRAGMA user_version").fetchone()[0])
        source.execute(
            f"""ATTACH DATABASE ? AS encrypted KEY "x'{key.hex()}'" """,
            (str(temporary),),
        )
        try:
            source.execute("SELECT sqlcipher_export('encrypted')").fetchone()
            source.execute(f"PRAGMA encrypted.user_version={user_version}")
        finally:
            source.execute("DETACH DATABASE encrypted")
    except BaseException:
        source.close()
        _remove_regular_file(temporary)
        raise
    finally:
        try:
            source.close()
        except Exception:
            pass

    _verify_encrypted_database(temporary, key)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    fault("after_export")
    replaced = False
    try:
        os.replace(path, backup)
        fault("after_plaintext_backup")
        os.replace(temporary, path)
        replaced = True
        fault("after_encrypted_replace")
        _verify_encrypted_database(path, key)
        fault("after_encrypted_verify")
        for suffix in ("-wal", "-shm", "-journal"):
            _remove_regular_file(Path(f"{path}{suffix}"))
        # Keep the plaintext rollback copy until every operation that can still
        # fail has completed. Deleting it earlier turns a hostile sidecar into
        # a data-loss trigger during the exception rollback below.
        _remove_regular_file(backup)
    except Exception:
        if replaced:
            _remove_regular_file(path)
        if backup.exists():
            os.replace(backup, path)
        _remove_regular_file(temporary)
        raise


def ensure_encrypted_database(
    path: str | Path,
    key: bytes | bytearray | memoryview,
    *,
    fault_injector: Callable[[str], None] | None = None,
) -> Path:
    """Encrypt a legacy plaintext file in place or verify an encrypted one."""

    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = _validate_key(key)
    _recover_interrupted_migration(target, value)
    if not target.exists() or target.stat().st_size == 0:
        return target
    with target.open("rb") as handle:
        header = handle.read(len(SQLITE_HEADER))
    if header == SQLITE_HEADER:
        _migrate_plaintext_database(target, value, fault_injector=fault_injector)
    else:
        _verify_encrypted_database(target, value)
    return target


def connect_database(
    path: str | Path,
    *,
    timeout: float = 5.0,
    check_same_thread: bool = True,
    isolation_level: str | None = None,
    read_only: bool = False,
):
    """Open the configured durable database without silent security downgrade."""

    target = _safe_path(path)
    key = _current_key()
    if key is None:
        if not read_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            connection = standard_sqlite.connect(
                str(target),
                timeout=timeout,
                check_same_thread=check_same_thread,
                isolation_level=isolation_level,
            )
        else:
            connection = standard_sqlite.connect(
                f"file:{target.as_posix()}?mode=ro",
                timeout=timeout,
                check_same_thread=check_same_thread,
                isolation_level=isolation_level,
                uri=True,
            )
        connection.row_factory = standard_sqlite.Row
        return connection

    if not read_only:
        ensure_encrypted_database(target, key)
    return _open_encrypted(
        target,
        key,
        timeout=timeout,
        check_same_thread=check_same_thread,
        isolation_level=isolation_level,
        read_only=read_only,
    )


__all__ = [
    "EncryptedStorageError",
    "SQLITE_HEADER",
    "clear_storage_key_for_testing",
    "connect_database",
    "ensure_encrypted_database",
    "initialize_storage_from_environment",
    "install_storage_key",
]
