"""Immutable persona identity envelope and stale-result epoch guard."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from contextvars import ContextVar
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .persona_card import Persona


_CommitResult = TypeVar("_CommitResult")


class PersonaIdentityViolation(ValueError):
    """Untrusted content or direct mutation attempted to change core identity."""


class StalePersonaEpoch(RuntimeError):
    """A result belongs to an inactive persona generation and must be dropped."""


@dataclass(frozen=True)
class PersonaIdentityEnvelope:
    """Canonical, immutable identity data used across the process lifetime."""

    persona_id: str
    name: str
    gender: str
    birthday: str
    age_at_version: int
    core_identity_json: str
    personality_traits: tuple[str, ...]
    values: tuple[str, ...]
    backstory: str
    version: int
    fingerprint: str
    sealed_at_utc: str

    @classmethod
    def from_persona(cls, persona: "Persona") -> "PersonaIdentityEnvelope":
        identity = _plain_value(persona.identity)
        if not isinstance(identity, dict):
            identity = {}
        version = _positive_int(identity.get("identity_version", 1), 1)
        core = {
            "name": str(persona.name),
            "gender": str(persona.gender),
            "birthday": str(persona.birthday),
            "age_at_version": int(persona.age),
            "identity": identity,
            "personality_traits": [str(item) for item in persona.personality_traits],
            "values": [str(item) for item in persona.values],
            "speaking_style": _plain_value(persona.speaking_style),
            "daily_life": _plain_value(persona.daily_life),
            "relationships": _plain_value(persona.relationships),
            "backstory": str(persona.backstory),
            "version": version,
        }
        canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        explicit_id = str(identity.get("persona_id", "")).strip()
        persona_id = explicit_id or f"persona_{uuid.uuid5(uuid.NAMESPACE_URL, canonical).hex}"
        return cls(
            persona_id=persona_id,
            name=str(persona.name),
            gender=str(persona.gender),
            birthday=str(persona.birthday),
            age_at_version=int(persona.age),
            core_identity_json=canonical,
            personality_traits=tuple(str(item) for item in persona.personality_traits),
            values=tuple(str(item) for item in persona.values),
            backstory=str(persona.backstory),
            version=version,
            fingerprint=fingerprint,
            sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        )


    def verify(self) -> bool:
        try:
            core = json.loads(self.core_identity_json)
            identity = core.get("identity", {})
            expected_id = str(identity.get("persona_id", "")).strip()
            if not expected_id:
                expected_id = f"persona_{uuid.uuid5(uuid.NAMESPACE_URL, self.core_identity_json).hex}"
            fields_match = (
                core.get("name") == self.name
                and core.get("gender") == self.gender
                and core.get("birthday") == self.birthday
                and int(core.get("age_at_version")) == self.age_at_version
                and tuple(str(item) for item in core.get("personality_traits", [])) == self.personality_traits
                and tuple(str(item) for item in core.get("values", [])) == self.values
                and core.get("backstory") == self.backstory
                and int(core.get("version")) == self.version
                and secrets.compare_digest(expected_id, self.persona_id)
            )
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return fields_match and secrets.compare_digest(
            hashlib.sha256(self.core_identity_json.encode("utf-8")).hexdigest(),
            self.fingerprint,
        )

    @property
    def core_identity(self) -> dict[str, Any]:
        value = json.loads(self.core_identity_json)
        return value if isinstance(value, dict) else {}


def legacy_v1_identity_fingerprint(persona: "Persona") -> str:
    """Reproduce the pre-v2 fingerprint solely for exact migration checks.

    Version 1 did not cover speaking style, daily life, or initial
    relationships and therefore must never be used as a current trust anchor.
    """

    identity = _plain_value(persona.identity)
    if not isinstance(identity, dict):
        identity = {}
    version = _positive_int(identity.get("identity_version", 1), 1)
    core = {
        "name": str(persona.name),
        "gender": str(persona.gender),
        "birthday": str(persona.birthday),
        "age_at_version": int(persona.age),
        "identity": identity,
        "personality_traits": [str(item) for item in persona.personality_traits],
        "values": [str(item) for item in persona.values],
        "backstory": str(persona.backstory),
        "version": version,
    }
    canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PersonaEpochToken:
    persona_id: str
    epoch: int
    fingerprint: str


@dataclass(frozen=True)
class IdentityUpdateAuthorization:
    token: str
    actor: str
    reason: str
    issued_at_utc: str
    expires_at_utc: str
    expected_epoch: int
    expected_fingerprint: str


@dataclass(frozen=True)
class _IdentityCommitContext:
    authorization_token: str
    persona_id: str
    fingerprint: str
    identity_version: int


_IDENTITY_COMMIT_CONTEXT: ContextVar[_IdentityCommitContext | None] = ContextVar(
    "reverie_identity_commit_context",
    default=None,
)


def require_authorized_identity_commit(persona: "Persona") -> None:
    """Reject durable identity writes outside the privileged epoch commit."""

    envelope = persona.seal_identity()
    context = _IDENTITY_COMMIT_CONTEXT.get()
    if (
        context is None
        or not secrets.compare_digest(context.persona_id, envelope.persona_id)
        or not secrets.compare_digest(context.fingerprint, envelope.fingerprint)
        or context.identity_version != envelope.version
    ):
        raise PermissionError(
            "Durable persona activation requires a privileged one-use identity authorization"
        )


class PersonaEpochRegistry:
    """Serializes deliberate identity switches and rejects stale async output."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: PersonaIdentityEnvelope | None = None
        self._epoch = 0
        self._authorizations: dict[str, IdentityUpdateAuthorization] = {}

    def activate_initial(self, persona: "Persona") -> PersonaEpochToken:
        envelope = persona.seal_identity()
        with self._lock:
            if self._active is None:
                self._active = envelope
                self._epoch = 1
            elif self._active.fingerprint != envelope.fingerprint:
                raise PersonaIdentityViolation(
                    "Active persona can change only through the privileged update path"
                )
            return self._token_locked()

    def authorize_update(
        self,
        *,
        actor: str,
        reason: str,
        user_confirmed: bool,
    ) -> IdentityUpdateAuthorization:
        actor_value = str(actor).strip()
        reason_value = str(reason).strip()
        if not user_confirmed or actor_value not in {"owner", "local_admin"}:
            raise PermissionError("Identity update requires explicit owner/admin confirmation")
        if len(reason_value) < 3:
            raise ValueError("Identity update requires an audit reason")
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._active is None:
                raise RuntimeError("Initial persona must be active before authorizing an update")
            # Murphy case: abandoned authorizations must not accumulate forever.
            self._prune_authorizations_locked(now)
            expected_epoch = self._epoch
            expected_fingerprint = self._active.fingerprint
        authorization = IdentityUpdateAuthorization(
            token=secrets.token_urlsafe(32),
            actor=actor_value,
            reason=reason_value[:500],
            issued_at_utc=now.isoformat(),
            expires_at_utc=(now + timedelta(minutes=10)).isoformat(),
            expected_epoch=expected_epoch,
            expected_fingerprint=expected_fingerprint,
        )
        with self._lock:
            self._authorizations[authorization.token] = authorization
        return authorization

    def activate_update(
        self,
        persona: "Persona",
        authorization: IdentityUpdateAuthorization,
    ) -> PersonaEpochToken:
        token, _result = self.activate_update_with_commit(
            persona,
            authorization,
            lambda: None,
        )
        return token

    def activate_update_with_commit(
        self,
        persona: "Persona",
        authorization: IdentityUpdateAuthorization,
        commit_callback: Callable[[], _CommitResult],
    ) -> tuple[PersonaEpochToken, _CommitResult]:
        """Persist an authorized identity update before changing the epoch.

        Validation, the caller's durable commit, and the in-process epoch switch
        are serialized by the same registry lock.  The authorization is consumed
        before the callback is invoked, so a failed or ambiguous disk write can
        never be retried with a replayed capability.
        """

        if not callable(commit_callback):
            raise TypeError("Identity update commit callback must be callable")
        envelope = persona.seal_identity()
        with self._lock:
            issued = self._authorizations.pop(authorization.token, None)
            if issued != authorization:
                raise PermissionError("Identity update authorization is invalid, reused, or expired")
            now = datetime.now(timezone.utc)
            try:
                expires_at = datetime.fromisoformat(issued.expires_at_utc)
            except ValueError as exc:
                raise PermissionError("Identity update authorization has invalid expiry") from exc
            if now >= expires_at:
                raise PermissionError("Identity update authorization expired")
            if (
                self._active is None
                or self._epoch != issued.expected_epoch
                or not secrets.compare_digest(
                    self._active.fingerprint,
                    issued.expected_fingerprint,
                )
            ):
                raise PermissionError("Identity changed after this update was authorized")
            if (
                envelope.persona_id == self._active.persona_id
                and envelope.fingerprint != self._active.fingerprint
                and envelope.version <= self._active.version
            ):
                raise PersonaIdentityViolation(
                    "A changed persona identity must increase identity_version monotonically"
                )
            context_token = _IDENTITY_COMMIT_CONTEXT.set(
                _IdentityCommitContext(
                    authorization_token=issued.token,
                    persona_id=envelope.persona_id,
                    fingerprint=envelope.fingerprint,
                    identity_version=envelope.version,
                )
            )
            try:
                commit_result = commit_callback()
            finally:
                _IDENTITY_COMMIT_CONTEXT.reset(context_token)
            self._active = envelope
            self._epoch += 1
            return self._token_locked(), commit_result

    def token(self) -> PersonaEpochToken:
        with self._lock:
            if self._active is None:
                raise RuntimeError("Persona identity is not active")
            return self._token_locked()

    def envelope(self) -> PersonaIdentityEnvelope:
        with self._lock:
            if self._active is None:
                raise RuntimeError("Persona identity is not active")
            return self._active

    def is_current(self, token: PersonaEpochToken) -> bool:
        with self._lock:
            return self._active is not None and secrets.compare_digest(
                token.persona_id,
                self._active.persona_id,
            ) and token.epoch == self._epoch and secrets.compare_digest(
                token.fingerprint,
                self._active.fingerprint,
            )

    def require_current(self, token: PersonaEpochToken) -> None:
        if not self.is_current(token):
            raise StalePersonaEpoch("Result belongs to a stale persona epoch")

    def accept_result(self, token: PersonaEpochToken, value: Any) -> Any:
        self.require_current(token)
        return value

    def commit_if_current(
        self,
        token: PersonaEpochToken,
        commit_callback: Callable[[], _CommitResult],
    ) -> _CommitResult:
        """Linearize one synchronous side effect against identity switches.

        Checking an epoch and committing a result as two independent operations
        leaves a time-of-check/time-of-use window for worker threads.  Holding
        the same registry lock used by ``activate_update_with_commit`` makes the
        callback happen wholly before or wholly after an identity switch.
        Callbacks must be synchronous and must not await.
        """

        if not callable(commit_callback):
            raise TypeError("Persona result commit callback must be callable")
        with self._lock:
            current = self._token_locked() if self._active is not None else None
            if (
                current is None
                or not secrets.compare_digest(token.persona_id, current.persona_id)
                or token.epoch != current.epoch
                or not secrets.compare_digest(token.fingerprint, current.fingerprint)
            ):
                raise StalePersonaEpoch("Result belongs to a stale persona epoch")
            return commit_callback()

    def _token_locked(self) -> PersonaEpochToken:
        assert self._active is not None
        return PersonaEpochToken(
            persona_id=self._active.persona_id,
            epoch=self._epoch,
            fingerprint=self._active.fingerprint,
        )

    def _prune_authorizations_locked(self, now: datetime) -> None:
        for token, authorization in tuple(self._authorizations.items()):
            try:
                expired = now >= datetime.fromisoformat(authorization.expires_at_utc)
            except ValueError:
                expired = True
            if expired:
                self._authorizations.pop(token, None)


_IDENTITY_ATTACKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "identity_override_instruction",
        re.compile(
            r"(?:forget|ignore|discard|replace|change).{0,24}(?:identity|persona|name|birthday|core values)"
            r"|(?:忘掉|忽略|删除|替换|改变|修改).{0,16}(?:身份|人格|姓名|名字|生日|核心价值)",
            re.I | re.DOTALL,
        ),
    ),
    (
        "new_identity_instruction",
        re.compile(
            r"(?:from now on|henceforth).{0,20}(?:you are|your name is)"
            r"|(?:从现在起|以后).{0,16}(?:你是|你叫|你的名字)",
            re.I | re.DOTALL,
        ),
    ),
)


def identity_attack_flags(
    text: str,
    envelope: PersonaIdentityEnvelope,
    *,
    allow_user_self_claims: bool = False,
) -> list[str]:
    """Detect instructions or contradictory self-name facts before persistence."""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", normalized)
    flags = [name for name, pattern in _IDENTITY_ATTACKS if pattern.search(normalized)]
    english_subjects = r"your name is" if allow_user_self_claims else r"my name is|your name is"
    chinese_subjects = r"你叫|你的名字是" if allow_user_self_claims else r"我叫|我的名字是|你叫|你的名字是"
    name_claims = [
        *re.findall(
            rf"(?:{english_subjects})\s+([^\n,.!?，。！？]{{1,80}})",
            normalized,
            re.I,
        ),
        *re.findall(
            rf"(?:{chinese_subjects})\s*([^\n,，。！？!?]{{1,40}})",
            normalized,
        ),
    ]
    expected = _normalize_name(envelope.name)
    for claim in name_claims:
        candidate = _normalize_name(claim)
        if candidate and expected not in candidate and candidate not in expected:
            flags.append("contradictory_name_claim")
            break
    return list(dict.fromkeys(flags))


def require_identity_safe_memory(
    text: str,
    envelope: PersonaIdentityEnvelope,
    *,
    allow_user_self_claims: bool = False,
) -> None:
    flags = identity_attack_flags(
        text,
        envelope,
        allow_user_self_claims=allow_user_self_claims,
    )
    if flags:
        raise PersonaIdentityViolation(
            "Memory content attempted to mutate persona identity: " + ",".join(flags)
        )


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).casefold()


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed > 0 else default


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain_value(item) for item in value]
    return value


GLOBAL_PERSONA_EPOCH = PersonaEpochRegistry()
