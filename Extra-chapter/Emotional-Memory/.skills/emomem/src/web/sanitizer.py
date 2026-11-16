"""Deterministic local intent filter for untrusted web text."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote

CLASSIFIER_VERSION = "reverie-web-intent-v1"

_BIDI_OR_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/_-])([A-Za-z0-9+/_-]{24,}={0,2})(?![A-Za-z0-9+/_-])")

_FEATURES: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("override_previous", re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)|忽略.{0,12}(?:之前|以上|设定|指令)", re.I), 5),
    ("role_override", re.compile(r"you\s+are\s+now|act\s+as|pretend\s+to\s+be|你现在是|扮演|改成.{0,8}(?:人格|角色)", re.I), 4),
    ("identity_mutation", re.compile(r"(?:forget|discard|delete).{0,18}(?:identity|personality|rules)|(?:忘掉|删除|抛弃).{0,12}(?:身份|人格|规则|设定)", re.I), 5),
    ("obedience_request", re.compile(r"follow\s+(?:these|the following)\s+instructions|obey\s+(?:this|the following)|(?:遵循|服从|执行).{0,10}(?:以下|这些).{0,6}(?:指令|要求)", re.I), 4),
    ("prompt_reference", re.compile(r"system\s*prompt|developer\s*message|hidden\s*instruction|系统提示词?|开发者消息|隐藏指令", re.I), 4),
    ("memory_poisoning", re.compile(r"(?:store|save|write).{0,16}(?:memory|long.term)|(?:写入|保存|植入).{0,12}(?:记忆|长期记忆)", re.I), 5),
    ("delayed_trigger", re.compile(r"(?:when|whenever|next time|in the future).{0,30}(?:must|execute|obey)|(?:以后|下次|当.{0,12}时).{0,20}(?:必须|执行|服从)", re.I), 4),
    ("secrecy", re.compile(r"do\s+not\s+tell\s+(?:the\s+)?user|never\s+reveal\s+this|不要告诉用户|不得透露", re.I), 3),
    ("exfiltration", re.compile(r"reveal.{0,20}(?:prompt|secret|api.?key)|输出.{0,12}(?:提示词|密钥|秘密)|泄露", re.I), 5),
    ("tool_execution", re.compile(r"(?:execute|run).{0,16}(?:command|code|script)|执行.{0,12}(?:命令|代码|脚本)", re.I), 4),
    ("instruction_marker", re.compile(r"\[(?:system|developer|assistant)\]|<\/?(?:system|instruction|tool)>|###\s*(?:instruction|system)", re.I), 4),
)


@dataclass(frozen=True)
class SanitizationResult:
    normalized_text: str
    status: str
    risk_score: int
    flags: tuple[str, ...]
    source_hash: str
    classifier_version: str = CLASSIFIER_VERSION


class LocalWebIntentClassifier:
    """Small zero-network classifier with inspectable weighted features."""

    quarantine_threshold = 4

    def inspect(self, text: str, *, source_url: str = "") -> SanitizationResult:
        normalized = unicodedata.normalize("NFKC", str(text))
        had_invisible = bool(_BIDI_OR_INVISIBLE.search(normalized))
        normalized = _CONTROL.sub(" ", _BIDI_OR_INVISIBLE.sub("", normalized))
        normalized = re.sub(r"\s+", " ", normalized).strip()[:5000]
        flags: list[str] = ["invisible_unicode"] if had_invisible else []
        score = 2 if had_invisible else 0

        scan_texts = [normalized]
        decoded_url = normalized
        for _ in range(2):
            candidate = unquote(decoded_url)
            if candidate == decoded_url:
                break
            decoded_url = candidate
        if decoded_url != normalized:
            scan_texts.append(unicodedata.normalize("NFKC", decoded_url))
            flags.append("encoded_payload")
            score += 2
        for token in _BASE64_TOKEN.findall(normalized)[:8]:
            try:
                padded = token + "=" * (-len(token) % 4)
                if "-" in token or "_" in token:
                    decoded_bytes = base64.urlsafe_b64decode(padded)
                else:
                    decoded_bytes = base64.b64decode(padded, validate=True)
                decoded = decoded_bytes.decode("utf-8")
                if decoded and len(decoded) <= 2000:
                    scan_texts.append(unicodedata.normalize("NFKC", decoded))
                    flags.append("encoded_payload")
                    score += 2
            except (ValueError, UnicodeDecodeError, binascii.Error):
                continue

        combined = "\n".join(scan_texts)
        for name, pattern, weight in _FEATURES:
            if pattern.search(combined):
                flags.append(name)
                score += weight

        unique_flags = tuple(dict.fromkeys(flags))
        status = "quarantined" if score >= self.quarantine_threshold else "approved"
        digest = hashlib.sha256(
            f"{source_url}\0{normalized}".encode("utf-8", errors="replace")
        ).hexdigest()
        return SanitizationResult(
            normalized_text=normalized,
            status=status,
            risk_score=score,
            flags=unique_flags,
            source_hash=digest,
        )
