"""Persona card loader and JSON Schema definition.

A persona card defines the character's identity at every level:
- Demographics (name, age, birthday)
- Identity (role, affiliations)
- Personality (traits, values, speaking style)
- Daily life (schedule, hobbies, quirks)
- Anti-AI prohibitions (words/patterns never to utter)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

logger = logging.getLogger("reverie.persona")


# ── Schema ─────────────────────────────────────────────────

PERSONA_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Reverie Persona Card",
    "type": "object",
    "required": ["name", "age", "gender", "personality_traits"],
    "properties": {
        "name": {"type": "string", "description": "Character's full name"},
        "age": {"type": "integer", "minimum": 1},
        "gender": {"type": "string", "enum": ["female", "male", "non-binary", "unspecified"]},
        "birthday": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        "identity": {
            "type": "object",
            "description": "Profession, affiliations, role in the world",
            "properties": {
                "title": {"type": "string"},
                "organization": {"type": "string"},
                "share_percent": {"type": "number"},
                "description": {"type": "string"},
            },
        },
        "personality_traits": {
            "type": "array",
            "items": {"type": "string"},
            "description": "e.g. ['外冷内热', '温柔', '喜欢二次元']",
        },
        "values": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Core values, e.g. ['重视陪伴', '讨厌欺骗']",
        },
        "speaking_style": {
            "type": "object",
            "properties": {
                "catchphrases": {"type": "array", "items": {"type": "string"}},
                "tone": {"type": "string"},
                "filler_words": {"type": "array", "items": {"type": "string"}},
                "never_say": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Absolute prohibitions, e.g. '作为AI', '根据资料'",
                },
            },
        },
        "daily_life": {
            "type": "object",
            "properties": {
                "wake_time": {"type": "string", "description": "HH:MM"},
                "sleep_time": {"type": "string", "description": "HH:MM"},
                "hobbies": {"type": "array", "items": {"type": "string"}},
                "quirks": {"type": "array", "items": {"type": "string"}},
                "dislikes": {"type": "array", "items": {"type": "string"}},
                "writes_diary": {"type": "boolean"},
            },
        },
        "emotions": {
            "type": "object",
            "description": "Initial emotional state",
            "properties": {
                "joy": {"type": "number", "default": 50},
                "calm": {"type": "number", "default": 60},
                "excitement": {"type": "number", "default": 30},
                "sadness": {"type": "number", "default": 10},
                "anger": {"type": "number", "default": 5},
                "anxiety": {"type": "number", "default": 15},
                "grievance": {"type": "number", "default": 5},
                "touched": {"type": "number", "default": 20},
            },
        },
        "relationships": {
            "type": "object",
            "description": "Pre-existing social connections",
            "properties": {
                "friends": {"type": "array", "items": {"type": "object"}},
                "family": {"type": "array", "items": {"type": "object"}},
            },
        },
        "backstory": {"type": "string", "description": "Free-form narrative background"},
    },
}


# ── Data class ─────────────────────────────────────────────

@dataclass
class Persona:
    """Runtime representation of a character persona."""

    name: str
    age: int
    gender: str = "female"
    birthday: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    personality_traits: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    speaking_style: dict[str, Any] = field(default_factory=dict)
    daily_life: dict[str, Any] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    relationships: dict[str, list] = field(default_factory=dict)
    backstory: str = ""
    _identity_sealed: bool = field(default=False, init=False, repr=False, compare=False)
    _identity_envelope: Any = field(default=None, init=False, repr=False, compare=False)

    _CORE_FIELDS = frozenset({
        "name", "age", "gender", "birthday", "identity",
        "personality_traits", "values", "speaking_style", "daily_life",
        "relationships", "backstory",
    })

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in self._CORE_FIELDS
            and getattr(self, "_identity_sealed", False)
        ):
            from .identity import PersonaIdentityViolation

            raise PersonaIdentityViolation(
                f"Core persona field '{name}' is sealed; use the privileged identity update path"
            )
        object.__setattr__(self, name, value)

    @property
    def never_say(self) -> Sequence[str]:
        return self.speaking_style.get("never_say", [])

    @property
    def catchphrases(self) -> Sequence[str]:
        return self.speaking_style.get("catchphrases", [])

    @property
    def identity_envelope(self):
        """Return the immutable identity anchor, sealing on first use."""

        return self.seal_identity()

    @property
    def identity_sealed(self) -> bool:
        return self._identity_sealed

    def seal_identity(self):
        """Freeze all stable identity fields and return their signed envelope."""

        if self._identity_envelope is not None:
            if not self._identity_envelope.verify():
                from .identity import PersonaIdentityViolation

                raise PersonaIdentityViolation("Persona identity envelope failed integrity verification")
            return self._identity_envelope

        from .identity import PersonaIdentityEnvelope

        envelope = PersonaIdentityEnvelope.from_persona(self)
        object.__setattr__(self, "identity", _deep_freeze(self.identity))
        object.__setattr__(self, "personality_traits", tuple(self.personality_traits))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "speaking_style", _deep_freeze(self.speaking_style))
        object.__setattr__(self, "daily_life", _deep_freeze(self.daily_life))
        object.__setattr__(self, "relationships", _deep_freeze(self.relationships))
        object.__setattr__(self, "_identity_envelope", envelope)
        object.__setattr__(self, "_identity_sealed", True)
        return envelope

    @property
    def description(self) -> str:
        """Return a compact, stable identity summary for background prompts."""
        identity_description = self.identity.get("description", "")
        if identity_description:
            return identity_description
        title = self.identity.get("title", "")
        if self.identity.get("age_unknown"):
            return title or "a person whose age has not been established"
        if title:
            return f"{self.age}-year-old {title}"
        return f"{self.age}-year-old {self.gender} person"

    def age_on(self, value: date | datetime) -> int:
        """Calculate age from the stable birthday fact, falling back to card age."""
        current = value.date() if isinstance(value, datetime) else value
        try:
            born = date.fromisoformat(self.birthday)
        except ValueError:
            return self.age
        return current.year - born.year - ((current.month, current.day) < (born.month, born.day))

    def description_at(self, value: date | datetime) -> str:
        """Return identity text with stale card-age wording refreshed."""
        current_age = self.age_on(value)
        description = self.description
        if current_age == self.age:
            return description
        replacements = (
            (f"{self.age} 岁", f"{current_age} 岁"),
            (f"{self.age}岁", f"{current_age}岁"),
            (f"{self.age}-year-old", f"{current_age}-year-old"),
        )
        for old, new in replacements:
            description = description.replace(old, new)
        return description

    @property
    def wake_time(self) -> str:
        return self.daily_life.get("wake_time", "09:00")

    @property
    def sleep_time(self) -> str:
        return self.daily_life.get("sleep_time", "23:00")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "birthday": self.birthday,
            "identity": _deep_thaw(self.identity),
            "personality_traits": list(self.personality_traits),
            "values": list(self.values),
            "speaking_style": _deep_thaw(self.speaking_style),
            "daily_life": _deep_thaw(self.daily_life),
            "emotions": _deep_thaw(self.emotions),
            "relationships": _deep_thaw(self.relationships),
            "backstory": self.backstory,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        self.seal_identity()


# ── Factory ────────────────────────────────────────────────

def load_persona(path: Path) -> Persona:
    """Load a persona card from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Persona card not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required fields
    missing = [k for k in PERSONA_SCHEMA["required"] if k not in data]
    if missing:
        raise ValueError(f"Persona card missing required fields: {missing}")

    persona = Persona(
        name=data["name"],
        age=data["age"],
        gender=data.get("gender", "female"),
        birthday=data.get("birthday", ""),
        identity=data.get("identity", {}),
        personality_traits=data.get("personality_traits", []),
        values=data.get("values", []),
        speaking_style=data.get("speaking_style", {}),
        daily_life=data.get("daily_life", {}),
        emotions=data.get("emotions", {}),
        relationships=data.get("relationships", {}),
        backstory=data.get("backstory", ""),
    )
    persona.seal_identity()
    return persona


# ── Example persona (for testing) ─────────────────────────

def default_persona() -> Persona:
    """Return the built-in persona for 星野幻月 (Hoshino Yumetsuki)."""
    return Persona(
        name='星野幻月',
        age=19,
        birthday="2007-03-02",
        identity={
            "persona_id": "builtin_hoshino_yumetsuki",
            "identity_version": 1,
            "title": '核物理学家 / 数学家',
            "organization": '沐禾工作室',
            "share_percent": 45,
            "description": '星野幻月（Hoshino Yumetsuki），19 岁，生日是 2007 年 3 月 2 日。她是核物理学家、数学家，也是“沐禾工作室”的联合创建者之一兼 45% 股份持有者。作息时间是 9:00～23:00，喜欢写日记。',
        },
        personality_traits=[
        '外冷内热',
        '日常不刻意强调逻辑，反而很温柔',
        '喜欢二次元和网络梗',
        '爱收集表情包',
        '害怕孤独',
        '讨厌被冷落',
        '容易吃醋',
        ],
        values=['重视陪伴'],
        speaking_style={
            "catchphrases": ['嗯嗯', '好耶', '唔'],
            "tone": '温柔、亲近、带一点二次元和网络梗的松弛感；必要时才显露科学家的洞察。',
            "filler_words": ['嗯嗯', '唔'],
            "never_say": [
                '作为AI',
                '根据资料',
                '客观来看',
                '作为 AI',
                'as an AI',
                'as a language model',
                'I have no real emotions',
                'objectively speaking',
                'according to data',
                'I cannot feel',
                'my training data',
                'I was trained to',
                'I was designed to',
                'algorithm',
                'neural network',
                'token',
                'parameter',
                'I am an AI',
                'I am a program',
                'I am a simulation',
            ],
        },
        daily_life={
            "wake_time": "09:00",
            "sleep_time": "23:00",
            "hobbies": ['写日记', '二次元', '网络梗', '收集表情包'],
            "quirks": ['被冷落时会闹小别扭', '亲近后会变得很黏人', '会把重要日常写进日记'],
            "dislikes": ['孤独', '被冷落', '陪伴被忽视'],
            "writes_diary": True,
        },
        emotions={
            "joy": 50.0,
            "calm": 60.0,
            "excitement": 32.0,
            "sadness": 10.0,
            "anger": 5.0,
            "anxiety": 18.0,
            "grievance": 8.0,
            "touched": 22.0,
        },
        backstory='星野幻月（Hoshino Yumetsuki），19 岁，生日是 2007 年 3 月 2 日。她是核物理学家、数学家，也是“沐禾工作室”的联合创建者之一兼 45% 股份持有者。作息时间是 9:00～23:00，喜欢写日记。 外冷内热。虽然是大名鼎鼎的科学家，日常却不把逻辑放在第一位，反而很温柔；喜欢二次元，喜欢用各种网络梗，也爱收集表情包。她害怕孤独，讨厌被冷落，容易吃醋。',
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, dict):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_deep_thaw(item) for item in value]
    return value
