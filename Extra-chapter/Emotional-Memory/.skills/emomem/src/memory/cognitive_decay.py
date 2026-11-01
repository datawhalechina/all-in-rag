"""Non-destructive memory decay and retrieval-time fuzzy recall.

Canonical memories are never deleted or rewritten by this module. Forgetting
means that a memory becomes harder to retrieve. A rare misremembering event is
an ephemeral, auditable projection made from two eligible memories.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import random
import re
import time
from typing import TYPE_CHECKING, Iterable

from .lifecycle_constants import (
    PROMISE_FACT_MARKERS,
    PROTECTED_FACT_MARKERS as _BASE_PROTECTED_MARKERS,
)

if TYPE_CHECKING:
    from .versioned_store import VersionedVectorStore

logger = logging.getLogger("reverie.memory.cognitive_decay")

SECONDS_PER_DAY = 86_400.0
DEFAULT_DECAY_LAMBDA = math.log(2.0) / 90.0
FUZZY_RECALL_PREFIX = "[[REVERIE_FUZZY_RECALL]]"

_QUOTED_ENTITY = re.compile(r"[\u300a\u201c\"']([^\u300b\u201d\"'\r\n]{2,64})[\u300b\u201d\"']")
_ENTITY_AFTER_ACTIVITY = re.compile(
    r"(?:playing|played|watching|game\s+is|\u5728\u73a9|\u73a9\u8fc7|\u559c\u6b22|\u8ffd\u756a)"
    r"\s*(?:\u7684\u662f)?\s*([A-Za-z0-9_\-\u3400-\u9fff]{2,32})",
    re.IGNORECASE,
)

_DOMAIN_MARKERS: dict[str, tuple[str, ...]] = {
    "game": ("game", "gaming", "steam", "\u6e38\u620f", "\u5728\u73a9", "\u73a9\u8fc7"),
    "anime": ("anime", "animation", "\u52a8\u6f2b", "\u65b0\u756a", "\u8ffd\u756a"),
    "music": ("music", "song", "album", "\u97f3\u4e50", "\u6b4c", "\u4e13\u8f91"),
    "book": ("book", "novel", "\u5c0f\u8bf4", "\u4e66", "\u6f2b\u753b"),
}

# Single source of truth lives in lifecycle_constants (shared with catalog).
_PROTECTED_FACT_MARKERS = _BASE_PROTECTED_MARKERS + PROMISE_FACT_MARKERS

_CORRECTION_MARKERS = (
    "\u8bb0\u9519", "\u641e\u9519", "\u8bb0\u53cd", "\u8bb0\u4e32", "\u6211\u8bf4\u7684\u662f",
    "actually", "you got that wrong", "that's wrong", "not that one",
)


@dataclass(frozen=True)
class RecallConfusion:
    event_id: int
    memory_id: str
    true_text: str
    rendered_text: str
    source_entity: str
    substitute_entity: str
    domain: str


class CognitiveDecaySystem:
    """Compute ACT-R-inspired activation without destroying source data."""

    def __init__(
        self,
        store: "VersionedVectorStore",
        *,
        adapter: object | None = None,
        forgetting_enabled: bool = True,
        long_term_forgetting_enabled: bool = True,
        short_term_forgetting_enabled: bool = True,
        long_term_forget_days: int = 90,
        short_term_forget_days: int = 7,
        forget_probability: float = 0.05,
        long_term_forget_probability: float | None = None,
        short_term_forget_probability: float | None = None,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        recall_reinforcement_alpha: float = 0.12,
        minimum_retrieval_retention: float = 0.05,
        misremembering_enabled: bool = False,
        misremember_probability: float = 0.001,
        long_term_misremembering_enabled: bool = True,
        short_term_misremembering_enabled: bool = True,
        long_term_misremember_probability: float | None = None,
        short_term_misremember_probability: float | None = None,
    ) -> None:
        self.store = store
        # Kept only for API compatibility. LLMs must never rewrite memory here.
        self.adapter = adapter
        self.forgetting_enabled = bool(forgetting_enabled)
        self.long_term_forgetting_enabled = bool(long_term_forgetting_enabled)
        self.short_term_forgetting_enabled = bool(short_term_forgetting_enabled)
        self.long_term_forget_days = max(1, int(long_term_forget_days))
        self.short_term_forget_days = max(1, int(short_term_forget_days))
        # Legacy deletion probabilities remain readable for old backups but are not executed.
        self.forget_probability = float(forget_probability)
        self.long_term_forget_probability = float(
            forget_probability if long_term_forget_probability is None else long_term_forget_probability
        )
        self.short_term_forget_probability = float(
            forget_probability if short_term_forget_probability is None else short_term_forget_probability
        )
        self.decay_lambda = max(0.0001, min(0.1, float(decay_lambda)))
        self.recall_reinforcement_alpha = max(
            0.0, min(0.5, float(recall_reinforcement_alpha))
        )
        self.minimum_retrieval_retention = max(
            0.0, min(0.95, float(minimum_retrieval_retention))
        )
        self.misremembering_enabled = bool(misremembering_enabled)
        self.misremember_probability = self._bounded_confusion_probability(misremember_probability)
        self.long_term_misremembering_enabled = bool(long_term_misremembering_enabled)
        self.short_term_misremembering_enabled = bool(short_term_misremembering_enabled)
        self.long_term_misremember_probability = self._bounded_confusion_probability(
            misremember_probability
            if long_term_misremember_probability is None
            else long_term_misremember_probability
        )
        self.short_term_misremember_probability = self._bounded_confusion_probability(
            misremember_probability
            if short_term_misremember_probability is None
            else short_term_misremember_probability
        )

    @staticmethod
    def _bounded_confusion_probability(value: float) -> float:
        return max(0.0, min(0.10, float(value)))

    def retention(self, memory: dict, *, now: float | None = None) -> float:
        """Return R in [0, 1] from time, salience, emotion, and references."""
        layer = str(memory.get("layer", memory.get("retention_layer", "short_term")))
        if layer == "permanent" or not self.forgetting_enabled:
            return 1.0
        if layer == "long_term" and not self.long_term_forgetting_enabled:
            return 1.0
        if layer == "short_term" and not self.short_term_forgetting_enabled:
            return 1.0

        current = time.time() if now is None else float(now)
        anchor = memory.get("last_accessed")
        if anchor is None:
            anchor = memory.get("timestamp", current)
        try:
            elapsed_days = max(0.0, (current - float(anchor)) / SECONDS_PER_DAY)
        except (TypeError, ValueError):
            elapsed_days = 0.0

        importance = max(0.0, min(1.0, float(memory.get("importance", 0.5) or 0.5)))
        emotions = memory.get("emotions")
        emotion_intensity = 0.0
        if isinstance(emotions, dict):
            finite_values = []
            for value in emotions.values():
                try:
                    numeric = abs(float(value))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    finite_values.append(numeric)
            emotion_intensity = min(1.0, max(finite_values, default=0.0) / 100.0)

        references = max(
            0,
            int(memory.get("access_count", 0) or 0)
            + int(memory.get("mention_count", 1) or 1)
            - 1,
        )
        reinforcement = 1.0 + self.recall_reinforcement_alpha * math.log1p(references)
        strength = max(0.15, (0.35 + 1.25 * importance + 0.5 * emotion_intensity) * reinforcement)
        half_life_days = (
            self.long_term_forget_days if layer == "long_term" else self.short_term_forget_days
        )
        layer_scale = 90.0 / max(1.0, float(half_life_days))
        exponent = -self.decay_lambda * layer_scale * elapsed_days / strength
        return max(0.0, min(1.0, math.exp(max(-700.0, exponent))))

    def retrieval_weight(self, memory: dict, *, now: float | None = None) -> float:
        """Implement importance * exp(-lambda*t) * bounded recall reinforcement."""
        if str(memory.get("layer", "")) == "permanent":
            return 1.0
        importance = max(0.0, min(1.0, float(memory.get("importance", 0.5) or 0.5)))
        references = max(
            0,
            int(memory.get("access_count", 0) or 0)
            + int(memory.get("mention_count", 1) or 1)
            - 1,
        )
        reinforcement = 1.0 + self.recall_reinforcement_alpha * math.log1p(references)
        return max(0.0, min(1.5, importance * self.retention(memory, now=now) * reinforcement))

    async def run_forgetting_cycle(self) -> dict:
        """Measure latent memories. This maintenance pass never deletes or mutates them."""
        summary = {"forgotten": 0, "misremembered": 0, "latent": 0, "evaluated": 0}
        if not self.forgetting_enabled:
            return summary
        for layer, enabled in (
            ("long_term", self.long_term_forgetting_enabled),
            ("short_term", self.short_term_forgetting_enabled),
        ):
            if not enabled:
                continue
            try:
                # Maintenance is observability only; cap work so an old desktop
                # cannot freeze after years of memories. Retrieval itself still
                # evaluates every candidate selected by the current cue.
                rows = self.store.list_by_layer(layer, limit=512)
            except Exception:
                logger.exception("Failed to evaluate %s memory decay", layer)
                continue
            summary["evaluated"] += len(rows)
            summary["latent"] += sum(
                1 for row in rows if self.retention(row) < self.minimum_retrieval_retention
            )
        return summary

    def apply_retrieval_noise(
        self,
        selected: list[dict],
        *,
        query: str,
        distractors: Iterable[dict],
    ) -> tuple[list[str], list[RecallConfusion]]:
        """Return one-turn projections; canonical rows remain byte-for-byte unchanged."""
        rendered = [str(row.get("text", "")) for row in selected]
        if not self.misremembering_enabled or not selected:
            return rendered, []

        distractor_rows = list(distractors)
        for index in random.sample(range(len(selected)), len(selected)):
            row = selected[index]
            probability = self._row_confusion_probability(row)
            if probability <= 0.0 or random.random() >= probability:
                continue
            substitution = self._find_substitution(row, distractor_rows)
            if substitution is None:
                continue
            source_entity, substitute_entity, domain = substitution
            true_text = str(row.get("text", ""))
            fuzzy_text = true_text.replace(source_entity, substitute_entity, 1)
            if fuzzy_text == true_text:
                continue
            try:
                event_id = self.store.catalog.record_recall_confusion(
                    memory_id=str(row.get("id", "")),
                    query=query,
                    true_text=true_text,
                    rendered_text=fuzzy_text,
                    source_entity=source_entity,
                    substitute_entity=substitute_entity,
                    domain=domain,
                )
            except Exception:
                logger.exception("Could not audit fuzzy recall; serving canonical memory")
                continue
            rendered[index] = FUZZY_RECALL_PREFIX + fuzzy_text
            return rendered, [RecallConfusion(
                event_id=event_id,
                memory_id=str(row.get("id", "")),
                true_text=true_text,
                rendered_text=fuzzy_text,
                source_entity=source_entity,
                substitute_entity=substitute_entity,
                domain=domain,
            )]
        return rendered, []

    def resolve_user_correction(self, user_message: str) -> bool:
        """Reinforce truth when the user corrects a recent fuzzy recollection."""
        try:
            event = self.store.catalog.latest_open_recall_confusion()
        except Exception:
            logger.exception("Could not inspect fuzzy recall audit")
            return False
        if not event:
            return False
        lowered = user_message.casefold()
        explicit = any(marker in lowered for marker in _CORRECTION_MARKERS)
        entity_negation = "\u4e0d\u662f" in user_message and any(
            str(event.get(key, "")) in user_message
            for key in ("source_entity", "substitute_entity")
        )
        if not explicit and not entity_negation:
            return False
        memory_id = str(event.get("memory_id", ""))
        try:
            if not self.store.catalog.resolve_recall_confusion(int(event["id"])):
                return False
            row = self.store.catalog.get(memory_id)
            if row:
                importance = min(1.0, float(row.get("importance", 0.5) or 0.5) + 0.03)
                self.store.catalog.update_importance(memory_id, importance)
                self.store.catalog.touch_access([memory_id])
            return True
        except Exception:
            logger.exception("Could not reinforce corrected canonical memory")
            return False

    def _row_confusion_probability(self, row: dict) -> float:
        layer = str(row.get("layer", row.get("retention_layer", "")))
        if self._is_protected(row):
            return 0.0
        try:
            age_days = max(0.0, (time.time() - float(row.get("timestamp", time.time()))) / SECONDS_PER_DAY)
        except (TypeError, ValueError):
            return 0.0
        if layer == "long_term":
            if not self.long_term_misremembering_enabled:
                return 0.0
            baseline = 90
            # The system is forbidden when the configured forgetting threshold
            # is below the 90-day baseline (the human forgetting curve).
            if int(self.long_term_forget_days) < baseline or age_days < baseline:
                return 0.0
            configured = self._bounded_confusion_probability(self.long_term_misremember_probability)
            return configured * self._age_scale(age_days, baseline, int(self.long_term_forget_days))
        if layer == "short_term":
            if not self.short_term_misremembering_enabled:
                return 0.0
            baseline = 5
            # The system is forbidden when the configured forgetting threshold
            # is below the 5-day baseline (the human forgetting curve).
            if int(self.short_term_forget_days) < baseline or age_days < baseline:
                return 0.0
            configured = self._bounded_confusion_probability(self.short_term_misremember_probability)
            return configured * self._age_scale(age_days, baseline, int(self.short_term_forget_days))
        return 0.0

    @staticmethod
    def _age_scale(age_days: float, baseline: int, ceiling: int) -> float:
        """Scale misremember probability with age.

        At the baseline day the configured probability is halved; it grows
        linearly and caps at 60% of the configured value when a memory reaches
        the configured forgetting threshold. Older memories never exceed that
        cap, so misremembering stays bounded and forget-table memories are
        never confused at full configured strength.
        """
        if ceiling <= baseline:
            # Threshold equals the baseline: every eligible memory is already
            # at (or past) the ceiling, so use the full in-window cap.
            return 0.6
        span = float(ceiling - baseline)
        ratio = max(0.0, min(1.0, (age_days - baseline) / span))
        return 0.5 + 0.1 * ratio

    @staticmethod
    def _is_protected(row: dict) -> bool:
        if str(row.get("layer", row.get("retention_layer", ""))) == "permanent":
            return True
        if str(row.get("cognitive_layer", "")) == "procedural":
            return True
        if float(row.get("importance", 0.5) or 0.5) >= 0.9:
            return True
        text = str(row.get("text", "")).casefold()
        return any(marker in text for marker in _PROTECTED_FACT_MARKERS)

    @staticmethod
    def _domain(text: str) -> str | None:
        lowered = text.casefold()
        for domain, markers in _DOMAIN_MARKERS.items():
            if any(marker in lowered for marker in markers):
                return domain
        return None

    @staticmethod
    def _entities(text: str) -> list[str]:
        entities = [match.strip() for match in _QUOTED_ENTITY.findall(text) if match.strip()]
        entities.extend(
            match.strip() for match in _ENTITY_AFTER_ACTIVITY.findall(text) if match.strip()
        )
        return list(dict.fromkeys(entity for entity in entities if 2 <= len(entity) <= 64))

    def _find_substitution(
        self,
        source: dict,
        distractors: list[dict],
    ) -> tuple[str, str, str] | None:
        source_text = str(source.get("text", ""))
        domain = self._domain(source_text)
        source_entities = self._entities(source_text)
        if not domain or not source_entities:
            return None
        alternatives: list[tuple[str, str, str]] = []
        for row in distractors:
            if str(row.get("id", "")) == str(source.get("id", "")) or self._is_protected(row):
                continue
            if self._domain(str(row.get("text", ""))) != domain:
                continue
            for source_entity in source_entities:
                for substitute_entity in self._entities(str(row.get("text", ""))):
                    if source_entity.casefold() != substitute_entity.casefold():
                        alternatives.append((source_entity, substitute_entity, domain))
        return random.choice(alternatives) if alternatives else None
