"""Three-layer memory classification and storage logic.

Permanent Memory:
  - Identity info (name, birthday, values, social connections)
  - NEVER forgotten, NEVER misremembered
  - Loaded from persona card on startup; injected into every prompt

Long-term Memory:
  - Important events, emotional conversations, conflicts, anniversaries
  - Stored in vector DB with high importance
  - Subject to forgetting (probabilistic, configurable cycle)
  - Subject to misremembering (confusion of details, configurable)

Short-term Memory:
  - Daily chatter, casual topics
  - Stored in vector DB with low importance
  - Rapid decay — configurable forgetting cycle (1–59 days)
"""

from __future__ import annotations

import logging
import hashlib
import time
import uuid
from typing import TYPE_CHECKING

from .versioned_store import VersionedVectorStore

if TYPE_CHECKING:
    from ..persona.persona_card import Persona

logger = logging.getLogger("reverie.memory.layers")


class MemoryLayers:
    """Manages the three-layer memory hierarchy."""

    def __init__(self, store: VersionedVectorStore, persona: "Persona") -> None:
        self.store = store
        self.persona = persona
        # Identity anchors are a derived cache of the active sealed persona.
        # Synchronize even when other permanent memories already exist so a
        # deliberate persona switch cannot leave the previous identity active.
        self._sync_permanent_identity()

    # ── Permanent memory ─────────────────────────────────

    def _identity_facts(self) -> list[str]:
        facts = [
            f"My name is {self.persona.name}.",
            f"I am a {self.persona.identity.get('title', 'person')}.",
            f"My personality: {', '.join(self.persona.personality_traits)}.",
            f"My values: {', '.join(self.persona.values)}.",
            *[f"Hobby: {h}" for h in self.persona.daily_life.get("hobbies", [])],
            *[f"Quirk: {q}" for q in self.persona.daily_life.get("quirks", [])],
        ]
        if not self.persona.identity.get("age_unknown"):
            facts.insert(1, f"I am {self.persona.age} years old.")
        if self.persona.birthday:
            facts.insert(2, f"My birthday is {self.persona.birthday}.")
        if self.persona.backstory:
            facts.append(f"Backstory: {self.persona.backstory}")
        return facts

    def _sync_permanent_identity(self) -> None:
        """Replace only derived persona anchors, preserving user memories."""

        envelope = self.persona.identity_envelope
        rows = self.store.list_by_layer("permanent")
        current_ids: set[str] = set()
        facts = self._identity_facts()
        for fact in facts:
            digest = hashlib.sha256(
                f"{envelope.fingerprint}\0{fact}".encode("utf-8")
            ).hexdigest()[:24]
            current_ids.add(f"identity_{digest}")

        for row in rows:
            if str(row.get("source_type", "")) not in {"persona_card", "persona_identity"}:
                continue
            memory_id = str(row.get("id", ""))
            if memory_id not in current_ids or str(row.get("source_hash", "")) != envelope.fingerprint:
                self.store.delete(memory_id)

        for fact in facts:
            digest = hashlib.sha256(
                f"{envelope.fingerprint}\0{fact}".encode("utf-8")
            ).hexdigest()[:24]
            cognitive_layer = "procedural" if fact.startswith(("My personality:", "My values:", "Hobby:", "Quirk:")) else "semantic"
            self.store.add(
                id=f"identity_{digest}",
                text=fact,
                layer="permanent",
                cognitive_layer=cognitive_layer,
                importance=1.0,
                source_type="persona_identity",
                source_uri=f"reverie-persona://{envelope.persona_id}/{envelope.version}",
                source_hash=envelope.fingerprint,
            )
        logger.info(
            "Synchronized %d immutable identity anchors for persona %s v%d",
            len(facts), envelope.persona_id, envelope.version,
        )

    # Compatibility name used by older callers/tests.
    def _seed_permanent_memory(self) -> None:
        self._sync_permanent_identity()

    def get_permanent_memories(self) -> list[str]:
        """Return all permanent memory texts (for prompt injection)."""
        results = self.store.list_by_layer("permanent")
        return [r["text"] for r in results]

    # ── Long-term memory ─────────────────────────────────

    def store_long_term(
        self,
        text: str,
        importance: float = 0.7,
        emotions: dict[str, float] | None = None,
        *,
        source_type: str = "local_interaction",
        source_uri: str = "",
        source_hash: str = "",
    ) -> str:
        """Store a long-term memory. Returns the assigned id."""
        # Keep the complete 128-bit UUID.  Truncating this to eight hex digits
        # made ordinary desktop-scale catalogs vulnerable to birthday
        # collisions, and ``MemoryCatalog.upsert`` would then silently replace
        # an unrelated memory with the same id.
        mem_id = f"lt_{uuid.uuid4().hex}"
        self.store.add(
            id=mem_id,
            text=text,
            layer="long_term",
            cognitive_layer="episodic" if text.startswith("事件记忆：") else "semantic",
            importance=importance,
            emotions=emotions,
            source_type=source_type,
            source_uri=source_uri,
            source_hash=source_hash,
        )
        return mem_id

    def search_long_term(self, query: str, k: int = 5) -> list[str]:
        """Search long-term memory for relevant memories."""
        qv = self.store.embed_query(query)
        results = self.store.search(qv, k=k, layers=["long_term", "permanent"])
        return [r["text"] for r in results]

    # ── Short-term memory ─────────────────────────────────

    def store_short_term(
        self,
        text: str,
        importance: float = 0.3,
        emotions: dict[str, float] | None = None,
        *,
        source_type: str = "local_interaction",
        source_uri: str = "",
        source_hash: str = "",
    ) -> str:
        """Store a short-term memory. Returns the assigned id."""
        mem_id = f"st_{uuid.uuid4().hex}"
        self.store.add(
            id=mem_id,
            text=text,
            layer="short_term",
            cognitive_layer="episodic" if text.startswith("事件记忆：") else "semantic",
            importance=importance,
            emotions=emotions,
            source_type=source_type,
            source_uri=source_uri,
            source_hash=source_hash,
        )
        return mem_id

    def search_short_term(self, query: str, k: int = 5) -> list[str]:
        """Search short-term memory for relevant memories."""
        qv = self.store.embed_query(query)
        results = self.store.search(qv, k=k, layers=["short_term"])
        return [r["text"] for r in results]

    # ── Unified search ────────────────────────────────────

    def search_all(self, query: str, k: int = 8) -> list[str]:
        """Search across all layers, prioritizing permanent > long > short."""
        qv = self.store.embed_query(query)
        results = self.store.search(qv, k=k)
        return [r["text"] for r in results]
