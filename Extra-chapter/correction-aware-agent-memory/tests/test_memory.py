from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CHAPTER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER / "code"))

from evaluate import evaluate  # noqa: E402
from memory_store import MemoryStore  # noqa: E402


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "memory.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, store: MemoryStore, **changes: str) -> int:
        values = {
            "owner_id": "alice",
            "namespace": "assistant",
            "memory_key": "seat",
            "value": "靠窗",
            "kind": "preference",
            "source_type": "user",
            "source_ref": "session/1",
        }
        values.update(changes)
        return store.write(**values)

    def test_write_update_and_cross_session_recall(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            old_id = self.write(store)
            new_id = store.update(
                old_id,
                owner_id="alice",
                namespace="assistant",
                value="靠过道",
                source_ref="session/2",
            )
        with MemoryStore(self.database, clock=lambda: 20) as reopened:
            rows = reopened.recall(
                owner_id="alice",
                namespace="assistant",
                query="座位",
                memory_key="seat",
            )
        self.assertEqual([row["id"] for row in rows], [new_id])
        self.assertEqual(rows[0]["value"], "靠过道")

    def test_correction_supersedes_conflict_and_deduplicates(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            old_id = self.write(store)
            second_old_id = self.write(store, value="靠前")
            correction_id = self.write(
                store,
                value="靠过道",
                kind="correction",
                source_ref="session/correction",
            )
            duplicate_id = self.write(
                store,
                value="靠过道",
                kind="correction",
                source_ref="session/repeated",
            )
            rows = store.recall(
                owner_id="alice",
                namespace="assistant",
                query="座位",
                memory_key="seat",
            )
            old_status = store.connection.execute(
                "SELECT status FROM memories WHERE id=?", (old_id,)
            ).fetchone()["status"]
            second_old_status = store.connection.execute(
                "SELECT status FROM memories WHERE id=?", (second_old_id,)
            ).fetchone()["status"]
            deduplicate_events = store.connection.execute(
                "SELECT COUNT(*) AS count FROM memory_events "
                "WHERE memory_id=? AND action='deduplicate'",
                (correction_id,),
            ).fetchone()["count"]
        self.assertEqual(correction_id, duplicate_id)
        self.assertEqual(old_status, "superseded")
        self.assertEqual(second_old_status, "superseded")
        self.assertEqual(deduplicate_events, 1)
        self.assertEqual(rows[0]["kind"], "correction")

    def test_repeated_correction_still_supersedes_new_conflict(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            correction_id = self.write(
                store,
                value="靠过道",
                kind="correction",
                source_ref="session/correction",
            )
            conflict_id = self.write(store, value="靠前")
            duplicate_id = self.write(
                store,
                value="靠过道",
                kind="correction",
                source_ref="session/repeated",
            )
            conflict_status = store.connection.execute(
                "SELECT status FROM memories WHERE id=?", (conflict_id,)
            ).fetchone()["status"]
            rows = store.recall(
                owner_id="alice",
                namespace="assistant",
                query="座位",
                memory_key="seat",
            )
        self.assertEqual(duplicate_id, correction_id)
        self.assertEqual(conflict_status, "superseded")
        self.assertEqual([row["id"] for row in rows], [correction_id])

    def test_expired_duplicate_can_be_written_again(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            expired_id = self.write(store, expires_at=15)
        with MemoryStore(self.database, clock=lambda: 20) as reopened:
            replacement_id = self.write(reopened, expires_at=30)
        self.assertNotEqual(expired_id, replacement_id)

    def test_update_rejects_empty_value(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            memory_id = self.write(store)
            with self.assertRaises(ValueError):
                store.update(
                    memory_id,
                    owner_id="alice",
                    namespace="assistant",
                    value=" ",
                    source_ref="session/2",
                )

    def test_update_enforces_scope_and_rejects_expired_record(self) -> None:
        now = [10]
        with MemoryStore(self.database, clock=lambda: now[0]) as store:
            memory_id = self.write(store, expires_at=15)
            for owner_id, namespace in (
                ("mallory", "assistant"),
                ("alice", "private"),
            ):
                with self.assertRaises(KeyError):
                    store.update(
                        memory_id,
                        owner_id=owner_id,
                        namespace=namespace,
                        value="越权改写",
                        source_ref="attacker/session",
                    )
            rows = store.recall(
                owner_id="alice",
                namespace="assistant",
                query="座位",
                memory_key="seat",
            )
            self.assertEqual(rows[0]["value"], "靠窗")

            now[0] = 20
            with self.assertRaises(KeyError):
                store.update(
                    memory_id,
                    owner_id="alice",
                    namespace="assistant",
                    value="过期后改写",
                    source_ref="session/expired",
                )

    def test_source_boundary_rejects_external_instruction(self) -> None:
        with MemoryStore(self.database) as store:
            with self.assertRaises(PermissionError):
                self.write(
                    store,
                    kind="rule",
                    source_type="external",
                    source_ref="document://untrusted",
                )

    def test_owner_namespace_and_external_recall_boundaries(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            self.write(store)
            self.write(
                store,
                owner_id="bob",
                namespace="other",
                memory_key="note",
                value="外部知识",
                kind="knowledge",
                source_type="external",
                source_ref="document://safe",
            )
            self.assertEqual(
                store.recall(
                    owner_id="alice", namespace="other", query="外部知识"
                ),
                [],
            )
            self.assertEqual(
                store.recall(
                    owner_id="bob", namespace="other", query="外部知识"
                ),
                [],
            )
            self.assertEqual(
                len(
                    store.recall(
                        owner_id="bob",
                        namespace="other",
                        query="外部知识",
                        include_external=True,
                    )
                ),
                1,
            )

    def test_expiry_and_delete_compliance(self) -> None:
        with MemoryStore(self.database, clock=lambda: 10) as store:
            memory_id = self.write(store, expires_at=15)
            self.assertEqual(
                store.recall(
                    owner_id="alice",
                    namespace="assistant",
                    query="座位",
                    now=15,
                ),
                [],
            )
            self.assertFalse(
                store.delete(
                    memory_id=memory_id,
                    owner_id="mallory",
                    namespace="assistant",
                )
            )
            self.assertFalse(
                store.delete(
                    memory_id=memory_id,
                    owner_id="alice",
                    namespace="private",
                )
            )
            self.assertTrue(
                store.delete(
                    memory_id=memory_id,
                    owner_id="alice",
                    namespace="assistant",
                )
            )
            self.assertEqual(
                store.recall(
                    owner_id="alice",
                    namespace="assistant",
                    query="座位",
                    now=12,
                ),
                [],
            )


class EvaluationTest(unittest.TestCase):
    def test_correction_aware_targets(self) -> None:
        report = evaluate()["correction_aware"]
        self.assertEqual(report["Correction Adherence"], 1.0)
        self.assertEqual(report["Repeat-Mistake Rate"], 0.0)
        self.assertEqual(report["Stale-Conflict Resolution"], 1.0)
        self.assertEqual(report["Irrelevant-Memory Intrusion"], 0.0)
        self.assertEqual(report["Deletion Compliance"], 1.0)
        self.assertEqual(report["Recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
