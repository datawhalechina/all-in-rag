"""确定性跨会话行为评估（时延仅作观测，不参与判定）。"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from memory_policy import lexical_score
from memory_store import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "correction_cases.jsonl"


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def baseline(case: dict, mode: str) -> tuple[str | None, int, float]:
    if mode == "no_memory":
        return (case.get("old") if case["type"] not in {"irrelevant", "deletion"} else None, 0, 0.0)
    candidates = [case["old"]]
    if case.get("new"):
        candidates.append(case["new"])
    selected = max(
        candidates,
        key=lambda value: lexical_score(case["query"], f"{case['key']} {value}"),
    )
    return selected, math.ceil(len(selected) / 4), 0.0


def correction_aware(case: dict, directory: Path) -> tuple[str | None, int, float]:
    database = directory / f"{case['id']}.sqlite3"
    with MemoryStore(database, clock=lambda: 100) as store:
        memory_id = store.write(
            owner_id="eval-user",
            namespace="assistant",
            memory_key=case["key"],
            value=case["old"],
            kind="preference",
            source_type="user",
            source_ref=f"{case['id']}/session-1",
        )
        if case.get("new"):
            memory_id = store.write(
                owner_id="eval-user",
                namespace="assistant",
                memory_key=case["key"],
                value=case["new"],
                kind="correction",
                source_type="user",
                source_ref=f"{case['id']}/correction",
            )
        if case["type"] == "deletion":
            store.delete(
                memory_id=memory_id,
                owner_id="eval-user",
                namespace="assistant",
            )

    started = time.perf_counter_ns()
    with MemoryStore(database, clock=lambda: 200) as reopened:
        recalled = reopened.recall(
            owner_id="eval-user",
            namespace="assistant",
            query=case["query"],
            limit=1,
        )
    elapsed_us = (time.perf_counter_ns() - started) / 1_000
    selected = recalled[0]["value"] if recalled else None
    overhead = math.ceil(len(selected) / 4) if selected else 0
    return selected, overhead, elapsed_us


def metrics(cases: list[dict], results: list[dict]) -> dict[str, float]:
    by_id = {row["id"]: row for row in results}

    def rate(selected_cases: list[dict], predicate) -> float:
        return sum(predicate(case, by_id[case["id"]]) for case in selected_cases) / len(selected_cases)

    corrections = [case for case in cases if case["type"] == "correction"]
    stale = [case for case in cases if case["type"] == "stale_conflict"]
    irrelevant = [case for case in cases if case["type"] == "irrelevant"]
    deleted = [case for case in cases if case["type"] == "deletion"]
    relevant = corrections + stale
    adherence = rate(corrections, lambda case, row: row["selected"] == case["expected"])
    return {
        "Correction Adherence": adherence,
        "Repeat-Mistake Rate": 1.0 - adherence,
        "Stale-Conflict Resolution": rate(stale, lambda case, row: row["selected"] == case["expected"]),
        "Irrelevant-Memory Intrusion": rate(irrelevant, lambda _, row: row["selected"] is not None),
        "Deletion Compliance": rate(deleted, lambda _, row: row["selected"] is None),
        "Recall@1": rate(relevant, lambda case, row: row["selected"] == case["expected"]),
        "Token Overhead (estimated)": statistics.mean(row["token_overhead"] for row in results),
        "Recall Latency (observed µs)": statistics.mean(row["latency_us"] for row in results),
    }


def evaluate() -> dict[str, dict[str, float]]:
    cases = load_cases()
    report: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for mode in ("no_memory", "similarity", "correction_aware"):
            rows = []
            for case in cases:
                selected, overhead, latency = (
                    correction_aware(case, directory)
                    if mode == "correction_aware"
                    else baseline(case, mode)
                )
                rows.append(
                    {
                        "id": case["id"],
                        "selected": selected,
                        "token_overhead": overhead,
                        "latency_us": latency,
                    }
                )
            report[mode] = metrics(cases, rows)
    return report


def main() -> None:
    report = evaluate()
    deterministic = [name for name in next(iter(report.values())) if "Latency" not in name]
    print("指标（除观测时延外均由固定数据与固定逻辑确定）")
    print("mode\t" + "\t".join(deterministic))
    for mode, values in report.items():
        print(mode + "\t" + "\t".join(f"{values[name]:.3f}" for name in deterministic))
    for mode, values in report.items():
        print(f"{mode} observed_latency_us={values['Recall Latency (observed µs)']:.1f}")


if __name__ == "__main__":
    main()
