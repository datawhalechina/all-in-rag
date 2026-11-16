"""A dependency-free toy demo of CRAG's three routing branches.

This is not the paper's learned evaluator or web-search implementation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s*")


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    documents: tuple[str, ...]


MOCK_WEB_RESULTS = (
    "[MOCK web] Corrective RAG evaluates retrieved evidence before generation.",
)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def overlap_score(query: str, document: str) -> float:
    """A deterministic lexical-overlap surrogate, not a relevance model."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & tokenize(document)) / len(query_tokens)


def choose_route(
    scores: Iterable[float], upper_threshold: float = 0.80, lower_threshold: float = 0.20
) -> str:
    values = list(scores)
    if not values:
        return "INCORRECT"
    if any(score > upper_threshold for score in values):
        return "CORRECT"
    if all(score < lower_threshold for score in values):
        return "INCORRECT"
    return "AMBIGUOUS"


def refine_internal(
    query: str, documents: Iterable[str], strip_threshold: float = 0.34
) -> list[str]:
    """Split into sentence-like strips and retain those above a toy threshold."""
    kept: list[str] = []
    for document in documents:
        for strip in SENTENCE_RE.split(document.strip()):
            strip = strip.strip()
            if strip and overlap_score(query, strip) >= strip_threshold:
                kept.append(strip)
    return kept


def run_case(case: Case) -> None:
    scores = [overlap_score(case.query, document) for document in case.documents]
    route = choose_route(scores)

    if route == "CORRECT":
        context = [f"[internal] {strip}" for strip in refine_internal(case.query, case.documents)]
    elif route == "INCORRECT":
        context = list(MOCK_WEB_RESULTS)
    else:
        internal = [
            f"[internal] {strip}" for strip in refine_internal(case.query, case.documents)
        ]
        context = internal + list(MOCK_WEB_RESULTS)

    rendered_scores = ", ".join(f"{score:.3f}" for score in scores)
    print(f"[{case.name}]")
    print(f"document scores: [{rendered_scores}]")
    print(f"route: {route}")
    print("context:")
    for item in context:
        print(f"- {item}")
    print()


def main() -> None:
    query = "CRAG evaluator confidence"
    cases = (
        Case(
            name="correct",
            query=query,
            documents=(
                "CRAG evaluator assigns confidence scores to retrieved documents. "
                "A mango is a tropical fruit.",
            ),
        ),
        Case(
            name="incorrect",
            query=query,
            documents=("Fresh mango trees need warm weather.",),
        ),
        Case(
            name="ambiguous",
            query=query,
            documents=("The CRAG evaluator estimates document relevance.",),
        ),
    )
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()


