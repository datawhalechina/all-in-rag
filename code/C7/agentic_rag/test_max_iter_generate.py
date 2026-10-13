"""Regression tests for Agentic RAG max-iteration termination.

When the retrieval budget is exhausted, the graph must still enter
``_generate`` instead of ending with an empty ``final_answer``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import AgenticRAG


PLAN = json.dumps(
    {
        "complexity": "simple",
        "sub_queries": ["question"],
        "dependencies": {},
        "recommended_tools": ["vector_search"],
        "initial_query": "question",
    }
)


def _critique(is_sufficient: bool, score: int, suggested_rewrite: str = "question rewritten") -> str:
    return json.dumps(
        {
            "relevance_score": score,
            "sufficiency_score": score,
            "is_sufficient": is_sufficient,
            "missing_aspects": [] if is_sufficient else ["more evidence"],
            "failure_reason": "" if is_sufficient else "not enough",
            "suggested_rewrite": suggested_rewrite,
        }
    )


@tool("vector_search")
def search(query: str) -> str:
    """Return deterministic retrieval evidence."""
    return f"Evidence for {query}"


def _run(responses: list[str], max_iterations: int, verbose: bool = False):
    llm = FakeListChatModel(responses=responses)
    return AgenticRAG(
        llm=llm, tools=[search], max_iterations=max_iterations, verbose=verbose
    ).run("question")


def _steps(result: dict) -> list[str]:
    return [step["step"] for step in result["trajectory"]]


def test_max_iter_still_generates_when_sufficient():
    result = _run([PLAN, _critique(True, 9), "Expected final answer"], max_iterations=1)
    assert result["answer"] == "Expected final answer"
    assert _steps(result) == ["planning", "retrieval_round_1", "critique", "generate"]
    assert result["iterations"] == 1


def test_max_iter_still_generates_when_insufficient():
    result = _run(
        [PLAN, _critique(False, 3), "Partial answer with incomplete-info note"],
        max_iterations=1,
    )
    assert result["answer"] == "Partial answer with incomplete-info note"
    assert "generate" in _steps(result)
    assert result["iterations"] == 1
    assert result["collected_docs_count"] == 1


def test_early_sufficient_does_not_use_full_budget():
    result = _run([PLAN, _critique(True, 9), "Early answer"], max_iterations=3)
    assert result["answer"] == "Early answer"
    assert _steps(result) == ["planning", "retrieval_round_1", "critique", "generate"]
    assert result["iterations"] == 1


def test_default_budget_generates_once_at_limit():
    result = _run(
        [
            PLAN,
            _critique(False, 3, "question round 2"),
            _critique(False, 4, "question round 3"),
            _critique(False, 5, "unused"),
            "Forced answer after three rounds",
        ],
        max_iterations=3,
    )
    assert result["answer"] == "Forced answer after three rounds"
    assert result["iterations"] == 3
    assert _steps(result).count("generate") == 1
    assert [s for s in _steps(result) if s.startswith("retrieval_round_")] == [
        "retrieval_round_1",
        "retrieval_round_2",
        "retrieval_round_3",
    ]


def test_stream_includes_generate_event():
    llm = FakeListChatModel(responses=[PLAN, _critique(True, 9), "Streamed answer"])
    events = list(
        AgenticRAG(
            llm=llm, tools=[search], max_iterations=1, verbose=False
        ).run_stream("question")
    )
    node_names = [name for event in events for name in event]
    assert "generate" in node_names
    generate_state = next(event["generate"] for event in events if "generate" in event)
    assert generate_state["final_answer"] == "Streamed answer"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
