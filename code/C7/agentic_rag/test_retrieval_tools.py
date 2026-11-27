"""Exercise the demo's retrieval tools without external requests."""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.mark.parametrize("enable_web_search", [False, True])
def test_demo_retrieves_with_configured_tools(monkeypatch, enable_web_search):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    expected_tools = ["vector_search"]
    if enable_web_search:
        monkeypatch.setenv("TAVILY_API_KEY", "unused-test-key")
        expected_tools.append("web_search")

    web_result = {
        "title": "Test source",
        "url": "https://example.com/",
        "content": "Web evidence",
        "score": 0.9,
    }
    web_request = Mock(return_value={"results": [web_result]})
    monkeypatch.setattr(TavilySearchAPIWrapper, "raw_results", web_request)
    demo = importlib.import_module("demo")

    vectorstore = Mock()
    evidence = Document(page_content="Local evidence")
    vectorstore.similarity_search_with_score.return_value = [(evidence, 0.1)]
    vectorstore.similarity_search.return_value = [evidence]
    monkeypatch.setattr(demo, "build_knowledge_base", lambda **kwargs: vectorstore)
    monkeypatch.setattr(demo, "TEST_QUERIES", demo.TEST_QUERIES[:1])

    llm = FakeListChatModel(
        responses=[
            json.dumps({"initial_query": "question", "recommended_tools": expected_tools}),
            json.dumps({"is_sufficient": True, "sufficiency_score": 9}),
            "Agent answer",
            "Baseline answer",
        ]
    )
    monkeypatch.setattr(demo, "ChatOpenAI", lambda **kwargs: llm)

    run_agent = demo.AgenticRAG.run
    trajectory = []

    def capture_trajectory(agent, query):
        result = run_agent(agent, query)
        trajectory.extend(result["trajectory"])
        return result

    monkeypatch.setattr(demo.AgenticRAG, "run", capture_trajectory)
    demo.main()

    retrieval = next(step for step in trajectory if step["step"] == "retrieval_round_1")
    assert [entry["tool"] for entry in retrieval["results"]] == expected_tools
    assert "Local evidence" in retrieval["results"][0]["result"]
    vectorstore.similarity_search_with_score.assert_called_once_with("question", k=5)
    if enable_web_search:
        assert "Web evidence" in str(retrieval["results"][1]["result"])
        web_request.assert_called_once()
    else:
        web_request.assert_not_called()
