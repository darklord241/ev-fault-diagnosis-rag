import json

import pytest

from src.common.models import QueryRequest
from src.orchestrator.agent import AgentOrchestrator


@pytest.mark.asyncio
@pytest.mark.parametrize("has_sources", [True, False])
async def test_orchestrator_fallback_returns_only_retrieved_evidence(monkeypatch, has_sources):
    evidence = [{
        "source_id": "fixture-source",
        "document_title": "Fixture manual",
        "content": "The manual describes a power-loss indicator.",
    }][:int(has_sources)]

    def unavailable(**kwargs):
        raise RuntimeError("Model unavailable")

    async def retrieve(*args):
        return json.dumps(evidence)

    monkeypatch.setattr("src.orchestrator.agent.LLMFactory.get_llm", unavailable)
    monkeypatch.setattr("src.orchestrator.mcp_client.tool_retrieve_technical_evidence", retrieve)
    result = await AgentOrchestrator().diagnose_fault(QueryRequest(query="Motor shuddering"))
    assert result.diagnostic_procedure == []
    assert result.identified_faults == []
    assert not result.recommended_remedy
    assert "grounding_score" not in result.model_dump()
    if has_sources:
        assert result.evidence_sources[0].content == evidence[0]["content"]
        assert result.root_cause_analysis == "A source passage is available for review."
        assert result.selected_source_id == evidence[0]["source_id"]
    else:
        assert result.evidence_sources == []
        assert result.selected_source_id is None
        assert result.root_cause_analysis == "No relevant source evidence was found."
