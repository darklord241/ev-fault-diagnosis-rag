import pytest
from src.orchestrator.agent import AgentOrchestrator
from src.orchestrator.grounding import GroundingEngine
from src.common.models import QueryRequest, EvidenceSource

@pytest.mark.asyncio
async def test_grounding_score_calculation():
    sample_evidence = [
        EvidenceSource(
            source_id="SRC-EV-BMS-001",
            document_title="BMS Guide",
            content="Check individual cell voltages. Delta V must not exceed 150mV. NTC thermistor is 10k.",
            relevance_score=0.9
        )
    ]
    response_text = "Refer to [SRC-EV-BMS-001]. Check if delta exceeds 150mV across cells and verify 10k thermistor."
    
    score = GroundingEngine.calculate_grounding_score(response_text, sample_evidence)
    assert 0.5 <= score <= 1.0

@pytest.mark.asyncio
async def test_orchestrator_diagnose_flow():
    orchestrator = AgentOrchestrator()
    req = QueryRequest(
        query="Motor shuddering at low RPM with error code E-08",
        vehicle_model="Ola S1",
        subsystem="Motor Controller"
    )
    result = await orchestrator.diagnose_fault(req)
    assert result.query == req.query
    assert len(result.evidence_sources) > 0
    assert result.grounding_score > 0.0
    assert len(result.diagnostic_procedure) > 0
