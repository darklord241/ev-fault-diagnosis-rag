import pytest
import json
from src.mcp_server.tools import (
    tool_retrieve_technical_evidence,
    tool_lookup_dtc_fault_code,
    tool_lookup_component_specs
)
from src.mcp_server.backend_adapter import backend_adapter

@pytest.mark.asyncio
async def test_retrieve_technical_evidence():
    result_str = await tool_retrieve_technical_evidence(
        query="throttle cut off after rain",
        subsystem="Throttle",
        top_k=2
    )
    data = json.loads(result_str)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "source_id" in data[0]
    assert "document_title" in data[0]
    assert "content" in data[0]

@pytest.mark.asyncio
async def test_lookup_dtc_fault_code():
    res_str = await tool_lookup_dtc_fault_code("P0A80")
    data = json.loads(res_str)
    assert data.get("code") == "P0A80"
    assert "BMS" in data.get("system", "")

@pytest.mark.asyncio
async def test_lookup_component_specs():
    res_str = await tool_lookup_component_specs("throttle")
    data = json.loads(res_str)
    assert "pinout_details" in data
    assert "nominal_operating_specs" in data
