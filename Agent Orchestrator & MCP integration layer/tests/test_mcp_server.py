import pytest
import json
import importlib
from src.mcp_server.tools import (
    tool_retrieve_technical_evidence,
    tool_lookup_dtc_fault_code,
    tool_lookup_component_specs
)
from src.mcp_server.backend_adapter import backend_adapter
backend_adapter_module = importlib.import_module("src.mcp_server.backend_adapter")

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

@pytest.mark.asyncio
async def test_live_backend_search_sends_bearer_api_key(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": []}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, *, json, headers):
            captured.update(url=url, payload=json, headers=headers)
            return Response()

    monkeypatch.setattr(
        backend_adapter_module.httpx,
        "AsyncClient",
        lambda **_: Client(),
    )
    adapter = backend_adapter_module.BackendRetrievalAdapter()
    adapter.use_mock = False
    adapter.base_url = "http://retrieval.example/api/v1/retrieval"
    adapter.api_key = "test-backend-key"

    results = await adapter.search_hybrid_evidence(
        query="battery loses power",
        vehicle_model="Ather 450X",
        subsystem="Battery/BMS",
    )

    assert results == []
    assert captured["url"] == "http://retrieval.example/api/v1/retrieval/search"
    assert captured["headers"]["Authorization"] == "Bearer test-backend-key"
    assert captured["payload"] == {
        "query": "battery loses power",
        "vehicle_model": "Ather 450X",
        "subsystem": "Battery/BMS",
        "top_k": 4,
    }


@pytest.mark.asyncio
async def test_live_backend_failure_does_not_fall_back_to_sample_evidence(monkeypatch):
    class Response:
        status_code = 503

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(
        backend_adapter_module.httpx,
        "AsyncClient",
        lambda **_: Client(),
    )
    adapter = backend_adapter_module.BackendRetrievalAdapter()
    adapter.use_mock = False
    adapter.base_url = "http://retrieval.example/api/v1/retrieval"
    adapter.api_key = "test-backend-key"

    with pytest.raises(RuntimeError, match="sample evidence was not used"):
        await adapter.search_hybrid_evidence(query="battery loses power")
