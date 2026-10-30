import pytest
import json
from src.mcp_server.tools import (
    tool_retrieve_technical_evidence,
    tool_lookup_dtc_fault_code,
    tool_lookup_component_specs
)
from src.mcp_server.backend_adapter import backend_adapter
import importlib
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
async def test_lookup_generic_dtc_fault_code():
    res_str = await tool_lookup_dtc_fault_code(" p0300 ")
    data = json.loads(res_str)
    assert data.get("code") == "P0300"
    assert "misfire" in data.get("description", "").lower()

@pytest.mark.asyncio
async def test_lookup_component_specs():
    res_str = await tool_lookup_component_specs("throttle")
    data = json.loads(res_str)
    assert "pinout_details" in data
    assert "nominal_operating_specs" in data

@pytest.mark.asyncio
async def test_live_pipeline_adapter_maps_backend_contract_and_passes_filters(monkeypatch):
    captured = {}
    source = {
        "source_id": "chunk-42",
        "document_title": "Battery service guide",
        "document_type": "service manual",
        "section": "Cell balancing",
        "page_number": 42,
        "vehicle_model": "Ather 450S",
        "subsystem": "BMS",
        "content": "Check cell voltage deviation under load.",
        "relevance_score": 0.73,
        "metadata": {
            "record_id": "record-42",
            "source_document": "service-guide.pdf",
            "content_type": "pdf",
            "provenance": {
                "url": "https://example.test/service-guide.pdf",
                "manufacturer": "Ather",
                "vehicle_model": "Ather 450S",
            },
            "chunk_metadata": {"page_number": 42},
        },
    }

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": [source]}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, *, json, headers):
            captured.update(url=url, payload=json, headers=headers)
            return Response()

    monkeypatch.setattr(backend_adapter_module.httpx, "AsyncClient", lambda **_: Client())
    adapter = backend_adapter_module.BackendRetrievalAdapter()
    adapter.use_mock = False
    adapter.base_url = "http://retrieval.test/api/v1/retrieval/"
    adapter.api_key = "test-key"

    chunks = await adapter.fetch_raw_chunks_from_backend(
        query="battery fault",
        vehicle_model="Ather 450S",
        subsystem="BMS",
        top_k=3,
    )

    assert captured["url"] == "http://retrieval.test/api/v1/retrieval/search"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"] == {
        "query": "battery fault",
        "vehicle_model": "Ather 450S",
        "subsystem": "BMS",
        "top_k": 3,
    }
    assert chunks[0]["chunk_id"] == "chunk-42"
    assert chunks[0]["title"] == "Battery service guide"
    assert chunks[0]["provenance"]["url"] == "https://example.test/service-guide.pdf"
    assert chunks[0]["score"] == 0.73

@pytest.mark.asyncio
async def test_live_pipeline_adapter_does_not_fall_back_on_backend_error(monkeypatch):
    class Response:
        status_code = 503

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(backend_adapter_module.httpx, "AsyncClient", lambda **_: Client())
    adapter = backend_adapter_module.BackendRetrievalAdapter()
    adapter.use_mock = False
    adapter.api_key = "test-key"

    with pytest.raises(RuntimeError, match="HTTP 503"):
        await adapter.fetch_raw_chunks_from_backend(query="battery fault")
