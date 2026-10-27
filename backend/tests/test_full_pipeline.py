from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient as RealAsyncClient

from backend import app as backend_module
from scraper.vector_search.retrieval import SearchResult
from src.api.main import app as agent_app
from src.mcp_server.backend_adapter import backend_adapter
from src.pipeline.history import diagnosis_history_store
from config.settings import settings


class FixtureRetriever:
    def search(self, query, top_k, *, record_filter):
        assert "battery" in query.lower()
        record = {
            "chunk_id": "fixture-battery-chunk",
            "record_id": "fixture-battery-record",
            "source_document": "fixture-manual.pdf",
            "source_type": "service manual",
            "content_type": "pdf",
            "title": "Battery troubleshooting section",
            "text": "The manual says to inspect battery connector continuity.",
            "provenance": {
                "url": "https://example.test/fixture-manual.pdf",
                "vehicle_model": "Ather 450S",
            },
            "metadata": {
                "page_number": 12,
                "section": "Battery connections",
                "subsystem": "BMS",
            },
        }
        return (
            [SearchResult(score=0.82, record=record)]
            if top_k > 0 and record_filter(record)
            else []
        )


class LayerRoutingClient:
    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def request(self, method, url, *, content=None, headers=None, json=None):
        if url.startswith("http://agent.test"):
            app = agent_app
        elif url.startswith("http://retrieval.test"):
            app = retrieval_app
        else:
            raise AssertionError(f"Unexpected service URL: {url}")

        async with RealAsyncClient(
            transport=ASGITransport(app=app),
            base_url=url.split("/api/", 1)[0],
        ) as client:
            return await client.request(
                method,
                url,
                content=content,
                headers=headers,
                json=json,
            )

    async def post(self, url, *, json, headers):
        return await self.request("POST", url, json=json, headers=headers)


retrieval_app = backend_module.create_app(
    retriever=FixtureRetriever(),
    backend_api_key="integration-test-key",
    agent_base_url="http://agent.test",
)


def test_full_website_gateway_retrieval_agent_and_database_flow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(backend_module.httpx, "AsyncClient", LayerRoutingClient)
    monkeypatch.setattr(backend_adapter, "use_mock", False)
    monkeypatch.setattr(
        backend_adapter,
        "base_url",
        "http://retrieval.test/api/v1/retrieval",
    )
    monkeypatch.setattr(backend_adapter, "api_key", "integration-test-key")
    monkeypatch.setattr(settings, "HYBRID_RETRIEVAL_USE_MOCK", False)
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        diagnosis_history_store,
        "path",
        tmp_path / "diagnosis-history.sqlite3",
    )

    website_gateway = backend_module.create_app(
        retriever=FixtureRetriever(),
        backend_api_key="integration-test-key",
        agent_base_url="http://agent.test",
    )
    with TestClient(website_gateway) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Battery loses power under acceleration",
                "vehicle_model": "Ather 450S",
                "subsystem": "BMS",
                "top_k": 3,
            },
        )
        history_response = client.get("/api/v1/history")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["query"] == "Battery loses power under acceleration"
    assert result["vehicle_model"] == "Ather 450S"
    assert result["sources"][0]["chunk_id"] == "fixture-battery-chunk"
    assert result["sources"][0]["url"] == "https://example.test/fixture-manual.pdf"
    assert result["sources"][0]["similarity_score"] == 0.82
    assert result["model_used"] == "deterministic-domain-engine"
    assert history_response.status_code == 200
    assert history_response.json()["items"][0]["result"]["case_id"] == result["case_id"]
