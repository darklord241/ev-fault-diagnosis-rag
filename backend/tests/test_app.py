import httpx

from fastapi.testclient import TestClient

from backend import app as backend_module
from scraper.vector_search.retrieval import SearchResult


def make_record(chunk_id, vehicle_model):
    return {
        "chunk_id": chunk_id,
        "record_id": f"{chunk_id}-record",
        "source_document": "service-manual.pdf",
        "source_type": "official service manual",
        "content_type": "pdf",
        "title": f"Manual section {chunk_id}",
        "text": "Check the battery controller diagnostic procedure.",
        "provenance": {
            "url": "https://example.test/manual",
            "vehicle_model": vehicle_model,
            "page_number": 12,
        },
        "metadata": {"section": "Battery", "subsystem": "BMS"},
    }


class FakeRetriever:
    def __init__(self):
        self.records = [
            make_record("ather-bms", "Ather 450X"),
            make_record("ola-bms", "Ola S1"),
            make_record("generic-bms", "Generic EV"),
        ]

    def search(self, query, top_k, *, record_filter):
        self.query = query
        matches = [record for record in self.records if record_filter(record)]
        return [
            SearchResult(score=0.91 - index * 0.1, record=record)
            for index, record in enumerate(matches[:top_k])
        ]


def test_live_search_requires_key_and_returns_vehicle_scoped_evidence():
    retriever = FakeRetriever()
    app = backend_module.create_app(
        retriever=retriever,
        backend_api_key="test-secret",
    )

    with TestClient(app) as client:
        unauthorized = client.post(
            "/api/v1/retrieval/search",
            json={"query": "battery fault", "top_k": 2},
        )
        response = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": "Bearer test-secret"},
            json={
                "query": "battery fault",
                "vehicle_model": "Ather 450X",
                "subsystem": "BMS",
                "top_k": 2,
            },
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert retriever.query == "BMS: battery fault"
    assert [item["source_id"] for item in response.json()["results"]] == [
        "ather-bms",
        "generic-bms",
    ]
    assert response.json()["results"][0]["document_title"] == "Manual section ather-bms"
    assert response.json()["results"][0]["page_number"] == 12
    assert response.json()["results"][0]["relevance_score"] == 0.91


def test_missing_corpus_reports_degraded_and_search_unavailable():
    def missing_corpus():
        raise FileNotFoundError("chunk corpus is missing")

    app = backend_module.create_app(
        retriever_factory=missing_corpus,
        backend_api_key="test-secret",
    )

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": "Bearer test-secret"},
            json={"query": "battery fault"},
        )

    assert health.status_code == 200
    assert health.json()["retrieval_ready"] is False
    assert health.json()["retrieval_error"]
    assert response.status_code == 503
    assert "Live retrieval is unavailable" in response.json()["detail"]


def test_diagnosis_gateway_forwards_to_agent_and_preserves_response(monkeypatch):
    captured = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def request(self, method, url, *, content, headers):
            captured.update(
                method=method,
                url=url,
                content=content,
                headers=headers,
            )
            return httpx.Response(
                201,
                json={"query": "battery fault", "evidence_sources": []},
            )

    monkeypatch.setattr(
        backend_module.httpx,
        "AsyncClient",
        lambda **_: Client(),
    )
    app = backend_module.create_app(agent_base_url="http://agent.test")
    payload = b'{"query":"battery fault"}'

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnose",
            content=payload,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 201
    assert response.json()["query"] == "battery fault"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://agent.test/api/v1/diagnose"
    assert captured["content"] == payload
    assert captured["headers"] == {"content-type": "application/json"}


def test_invalid_search_payload_is_rejected():
    app = backend_module.create_app(
        retriever=FakeRetriever(),
        backend_api_key="test-secret",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/retrieval/search",
            headers={"Authorization": "Bearer test-secret"},
            json={"query": "   "},
        )

    assert response.status_code == 422
