import json

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app
from src.pipeline.cleaner import RetrievalCleaner
from src.pipeline.history import diagnosis_history_store
from src.pipeline.service import live_pipeline_service
from src.pipeline.summarizer import EVSummarizer
from src.mcp_server.backend_adapter import backend_adapter

SAMPLE_USER_CHUNK = {
    "chunk_id": "2025-450S-QSG_page_1_chunk_0",
    "record_id": "2025-450S-QSG_page_1",
    "source_document": "2025-450S-QSG",
    "source_type": "official",
    "content_type": "pdf",
    "title": "2025-450S-QSG",
    "text": "Q U I C K S TA R T G U I D E - 700W charger is backwards compatible with Ather 450S.",
    "provenance": {
        "document_id": "2025-450S-QSG",
        "url": "https://media.atherenergy.com/2025-450S-QSG.pdf",
        "source_type": "official",
        "document_type": "quick_start_guide",
        "manufacturer": "Ather Energy",
        "vehicle_model": "Ather 450S",
        "retrieved_at": "2026-10-26T07:21:12.151865+00:00"
    },
    "metadata": {
        "category": None,
        "group": None,
        "page": 1
    }
}

@pytest.fixture
def history_database(tmp_path, monkeypatch):
    monkeypatch.setattr(
        diagnosis_history_store,
        "path",
        tmp_path / "diagnosis-history.sqlite3",
    )

@pytest.mark.asyncio
async def test_clean_api_chunks():
    chunks = [SAMPLE_USER_CHUNK]
    evidence = RetrievalCleaner.clean_api_chunks(chunks, query_text="Can I use 700W charger?")
    
    assert evidence.raw_chunk_count == 1
    assert len(evidence.cleaned_chunks) == 1
    cleaned = evidence.cleaned_chunks[0]
    assert cleaned.chunk_id == "2025-450S-QSG_page_1_chunk_0"
    assert "700W charger is backwards compatible" in cleaned.content
    assert cleaned.vehicle_model == "Ather 450S"
    assert cleaned.url == "https://media.atherenergy.com/2025-450S-QSG.pdf"

def test_clean_api_chunks_maps_backend_evidence_contract():
    evidence = RetrievalCleaner.clean_api_chunks(
        [{
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
        }],
        query_text="battery fault",
    )

    chunk = evidence.cleaned_chunks[0]
    assert chunk.chunk_id == "chunk-42"
    assert chunk.title == "Battery service guide"
    assert chunk.document_id == "record-42"
    assert chunk.url == "https://example.test/service-guide.pdf"
    assert chunk.vehicle_model == "Ather 450S"
    assert chunk.similarity_score == 0.73
    assert chunk.source_metadata["chunk_metadata"]["page_number"] == 42

def test_clean_api_chunks_rejects_missing_ids_and_empty_content():
    for chunk in (
        {"content": "evidence without stable source id"},
        {"source_id": "empty-content", "content": "   "},
    ):
        with pytest.raises(ValueError):
            RetrievalCleaner.clean_api_chunks([chunk])


@pytest.mark.asyncio
async def test_gemini_text_blocks_are_parsed_as_structured_summary(monkeypatch):
    class FakeResponse:
        content = [{
            "type": "text",
            "text": json.dumps({
                "direct_answer": "The cited manual describes a low-charge power reduction.",
                "key_takeaways": ["Check the dashboard charge level."],
                "technical_specs": {},
                "conditions_and_safety": ["Use an authorized service center for repairs."],
                "formatted_markdown": "### Answer\nThe manual describes a low-charge power reduction.",
                "grounding_confidence": 0.8,
            }),
        }]

    class FakeLLM:
        async def ainvoke(self, _messages):
            return FakeResponse()

    monkeypatch.setattr(
        "src.pipeline.summarizer.LLMFactory.get_llm",
        lambda **_kwargs: FakeLLM(),
    )
    evidence = RetrievalCleaner.clean_api_chunks(
        [SAMPLE_USER_CHUNK],
        query_text="Battery loses power under acceleration",
        case_id="gemini-test",
    )
    summarizer = EVSummarizer(
        provider="gemini",
        model="gemini-3.5-flash-lite",
    )

    result = await summarizer._call_llm_summarizer(
        evidence,
        vehicle_model="Ather 450S",
    )

    assert result.model_used == "gemini-3.5-flash-lite"
    assert result.direct_answer.startswith("The cited manual")
    assert result.sources[0].chunk_id == SAMPLE_USER_CHUNK["chunk_id"]
    assert result.sources[0].page_number == 1
    assert "700W charger is backwards compatible" in result.sources[0].content


@pytest.mark.asyncio
async def test_knowledge_search_returns_five_readable_ranked_chunks(monkeypatch):
    requested = {}

    async def fetch_chunks(**kwargs):
        requested.update(kwargs)
        return [
            {
                "chunk_id": f"manual_page_{index}_chunk_0",
                "title": f"Service manual {index}",
                "source_type": "official",
                "content_type": "pdf",
                "text": f"Full evidence text for page {index}.",
                "score": 0.9 - index / 100,
                "provenance": {"url": f"https://example.test/manual#{index}"},
                "metadata": {"page": index, "section": "Battery service"},
            }
            for index in range(1, 8)
        ]

    monkeypatch.setattr(backend_adapter, "fetch_raw_chunks_from_backend", fetch_chunks)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/knowledge/search",
            json={
                "query": "battery service",
                "vehicle_model": "Ather 450S",
                "subsystem": "BMS",
            },
        )

    assert response.status_code == 200
    assert requested == {
        "query": "battery service",
        "top_k": 5,
        "vehicle_model": "Ather 450S",
        "subsystem": "BMS",
    }
    results = response.json()["chunks"]
    assert len(results) == 5
    assert results[0]["page_number"] == 1
    assert results[0]["content"] == "Full evidence text for page 1."
    assert results[0]["url"] == "https://example.test/manual#1"


@pytest.mark.asyncio
async def test_empty_evidence_returns_explicit_no_evidence_payload():
    evidence = RetrievalCleaner.clean_api_chunks(
        [],
        query_text="battery fault",
        case_id="no-evidence",
    )

    result = await EVSummarizer().summarize(evidence, vehicle_model="Ather 450S")

    assert result.model_used == "no-evidence"
    assert result.grounding_score == 0
    assert result.sources == []
    assert result.vehicle_model == "Ather 450S"
    assert "No relevant source evidence" in result.direct_answer

@pytest.mark.asyncio
async def test_live_pipeline_service(history_database, monkeypatch):
    monkeypatch.setattr("src.pipeline.summarizer.settings.GOOGLE_API_KEY", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    card = await live_pipeline_service.process_query_live(
        query="Can I use my existing 700W charger with the Ather 450S?",
        vehicle_model="Ather 450S",
        top_k=5
    )
    assert card.direct_answer
    assert len(card.key_takeaways) > 0
    assert len(card.sources) > 0
    assert card.grounding_score >= 0.8
    assert diagnosis_history_store.list_recent()[0]["case_id"] == card.case_id

@pytest.mark.asyncio
async def test_api_live_query_endpoint(history_database, monkeypatch):
    monkeypatch.setattr("src.pipeline.summarizer.settings.GOOGLE_API_KEY", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/query", json={
            "query": "What home charging time is listed for the Ather 450S?",
            "vehicle_model": "Ather 450S",
            "subsystem": "Charging",
            "top_k": 5
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "direct_answer" in data
        assert "key_takeaways" in data
        assert "sources" in data
        assert len(data["sources"]) > 0
        history = await client.get("/api/v1/history")
        assert history.status_code == 200
        assert history.json()["items"][0]["query"] == "What home charging time is listed for the Ather 450S?"
        cleared = await client.delete("/api/v1/history")
        assert cleared.status_code == 200
        assert cleared.json()["deleted"] == 1

@pytest.mark.asyncio
async def test_api_direct_chunks_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/process-chunks", json={
            "query": "Is FallSafe active?",
            "chunks": [SAMPLE_USER_CHUNK]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "direct_answer" in data
        assert data["sources"][0]["chunk_id"] == "2025-450S-QSG_page_1_chunk_0"
