import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app
from src.pipeline.cleaner import RetrievalCleaner
from src.pipeline.service import live_pipeline_service

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

@pytest.mark.asyncio
async def test_live_pipeline_service():
    card = await live_pipeline_service.process_query_live(
        query="Can I use my existing 700W charger with the Ather 450S?",
        vehicle_model="Ather 450S",
        top_k=5
    )
    assert card.direct_answer
    assert len(card.key_takeaways) > 0
    assert len(card.sources) > 0
    assert card.grounding_score >= 0.8

@pytest.mark.asyncio
async def test_api_live_query_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/query", json={
            "query": "What home charging time is listed for the Ather 450S?",
            "vehicle_model": "Ather 450S",
            "top_k": 5
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "direct_answer" in data
        assert "key_takeaways" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

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
