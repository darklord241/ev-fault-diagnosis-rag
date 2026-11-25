import pytest
from src.pipeline.cleaner import RetrievalCleaner
from src.pipeline.summarizer import EVSummarizer
from src.pipeline.formatter import WebsiteFormatter
from src.pipeline.batch_processor import BatchPipelineProcessor
from src.pipeline.models import CleanedChunk, QueryCleanedEvidence

@pytest.mark.asyncio
async def test_cleaner_text_and_deduplication():
    raw_html_text = "<pre>Can I use my existing Ather&#x27;s 700W charger? &quot;Yes!&quot;</pre>"
    cleaned = RetrievalCleaner.clean_text(raw_html_text)
    assert "<pre>" not in cleaned
    assert "Ather's" in cleaned
    assert '"Yes!"' in cleaned

    chunks = [
        CleanedChunk(
            chunk_id="chunk_1",
            title="Charger compatibility",
            content="Use the 700W charger. Recharging takes 4 hr 30 min.",
            similarity_score=0.89,
            url="https://ather.com/url1"
        ),
        CleanedChunk(
            chunk_id="chunk_2",
            title="Charger compatibility duplicate",
            content="Use the 700W charger. Recharging takes 4 hr 30 min.",
            similarity_score=0.88,
            url="https://ather.com/url2"
        )
    ]

    deduped = RetrievalCleaner.deduplicate_chunks(chunks)
    assert len(deduped) == 1
    assert deduped[0].duplicate_sources_merged == 2
    assert "https://ather.com/url2" in deduped[0].alternate_urls

@pytest.mark.asyncio
async def test_summarizer_fallback_and_schema(monkeypatch):
    monkeypatch.setattr("src.pipeline.summarizer.settings.GOOGLE_API_KEY", None)
    monkeypatch.setattr("src.pipeline.summarizer.settings.OPENAI_API_KEY", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    summarizer = EVSummarizer()
    chunk = CleanedChunk(
        chunk_id="c1",
        title="700W charger",
        content="700W charger is compatible. 0-100% takes 4 hr 30 min.",
        similarity_score=0.9
    )
    evidence = QueryCleanedEvidence(
        case_id="charging-compatible-700w",
        query="Can I use my existing 700W charger?",
        raw_chunk_count=1,
        cleaned_chunks=[chunk]
    )
    res = await summarizer.summarize(evidence)
    assert res.model_used == "deterministic-excerpt"
    assert res.sources[0].excerpt == chunk.content
    assert res.sources[0].chunk_id == chunk.chunk_id
    assert not res.technical_specs
    assert "grounding_score" not in res.model_dump()

@pytest.mark.asyncio
async def test_batch_processor_run(tmp_path):
    processor = BatchPipelineProcessor(
        output_json=tmp_path / "website_payload.json",
        output_md=tmp_path / "website_payload.md",
    )
    delivery = await processor.run_pipeline()
    assert delivery.total_cases == 5
    assert len(delivery.cases) == 5
    assert delivery.cases[0].case_id == "charging-compatible-700w"
