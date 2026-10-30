import json
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi import Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.common.models import QueryRequest, GroundedAnswer, DTCRecord
from src.pipeline.models import WebsiteCardPayload, BatchWebsiteDelivery, CleanedChunk, QueryCleanedEvidence
from src.pipeline.cleaner import RetrievalCleaner
from src.pipeline.summarizer import EVSummarizer
from src.pipeline.batch_processor import BatchPipelineProcessor
from src.pipeline.service import live_pipeline_service
from src.pipeline.history import diagnosis_history_store
from src.orchestrator.agent import AgentOrchestrator
from src.mcp_server.backend_adapter import backend_adapter
from src.common.logging import logger
from config.settings import settings

router = APIRouter(prefix="/api/v1", tags=["EV Fault Diagnosis, Retrieval & Website Delivery"])

# Note
orchestrator = AgentOrchestrator()
batch_processor = BatchPipelineProcessor()
summarizer = EVSummarizer()
cleaner = RetrievalCleaner()

class LiveQueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    vehicle_model: Optional[str] = None
    subsystem: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)
    case_id: Optional[str] = None

class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    vehicle_model: Optional[str] = Field(default=None, max_length=200)
    subsystem: Optional[str] = Field(default=None, max_length=200)

class KnowledgeChunkResult(BaseModel):
    chunk_id: str
    title: str
    source_type: str
    content_type: str
    content: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    vehicle_model: Optional[str] = None
    subsystem: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float

class KnowledgeSearchResponse(BaseModel):
    query: str
    chunks: List[KnowledgeChunkResult]

class DirectChunksRequest(BaseModel):
    query: str
    chunks: List[Dict[str, Any]]
    case_id: Optional[str] = None

class OnDemandSummarizeRequest(BaseModel):
    case_id: str
    query: str
    raw_markdown_or_text: str

@router.get("/health", summary="Health and Status Check")
async def health_check() -> Dict[str, Any]:
    """Returns the operational status of the Orchestrator, LLM Provider, and MCP layer."""
    return {
        "status": "healthy",
        "service": "EV Fault Diagnosis Agent Orchestrator & Evidence Pipeline",
        "default_llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "default_llm_model": settings.DEFAULT_LLM_MODEL,
        "mcp_transport": settings.MCP_TRANSPORT,
        "backend_retrieval_mode": "mock_domain_db" if settings.HYBRID_RETRIEVAL_USE_MOCK else "live_hybrid_api",
        "active_mcp_tools": [t.name for t in orchestrator.tools]
    }

@router.post("/diagnose", response_model=GroundedAnswer, summary="Execute AI Fault Diagnosis")
async def diagnose_fault(request: QueryRequest) -> GroundedAnswer:
    """
    Main endpoint for Frontend/Dashboard interactive diagnosis.
    Processes a diagnostic query through the Agent Orchestrator + MCP Retrieval tools.
    """
    try:
        result = await orchestrator.diagnose_fault(request)
        return result
    except Exception as e:
        logger.error(f"Error during fault diagnosis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diagnostic orchestrator error: {str(e)}")

# Note
# Note
# Note

@router.post("/query", response_model=WebsiteCardPayload, summary="Live Query: Calls Backend API (5 chunks) -> Cleans -> Gemini Summarizes -> Returns Website Payload")
async def process_live_query(req: LiveQueryRequest) -> WebsiteCardPayload:
    """
    Live end-to-end endpoint for the website:
    1. Calls generic backend retrieval API to get 5 separate chunks.
    2. Cleans HTML tags, whitespace, and deduplicates.
    3. Runs Gemini LLM to synthesize direct answer, specs, takeaways, and citations.
    4. Delivers formatted WebsiteCardPayload for instant UI rendering.
    """
    try:
        result = await live_pipeline_service.process_query_live(
            query=req.query,
            vehicle_model=req.vehicle_model,
            top_k=req.top_k,
            case_id=req.case_id,
            subsystem=req.subsystem,
        )
        return result
    except Exception as e:
        logger.error(f"Error processing live query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Live query processing error: {str(e)}")

@router.post(
    "/knowledge/search",
    response_model=KnowledgeSearchResponse,
    summary="Retrieve the five most relevant source chunks for the knowledge base",
)
async def search_knowledge_base(req: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    try:
        raw_chunks = await backend_adapter.fetch_raw_chunks_from_backend(
            query=req.query,
            top_k=5,
            vehicle_model=req.vehicle_model,
            subsystem=req.subsystem,
        )
        chunks = []
        for item in raw_chunks[:5]:
            provenance = item.get("provenance") or {}
            metadata = item.get("metadata") or {}
            if not isinstance(provenance, dict):
                provenance = {}
            if not isinstance(metadata, dict):
                metadata = {}
            page_number = metadata.get("page_number") or metadata.get("page")
            if isinstance(page_number, bool) or not isinstance(page_number, int):
                page_number = None
            section = metadata.get("section") or item.get("section")
            if not isinstance(section, str):
                section = None
            vehicle_model = provenance.get("vehicle_model") or item.get("vehicle_model")
            subsystem = metadata.get("subsystem") or item.get("subsystem")
            url = provenance.get("url") or item.get("url")
            chunks.append(
                KnowledgeChunkResult(
                    chunk_id=item["chunk_id"],
                    title=item.get("title") or item.get("source_document") or "Untitled source",
                    source_type=item.get("source_type") or "unknown",
                    content_type=item.get("content_type") or "unknown",
                    content=item["text"],
                    page_number=page_number,
                    section=section,
                    vehicle_model=vehicle_model if isinstance(vehicle_model, str) else None,
                    subsystem=subsystem if isinstance(subsystem, str) else None,
                    url=url if isinstance(url, str) else None,
                    relevance_score=float(item.get("score", 0.0)),
                )
            )
        return KnowledgeSearchResponse(query=req.query, chunks=chunks)
    except Exception as e:
        logger.error("Knowledge-base retrieval failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Knowledge-base retrieval is unavailable. Check retrieval backend readiness.",
        ) from e

@router.get("/history", summary="Get recent live diagnosis history")
async def get_diagnosis_history(limit: int = Query(default=20, ge=1, le=50)):
    """Return locally stored website query results, newest first."""
    try:
        return {
            "items": await asyncio.to_thread(
                diagnosis_history_store.list_recent,
                limit,
            )
        }
    except Exception as e:
        logger.error("Could not read diagnosis history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read diagnosis history.")

@router.delete("/history", summary="Clear local live diagnosis history")
async def clear_diagnosis_history():
    """Delete locally stored website query results."""
    try:
        deleted = await asyncio.to_thread(diagnosis_history_store.clear)
        return {"deleted": deleted}
    except Exception as e:
        logger.error("Could not clear diagnosis history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not clear diagnosis history.")

@router.post("/process-chunks", response_model=WebsiteCardPayload, summary="Process 5 Separate Chunks Directly: Cleans -> Gemini Summarizes -> Returns Website Payload")
async def process_chunks_direct(req: DirectChunksRequest) -> WebsiteCardPayload:
    """
    Accepts 5 separate chunks directly from the backend API teammate,
    cleans them, generates a Gemini grounded summary, and formats for website display.
    """
    try:
        result = await live_pipeline_service.process_raw_chunks_live(
            query=req.query,
            raw_chunks=req.chunks,
            case_id=req.case_id
        )
        return result
    except Exception as e:
        logger.error(f"Error processing raw chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Direct chunk processing error: {str(e)}")

# Note
# Note
# Note

@router.get("/eval-answers", response_model=BatchWebsiteDelivery, summary="Get Cleaned & Summarized Eval Answers for Website")
async def get_eval_answers():
    """
    Delivers all cleaned, deduplicated, and Gemini-summarized answers
    ready for the website teammate to display on cards, tables, and markdown views.
    """
    payload_file = Path("data/website_payload.json")
    if not payload_file.exists():
        logger.info("Website payload not found on disk. Generating fresh batch...")
        delivery = await batch_processor.run_pipeline()
        return delivery

    try:
        with open(payload_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BatchWebsiteDelivery(**data)
    except Exception as e:
        logger.error(f"Error reading website payload: {e}")
        delivery = await batch_processor.run_pipeline()
        return delivery

@router.get("/eval-answers/{case_id}", response_model=WebsiteCardPayload, summary="Get Specific Case Answer for Website")
async def get_case_answer(case_id: str):
    """Fetches a specific evaluation case payload by case ID (e.g. charging-compatible-700w)."""
    batch = await get_eval_answers()
    for case in batch.cases:
        if case.case_id.lower() == case_id.lower():
            return case
    raise HTTPException(status_code=404, detail=f"Case ID '{case_id}' not found.")

@router.post("/pipeline/run-batch", response_model=BatchWebsiteDelivery, summary="Run Full Cleaning, Summarizing & Website Export Pipeline")
async def trigger_pipeline_batch():
    """
    Runs the entire pipeline on data/retrieval_baseline.md:
    1. Parses & cleans raw output from previous person.
    2. Deduplicates identical chunks.
    3. Runs LLM / Gemini summarization with grounding.
    4. Formats and saves data/website_payload.json and data/website_payload.md.
    """
    try:
        delivery = await batch_processor.run_pipeline()
        return delivery
    except Exception as e:
        logger.error(f"Batch pipeline execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@router.post("/summarize", response_model=WebsiteCardPayload, summary="On-Demand Cleaning & Summarization of Raw Retrieval Text")
async def summarize_on_demand(req: OnDemandSummarizeRequest):
    """
    Cleans raw markdown or text from retrieval person on the fly,
    runs Gemini summarization, and returns structured website payload.
    """
    try:
        cleaned_text = cleaner.clean_text(req.raw_markdown_or_text)
        chunk = CleanedChunk(
            chunk_id=f"chunk_{req.case_id}_0",
            title=req.query,
            content=cleaned_text,
            similarity_score=0.90,
            content_type="retrieval_text"
        )
        evidence = QueryCleanedEvidence(
            case_id=req.case_id,
            query=req.query,
            raw_chunk_count=1,
            cleaned_chunks=[chunk]
        )
        return await summarizer.summarize(evidence)
    except Exception as e:
        logger.error(f"On-demand summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

# Note
# Note
# Note

@router.get("/mcp/tools", summary="List MCP Retrieval Tools")
async def list_mcp_tools() -> Dict[str, Any]:
    """Exposes all connected MCP retrieval tools and descriptions."""
    return {
        "mcp_tools": [
            {
                "name": t.name,
                "description": t.description,
                "args_schema": t.args_schema.model_json_schema() if t.args_schema and hasattr(t.args_schema, "model_json_schema") else (t.args_schema.schema() if t.args_schema else {})
            }
            for t in orchestrator.tools
        ]
    }

@router.get("/dtc/{code}", summary="Direct DTC Code Lookup")
async def lookup_dtc(code: str) -> Dict[str, Any]:
    """Direct lookup for electric two-wheeler Diagnostic Trouble Codes."""
    dtc_record = await backend_adapter.get_dtc_details(code)
    if not dtc_record:
        raise HTTPException(status_code=404, detail=f"DTC code '{code}' not found.")
    return dtc_record.model_dump()
