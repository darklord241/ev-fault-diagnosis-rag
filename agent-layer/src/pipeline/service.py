import asyncio
import uuid
from typing import List, Dict, Any, Optional

from src.pipeline.models import WebsiteCardPayload, QueryCleanedEvidence
from src.pipeline.cleaner import RetrievalCleaner
from src.pipeline.summarizer import EVSummarizer
from src.pipeline.history import diagnosis_history_store
from src.mcp_server.backend_adapter import backend_adapter
from src.common.logging import logger

class LivePipelineService:
    """
    Orchestrates the live online flow:
    1. Calls the generic Backend API to fetch N chunks (default 5).
    2. Cleans HTML, unescapes characters, and deduplicates redundant chunks.
    3. Runs LLM / Gemini summarization with grounded citations and specs.
    4. Delivers formatted WebsiteCardPayload to the website developer.
    """

    def __init__(self):
        self.cleaner = RetrievalCleaner()
        self.summarizer = EVSummarizer()

    async def process_query_live(
        self,
        query: str,
        vehicle_model: Optional[str] = None,
        top_k: int = 5,
        case_id: Optional[str] = None,
        subsystem: Optional[str] = None,
    ) -> WebsiteCardPayload:
        """
        Full live flow: Query -> Backend API (5 chunks) -> Cleaner -> Gemini Summarizer -> Website Payload.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10.")

        cid = case_id or f"live-{uuid.uuid4().hex[:8]}"
        logger.info(f"Processing live query: '{query}' (model: {vehicle_model}, top_k: {top_k})")

        # Note
        raw_chunks = await backend_adapter.fetch_raw_chunks_from_backend(
            query=query,
            top_k=top_k,
            vehicle_model=vehicle_model,
            subsystem=subsystem,
        )
        logger.info(f"Received {len(raw_chunks)} chunks from backend retrieval layer")

        # Note
        cleaned_evidence = self.cleaner.clean_api_chunks(
            raw_chunks=raw_chunks,
            query_text=query,
            case_id=cid
        )

        # Note
        card_payload = await self.summarizer.summarize(
            cleaned_evidence,
            vehicle_model=vehicle_model,
        )
        await asyncio.to_thread(
            diagnosis_history_store.save,
            query=query,
            vehicle_model=vehicle_model,
            subsystem=subsystem,
            result=card_payload,
        )

        logger.info(f"Successfully generated grounded website payload for '{query}' (Grounding: {card_payload.grounding_score})")
        return card_payload

    async def process_raw_chunks_live(
        self,
        query: str,
        raw_chunks: List[Dict[str, Any]],
        case_id: Optional[str] = None
    ) -> WebsiteCardPayload:
        """
        Direct chunk processing: accepts a list of 5 chunks provided directly in request body,
        cleans them, and produces the website summary.
        """
        cid = case_id or f"direct-{uuid.uuid4().hex[:8]}"
        logger.info(f"Processing {len(raw_chunks)} direct chunks for query: '{query}'")

        cleaned_evidence = self.cleaner.clean_api_chunks(
            raw_chunks=raw_chunks,
            query_text=query,
            case_id=cid
        )

        card_payload = await self.summarizer.summarize(cleaned_evidence)
        return card_payload

live_pipeline_service = LivePipelineService()
