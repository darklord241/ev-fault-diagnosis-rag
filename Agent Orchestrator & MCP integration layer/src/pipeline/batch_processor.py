import os
import json
import asyncio
from pathlib import Path
from typing import List, Optional

from src.pipeline.models import BatchWebsiteDelivery, WebsiteCardPayload
from src.pipeline.cleaner import RetrievalCleaner
from src.pipeline.summarizer import EVSummarizer
from src.pipeline.formatter import WebsiteFormatter
from src.common.logging import logger

class BatchPipelineProcessor:
    """
    Coordinates end-to-end processing:
    1. Reads raw retrieval results from previous person.
    2. Cleans & deduplicates evidence chunks.
    3. Runs LLM / Gemini summarization with grounding.
    4. Formats payloads for the website teammate.
    5. Saves data/website_payload.json & data/website_payload.md.
    """

    def __init__(
        self,
        baseline_file: str = "data/retrieval_baseline.md",
        eval_file: str = "data/eval.json",
        output_json: str = "data/website_payload.json",
        output_md: str = "data/website_payload.md"
    ):
        self.baseline_file = Path(baseline_file)
        if not self.baseline_file.exists():
            # Fall back to project root resolution
            proj_root = Path(__file__).resolve().parent.parent.parent
            alt_path = proj_root / baseline_file
            if alt_path.exists():
                self.baseline_file = alt_path

        self.eval_file = Path(eval_file)
        if not self.eval_file.exists():
            proj_root = Path(__file__).resolve().parent.parent.parent
            alt_eval = proj_root / eval_file
            if alt_eval.exists():
                self.eval_file = alt_eval

        self.output_json = Path(output_json)
        self.output_md = Path(output_md)
        self.cleaner = RetrievalCleaner()
        self.summarizer = EVSummarizer()
        self.formatter = WebsiteFormatter()

    async def run_pipeline(self) -> BatchWebsiteDelivery:
        """Executes full batch processing pipeline."""
        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Retrieval baseline file not found: {self.baseline_file}")

        logger.info(f"Loading raw retrieval results from: {self.baseline_file}")
        with open(self.baseline_file, "r", encoding="utf-8") as f:
            raw_markdown = f.read()

        # Step 1: Clean & Deduplicate
        logger.info("Cleaning and deduplicating candidate chunks...")
        cleaned_queries = self.cleaner.parse_markdown_baseline(raw_markdown)
        logger.info(f"Parsed {len(cleaned_queries)} evaluation queries.")

        # Step 2: Summarize each query with Gemini / LLM
        summarized_cases: List[WebsiteCardPayload] = []
        for q in cleaned_queries:
            logger.info(f"Summarizing case '{q.case_id}' with {len(q.cleaned_chunks)} unique chunks...")
            case_payload = await self.summarizer.summarize(q)
            summarized_cases.append(case_payload)

        # Step 3: Format into batch delivery payload
        delivery_batch = self.formatter.create_batch_delivery(summarized_cases)

        # Step 4: Save JSON and Markdown artifacts
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(delivery_batch.model_dump(), f, indent=2, default=str)
        logger.info(f"✅ Saved website delivery JSON to: {self.output_json}")

        md_report = self.formatter.generate_markdown_report(summarized_cases)
        with open(self.output_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info(f"✅ Saved website Markdown report to: {self.output_md}")

        return delivery_batch
