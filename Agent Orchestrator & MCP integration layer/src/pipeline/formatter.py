import json
import httpx
from typing import List, Optional, Dict, Any
from src.pipeline.models import WebsiteCardPayload, BatchWebsiteDelivery
from src.common.logging import logger

class WebsiteFormatter:
    """
    Formats summarized results into frontend-ready contracts
    and handles delivery to the teammate in charge of the website.
    """

    @staticmethod
    def create_batch_delivery(cases: List[WebsiteCardPayload]) -> BatchWebsiteDelivery:
        """Constructs standard BatchWebsiteDelivery schema."""
        return BatchWebsiteDelivery(
            version="1.0.0",
            project_title="AI-Assisted Fault Diagnosis and Technical Information Retrieval for Electric Two-Wheelers",
            total_cases=len(cases),
            cases=cases
        )

    @staticmethod
    def generate_markdown_report(cases: List[WebsiteCardPayload]) -> str:
        """Generates a complete, beautiful Markdown document summarizing all cases."""
        lines = [
            "# ⚡ Electric Two-Wheeler Technical Evidence & Diagnosis Report",
            "### AI-Assisted Retrieval, Cleaning & Grounded Summaries for Website Display",
            "",
            "> Processed by Agent Orchestrator & Evidence Pipeline (Cleaned, Deduplicated, Grounded via Gemini)",
            "",
            "---",
            ""
        ]

        for idx, case in enumerate(cases, 1):
            lines.append(f"## {idx}. Case: `{case.case_id}`")
            lines.append(f"**Query:** *\"{case.query}\"*  ")
            lines.append(f"**Identified Model:** `{case.vehicle_model or 'Ather 450 Series'}` | **Grounding Score:** `{case.grounding_score:.2f}` | **Chunks:** `{case.unique_chunks_used}` unique (deduped from `{case.raw_chunks_retrieved}`)")
            lines.append("")
            lines.append(f"> **Direct Summary:** {case.direct_answer}")
            lines.append("")
            
            lines.append("### 📋 Key Takeaways")
            for t in case.key_takeaways:
                lines.append(f"- {t}")
            lines.append("")

            if case.technical_specs:
                lines.append("### ⚙️ Technical Specifications & Limits")
                lines.append("| Specification | Value |")
                lines.append("| :--- | :--- |")
                for k, v in case.technical_specs.items():
                    lines.append(f"| **{k}** | {v} |")
                lines.append("")

            if case.conditions_and_safety:
                lines.append("### ⚠️ Conditions & Roadside Safety")
                for c in case.conditions_and_safety:
                    lines.append(f"- {c}")
                lines.append("")

            if case.sources:
                lines.append("### 📚 Official Document Sources")
                for s in case.sources:
                    url_part = f" ([Official Link]({s.url}))" if s.url else ""
                    lines.append(f"- **{s.title}** (Sim: `{s.similarity_score:.4f}`){url_part} — `{s.chunk_id}`")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    async def push_to_website_webhook(
        payload: BatchWebsiteDelivery,
        webhook_url: str,
        auth_token: Optional[str] = None
    ) -> bool:
        """Pushes the formatted batch payload to the website person's API/webhook."""
        try:
            headers = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json=payload.model_dump(), headers=headers)
                if resp.status_code in [200, 201, 202]:
                    logger.info(f"Successfully pushed payload to website webhook: {webhook_url}")
                    return True
                else:
                    logger.warning(f"Website webhook returned HTTP {resp.status_code}: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to push to website webhook ({webhook_url}): {e}")
            return False
