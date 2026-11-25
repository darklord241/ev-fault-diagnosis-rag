from src.common.models import EvidenceSource, GroundedAnswer


class GroundingEngine:
    """Assembles an evidence response from a validated source selection."""

    @staticmethod
    def synthesize_grounded_answer(
        query: str,
        vehicle_model: str | None,
        response_text: str,
        retrieved_evidence_raw: list[dict],
        execution_time: float,
        reasoning_trace: list[str],
        retrieval_mode: str,
    ) -> GroundedAnswer:
        sources = [EvidenceSource.model_validate(item) for item in retrieved_evidence_raw]
        selected_id = response_text.strip()
        selected = next(
            (source for source in sources if source.source_id == selected_id),
            sources[0] if sources else None,
        )
        return GroundedAnswer(
            query=query,
            vehicle_model=vehicle_model,
            root_cause_analysis=(
                "A source passage is available for review."
                if selected else "No relevant source evidence was found."
            ),
            selected_source_id=selected.source_id if selected else None,
            recommended_remedy="",
            retrieval_mode=retrieval_mode,
            evidence_sources=sources,
            execution_time_seconds=round(execution_time, 2),
            agent_reasoning_trace=reasoning_trace,
        )
