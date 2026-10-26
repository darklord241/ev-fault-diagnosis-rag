import re
from typing import List, Dict, Any, Tuple
from src.common.models import EvidenceSource, DiagnosticStep, GroundedAnswer
from src.common.logging import logger

class GroundingEngine:
    """
    Verifies that generated responses are strictly grounded in retrieved MCP evidence,
    extracts diagnostic steps, and computes a quantitative grounding score.
    """

    @staticmethod
    def calculate_grounding_score(
        response_text: str,
        retrieved_sources: List[EvidenceSource]
    ) -> float:
        """
        Calculates a grounding score between 0.0 and 1.0 based on:
        1. Explicit citation presence of retrieved source IDs in the text
        2. Keyword / semantic overlap between the response and retrieved source content
        3. Average relevance score of retrieved documents
        """
        if not retrieved_sources:
            return 0.35  # Baseline score when relying solely on parametric knowledge

        # 1. Check explicit citation matching
        cited_sources = 0
        for src in retrieved_sources:
            # Check for source_id or document_title fragment
            if src.source_id in response_text or (src.section and src.section[:15].lower() in response_text.lower()):
                cited_sources += 1

        citation_ratio = cited_sources / len(retrieved_sources)

        # 2. Key technical terms overlap (voltages, pins, components)
        technical_matches = 0
        all_source_text = " ".join([s.content for s in retrieved_sources]).lower()
        response_lower = response_text.lower()

        # Extract technical numbers/ranges like 5.0v, 150mv, 10k, pin 1, hall
        tokens = set(re.findall(r"\b(?:\d+(?:\.\d+)?[v|mv|k\u03a9|\u03a9|a|%]|pin \d+|mosfet|hall|bms|can-h|dtc|soc)\b", all_source_text))
        if tokens:
            for t in tokens:
                if t in response_lower:
                    technical_matches += 1
            overlap_ratio = min(1.0, technical_matches / max(1, len(tokens) * 0.4))
        else:
            overlap_ratio = 0.5

        # 3. Average relevance score from retrieval
        avg_retrieval_relevance = sum(s.relevance_score for s in retrieved_sources) / len(retrieved_sources)

        # Weighted grounding score formula
        grounding_score = (0.45 * citation_ratio) + (0.35 * overlap_ratio) + (0.20 * avg_retrieval_relevance)
        grounding_score = max(0.1, min(0.99, round(grounding_score, 3)))

        logger.info(f"Grounding Score: {grounding_score} (Citations: {cited_sources}/{len(retrieved_sources)}, Overlap: {overlap_ratio:.2f})")
        return grounding_score

    @staticmethod
    def parse_diagnostic_steps(response_text: str, evidence_sources: List[EvidenceSource]) -> List[DiagnosticStep]:
        """Parses structured diagnostic verification steps from the generated response."""
        steps: List[DiagnosticStep] = []
        lines = response_text.split("\n")
        step_counter = 1

        source_id_list = [s.source_id for s in evidence_sources]

        for line in lines:
            line_clean = line.strip()
            # Match step patterns like: "1. Check multimeter voltage...", "Step 1: ...", "- Step 1: ..."
            match = re.match(r"^(?:(?:Step\s*)?(\d+)[\.:\)]|\* Step \d+:|- Step \d+:)\s*(.+)$", line_clean, re.IGNORECASE)
            if match:
                action_text = match.group(2) if match.group(2) else line_clean
                
                # Extract any cited source IDs in the step
                step_citations = [sid for sid in source_id_list if sid in action_text]

                # Component detection heuristic
                comp_match = re.search(r"\b(throttle|bms|motor|controller|mcu|inverter|hall sensor|fuse|battery|charger|wire harness|can bus)\b", action_text, re.IGNORECASE)
                component = comp_match.group(1).title() if comp_match else "Subsystem / Wiring"

                # Expected reading detection
                reading_match = re.search(r"(\d+(?:\.\d+)?\s*(?:V|mV|kΩ|Ω|A|°C|%)(?:\s*(?:-|to)\s*\d+(?:\.\d+)?\s*(?:V|mV|kΩ|Ω|A|°C|%))?)", action_text)
                expected_reading = reading_match.group(1) if reading_match else None

                steps.append(DiagnosticStep(
                    step_number=step_counter,
                    action=action_text,
                    component_checked=component,
                    expected_reading=expected_reading,
                    citation_source_ids=step_citations
                ))
                step_counter += 1

        return steps

    @classmethod
    def synthesize_grounded_answer(
        cls,
        query: str,
        vehicle_model: str,
        response_text: str,
        retrieved_evidence_raw: List[Dict[str, Any]],
        execution_time: float,
        reasoning_trace: List[str]
    ) -> GroundedAnswer:
        """Assembles the final GroundedAnswer response object."""
        evidence_sources = [EvidenceSource(**item) for item in retrieved_evidence_raw]
        grounding_score = cls.calculate_grounding_score(response_text, evidence_sources)
        diagnostic_steps = cls.parse_diagnostic_steps(response_text, evidence_sources)

        # Extract safety warnings
        safety_warnings = []
        for line in response_text.split("\n"):
            if any(w in line.lower() for w in ["warning", "caution", "danger", "high voltage", "disconnect pack", "safety"]):
                clean_line = re.sub(r"^[-*#\s:]+", "", line).strip()
                if len(clean_line) > 10 and clean_line not in safety_warnings:
                    safety_warnings.append(clean_line)

        # Extract identified faults
        identified_faults = []
        for line in response_text.split("\n"):
            if "fault" in line.lower() or "root cause" in line.lower() or "error" in line.lower():
                clean_line = re.sub(r"^[-*#\s:]+", "", line).strip()
                if len(clean_line) > 15 and len(clean_line) < 150:
                    identified_faults.append(clean_line)
                    if len(identified_faults) >= 3:
                        break

        if not identified_faults:
            identified_faults = ["Potential Subsystem Malfunction or Signal Drift"]

        return GroundedAnswer(
            query=query,
            vehicle_model=vehicle_model,
            identified_faults=identified_faults,
            root_cause_analysis=response_text,
            diagnostic_procedure=diagnostic_steps,
            recommended_remedy="Follow step-by-step verification, verify nominal multimeter readings, and repair/replace flagged component as detailed.",
            safety_warnings=safety_warnings if safety_warnings else ["Observe high-voltage electrical safety guidelines when inspecting EV powertrain connectors."],
            grounding_score=grounding_score,
            evidence_sources=evidence_sources,
            execution_time_seconds=round(execution_time, 2),
            agent_reasoning_trace=reasoning_trace
        )
