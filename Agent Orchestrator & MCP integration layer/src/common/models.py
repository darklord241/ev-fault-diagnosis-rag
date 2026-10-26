from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class EvidenceSource(BaseModel):
    """Represents a grounded piece of technical documentation evidence."""
    source_id: str = Field(..., description="Unique ID for the retrieved chunk or manual section")
    document_title: str = Field(..., description="Name or title of the service manual / technical doc")
    document_type: str = Field("Service Manual", description="e.g. Service Manual, Datasheet, DTC Table, Wiring Diagram")
    section: Optional[str] = Field(None, description="Section heading or chapter in document")
    page_number: Optional[int] = Field(None, description="Page number in the original PDF/Doc")
    vehicle_model: Optional[str] = Field(None, description="Specific 2-wheeler model/variant if applicable")
    subsystem: Optional[str] = Field(None, description="Subsystem: BMS, Motor Controller, Throttle, Charger, etc.")
    content: str = Field(..., description="The relevant text content or excerpt extracted")
    relevance_score: float = Field(0.0, description="Hybrid retrieval similarity / reranker score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")

class DiagnosticStep(BaseModel):
    """Step-by-step diagnostic or troubleshooting instruction."""
    step_number: int = Field(..., description="Step order sequence")
    action: str = Field(..., description="Clear diagnostic or verification action to perform")
    component_checked: str = Field(..., description="Component or harness pin checked")
    expected_reading: Optional[str] = Field(None, description="Expected multimeter/sensor/DTC reading")
    caution: Optional[str] = Field(None, description="Safety precautions (high voltage, ESD, etc.)")
    citation_source_ids: List[str] = Field(default_factory=list, description="IDs of EvidenceSource supporting this step")

class GroundedAnswer(BaseModel):
    """Grounded diagnostic response delivered to the Dashboard/User."""
    query: str = Field(..., description="Original user diagnostic query")
    vehicle_model: Optional[str] = Field(None, description="Identified vehicle model")
    identified_faults: List[str] = Field(default_factory=list, description="Likely faults identified")
    root_cause_analysis: str = Field(..., description="Detailed explanation of the failure mechanism")
    diagnostic_procedure: List[DiagnosticStep] = Field(default_factory=list, description="Ordered verification steps")
    recommended_remedy: str = Field(..., description="Repair, replacement, or recalibration action")
    safety_warnings: List[str] = Field(default_factory=list, description="Critical safety warnings (HV isolation, etc.)")
    grounding_score: float = Field(..., description="Calculated grounding confidence score (0.0 to 1.0)")
    evidence_sources: List[EvidenceSource] = Field(default_factory=list, description="List of technical documents cited")
    generated_at: datetime = Field(default_factory=utc_now)
    execution_time_seconds: Optional[float] = Field(None, description="Latency in seconds")
    agent_reasoning_trace: Optional[List[str]] = Field(default_factory=list, description="ReAct tool call execution trace")

class QueryRequest(BaseModel):
    """Input payload from Dashboard or external clients."""
    query: str = Field(..., description="Fault symptom, DTC code, or technical question")
    vehicle_model: Optional[str] = Field(None, description="Electric two-wheeler model (e.g. Ather 450X, Ola S1 Pro, TVS iQube, Generic BLDC)")
    subsystem: Optional[str] = Field(None, description="Target subsystem if known (e.g., Battery/BMS, MCU, Throttle, Wiring)")
    session_id: Optional[str] = Field(None, description="Session ID for conversational memory")
    llm_provider: Optional[str] = Field(None, description="Override LLM provider (gemini, openai, anthropic, ollama)")
    llm_model: Optional[str] = Field(None, description="Override LLM model name")

class DTCRecord(BaseModel):
    """Diagnostic Trouble Code entry."""
    code: str = Field(..., description="DTC Code (e.g. P0A80, E-07, U0100)")
    system: str = Field(..., description="Subsystem (BMS, Inverter, Throttle, Dashboard)")
    description: str = Field(..., description="Short fault description")
    symptoms: List[str] = Field(default_factory=list)
    possible_causes: List[str] = Field(default_factory=list)
    suggested_checks: List[str] = Field(default_factory=list)
    source_manual: str = Field(..., description="Reference service manual")

class ComponentSpec(BaseModel):
    """Technical specifications of an EV component."""
    component_name: str = Field(..., description="e.g., Hall Effect Throttle, 48V Li-ion BMS, BLDC Hub Motor")
    subsystem: str = Field(...)
    nominal_operating_specs: Dict[str, Any] = Field(default_factory=dict)
    pinout_details: Dict[str, str] = Field(default_factory=dict)
    troubleshooting_thresholds: Dict[str, str] = Field(default_factory=dict)
    source_reference: str = Field(...)
