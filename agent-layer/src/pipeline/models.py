from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class CleanedChunk(BaseModel):
    """Cleaned, deduplicated chunk of evidence extracted from the retrieval person's output."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    title: str = Field(..., description="Cleaned title or question headline")
    content: str = Field(..., description="Cleaned text content stripped of HTML tags/entities")
    similarity_score: float = Field(..., description="Cosine similarity score from retriever (e.g. MiniLM)")
    content_type: str = Field("faq", description="Document or content type (faq, warranty, manual, etc.)")
    document_id: Optional[str] = Field(None, description="Original document ID")
    url: Optional[str] = Field(None, description="Source official URL")
    manufacturer: Optional[str] = Field(None, description="Vehicle manufacturer")
    vehicle_model: Optional[str] = Field(None, description="Target vehicle model if specified")
    category: Optional[str] = Field(None, description="Topic category (e.g. charging-grid, explore-buy)")
    group: Optional[str] = Field(None, description="Topic group (e.g. Ather 450 Series, Battery Warranty)")
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    duplicate_sources_merged: int = Field(1, description="Number of duplicate chunks merged into this chunk")
    alternate_urls: List[str] = Field(default_factory=list, description="Other URLs where identical content appeared")

class QueryCleanedEvidence(BaseModel):
    """Container for a query and its cleaned, deduplicated evidence chunks."""
    case_id: str = Field(..., description="Case or query ID (e.g. charging-compatible-700w)")
    query: str = Field(..., description="User question or fault query")
    raw_chunk_count: int = Field(..., description="Total candidate chunks received from retrieval person")
    cleaned_chunks: List[CleanedChunk] = Field(default_factory=list, description="Deduplicated, cleaned chunks")

class SourceCitation(BaseModel):
    """Citation metadata for website display."""
    title: str
    chunk_id: str
    url: Optional[str] = None
    page_number: Optional[int] = None
    content: Optional[str] = None
    similarity_score: float
    relevance_rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebsiteCardPayload(BaseModel):
    """
    Website-ready payload containing both rich Markdown and broken-down UI fields
    for the frontend teammate.
    """
    case_id: str = Field(..., description="Unique case identifier")
    query: str = Field(..., description="User query / technical question")
    vehicle_model: Optional[str] = Field(None, description="Vehicle model identified")
    
    # Note
    direct_answer: str = Field(..., description="Direct, concise answer in 1-3 sentences")
    key_takeaways: List[str] = Field(default_factory=list, description="Bullet points for quick skimming")
    technical_specs: Dict[str, str] = Field(default_factory=dict, description="Key metrics (e.g. charge times, warranty, cutoffs)")
    conditions_and_safety: List[str] = Field(default_factory=list, description="Safety precautions, limitations, or warranty terms")
    
    # Note
    formatted_markdown: str = Field(..., description="Complete beautifully formatted Markdown document")
    
    # Note
    sources: List[SourceCitation] = Field(default_factory=list, description="Supporting official sources with URLs")
    
    # Note
    grounding_score: float = Field(1.0, description="Calculated grounding confidence score (0.0 to 1.0)")
    raw_chunks_retrieved: int = Field(0, description="Chunks received from retriever before deduplication")
    unique_chunks_used: int = Field(0, description="Unique chunks after cleaning & deduplication")
    model_used: str = Field("gemini-3.5-flash-lite", description="LLM used for synthesis")
    processed_at: datetime = Field(default_factory=utc_now)

class BatchWebsiteDelivery(BaseModel):
    """Complete batch dataset payload delivered to the website person."""
    version: str = "1.0.0"
    project_title: str = "AI-Assisted Fault Diagnosis and Technical Information Retrieval for Electric Two-Wheelers"
    generated_at: datetime = Field(default_factory=utc_now)
    total_cases: int = Field(...)
    cases: List[WebsiteCardPayload] = Field(default_factory=list)
