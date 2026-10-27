/* ── TypeScript interfaces matching backend schemas ── */

export interface QueryRequest {
  query: string;
  vehicle_model?: string;
  subsystem?: string;
  session_id?: string;
  llm_provider?: string;
  llm_model?: string;
}

export interface LiveQueryRequest {
  query: string;
  vehicle_model?: string;
  subsystem?: string;
  top_k?: number;
  case_id?: string;
}

export interface DiagnosticStep {
  step_number: number;
  action: string;
  component_checked: string;
  expected_reading?: string;
  caution?: string;
  citation_source_ids: string[];
}

export interface EvidenceSource {
  source_id: string;
  document_title: string;
  document_type: string;
  section?: string;
  page_number?: number;
  vehicle_model?: string;
  subsystem?: string;
  content: string;
  relevance_score: number;
  metadata: Record<string, unknown>;
}

export interface GroundedAnswer {
  query: string;
  vehicle_model?: string;
  identified_faults: string[];
  root_cause_analysis: string;
  diagnostic_procedure: DiagnosticStep[];
  recommended_remedy: string;
  safety_warnings: string[];
  grounding_score: number;
  evidence_sources: EvidenceSource[];
  generated_at: string;
  execution_time_seconds?: number;
  agent_reasoning_trace?: string[];
}

export interface SourceCitation {
  title: string;
  chunk_id: string;
  url?: string;
  similarity_score: number;
  relevance_rationale?: string;
  metadata?: Record<string, unknown>;
}

export interface WebsiteCardPayload {
  case_id: string;
  query: string;
  vehicle_model?: string;
  direct_answer: string;
  key_takeaways: string[];
  technical_specs: Record<string, string>;
  conditions_and_safety: string[];
  formatted_markdown: string;
  sources: SourceCitation[];
  grounding_score: number;
  raw_chunks_retrieved: number;
  unique_chunks_used: number;
  model_used: string;
  processed_at: string;
}

export interface DiagnosisHistoryEntry {
  case_id: string;
  query: string;
  vehicle_model?: string;
  subsystem?: string;
  processed_at: string;
  result: WebsiteCardPayload;
}

export interface BatchWebsiteDelivery {
  version: string;
  project_title: string;
  generated_at: string;
  total_cases: number;
  cases: WebsiteCardPayload[];
}

export interface DTCRecord {
  code: string;
  system: string;
  description: string;
  symptoms: string[];
  possible_causes: string[];
  suggested_checks: string[];
  source_manual: string;
}

export interface HealthResponse {
  status: string;
  service?: string;
  default_llm_provider?: string;
  default_llm_model?: string;
  mcp_transport?: string;
  backend_retrieval_mode?: string;
  retrieval_backend_ready?: boolean;
  retrieval_backend_error?: string | null;
  active_mcp_tools?: string[];
  [key: string]: unknown;
}

export interface MCPTool {
  name: string;
  description?: string;
  [key: string]: unknown;
}
