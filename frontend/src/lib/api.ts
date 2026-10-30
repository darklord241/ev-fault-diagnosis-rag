import type {
  QueryRequest,
  GroundedAnswer,
  BatchWebsiteDelivery,
  WebsiteCardPayload,
  DTCRecord,
  HealthResponse,
  MCPTool,
  LiveQueryRequest,
  DiagnosisHistoryEntry,
  KnowledgeSearchRequest,
  KnowledgeSearchResponse,
} from "./types";

/* Note */

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let msg = `Request failed: ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail ?? body.message ?? msg;
    } catch {
      /* Note */
    }
    throw new ApiError(msg, res.status);
  }

  return res.json() as Promise<T>;
}

/* Note */

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health");
}

export async function diagnose(
  payload: QueryRequest
): Promise<GroundedAnswer> {
  return request<GroundedAnswer>("/api/v1/diagnose", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function queryLive(
  payload: LiveQueryRequest
): Promise<WebsiteCardPayload> {
  return request<WebsiteCardPayload>("/api/v1/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchDiagnosisHistory(): Promise<{
  items: DiagnosisHistoryEntry[];
}> {
  return request<{ items: DiagnosisHistoryEntry[] }>("/api/v1/history");
}

export async function clearDiagnosisHistory(): Promise<{ deleted: number }> {
  return request<{ deleted: number }>("/api/v1/history", {
    method: "DELETE",
  });
}

export async function searchKnowledge(
  payload: KnowledgeSearchRequest
): Promise<KnowledgeSearchResponse> {
  return request<KnowledgeSearchResponse>("/api/v1/knowledge/search", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchEvalAnswers(): Promise<BatchWebsiteDelivery> {
  return request<BatchWebsiteDelivery>("/api/v1/eval-answers");
}

export async function fetchCaseAnswer(
  caseId: string
): Promise<WebsiteCardPayload> {
  return request<WebsiteCardPayload>(
    `/api/v1/eval-answers/${encodeURIComponent(caseId)}`
  );
}

export async function runBatchPipeline(): Promise<BatchWebsiteDelivery> {
  return request<BatchWebsiteDelivery>("/api/v1/pipeline/run-batch", {
    method: "POST",
  });
}

export async function summarizeOnDemand(
  payload: { case_id: string; query: string; raw_markdown_or_text: string }
): Promise<WebsiteCardPayload> {
  return request<WebsiteCardPayload>("/api/v1/summarize", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchMCPTools(): Promise<{ mcp_tools: MCPTool[] }> {
  return request<{ mcp_tools: MCPTool[] }>("/api/v1/mcp/tools");
}

export async function lookupDTC(code: string): Promise<DTCRecord> {
  return request<DTCRecord>(
    `/api/v1/dtc/${encodeURIComponent(code)}`
  );
}
