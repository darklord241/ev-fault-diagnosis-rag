"""Backend gateway and HTTP API for the repository's vector-search corpus."""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

logger = logging.getLogger("ev_fault_backend")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPOSITORY_ROOT / "backend" / ".env")
ALLOWED_AGENT_PATHS = {
    "diagnose",
    "eval-answers",
    "health",
    "history",
    "knowledge/search",
    "mcp/tools",
    "pipeline/run-batch",
    "query",
    "summarize",
}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    vehicle_model: Optional[str] = Field(default=None, max_length=200)
    subsystem: Optional[str] = Field(default=None, max_length=200)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def _repo_path(environment_key: str, default: str) -> Path:
    configured = Path(os.getenv(environment_key, default)).expanduser()
    return configured if configured.is_absolute() else REPOSITORY_ROOT / configured


def _create_retriever():
    from scraper.vector_search.embedding import MiniLMEmbedder
    from scraper.vector_search.retriever import Retriever
    from scraper.vector_search.retrieval import load_chunks

    chunk_directory = _repo_path("EV_CHUNKED_DIR", "scraper/data/chunked")
    records = load_chunks(chunk_directory)
    if not records:
        raise RuntimeError(f"No JSONL chunks found in {chunk_directory}")

    return Retriever(
        records,
        MiniLMEmbedder(local_files_only=True),
        _repo_path(
            "EV_EMBEDDING_CACHE_PATH",
            "scraper/data/embeddings/chunk_vectors.json",
        ),
    )


def _vehicle_matches(record: Mapping[str, object], requested: Optional[str]) -> bool:
    if not requested:
        return True

    provenance = record.get("provenance")
    metadata = record.get("metadata")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    document_model = provenance.get("vehicle_model") or metadata.get("vehicle_model")
    if not isinstance(document_model, str) or not document_model.strip():
        return True

    normalized_model = document_model.casefold()
    normalized_request = requested.casefold()
    generic_labels = ("generic", "all models", "all vehicles", "standard")
    return (
        normalized_request in normalized_model
        or normalized_model in normalized_request
        or any(label in normalized_model for label in generic_labels)
    )


def _evidence_from_result(result) -> dict:
    record = result.record
    provenance = record.get("provenance", {})
    metadata = record.get("metadata", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    page_number = (
        metadata.get("page_number")
        or metadata.get("page")
        or provenance.get("page_number")
    )
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        page_number = None
    section = (
        metadata.get("section")
        or metadata.get("group")
        or metadata.get("category")
    )
    vehicle_model = (
        provenance.get("vehicle_model") or metadata.get("vehicle_model")
    )
    subsystem = (
        metadata.get("subsystem")
        or metadata.get("category")
        or metadata.get("group")
    )
    return {
        "source_id": record["chunk_id"],
        "document_title": record["title"],
        "document_type": record["source_type"],
        "section": section if isinstance(section, str) else None,
        "page_number": page_number,
        "vehicle_model": vehicle_model if isinstance(vehicle_model, str) else None,
        "subsystem": subsystem if isinstance(subsystem, str) else None,
        "content": record["text"],
        "relevance_score": max(0.0, min(1.0, float(result.score))),
        "metadata": {
            "record_id": record["record_id"],
            "source_document": record["source_document"],
            "content_type": record["content_type"],
            "provenance": dict(provenance),
            "chunk_metadata": dict(metadata),
        },
    }


def create_app(
    *,
    retriever=None,
    retriever_factory: Optional[Callable] = None,
    backend_api_key: Optional[str] = None,
    agent_base_url: Optional[str] = None,
) -> FastAPI:
    """Build an app with optional dependencies for deterministic tests."""
    configured_key = (
        backend_api_key
        if backend_api_key is not None
        else os.getenv("BACKEND_API_KEY")
    )
    configured_agent_url = (
        agent_base_url
        or os.getenv(
            "AGENT_ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8080"
        )
    ).rstrip("/")
    factory = retriever_factory or _create_retriever

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.retriever is None:
            try:
                app.state.retriever = factory()
                app.state.retrieval_error = None
            except Exception as error:
                app.state.retrieval_error = (
                    "Retrieval index initialization failed; check backend logs."
                )
                logger.exception("Retrieval corpus could not be initialized")
        yield

    app = FastAPI(
        title="EV Fault Diagnosis Backend",
        description="Diagnosis API gateway and corpus-backed retrieval service.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.retriever = retriever
    app.state.retrieval_error = None

    async def require_backend_key(
        authorization: Optional[str] = Header(default=None),
    ) -> None:
        if not configured_key:
            raise HTTPException(
                status_code=503,
                detail="Backend retrieval authentication is not configured.",
            )
        expected = f"Bearer {configured_key}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid backend API key.")

    @app.get("/health")
    async def health():
        available = app.state.retriever is not None
        return {
            "status": "healthy" if available else "degraded",
            "service": "EV Fault Diagnosis Backend",
            "retrieval_ready": available,
            "retrieval_error": app.state.retrieval_error,
            "agent_orchestrator_url": configured_agent_url,
        }

    @app.post(
        "/api/v1/retrieval/search",
        dependencies=[Depends(require_backend_key)],
    )
    async def search(request: SearchRequest, http_request: Request):
        active_retriever = http_request.app.state.retriever
        if active_retriever is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Live retrieval is unavailable. Prepare the chunked corpus "
                    "and local embedding model, then restart the backend."
                ),
            )

        query = (
            f"{request.subsystem}: {request.query}"
            if request.subsystem
            else request.query
        )
        results = await run_in_threadpool(
            active_retriever.search,
            query,
            request.top_k,
            record_filter=lambda record: _vehicle_matches(
                record, request.vehicle_model
            ),
        )
        return {"results": [_evidence_from_result(result) for result in results]}

    @app.api_route(
        "/api/v1/{path:path}",
        methods=["GET", "POST", "DELETE"],
        include_in_schema=False,
    )
    async def forward_to_agent(path: str, request: Request):
        if path not in ALLOWED_AGENT_PATHS and not (
            path.startswith("eval-answers/")
            or path.startswith("dtc/")
        ):
            raise HTTPException(status_code=404, detail="API route not found.")

        target = f"{configured_agent_url}/api/v1/{quote(path, safe='/')}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        headers = {}
        content_type = request.headers.get("content-type")
        if content_type:
            headers["content-type"] = content_type

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                upstream = await client.request(
                    request.method,
                    target,
                    content=await request.body(),
                    headers=headers,
                )
        except httpx.RequestError as error:
            logger.warning("Agent orchestrator request failed: %s", error)
            raise HTTPException(
                status_code=503,
                detail="Agent orchestrator is unavailable.",
            ) from error

        if path == "health" and upstream.status_code == 200:
            health_data = upstream.json()
            if not isinstance(health_data, dict):
                raise HTTPException(
                    status_code=502,
                    detail="Agent health endpoint returned an invalid response.",
                )
            health_data["retrieval_backend_ready"] = app.state.retriever is not None
            health_data["retrieval_backend_error"] = app.state.retrieval_error
            return JSONResponse(
                content=health_data,
                status_code=upstream.status_code,
            )

        response_headers = {}
        upstream_content_type = upstream.headers.get("content-type")
        if upstream_content_type:
            response_headers["content-type"] = upstream_content_type
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    return app


app = create_app()
