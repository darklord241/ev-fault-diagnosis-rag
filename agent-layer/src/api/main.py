from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from config.settings import settings
from src.common.logging import logger

def create_app() -> FastAPI:
    app = FastAPI(
        title="Electric Two-Wheeler Fault Diagnosis - Agent Orchestrator API",
        description=(
            "Agent Orchestrator connecting LLM Reasoning Engines with Model Context Protocol (MCP) "
            "Hybrid Retrieval Tools for Electric Two-Wheeler Technical Fault Diagnosis."
        ),
        version="1.0.0"
    )

    # Note
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Orchestrator API Server at http://{settings.ORCHESTRATOR_API_HOST}:{settings.ORCHESTRATOR_API_PORT}")
    uvicorn.run(
        "src.api.main:app",
        host=settings.ORCHESTRATOR_API_HOST,
        port=settings.ORCHESTRATOR_API_PORT,
        reload=True
    )
