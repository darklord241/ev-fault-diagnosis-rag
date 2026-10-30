import sys
import os
import uvicorn

# Note
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings

if __name__ == "__main__":
    print(f"================================================================")
    print(f" Starting EV Diagnostic Agent Orchestrator API Gateway")
    print(f" Dashboard Integration URL: http://{settings.ORCHESTRATOR_API_HOST}:{settings.ORCHESTRATOR_API_PORT}")
    print(f" Swagger UI / OpenAPI Docs:  http://127.0.0.1:{settings.ORCHESTRATOR_API_PORT}/docs")
    print(f" Default LLM Provider:      {settings.DEFAULT_LLM_PROVIDER} ({settings.DEFAULT_LLM_MODEL})")
    print(f"================================================================")
    uvicorn.run(
        "src.api.main:app",
        host=settings.ORCHESTRATOR_API_HOST,
        port=settings.ORCHESTRATOR_API_PORT,
        reload=True
    )
