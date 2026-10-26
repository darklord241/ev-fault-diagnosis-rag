import pytest
from langchain_core.messages import AIMessage

from config.settings import settings
from src.common.models import QueryRequest
from src.orchestrator import agent as agent_module


@pytest.mark.asyncio
async def test_live_retrieval_failure_does_not_return_an_ungrounded_answer(monkeypatch):
    class FailingTool:
        async def ainvoke(self, _arguments):
            raise RuntimeError("live retrieval unavailable")

    class Model:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_technical_evidence",
                        "args": {"query": "battery fault"},
                        "id": "retrieve-1",
                    }
                ],
            )

    class MCPClient:
        def clear_session_cache(self):
            return None

        async def _wrapped_retrieve_evidence(self, **_arguments):
            raise RuntimeError("live retrieval unavailable")

    monkeypatch.setattr(settings, "HYBRID_RETRIEVAL_USE_MOCK", False)
    monkeypatch.setattr(agent_module.LLMFactory, "get_llm", lambda **_kwargs: Model())
    orchestrator = agent_module.AgentOrchestrator.__new__(
        agent_module.AgentOrchestrator
    )
    orchestrator.provider = "test"
    orchestrator.model_name = "test"
    orchestrator.temperature = None
    orchestrator.mcp_client = MCPClient()
    orchestrator.tools = [FailingTool()]
    orchestrator.tools_by_name = {
        "retrieve_technical_evidence": orchestrator.tools[0]
    }

    with pytest.raises(RuntimeError, match="live retrieval unavailable"):
        await orchestrator.diagnose_fault(QueryRequest(query="battery fault"))
