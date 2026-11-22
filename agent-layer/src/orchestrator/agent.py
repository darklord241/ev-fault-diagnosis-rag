import time
import json
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from src.common.models import QueryRequest, GroundedAnswer, EvidenceSource
from src.common.logging import logger
from src.orchestrator.llm_factory import LLMFactory
from src.orchestrator.mcp_client import MCPClientManager
from src.orchestrator.grounding import GroundingEngine
from src.mcp_server.backend_adapter import backend_adapter
from src.orchestrator.prompt_templates import SYSTEM_PROMPT, USER_QUERY_TEMPLATE
from config.settings import settings

class AgentOrchestrator:
    """
    Central Agent Orchestrator connecting the LLM Reasoning Engine with
    the MCP Retrieval Layer to diagnose electric 2-wheeler faults.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        self.provider = llm_provider or settings.DEFAULT_LLM_PROVIDER
        self.model_name = llm_model or settings.DEFAULT_LLM_MODEL
        self.temperature = temperature
        self.mcp_client = MCPClientManager()
        self.tools = self.mcp_client.load_langchain_tools()
        self.tools_by_name = {t.name: t for t in self.tools}

    async def diagnose_fault(self, request: QueryRequest) -> GroundedAnswer:
        """
        Main entry point: executes reasoning loop, retrieves technical evidence via MCP tools,
        and selects a retrieved source passage.
        """
        start_time = time.time()
        self.mcp_client.clear_session_cache()
        reasoning_trace: List[str] = []

        logger.info(f"Starting fault diagnosis for query: '{request.query}' (Vehicle: {request.vehicle_model}, Subsystem: {request.subsystem})")

        provider = request.llm_provider or self.provider
        model_name = request.llm_model or self.model_name

        try:
            llm = LLMFactory.get_llm(
                provider=provider,
                model_name=model_name,
                temperature=self.temperature
            )
            # Bind MCP tools to LLM
            llm_with_tools = llm.bind_tools(self.tools)

            user_content = USER_QUERY_TEMPLATE.format(
                query=request.query,
                vehicle_model=request.vehicle_model or "Standard Electric Two-Wheeler",
                subsystem=request.subsystem or "Not Specified / Auto-Detect"
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content)
            ]

            # ReAct Execution Loop (max 5 iterations)
            max_iterations = 5
            final_response_text = ""

            for i in range(max_iterations):
                logger.info(f"[Orchestrator Loop] Iteration {i+1}")
                ai_msg = await llm_with_tools.ainvoke(messages)
                messages.append(ai_msg)

                # Check if the model made tool calls
                if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call["id"]

                        trace_entry = f"Tool Call: {tool_name}({json.dumps(tool_args)})"
                        reasoning_trace.append(trace_entry)
                        logger.info(f"[Agent Tool Execution] {trace_entry}")

                        if tool_name in self.tools_by_name:
                            tool_fn = self.tools_by_name[tool_name]
                            try:
                                tool_result = await tool_fn.ainvoke(tool_args)
                            except Exception as err:
                                tool_result = json.dumps({"error": str(err)})
                        else:
                            tool_result = json.dumps({"error": f"Tool {tool_name} not found"})

                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id,
                            name=tool_name
                        ))
                else:
                    # Model produced final synthesis
                    final_response_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
                    break

        except Exception as e:
            logger.warning(f"LLM tool-calling loop encountered error or API key missing: {e}. Using source passages.")
            reasoning_trace.append(f"Fallback Mode Activated: {e}")
            final_response_text = ""

        if not self.mcp_client.get_retrieved_evidence_cache():
            reasoning_trace.append("Fallback: retrieving source passages for manual review")
            await self.mcp_client._wrapped_retrieve_evidence(
                query=request.query,
                vehicle_model=request.vehicle_model,
                subsystem=request.subsystem,
                top_k=3,
            )
            final_response_text = ""

        execution_time = time.time() - start_time
        retrieved_raw = self.mcp_client.get_retrieved_evidence_cache()

        grounded_answer = GroundingEngine.synthesize_grounded_answer(
            query=request.query,
            vehicle_model=request.vehicle_model,
            response_text=final_response_text,
            retrieved_evidence_raw=retrieved_raw,
            execution_time=execution_time,
            reasoning_trace=reasoning_trace,
            retrieval_mode="demo" if backend_adapter.use_mock else "live",
        )

        logger.info(f"Evidence selection completed in {execution_time:.2f}s")
        return grounded_answer
