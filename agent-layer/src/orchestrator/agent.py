import time
import json
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from src.common.models import QueryRequest, GroundedAnswer, EvidenceSource
from src.common.logging import logger
from src.orchestrator.llm_factory import LLMFactory
from src.orchestrator.mcp_client import MCPClientManager
from src.orchestrator.grounding import GroundingEngine
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
        and constructs a source-grounded diagnosis.
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
            # Note
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

            # Note
            max_iterations = 5
            final_response_text = ""

            for i in range(max_iterations):
                logger.info(f"[Orchestrator Loop] Iteration {i+1}")
                ai_msg = await llm_with_tools.ainvoke(messages)
                messages.append(ai_msg)

                # Note
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
                    # Note
                    final_response_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
                    break

            if not final_response_text and messages:
                final_response_text = messages[-1].content

        except Exception as e:
            logger.warning(f"LLM tool-calling loop encountered error or API key missing: {e}. Executing fallback grounded synthesis pipeline.")
            reasoning_trace.append(f"Fallback Mode Activated: {e}")
            final_response_text = await self._execute_fallback_diagnosis(request, reasoning_trace)

        execution_time = time.time() - start_time
        retrieved_raw = self.mcp_client.get_retrieved_evidence_cache()

        grounded_answer = GroundingEngine.synthesize_grounded_answer(
            query=request.query,
            vehicle_model=request.vehicle_model or "Electric 2-Wheeler",
            response_text=final_response_text,
            retrieved_evidence_raw=retrieved_raw,
            execution_time=execution_time,
            reasoning_trace=reasoning_trace
        )

        logger.info(f"Diagnosis completed in {execution_time:.2f}s with Grounding Score: {grounded_answer.grounding_score}")
        return grounded_answer

    async def _execute_fallback_diagnosis(self, request: QueryRequest, reasoning_trace: List[str]) -> str:
        """
        Direct deterministic fallback that directly triggers MCP retrieval tools,
        synthesizes grounded results, and allows testing even without an external LLM API key.
        """
        # Note
        reasoning_trace.append("Fallback: Invoking MCP retrieval tool `retrieve_technical_evidence`")
        raw_evidence = await self.mcp_client._wrapped_retrieve_evidence(
            query=request.query,
            vehicle_model=request.vehicle_model,
            subsystem=request.subsystem,
            top_k=3
        )
        sources = json.loads(raw_evidence)

        # Note
        for dtc in ["P0A80", "E-08", "E-07", "U0100"]:
            if dtc.lower() in request.query.lower():
                reasoning_trace.append(f"Fallback: Invoking MCP tool `lookup_dtc_fault_code` for {dtc}")
                await self.mcp_client._wrapped_lookup_dtc(dtc)

        # Note
        primary_source = sources[0] if sources else None
        source_id = primary_source.get("source_id", "SRC-EV-GEN-001") if primary_source else "SRC-EV-GEN-001"
        source_title = primary_source.get("document_title", "Service Manual") if primary_source else "Service Manual"
        source_content = primary_source.get("content", "") if primary_source else ""

        synthesized_text = f"""## 🔍 Fault Identification
Based on the symptom **"{request.query}"**, the primary fault is localized to the **{request.subsystem or 'Powertrain / Electrical'}** subsystem. Grounded reference: `[{source_id}]`.

## ⚙️ Root Cause Analysis
Analysis from technical documentation indicates:
{source_content}

## 🛠️ Step-by-Step Diagnostic Procedure
1. Disconnect main battery pack isolation switch and measure resting voltage across terminals with a calibrated multimeter (Refer to `[{source_id}]`).
2. Probe signal wiring harness for supply voltage: Pin 1 (Red) should read 5.0V ± 0.2V DC relative to Pin 2 (GND).
3. Check signal output line while rotating wheel or operating throttle. Confirm reading toggles cleanly or sweeps between 0.8V and 4.2V.
4. Verify CAN bus termination resistance across CAN-H and CAN-L (Nominal 60Ω with ignition OFF).

## 🔧 Recommended Remedy & Part Replacement
- If sensor voltage is stuck at 0V or >4.6V, inspect harness for broken wire or water ingress.
- If individual cell voltage delta exceeds 150mV, initiate a balanced slow charging cycle (CV mode at 0.1C) for 12 hours. Replace degraded cell modules if delta remains >200mV.

## ⚠️ Safety Precautions
- **DANGER - High Voltage System**: Ensure the vehicle ignition is OFF and wait 5 minutes for inverter capacitor discharge before disconnecting high-voltage connectors.
- Never bypass BMS discharge MOSFETs or safety interlocks.

## 📚 Supporting Technical Sources
- `[{source_id}]`: {source_title}
"""
        return synthesized_text
