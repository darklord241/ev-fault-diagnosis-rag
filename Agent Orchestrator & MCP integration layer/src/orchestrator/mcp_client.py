import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from langchain_core.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

from src.mcp_server.tools import (
    tool_retrieve_technical_evidence,
    tool_lookup_dtc_fault_code,
    tool_lookup_component_specs
)
from src.common.logging import logger
from config.settings import settings

# Tool input schemas for LangChain StructuredTool integration
class RetrieveEvidenceInput(BaseModel):
    query: str = Field(..., description="Fault symptom, error symptom, or technical question to search in manuals")
    vehicle_model: Optional[str] = Field(None, description="Electric two-wheeler model (e.g. Ather 450X, Ola S1, TVS iQube, Generic)")
    subsystem: Optional[str] = Field(None, description="Subsystem if known: 'BMS', 'Motor Controller', 'Throttle', 'Charger'")
    top_k: int = Field(4, description="Number of technical evidence passages to retrieve")

class LookupDTCInput(BaseModel):
    dtc_code: str = Field(..., description="DTC / error code to look up (e.g. 'P0A80', 'E-08', 'E-07', 'U0100')")

class LookupComponentInput(BaseModel):
    component_name: str = Field(..., description="Component name to look up specs & pinouts for (e.g. 'throttle', 'bms', 'hall_sensor')")

class MCPClientManager:
    """
    Manages connections to MCP (Model Context Protocol) Servers and converts
    MCP tools into LangChain-compatible tools.
    """

    def __init__(self, transport: str = "sse", server_url: Optional[str] = None):
        self.transport = transport or settings.MCP_TRANSPORT
        self.server_url = server_url or settings.MCP_SSE_ENDPOINT
        self.connected_tools: List[StructuredTool] = []
        self._retrieved_sources_accumulator: List[Dict[str, Any]] = []

    def get_retrieved_evidence_cache(self) -> List[Dict[str, Any]]:
        """Return all evidence chunks retrieved during the current agent session."""
        return self._retrieved_sources_accumulator

    def clear_session_cache(self):
        """Clear cached evidence between queries."""
        self._retrieved_sources_accumulator = []

    async def _wrapped_retrieve_evidence(self, query: str, vehicle_model: Optional[str] = None, subsystem: Optional[str] = None, top_k: int = 4) -> str:
        """Invokes MCP retrieval tool and records evidence for downstream grounding calculation."""
        res_json = await tool_retrieve_technical_evidence(query, vehicle_model, subsystem, top_k)
        try:
            items = json.loads(res_json)
            if isinstance(items, list):
                for item in items:
                    if item not in self._retrieved_sources_accumulator:
                        self._retrieved_sources_accumulator.append(item)
        except Exception as e:
            logger.debug(f"Could not parse evidence items into cache: {e}")
        return res_json

    async def _wrapped_lookup_dtc(self, dtc_code: str) -> str:
        return await tool_lookup_dtc_fault_code(dtc_code)

    async def _wrapped_lookup_component(self, component_name: str) -> str:
        return await tool_lookup_component_specs(component_name)

    def load_langchain_tools(self) -> List[StructuredTool]:
        """
        Builds and returns LangChain StructuredTool instances wired directly to MCP tool handlers.
        """
        tools = [
            StructuredTool.from_function(
                coroutine=self._wrapped_retrieve_evidence,
                name="retrieve_technical_evidence",
                description="Search electric 2-wheeler service manuals and technical documents for symptoms, fault causes, and troubleshooting procedures.",
                args_schema=RetrieveEvidenceInput
            ),
            StructuredTool.from_function(
                coroutine=self._wrapped_lookup_dtc,
                name="lookup_dtc_fault_code",
                description="Look up official Electric Two-Wheeler Diagnostic Trouble Codes (DTCs) such as P0A80, E-08, E-07, U0100.",
                args_schema=LookupDTCInput
            ),
            StructuredTool.from_function(
                coroutine=self._wrapped_lookup_component,
                name="lookup_component_specs",
                description="Look up pinout diagrams, nominal DC voltage thresholds, and sensor wiring specifications for EV components (throttle, BMS, hall sensors, etc.).",
                args_schema=LookupComponentInput
            )
        ]
        self.connected_tools = tools
        logger.info(f"Loaded {len(tools)} MCP tools for LangChain agent orchestrator")
        return tools
