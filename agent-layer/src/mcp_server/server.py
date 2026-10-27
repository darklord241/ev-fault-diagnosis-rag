import sys
import argparse
from typing import Optional

try:
    from mcp.server import MCPServer
    mcp_server = MCPServer(
        name="ev-retrieval-service",
        instructions="Electric Two-Wheeler Technical Knowledge Base & Hybrid Retrieval Tool Server"
    )
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
        mcp_server = FastMCP(
            "ev-retrieval-service",
            instructions="Electric Two-Wheeler Technical Knowledge Base & Hybrid Retrieval Tool Server"
        )
    except Exception:
        class DummyMCPServer:
            def tool(self, *args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator
            def run(self, *args, **kwargs):
                pass
        mcp_server = DummyMCPServer()

from src.mcp_server.tools import (
    tool_retrieve_technical_evidence,
    tool_lookup_dtc_fault_code,
    tool_lookup_component_specs
)
from src.common.logging import logger
from config.settings import settings

@mcp_server.tool(
    name="retrieve_technical_evidence",
    description="Retrieve technical service manuals, troubleshooting guides, and knowledge chunks for electric 2-wheeler faults."
)
async def retrieve_technical_evidence(
    query: str,
    vehicle_model: Optional[str] = None,
    subsystem: Optional[str] = None,
    top_k: int = 4
) -> str:
    """Retrieve technical documentation evidence for electric 2-wheeler diagnosis."""
    return await tool_retrieve_technical_evidence(query, vehicle_model, subsystem, top_k)

@mcp_server.tool(
    name="lookup_dtc_fault_code",
    description="Look up official EV Diagnostic Trouble Codes (DTCs), cause analysis, and suggested checks."
)
async def lookup_dtc_fault_code(dtc_code: str) -> str:
    """Look up an electric 2-wheeler DTC fault code."""
    return await tool_lookup_dtc_fault_code(dtc_code)

@mcp_server.tool(
    name="lookup_component_specs",
    description="Look up pinout, operating voltages, and diagnostic thresholds for EV components."
)
async def lookup_component_specs(component_name: str) -> str:
    """Look up component pinouts and operating thresholds."""
    return await tool_lookup_component_specs(component_name)

def run_server():
    parser = argparse.ArgumentParser(description="Electric Two-Wheeler MCP Retrieval Server")
    parser.add_argument("--transport", choices=["sse", "stdio"], default=settings.MCP_TRANSPORT, help="MCP transport protocol")
    parser.add_argument("--host", default=settings.MCP_SERVER_HOST, help="Host address for SSE transport")
    parser.add_argument("--port", type=int, default=settings.MCP_SERVER_PORT, help="Port for SSE transport")
    args = parser.parse_args()

    logger.info(f"Starting EV Retrieval MCP Server on transport={args.transport} (Host: {args.host}:{args.port})")

    if hasattr(mcp_server, "settings"):
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
    mcp_server.run(transport=args.transport)

if __name__ == "__main__":
    run_server()
