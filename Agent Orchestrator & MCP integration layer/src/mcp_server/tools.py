import json
from typing import Optional, Dict, Any, List
from src.mcp_server.backend_adapter import backend_adapter
from src.common.logging import logger

async def tool_retrieve_technical_evidence(
    query: str,
    vehicle_model: Optional[str] = None,
    subsystem: Optional[str] = None,
    top_k: int = 4
) -> str:
    """
    Search the technical knowledge base (service manuals, troubleshooting trees, circuit diagrams)
    for evidence regarding electric two-wheeler faults.

    Args:
        query: Symptom, fault description, or question (e.g. 'throttle unresponsive after rain', 'motor shuddering at low speed')
        vehicle_model: Optional vehicle model (e.g. 'Ather 450X', 'Ola S1', 'Generic 48V')
        subsystem: Optional subsystem (e.g. 'BMS', 'Motor Controller', 'Throttle', 'Charger')
        top_k: Number of relevant evidence chunks to return (default 4)

    Returns:
        JSON string representing the list of retrieved evidence sources with metadata.
    """
    logger.info(f"[MCP Tool] retrieve_technical_evidence: query='{query}', model='{vehicle_model}', subsystem='{subsystem}'")
    sources = await backend_adapter.search_hybrid_evidence(
        query=query,
        vehicle_model=vehicle_model,
        subsystem=subsystem,
        top_k=top_k
    )
    result = [s.model_dump() for s in sources]
    return json.dumps(result, indent=2)

async def tool_lookup_dtc_fault_code(dtc_code: str) -> str:
    """
    Look up an official Diagnostic Trouble Code (DTC) or error code for Electric 2-Wheelers.

    Args:
        dtc_code: Error code string (e.g., 'P0A80', 'E-08', 'E-07', 'U0100')

    Returns:
        JSON string with code definition, causes, symptoms, and suggested checks.
    """
    logger.info(f"[MCP Tool] lookup_dtc_fault_code: code='{dtc_code}'")
    record = await backend_adapter.get_dtc_details(dtc_code)
    if record:
        return json.dumps(record.model_dump(), indent=2)
    return json.dumps({"status": "not_found", "message": f"No DTC record found for code: {dtc_code}"})

async def tool_lookup_component_specs(component_name: str) -> str:
    """
    Look up electrical specifications, nominal voltage/current ranges, and pinout details for an EV component.

    Args:
        component_name: Name of component (e.g. 'throttle', 'bms', 'hall_sensor', 'inverter')

    Returns:
        JSON string with component nominal specifications, pinouts, and test thresholds.
    """
    logger.info(f"[MCP Tool] lookup_component_specs: component='{component_name}'")
    spec = await backend_adapter.get_component_specifications(component_name)
    if spec:
        return json.dumps(spec.model_dump(), indent=2)
    return json.dumps({"status": "not_found", "message": f"No technical specification found for: {component_name}"})
