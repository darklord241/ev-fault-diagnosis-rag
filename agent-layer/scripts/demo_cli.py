import asyncio
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator.agent import AgentOrchestrator
from src.common.models import QueryRequest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

async def run_demo():
    console.print(Panel.fit(
        "[bold cyan]⚡ AI-Assisted Fault Diagnosis & MCP Retrieval for Electric Two-Wheelers[/bold cyan]\n"
        "[dim]Agent Orchestrator <-> Model Context Protocol (MCP) Integration Engine[/dim]",
        border_style="cyan"
    ))

    orchestrator = AgentOrchestrator()

    sample_queries = [
        ("Throttle is completely unresponsive after driving through heavy rain", "Generic 48V E-Scooter", "Throttle"),
        ("Vehicle cuts power suddenly when accelerating uphill; error P0A80 on cluster", "Ather 450X", "Battery Management System"),
        ("Rear motor makes shuddering growl at low RPM and throws error code E-08", "Ola S1 Pro", "Motor Controller"),
        ("Headlight and horn stopped working, but scooter still drives normally", "Generic Electric Scooter", "Auxiliary Electrics")
    ]

    console.print("\n[bold yellow]Sample Diagnostic Queries:[/bold yellow]")
    for idx, (q, model, subsys) in enumerate(sample_queries, 1):
        console.print(f" [cyan]{idx}[/cyan]. {q} [dim]({model} | {subsys})[/dim]")

    console.print("\nType a query number [1-4], type your own query, or 'exit' to quit:")
    
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not user_input or user_input.lower() == "exit":
        return

    if user_input.isdigit() and 1 <= int(user_input) <= len(sample_queries):
        chosen_query, chosen_model, chosen_subsys = sample_queries[int(user_input) - 1]
    else:
        chosen_query = user_input
        chosen_model = "Generic Electric Two-Wheeler"
        chosen_subsys = None

    req = QueryRequest(
        query=chosen_query,
        vehicle_model=chosen_model,
        subsystem=chosen_subsys
    )

    with console.status(f"[bold green]Orchestrating agent reasoning & MCP retrieval for '{chosen_query}'...", spinner="dots"):
        result = await orchestrator.diagnose_fault(req)

    console.print("\n" + "="*80)
    console.print(f"[bold green]✅ Diagnostic Result Generated in {result.execution_time_seconds}s (Grounding Score: {result.grounding_score})[/bold green]")
    console.print("="*80 + "\n")

    # Display Root Cause / Synthesized Answer
    console.print(Panel(Markdown(result.root_cause_analysis), title="[bold magenta]🛠️ Grounded Diagnostic Analysis[/bold magenta]", border_style="magenta"))

    # Display Diagnostic Steps Table
    if result.diagnostic_procedure:
        table = Table(title="📋 Step-by-Step Diagnostic Verification Procedure", border_style="blue")
        table.add_column("Step", style="bold cyan", width=6)
        table.add_column("Action", style="white")
        table.add_column("Component", style="yellow")
        table.add_column("Expected Reading", style="green")

        for step in result.diagnostic_procedure:
            table.add_row(
                str(step.step_number),
                step.action,
                step.component_checked,
                step.expected_reading or "N/A"
            )
        console.print(table)

    # Display Supporting Evidence Sources
    if result.evidence_sources:
        src_table = Table(title="📚 Supporting Technical Documentation Evidence", border_style="green")
        src_table.add_column("Source ID", style="bold yellow", width=16)
        src_table.add_column("Document Title", style="cyan")
        src_table.add_column("Section / Page", style="dim")
        src_table.add_column("Relevance", style="green", width=10)

        for src in result.evidence_sources:
            src_table.add_row(
                src.source_id,
                src.document_title,
                f"{src.section or 'General'} (p. {src.page_number or 'N/A'})",
                f"{src.relevance_score:.2f}"
            )
        console.print(src_table)

    # Safety Warnings
    if result.safety_warnings:
        warn_text = "\n".join([f"• {w}" for w in result.safety_warnings])
        console.print(Panel(warn_text, title="[bold red]⚠️ Critical Safety Precautions[/bold red]", border_style="red"))

if __name__ == "__main__":
    asyncio.run(run_demo())
