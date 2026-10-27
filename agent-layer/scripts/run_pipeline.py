import asyncio
import sys
import os
from pathlib import Path

# Configure utf-8 standard output for Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.batch_processor import BatchPipelineProcessor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(force_terminal=True, legacy_windows=False)

async def main():
    console.print(Panel.fit(
        "[bold cyan]EV Technical Evidence & Diagnosis Pipeline[/bold cyan]\n"
        "[dim]Cleaner -> LLM / Gemini Summarizer -> Website Contract Formatter[/dim]",
        border_style="cyan"
    ))

    processor = BatchPipelineProcessor()
    
    with console.status("[bold green]Processing raw retrieval baseline, cleaning chunks, and generating grounded summaries...", spinner="dots"):
        batch = await processor.run_pipeline()

    console.print(f"\n[bold green]Pipeline Successfully Processed {batch.total_cases} Cases![/bold green]\n")

    summary_table = Table(title="Pipeline Cleaning & Summarization Results", border_style="blue")
    summary_table.add_column("Case ID", style="bold yellow", width=22)
    summary_table.add_column("User Question", style="white", width=35)
    summary_table.add_column("Raw Chunks", style="dim", justify="center", width=12)
    summary_table.add_column("Unique Chunks", style="bold green", justify="center", width=14)
    summary_table.add_column("Grounding", style="cyan", justify="center", width=10)

    for case in batch.cases:
        summary_table.add_row(
            case.case_id,
            case.query[:35] + ("..." if len(case.query) > 35 else ""),
            str(case.raw_chunks_retrieved),
            f"{case.unique_chunks_used} (deduped)",
            f"{case.grounding_score:.2f}"
        )
    console.print(summary_table)

    console.print("\n[bold magenta]Sample Website Card Payload (Case 1):[/bold magenta]")
    sample_case = batch.cases[0]
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {sample_case.query}\n\n"
        f"[bold green]Direct Answer:[/bold green] {sample_case.direct_answer}\n\n"
        f"[bold yellow]Key Takeaways:[/bold yellow]\n" + "\n".join([f"- {t}" for t in sample_case.key_takeaways]) + "\n\n"
        f"[bold blue]Technical Specs:[/bold blue]\n" + "\n".join([f"- [bold]{k}[/bold]: {v}" for k, v in sample_case.technical_specs.items()]),
        title=f"[bold]Case: {sample_case.case_id}[/bold]",
        border_style="magenta"
    ))

    console.print("\n[bold green]Generated Website Artifacts:[/bold green]")
    console.print(" 1. Website JSON payload: [cyan]data/website_payload.json[/cyan] (Ready for REST API / website consumption)")
    console.print(" 2. Markdown report:      [cyan]data/website_payload.md[/cyan] (Human-readable / report appendix)")
    console.print("\n[dim]To serve this via API to your website teammate, run:[/dim] [bold yellow]python scripts/run_orchestrator_api.py[/bold yellow]\n")

if __name__ == "__main__":
    asyncio.run(main())
