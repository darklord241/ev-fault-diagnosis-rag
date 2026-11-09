"""Shared filesystem and run metadata helpers for experiment runners."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results"


def create_run_directory(
    output_root: Path,
    run_id: str | None = None,
) -> tuple[str, Path]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        output_dir = root / run_id
        output_dir.mkdir(parents=False, exist_ok=False)
        return run_id, output_dir

    existing_ids = [
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdecimal()
    ]
    next_id = max(existing_ids, default=0) + 1
    while True:
        run_id = f"{next_id:02d}"
        output_dir = root / run_id
        try:
            output_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            next_id += 1
            continue
        return run_id, output_dir


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository_root: Path = REPOSITORY_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    require_nonempty: bool = False,
) -> None:
    if require_nonempty and not rows:
        raise ValueError(f"refusing to write an empty results CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: object,
    *,
    ensure_ascii: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )


def write_result_tables(
    output_dir: Path,
    aggregate_rows: Sequence[Mapping[str, object]],
    per_query_rows: Sequence[Mapping[str, object]],
) -> None:
    write_csv(
        output_dir / "aggregate_results.csv",
        aggregate_rows,
        require_nonempty=True,
    )
    write_csv(
        output_dir / "per_query_results.csv",
        per_query_rows,
        require_nonempty=True,
    )


def write_run_metadata(
    output_dir: Path,
    *,
    run_id: str,
    config_path: Path,
    input_files: Mapping[str, Path],
    corpus: Mapping[str, object],
    benchmark: Mapping[str, object],
    packages: Sequence[str] = (),
    experiment_details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    input_metadata = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(path),
        }
        for name, path in input_files.items()
    }
    package_versions = {}
    for package in packages:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    metadata: dict[str, object] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "config_sha256": file_sha256(config_path),
        "input_files": input_metadata,
        "corpus": dict(corpus),
        "benchmark": dict(benchmark),
    }
    if experiment_details is not None:
        metadata["experiment_details"] = dict(experiment_details)
    write_json(Path(output_dir) / "metadata.json", metadata)
    return metadata


def write_metric_plot(
    path: Path,
    aggregate_rows: Sequence[Mapping[str, object]],
    series_keys: Sequence[str],
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = {
        str(row["model_key"]): row
        for row in aggregate_rows
        if row["query_type"] == "overall"
    }
    missing = [key for key in series_keys if key not in overall]
    if missing:
        raise ValueError(f"aggregate results are missing overall rows for {missing}")

    metrics = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5")
    width = 0.8 / len(metrics)
    positions = list(range(len(series_keys)))
    figure, axis = plt.subplots(figsize=(11, 6))
    for metric_index, metric in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [overall[key][metric] for key in series_keys]
        if any(value is None for value in values):
            raise ValueError(f"aggregate metric {metric} contains an empty value")
        axis.bar(
            [position + offset for position in positions],
            [float(value) for value in values],
            width,
            label=metric.replace("_", " "),
        )
    axis.set_xticks(positions, series_keys, rotation=20, ha="right")
    axis.set(ylabel="Score", ylim=(0, 1))
    axis.set_title(title)
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
