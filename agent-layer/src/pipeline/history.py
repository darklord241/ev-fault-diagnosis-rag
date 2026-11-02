"""SQLite persistence for recent website diagnosis requests and results."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from src.pipeline.models import WebsiteCardPayload

AGENT_ROOT = Path(__file__).resolve().parents[2]


class DiagnosisHistoryStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        configured_path = path or Path(settings.DIAGNOSIS_DB_PATH).expanduser()
        self.path = (
            configured_path
            if configured_path.is_absolute()
            else AGENT_ROOT / configured_path
        )

    def save(
        self,
        *,
        query: str,
        vehicle_model: Optional[str],
        subsystem: Optional[str],
        result: WebsiteCardPayload,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=5.0) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    vehicle_model TEXT,
                    subsystem TEXT,
                    processed_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO diagnosis_history (
                    case_id, query, vehicle_model, subsystem, processed_at, result_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.case_id,
                    query,
                    vehicle_model,
                    subsystem,
                    result.processed_at.isoformat(),
                    result.model_dump_json(),
                ),
            )

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        with sqlite3.connect(self.path, timeout=5.0) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT case_id, query, vehicle_model, subsystem, processed_at, result_json
                FROM diagnosis_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "case_id": row["case_id"],
                "query": row["query"],
                "vehicle_model": row["vehicle_model"],
                "subsystem": row["subsystem"],
                "processed_at": row["processed_at"],
                "result": json.loads(row["result_json"]),
            }
            for row in rows
        ]

    def clear(self) -> int:
        if not self.path.exists():
            return 0

        with sqlite3.connect(self.path, timeout=5.0) as connection:
            cursor = connection.execute("DELETE FROM diagnosis_history")
            return cursor.rowcount


diagnosis_history_store = DiagnosisHistoryStore()
