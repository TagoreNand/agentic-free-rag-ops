from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DbPaths:
    root_dir: Path
    db_path: Path


def get_db_paths() -> DbPaths:
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    data_dir = project_root / "backend" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return DbPaths(root_dir=project_root, db_path=data_dir / "rag.db")


@contextmanager
def connect(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
              task_id TEXT PRIMARY KEY,
              goal TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
              step_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              agent TEXT NOT NULL,
              step_type TEXT NOT NULL,
              status TEXT NOT NULL,
              output TEXT,
              error TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
              doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
              task_id TEXT,
              text TEXT NOT NULL,
              metadata TEXT,
              embedding BLOB NOT NULL
            )
            """
        )


def upsert_task(
    db_path: Path,
    task_id: str,
    goal: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(task_id, goal, status, created_at, updated_at, error)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
              goal=excluded.goal,
              status=excluded.status,
              updated_at=excluded.updated_at,
              error=excluded.error
            """,
            (task_id, goal, status, now, now, error),
        )


def set_task_status(db_path: Path, task_id: str, status: str, error: Optional[str] = None) -> None:
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status=?, updated_at=?, error=? WHERE task_id=?",
            (status, now, error, task_id),
        )


def insert_step(
    db_path: Path,
    step_id: str,
    task_id: str,
    agent: str,
    step_type: str,
    status: str,
) -> None:
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO steps(step_id, task_id, agent, step_type, status, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (step_id, task_id, agent, step_type, status, now, now),
        )


def update_step(
    db_path: Path,
    step_id: str,
    status: str,
    output: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE steps
            SET status=?, updated_at=?, output=COALESCE(?, output), error=?
            WHERE step_id=?
            """,
            (status, now, output, error, step_id),
        )


def list_task_steps(db_path: Path, task_id: str) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT step_id, agent, step_type, status, output, error, created_at, updated_at FROM steps WHERE task_id=? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()
    return rows


def get_task(db_path: Path, task_id: str) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    return row


def insert_docs(
    db_path: Path,
    task_id: Optional[str],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    with connect(db_path) as conn:
        for text, meta, emb in zip(texts, metadatas, embeddings):
            vec = np.array(emb, dtype=np.float32)
            blob = vec.tobytes()
            conn.execute(
                """
                INSERT INTO docs(task_id, text, metadata, embedding)
                VALUES(?, ?, ?, ?)
                """,
                (task_id, text, json.dumps(meta, ensure_ascii=False), blob),
            )


def iter_docs(db_path: Path, task_id: Optional[str] = None) -> Iterable[Tuple[str, Dict[str, Any], str, np.ndarray]]:
    with connect(db_path) as conn:
        if task_id is None:
            rows = conn.execute("SELECT doc_id, text, metadata, embedding FROM docs").fetchall()
        else:
            rows = conn.execute(
                "SELECT doc_id, text, metadata, embedding FROM docs WHERE task_id=?",
                (task_id,),
            ).fetchall()

        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            blob = row["embedding"]
            # We don't know embedding dimension without additional metadata,
            # but we can infer from blob length and float32 size.
            dim = len(blob) // 4
            vec = np.frombuffer(blob, dtype=np.float32).reshape((dim,))
            yield (str(row["doc_id"]), metadata, row["text"], vec)

