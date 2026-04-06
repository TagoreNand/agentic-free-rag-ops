from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .agent_system import TaskManager
from .db import get_db_paths, init_db
from .models import TaskCreateRequest, TaskResponse


app = FastAPI(title="Agentic Free RAG Ops", version="0.1.0")

DB_PATHS = get_db_paths()
init_db(DB_PATHS.db_path)

# Shared TaskManager instance
task_manager = TaskManager(db_paths=DB_PATHS)

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("Missing frontend assets.", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/v1/tasks", response_model=TaskResponse)
def create_task(req: TaskCreateRequest, background: BackgroundTasks) -> TaskResponse:
    task = task_manager.create_task(
        goal=req.goal,
        max_steps=req.max_steps,
        enable_code_run=req.enable_code_run,
    )

    # Run agent flow in background
    background.add_task(task_manager.run_task, task.task_id)
    return task_manager.get_task_response(task.task_id)


@app.get("/v1/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str) -> TaskResponse:
    task = task_manager.get_task_response(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

