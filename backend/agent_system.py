from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .db import DbPaths, get_task, init_db, insert_step, list_task_steps, set_task_status, upsert_task, update_step
from .llm_ollama import OllamaConfig, ollama_chat
from .models import TaskResponse
from .rag import LocalRag, RagConfig
from .tools_web import fetch_url, wikipedia_page_summary, wikipedia_search
from .tools_code import run_python_snippet


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to extract the first {...} object.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # Remove ```python or ``` fences
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        if t.endswith("```"):
            t = t[: -len("```")].strip()
    return t


def _format_retrieval_context(blocks: List[Tuple[str, Dict[str, Any]]]) -> str:
    out: List[str] = []
    for idx, (text, meta) in enumerate(blocks, start=1):
        title = meta.get("title") or "Unknown title"
        url = meta.get("url") or meta.get("source_url") or ""
        out.append(f"[{idx}] {title} ({url})\n{text}")
    return "\n\n".join(out)


@dataclass(frozen=True)
class AgentSettings:
    ollama_base_url: str
    llm_model: str
    embed_model: str
    rag_top_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int
    wiki_search_endpoint: str
    wiki_summary_endpoint: str
    max_steps: int
    rag_max_sources: int
    enable_code_run_default: bool
    max_code_run_seconds: int


def load_settings(db_paths: DbPaths) -> AgentSettings:
    # Loads .env if present at the repo root.
    repo_root = db_paths.root_dir
    load_dotenv(str(repo_root / ".env"), override=False)

    return AgentSettings(
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=_env("LLM_MODEL", "llama3"),
        embed_model=_env("EMBED_MODEL", "nomic-embed-text"),
        rag_top_k=_env_int("RAG_TOP_K", 5),
        rag_chunk_size=_env_int("RAG_CHUNK_SIZE", 1200),
        rag_chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 200),
        wiki_search_endpoint=_env("WIKI_SEARCH_ENDPOINT", "https://en.wikipedia.org/w/api.php"),
        wiki_summary_endpoint=_env(
            "WIKI_PAGE_SUMMARY_ENDPOINT", "https://en.wikipedia.org/api/rest_v1/page/summary"
        ),
        max_steps=_env_int("MAX_STEPS", 6),
        rag_max_sources=_env_int("RAG_MAX_SOURCES", 6),
        enable_code_run_default=_env_bool("ENABLE_CODE_RUN", False),
        max_code_run_seconds=_env_int("MAX_CODE_RUN_SECONDS", 20),
    )


class TaskManager:
    def __init__(self, db_paths: DbPaths):
        self.db_paths = db_paths
        self.settings = load_settings(db_paths)
        # Per-task runtime options (kept in-memory; FastAPI background tasks run in-process).
        self.task_options: Dict[str, Dict[str, Any]] = {}

        self.ollama_cfg = OllamaConfig(
            base_url=self.settings.ollama_base_url,
            model=self.settings.llm_model,
            embed_model=self.settings.embed_model,
        )
        self.rag = LocalRag(
            db_paths=db_paths,
            ollama_cfg=self.ollama_cfg,
            rag_cfg=RagConfig(
                top_k=self.settings.rag_top_k,
                chunk_size=self.settings.rag_chunk_size,
                chunk_overlap=self.settings.rag_chunk_overlap,
            ),
        )

    def create_task(self, goal: str, max_steps: int, enable_code_run: bool) -> TaskResponse:
        task_id = str(uuid.uuid4())
        upsert_task(self.db_paths.db_path, task_id, goal, status="queued", error=None)
        self.task_options[task_id] = {"max_steps": max_steps, "enable_code_run": enable_code_run}
        # Store runtime options in memory by embedding into the task_id mapping is not persisted.
        # Instead, we will infer options from request env and keep them minimal.
        # For per-task options, we attach them to the goal prompt in runtime (done in run_task).
        return self.get_task_response(task_id)  # steps populated later

    def get_task_response(self, task_id: str) -> Optional[TaskResponse]:
        t = get_task(self.db_paths.db_path, task_id)
        if not t:
            return None
        steps = list_task_steps(self.db_paths.db_path, task_id)
        steps_out = []
        for row in steps:
            steps_out.append(
                {
                    "step_id": row["step_id"],
                    "agent": row["agent"],
                    "step_type": row["step_type"],
                    "status": row["status"],
                    "output": (row["output"] or "")[:400],
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return TaskResponse(
            task_id=t["task_id"],
            goal=t["goal"],
            status=t["status"],  # type: ignore[arg-type]
            created_at=t["created_at"],
            updated_at=t["updated_at"],
            error=t["error"],
            steps=steps_out,
            result=(next((s["output"] for s in steps_out if s["step_type"] == "finalize"), None)),
        )

    def _add_step(self, task_id: str, agent: str, step_type: str, status: str) -> str:
        step_id = str(uuid.uuid4())
        insert_step(self.db_paths.db_path, step_id, task_id, agent, step_type, status=status)
        return step_id

    def _update_step(self, step_id: str, status: str, output: Optional[str] = None, error: Optional[str] = None) -> None:
        update_step(self.db_paths.db_path, step_id, status=status, output=output, error=error)

    def run_task(self, task_id: str) -> None:
        t = get_task(self.db_paths.db_path, task_id)
        if not t:
            return

        goal: str = t["goal"]
        opts = self.task_options.get(task_id, {})
        max_steps = int(opts.get("max_steps", self.settings.max_steps))
        enable_code_run = bool(opts.get("enable_code_run", self.settings.enable_code_run_default))

        set_task_status(self.db_paths.db_path, task_id, "running")

        step_count = 0
        try:
            # 1) Supervisor planning step
            if step_count >= max_steps:
                raise RuntimeError("Reached max steps before starting.")
            step_count += 1

            supervisor_step = self._add_step(task_id, "supervisor", "web_research", "running")
            try:
                plan = self._supervisor_plan(goal)
            except Exception as e:
                plan = None

            if not plan:
                plan = {"queries": [goal], "include_code_demo": False}

            queries = plan.get("queries") or [goal]
            queries = [str(q).strip() for q in queries if str(q).strip()][: self.settings.rag_max_sources]
            include_code_demo = bool(plan.get("include_code_demo", False))
            supervisor_out = json.dumps(
                {"queries": queries, "include_code_demo": include_code_demo},
                ensure_ascii=False,
                indent=2,
            )
            self._update_step(supervisor_step, "succeeded", output=supervisor_out)

            # 2) Research agent fetch + ingest
            if step_count >= max_steps:
                raise RuntimeError("Reached max steps before research.")
            step_count += 1

            research_step = self._add_step(task_id, "researcher", "web_research", "running")
            fetched_sources: List[Dict[str, Any]] = []
            for q in queries:
                hits = wikipedia_search(q, limit=3, endpoint=self.settings.wiki_search_endpoint)
                for hit in hits[:2]:
                    title = hit.get("title")
                    page_url = hit.get("page_url")
                    if not title:
                        continue
                    summary = wikipedia_page_summary(
                        title,
                        summary_endpoint=self.settings.wiki_summary_endpoint,
                    )
                    if not summary or len(summary) < 80:
                        continue

                    meta = {"source": "wikipedia", "title": title, "url": page_url, "query": q}
                    self.rag.ingest_text(task_id=task_id, text=summary, metadata=meta)
                    fetched_sources.append(meta)

                    # Keep ingestion bounded for responsiveness.
                    if len(fetched_sources) >= self.settings.rag_max_sources:
                        break
                if len(fetched_sources) >= self.settings.rag_max_sources:
                    break

            self._update_step(
                research_step,
                "succeeded",
                output=json.dumps({"sources_ingested": fetched_sources}, ensure_ascii=False, indent=2)[:4000],
            )

            # 3) Rager step: retrieval + context assembly
            if step_count >= max_steps:
                raise RuntimeError("Reached max steps before RAG.")
            step_count += 1

            rager_step = self._add_step(task_id, "rager", "rag_index", "running")
            blocks = self.rag.retrieve(goal, task_id=task_id)
            context = _format_retrieval_context(blocks)
            self._update_step(
                rager_step,
                "succeeded",
                output=(context[:4000] if context else "No RAG context retrieved."),
            )

            # 4) Synthesis step
            if step_count >= max_steps:
                raise RuntimeError("Reached max steps before synthesis.")
            step_count += 1

            synth_step = self._add_step(task_id, "analyst", "synthesize_report", "running")
            report_markdown = None
            sources_used = []
            revision_notes = None
            retry = 0

            while retry <= 2:
                if retry == 0:
                    revision_notes = None
                else:
                    # Keep the revised prompt tight to reduce drift.
                    pass

                synth_payload = self._synthesize_report(goal, context=context, revision_notes=revision_notes)
                if isinstance(synth_payload, dict):
                    report_markdown = synth_payload.get("report_markdown") or synth_payload.get("report")  # type: ignore[assignment]
                    sources_used = synth_payload.get("sources_used") or []
                else:
                    report_markdown = str(synth_payload)

                # Reflection
                reflect = self._reflect_report(goal, report_markdown or "")
                if not reflect:
                    break
                needs_revision = bool(reflect.get("needs_revision"))
                if not needs_revision:
                    break
                revision_notes = reflect.get("revision_instructions") or ""
                retry += 1

            if not report_markdown:
                report_markdown = f"Unable to synthesize a report for: {goal}"

            self._update_step(
                synth_step,
                "succeeded",
                output=json.dumps({"sources_used": sources_used}, ensure_ascii=False, indent=2)[:2000]
                + "\n\n"
                + (report_markdown[:3500]),
            )

            # 5) Optional code demo step (does not run code unless enabled)
            if include_code_demo and (step_count < max_steps):
                step_count += 1
                code_step = self._add_step(task_id, "coder", "code_demo", "running")
                try:
                    code_text = self._code_demo(goal)
                    run_result = None
                    if enable_code_run:
                        run_result = run_python_snippet(code_text, timeout_s=self.settings.max_code_run_seconds)
                    if run_result:
                        out = f"--- CODE ---\n{code_text[:2500]}\n\n--- RUN ---\nexit={run_result.exit_code}\nSTDOUT:\n{run_result.stdout[-2000:]}\nSTDERR:\n{run_result.stderr[-2000:]}"
                    else:
                        out = f"--- CODE (not executed) ---\n{code_text[:3500]}"
                    self._update_step(code_step, "succeeded", output=out[:4000])
                except Exception as e:
                    self._update_step(code_step, "failed", error=str(e))

            # 6) Finalize
            finalize_step = self._add_step(task_id, "reflector", "finalize", "running")
            self._update_step(finalize_step, "succeeded", output=(report_markdown or "")[:8000])
            set_task_status(self.db_paths.db_path, task_id, "succeeded", error=None)
        except Exception as e:
            set_task_status(self.db_paths.db_path, task_id, "failed", error=str(e))

    def _supervisor_plan(self, goal: str) -> Optional[Dict[str, Any]]:
        system = (
            "You are a supervisor for an agentic research system. "
            "Create a plan for Wikipedia-based research. "
            "Output MUST be valid JSON with exactly these keys: "
            "`queries` (array of short search queries, max 5), "
            "`include_code_demo` (boolean)."
        )
        user = f"Goal: {goal}\n\nProduce JSON now."
        text = ollama_chat(
            self.ollama_cfg,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            format_json=True,
        )
        return _safe_json_loads(text)

    def _synthesize_report(
        self,
        goal: str,
        context: str,
        revision_notes: Optional[str] = None,
    ) -> Dict[str, Any] | str:
        system = (
            "You are an expert analyst. You must write a high-quality report using ONLY the provided context. "
            "Return ONLY valid JSON with keys: "
            "`report_markdown` (string in markdown), "
            "`sources_used` (array of URLs found in context)."
        )
        if revision_notes:
            user = (
                f"Goal: {goal}\n\n"
                f"Revision instructions from reviewer:\n{revision_notes}\n\n"
                f"Context:\n{context}\n\n"
                "Write the revised report."
            )
        else:
            user = (
                f"Goal: {goal}\n\n"
                f"Context:\n{context}\n\n"
                "Write the report."
            )
        text = ollama_chat(
            self.ollama_cfg,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            format_json=True,
        )
        parsed = _safe_json_loads(text)
        return parsed if parsed else text

    def _reflect_report(self, goal: str, report_markdown: str) -> Optional[Dict[str, Any]]:
        system = (
            "You are a strict reviewer. Evaluate the report against the goal. "
            "Return ONLY valid JSON with keys: "
            "`needs_revision` (boolean), "
            "`revision_instructions` (string). "
            "If the report is good, set needs_revision=false and revision_instructions=''."
        )
        rubric = (
            "Checklist: (1) Clear introduction, (2) Key points/comparison, (3) Evidence that uses provided sources, "
            "(4) Recommendation or conclusion, (5) Limitations/risks."
        )
        user = f"Goal: {goal}\n\nRubric:\n{rubric}\n\nReport:\n{report_markdown}\n\nReview and respond with JSON."
        text = ollama_chat(
            self.ollama_cfg,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            format_json=True,
        )
        return _safe_json_loads(text)

    def _code_demo(self, goal: str) -> str:
        system = (
            "You are a code agent. Write a single Python 3 script for demonstrating the project's local RAG retrieval. "
            "The script must: "
            "(1) import LocalRag and required modules from the repo, "
            "(2) load RAG DB and retrieve top passages for a query related to the goal, "
            "(3) print the top 3 passages' metadata and first 300 characters. "
            "Output ONLY the python code (no markdown fences)."
        )
        user = f"Goal: {goal}\n\nChoose a query that is relevant and printable."
        text = ollama_chat(
            self.ollama_cfg,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            format_json=False,
        )
        return _strip_code_fences(text)

