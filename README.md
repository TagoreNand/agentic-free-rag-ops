# Agentic Free RAG Ops (Ollama-based)

An advanced, GitHub-ready example of an **agentic** AI system that can:

- Plan a multi-step task (planner/supervisor agent)
- Use multiple specialist agents (researcher, code, analyst, reflector)
- Fetch and read web content (free/public sources)
- Build a local RAG index (embeddings via **Ollama** + SQLite storage)
- Generate a final report and persist task/step logs

This project is designed to use **free/local** components:

- LLM: **Ollama** (run locally)
- Embeddings: **Ollama embeddings** (run locally)
- Web fetching: public HTTP endpoints (no API keys)

## Features

- FastAPI API + a minimal web UI
- Background execution for tasks
- SQLite persistence for tasks/steps
- RAG index stored locally
- Reflection loop to improve quality (bounded retries)

## Prerequisites

1. Install **Ollama**: https://ollama.com
2. Pull models (examples):
   - `ollama pull llama3`
   - `ollama pull nomic-embed-text`

3. Python 3.10+ installed

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Copy env file:

```bash
# Windows (PowerShell)
copy .env.example .env

# Or (bash/mac)
cp .env.example .env
```

Edit `.env` if you want different models or settings.

## Run

```bash
uvicorn backend.app:app --reload --port 8000
```

Open:

- http://localhost:8000

## API quickstart

Create a task:

```bash
curl -X POST http://localhost:8000/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal":"Compare 3 open-source vector databases and summarize tradeoffs. Provide a short recommendation."}'
```

Check status:

```bash
curl http://localhost:8000/v1/tasks/<task_id>
```

## Notes / Limitations

- The tool-using “code agent” executes Python in a subprocess. It is sandboxed only lightly; do not run untrusted code.
- Web fetching relies on network access available on your machine.
- This is an educational/portfolio project—extend it for production hardening (timeouts, rate limiting, auth, better scraping, etc.).

