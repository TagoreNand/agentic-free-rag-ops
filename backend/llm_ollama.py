from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    embed_model: str
    timeout_s: int = 120


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def ollama_chat(
    cfg: OllamaConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    format_json: bool = False,
) -> str:
    """
    Calls Ollama /api/chat and returns the assistant message content.

    `format_json=true` adds a hint to keep output valid JSON (not a guarantee).
    """
    url = cfg.base_url.rstrip("/") + "/api/chat"
    system = None
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content")

    prompt_messages = messages
    if format_json:
        hint = (
            "\n\nReturn ONLY valid JSON. No markdown, no commentary. "
            "All strings must be valid JSON strings."
        )
        if system:
            prompt_messages = [dict(messages[0]), *messages[1:]]
            prompt_messages[0]["content"] = system + hint
        else:
            prompt_messages = [{"role": "system", "content": hint}, *messages]

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": prompt_messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = _post_json(url, payload, timeout_s=cfg.timeout_s)
    # Ollama returns: { "message": { "role": "...", "content": "..." }, ... }
    return data["message"]["content"]


def ollama_embeddings(cfg: OllamaConfig, texts: List[str]) -> List[List[float]]:
    url = cfg.base_url.rstrip("/") + "/api/embeddings"
    payload = {"model": cfg.embed_model, "prompt": texts[0] if len(texts) == 1 else texts}
    # Ollama supports batching by passing a list in some versions; if not, we fall back.
    try:
        data = _post_json(url, payload, timeout_s=cfg.timeout_s)
        if isinstance(data.get("embeddings"), list) and data["embeddings"] and isinstance(
            data["embeddings"][0], list
        ):
            return data["embeddings"]
        # Some versions return {"embedding": [...]} for a single prompt.
        if "embedding" in data:
            return [data["embedding"]]
    except requests.HTTPError:
        pass

    # Fallback: embed one-by-one
    out: List[List[float]] = []
    for t in texts:
        single_payload = {"model": cfg.embed_model, "prompt": t}
        data = _post_json(url, single_payload, timeout_s=cfg.timeout_s)
        if "embedding" not in data:
            raise RuntimeError(f"Unexpected embeddings response: {json.dumps(data)[:300]}")
        out.append(data["embedding"])
    return out

