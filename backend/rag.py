from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .db import DbPaths, insert_docs, iter_docs
from .llm_ollama import OllamaConfig, ollama_embeddings


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200


class LocalRag:
    def __init__(self, db_paths: DbPaths, ollama_cfg: OllamaConfig, rag_cfg: RagConfig):
        self.db_paths = db_paths
        self.ollama_cfg = ollama_cfg
        self.rag_cfg = rag_cfg

    def ingest_text(
        self,
        task_id: Optional[str],
        text: str,
        metadata: Dict[str, Any],
    ) -> int:
        chunks = chunk_text(text, self.rag_cfg.chunk_size, self.rag_cfg.chunk_overlap)
        if not chunks:
            return 0

        # Embed in small batches to avoid request payload limits.
        all_embeddings: List[List[float]] = []
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embs = ollama_embeddings(self.ollama_cfg, batch)
            all_embeddings.extend(embs)

        metadatas: List[Dict[str, Any]] = []
        for idx, _ in enumerate(chunks):
            m = dict(metadata)
            m["chunk_index"] = idx
            metadatas.append(m)

        insert_docs(
            self.db_paths.db_path,
            task_id=task_id,
            texts=chunks,
            metadatas=metadatas,
            embeddings=all_embeddings,
        )
        return len(chunks)

    def retrieve(self, query: str, task_id: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        q = _normalize_whitespace(query)
        if not q:
            return []

        query_emb = ollama_embeddings(self.ollama_cfg, [q])[0]
        qvec = np.array(query_emb, dtype=np.float32)

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for _doc_id, metadata, text, vec in iter_docs(self.db_paths.db_path, task_id=task_id):
            score = cosine_similarity(qvec, vec)
            scored.append((score, text, metadata))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.rag_cfg.top_k]
        return [(t, m) for _, t, m in top]

