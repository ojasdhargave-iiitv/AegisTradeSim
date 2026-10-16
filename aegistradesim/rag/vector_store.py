"""Vector storage and similarity retrieval for AegisTradeSim RAG."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try importing chromadb if user installed it, else use built-in SQLite/JSON vector store
try:
    import chromadb  # type: ignore[import-not-found,import-untyped]
    _HAS_CHROMADB = True
except (ImportError, ModuleNotFoundError):
    chromadb = None  # type: ignore[assignment]
    _HAS_CHROMADB = False



class LocalEmbedder:
    """Embedder that connects to local Ollama or falls back to term-frequency vectors."""

    def __init__(self, model_name: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/embeddings"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Fetch embeddings from Ollama or compute normalized TF vectors."""
        embeddings: list[list[float]] = []
        for text in texts:
            emb = self._embed_ollama(text)
            if emb is None:
                emb = self._sparse_hash_vector(text)
            embeddings.append(emb)
        return embeddings

    def _embed_ollama(self, text: str) -> list[float] | None:
        try:
            payload = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("embedding")
        except Exception:
            return None

    @staticmethod
    def _sparse_hash_vector(text: str, dim: int = 256) -> list[float]:
        """Deterministic term-frequency vector fallback (256-dim unit vector)."""
        vec = [0.0] * dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        counts = Counter(words)
        for word, count in counts.items():
            idx = abs(hash(word)) % dim
            vec[idx] += float(count)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two numeric lists."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class ReportVectorStore:
    """Hybrid Vector Store: Uses ChromaDB if installed, with a self-contained persistent fallback."""

    def __init__(self, persist_dir: str | Path | None = None):
        if persist_dir is None:
            home = os.path.expanduser("~")
            persist_dir = Path(home) / ".aegistradesim" / "rag_store" / "vectors"
        self.persist_path = Path(persist_dir).expanduser()
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embedder = LocalEmbedder()
        self._fallback_file = self.persist_path / "fallback_vectors.json"

        self.chroma_collection = None
        if _HAS_CHROMADB:
            try:
                client = chromadb.PersistentClient(path=str(self.persist_path))
                self.chroma_collection = client.get_or_create_collection(
                    name="trading_reports",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("ChromaDB initialization failed (%s); using fallback store.", e)
                self.chroma_collection = None

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Insert or update chunks into the vector store."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        if self.chroma_collection is not None:
            ids = [c["chunk_id"] for c in chunks]
            metadatas = [
                {
                    "ticker": c.get("ticker", ""),
                    "trade_date": c.get("trade_date", ""),
                    "chunk_type": c.get("chunk_type", ""),
                    "source_path": c.get("source_path", ""),
                }
                for c in chunks
            ]
            self.chroma_collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return

        # Fallback JSON vector storage
        store = self._load_fallback_store()
        for c, emb in zip(chunks, embeddings):
            store[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "ticker": c.get("ticker", ""),
                "trade_date": c.get("trade_date", ""),
                "chunk_type": c.get("chunk_type", ""),
                "source_path": c.get("source_path", ""),
                "embedding": emb,
            }
        self._save_fallback_store(store)

    def search_similar(
        self,
        ticker: str,
        query: str,
        top_k: int = 3,
        chunk_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve most similar chunks for a given ticker and query."""
        ticker = ticker.upper()

        if self.chroma_collection is not None:
            where_filter: dict[str, Any] = {"ticker": ticker}
            if chunk_type:
                where_filter = {"$and": [{"ticker": ticker}, {"chunk_type": chunk_type}]}

            query_emb = self.embedder.embed_texts([query])[0]
            try:
                results = self.chroma_collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where=where_filter,
                )
                hits: list[dict[str, Any]] = []
                if results and results["documents"]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0]
                    dists = results.get("distances", [[0.0] * len(docs)])[0]
                    for doc, meta, dist in zip(docs, metas, dists):
                        hits.append({
                            "text": doc,
                            "metadata": meta,
                            "score": round(max(0.0, 1.0 - float(dist)), 4),
                        })
                return hits
            except Exception as e:
                logger.warning("Chroma search error (%s); falling back to file search.", e)

        # Fallback scan
        store = self._load_fallback_store()
        query_emb = self.embedder.embed_texts([query])[0]

        candidates = []
        for item in store.values():
            if item.get("ticker") != ticker:
                continue
            if chunk_type and item.get("chunk_type") != chunk_type:
                continue
            sim = cosine_similarity(query_emb, item.get("embedding", []))
            candidates.append({
                "text": item["text"],
                "metadata": {
                    "ticker": item["ticker"],
                    "trade_date": item["trade_date"],
                    "chunk_type": item["chunk_type"],
                    "source_path": item["source_path"],
                },
                "score": round(sim, 4),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def count_chunks(self) -> int:
        """Return total number of chunks stored in vector store."""
        if self.chroma_collection is not None:
            try:
                return int(self.chroma_collection.count())
            except Exception:
                pass
        return len(self._load_fallback_store())

    def count_ticker_chunks(self, ticker: str) -> int:
        """Return number of chunks stored for a specific ticker."""
        ticker = ticker.upper()
        if self.chroma_collection is not None:
            try:
                res = self.chroma_collection.get(where={"ticker": ticker})
                return len(res.get("ids", []))
            except Exception:
                pass
        store = self._load_fallback_store()
        return sum(1 for item in store.values() if item.get("ticker") == ticker)

    def _load_fallback_store(self) -> dict[str, Any]:
        if self._fallback_file.exists():
            try:
                return json.loads(self._fallback_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_fallback_store(self, data: dict[str, Any]) -> None:
        self._fallback_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

