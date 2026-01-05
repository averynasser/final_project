from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from .base_agent import BaseAgent
from app.core.llm import chat_completion
from app.core.config import RAG_PRODUCTS_PATH, EMBEDDING_MODEL


# ---- ENV LOADING (robust, Docker-safe) ----
# Prefer .env at project root. If your config already loads it, this is harmless.
def _load_env_once() -> None:
    # /final_project/app/agents/rag_agent.py -> parents[2] == project root (/final_project)
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # fallback: try current working dir .env (dev convenience)
        load_dotenv()


_load_env_once()


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


class RAGAgent(BaseAgent):
    """
    Agent untuk melakukan RAG (Retrieval-Augmented Generation) lokal:

    - search(query): cari dokumen relevan dari Qdrant
    - answer(query): buat jawaban LLM berbasis konteks hasil search
    """

    def __init__(
        self,
        collection_name: str = "olist_products",
        top_k: int = 5,
    ):
        super().__init__(name="RAGAgent", role="RAG over product review docs")

        self.collection_name = collection_name
        self.top_k = top_k

        # ---- OpenAI Embedding Client ----
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY tidak ditemukan. Pastikan sudah di-set via .env atau environment variable.")
        self._embed_client = OpenAI(api_key=api_key)
        self._embed_model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

        # ---- Qdrant Client ----
        qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
        qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY belum di-set. Pastikan ada di .env atau environment variable.")
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # ---- Load rag_products.csv (DO NOT hardcode app/agents/...) ----
        # Priority:
        # 1) RAG_PRODUCTS_PATH from config (recommended: data/processed/rag_products.csv)
        # 2) common fallbacks (in case you still keep a copy somewhere)
        candidates: List[Path] = []

        try:
            candidates.append(Path(RAG_PRODUCTS_PATH))
        except Exception:
            pass

        # Fallbacks (not recommended long-term, but helps migration)
        project_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                project_root / "data" / "processed" / "rag_products.csv",
                project_root / "app" / "agents" / "rag_products.csv",
                Path.cwd() / "data" / "processed" / "rag_products.csv",
                Path.cwd() / "app" / "agents" / "rag_products.csv",
            ]
        )

        rag_path: Path | None = None
        for p in candidates:
            if p and p.exists():
                rag_path = p
                break

        if rag_path is None:
            tried = "\n".join([str(p) for p in candidates if p is not None])
            raise FileNotFoundError(
                "rag_products.csv tidak ditemukan. Lokasi yang dicoba:\n"
                f"{tried}\n\n"
                "Solusi: simpan file di data/processed/rag_products.csv dan pastikan ter-copy ke Docker image."
            )

        self._df = pd.read_csv(rag_path)

        if "doc_id" not in self._df.columns:
            raise ValueError("rag_products.csv harus punya kolom 'doc_id'")

        self._df.set_index("doc_id", inplace=True)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._embed_client.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def search(self, query: str) -> List[Document]:
        [vec] = self._embed([query])

        resp = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=self.top_k,
            with_payload=True,
        )
        hits = resp.points

        docs: List[Document] = []
        for h in hits:
            doc_id = str(h.id)

            if doc_id not in self._df.index:
                metadata = h.payload or {}
                text = metadata.get("doc_text", "")
            else:
                row = self._df.loc[doc_id]
                text = row.get("doc_text", "")
                metadata = {
                    "product_id": row.get("product_id", ""),
                    "product_category_en": row.get("product_category_en", ""),
                    "seller_id": row.get("seller_id", ""),
                    "seller_city": row.get("seller_city", ""),
                    "avg_review_score": float(row.get("avg_review_score", 0) or 0),
                    "score": h.score,
                }

            docs.append(Document(id=doc_id, text=text, metadata=metadata))

        return docs

    def answer(self, query: str) -> Dict[str, Any]:
        docs = self.search(query)

        context_lines = []
        for i, d in enumerate(docs, start=1):
            m = d.metadata or {}

            # Build HUMAN-READABLE PRODUCT LABEL
            category = m.get("product_category_en", "") or "Unknown category"
            city = m.get("seller_city", "") or "Unknown city"
            rating = m.get("avg_review_score", 0)

            product_label = f"{category} product (seller in {city})"

            context_lines.append(
                f"{i}. "
                f"label={product_label} | "
                f"product_id={m.get('product_id', '')} | "
                f"avg_review_score={rating}"
            )

        context_text = "\n".join(context_lines) if context_lines else "(no candidates found)"

        system_prompt = (
            "You are an e-commerce product analyst.\n"
            "You receive candidate products with metadata.\n"
            "IMPORTANT RULES:\n"
            "- The dataset does NOT contain product names.\n"
            "- NEVER invent product names.\n"
            "- Use the provided label (category + seller city) instead.\n"
            "- Be clear and honest about what the product represents."
        )

        user_msg = (
            "Jawaban harus menggunakan bahasa yang sama dengan pertanyaan user.\n\n"
            f"Pertanyaan user:\n{query}\n\n"
            "Daftar kandidat produk:\n"
            f"{context_text}\n\n"
            "Format output:\n"
            "1) Paragraf rekomendasi singkat.\n"
            "2) Bullet list produk dengan format:\n"
            "- Produk: <label>, Rating: <avg_review_score>, Product ID: <product_id>\n"
            "3) Jangan menyebutkan bahwa data tidak punya nama produk kecuali ditanya."
        )

        answer_text = chat_completion(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=600,
        )

        sources: List[Dict[str, Any]] = []
        for d in docs:
            m = dict(d.metadata or {})
            m["doc_id"] = d.id
            # add explicit label for UI/debug
            m["product_label"] = f"{m.get('product_category_en','')} product (seller in {m.get('seller_city','')})"
            sources.append(m)

        return {
            "answer": answer_text,
            "sources": sources,
        }

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        result = self.answer(task)
        new_ctx = context.copy()
        new_ctx["rag_answer"] = result["answer"]
        new_ctx["rag_sources"] = result["sources"]

        return {
            "agent": self.name,
            "role": self.role,
            "summary": "RAG answer generated.",
            "answer": result["answer"],
            "sources": result["sources"],
            "context": new_ctx,
        }
