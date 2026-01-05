from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from .base_agent import BaseAgent
from app.core.llm import chat_completion
from app.core.config import PROJECT_ROOT


class SQLAgent(BaseAgent):
    """
    Agent untuk menjawab pertanyaan berbasis SQL di atas SQLite olist.db.

    Fitur:
    - Pertanyaan bisa Bahasa Indonesia atau Inggris
    - Jawaban LLM bisa dipilih: Indonesia / Inggris
    - Auto-suggest kategori (dari product_category_name_english)
    - Fallback query otomatis kalau hasil pertama kosong
    - Keamanan query (SELECT-only)
    """

    # mapping kasar nama kategori Indonesia -> nilai english di product_category_name_english
    CATEGORY_ALIASES: Dict[str, str] = {
        "elektronik": "electronics",
        "elektronik rumah tangga": "home_appliances",
        "fashion pria": "mens_fashion",
        "fashion wanita": "womens_fashion",
        "buku": "books_general_interest",
        # Tambah sendiri kalau perlu
    }

    # kata-kata SQL berbahaya yang tidak boleh muncul
    UNSAFE_WORDS = [
        "update ", "delete ", "insert ", "alter ",
        "drop ", "create ", "replace ", "truncate "
    ]

    def __init__(self, db_path: Path | None = None, top_n: int = 20):
        super().__init__(name="SQLAgent", role="SQL over olist.db")

        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.db_path = db_path or (PROJECT_ROOT / "app" / "db" / "olist.db")
        self.top_n = top_n

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database tidak ditemukan di {self.db_path}. "
                "Jalankan build_sqlite.py dulu."
            )

        # Muat daftar kategori sah (auto-suggest & prompt LLM)
        self._categories = self._load_categories()

        # Schema yang dikasih ke LLM
        cats_preview = ", ".join(self._categories[:50]) or "(kosong)"
        self.schema_description = f"""
Tabel utama: fact_order_items

Kolom fact_order_items:
- order_id (TEXT)
- order_item_id (INTEGER)
- product_id (TEXT)
- seller_id (TEXT)
- seller_city (TEXT)
- product_category_name (TEXT, nama kategori asli)
- product_category_name_english (TEXT, contoh: 'electronics', 'computers_accessories')
- review_comment_message (TEXT)
- review_score (INTEGER)

Daftar contoh nilai product_category_name_english:
{cats_preview}

Aturan:
- Hanya boleh menggunakan SELECT, WHERE, GROUP BY, ORDER BY, LIMIT.
- Dilarang UPDATE/DELETE/INSERT/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Gunakan nama tabel: fact_order_items.
- Jika memfilter kategori produk, SELALU gunakan kolom product_category_name_english.
- Nama kategori di kolom product_category_name_english menggunakan Bahasa Inggris
  seperti 'electronics', 'computers_accessories', 'books_general_interest', dll.
- Jika user menyebut kategori dalam Bahasa Indonesia, terjemahkan ke bentuk
  Bahasa Inggris yang paling sesuai sebelum menulis SQL.
- Pertanyaan user bisa dalam Bahasa Indonesia atau Inggris.
- Tambahkan LIMIT {top_n} di akhir query.
"""

    # ------------------------------------------------------------------
    # 0. Public property untuk UI (auto-suggest kategori)
    # ------------------------------------------------------------------
    @property
    def categories(self) -> List[str]:
        return self._categories

    def _load_categories(self) -> List[str]:
        """Ambil daftar distinct product_category_name_english dari DB."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT product_category_name_english
                FROM fact_order_items
                WHERE product_category_name_english IS NOT NULL
                ORDER BY product_category_name_english
                """
            )
            rows = cur.fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 1. Normalisasi pertanyaan (mapping kategori Indonesia -> English)
    # ------------------------------------------------------------------
    def _normalize_question_text(self, question: str) -> str:
        """
        Jika pertanyaan mengandung nama kategori dalam Bahasa Indonesia,
        tambahkan catatan eksplisit untuk membantu LLM pakai kategori English.
        """
        q_lower = question.lower()
        notes: List[str] = []

        for indo, eng in self.CATEGORY_ALIASES.items():
            if indo in q_lower:
                notes.append(
                    f"Kata '{indo}' di sini merujuk ke kategori English "
                    f"'{eng}' pada kolom product_category_name_english."
                )

        if not notes:
            return question

        note_text = "\n".join(notes)
        return question + "\n\nCatatan kategori:\n" + note_text

    # ------------------------------------------------------------------
    # 2. Generate SQL dari LLM (pertanyaan bisa ID/EN)
    # ------------------------------------------------------------------
    def _generate_sql(self, question: str) -> str:
        system_prompt = (
            "You are an SQL assistant for SQLite. "
            "Your task is to write ONE valid SELECT query for SQLite. "
            "NEVER output UPDATE/DELETE/INSERT/ALTER/DROP/CREATE/REPLACE/TRUNCATE. "
            "Output only the SQL, without any explanation."
        )

        cats_text = ", ".join(self._categories[:50]) or "(no categories found)"

        user_msg = (
            f"Schema:\n{self.schema_description}\n\n"
            f"Valid product_category_name_english values (examples):\n{cats_text}\n\n"
            "The user's question can be in Indonesian or English.\n\n"
            f"User question:\n{question}\n\n"
            "Write ONE SELECT query that answers the question. "
            "If the user mentions a product category in Indonesian, map it to the "
            "closest English value in product_category_name_english "
            "(e.g., 'elektronik' -> 'electronics'). "
            "The query must be syntactically valid for SQLite."
        )

        sql = chat_completion(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=400,
        ).strip()

        # bersihkan formatting ```sql ... ```
        sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
        sql = sql.replace("```", "").strip()

        return sql

    # ------------------------------------------------------------------
    # 3. Generate SQL fallback kalau hasil kosong
    # ------------------------------------------------------------------
    def _generate_fallback_sql(self, question: str, first_sql: str) -> str:
        """
        Minta LLM membuat query alternatif yang lebih longgar
        ketika query pertama mengembalikan hasil kosong.
        """
        system_prompt = (
            "You are an SQL assistant for SQLite. "
            "The first SELECT query returned 0 rows. "
            "Write ONE alternative SELECT query that is still relevant, "
            "but with slightly looser conditions (for example, remove overly "
            "specific category filters or use LIKE). "
            "The query must still be read-only. Do NOT use any write operations."
        )

        cats_text = ", ".join(self._categories[:50]) or "(no categories found)"

        user_msg = (
            f"Schema:\n{self.schema_description}\n\n"
            f"Valid product_category_name_english values (examples):\n{cats_text}\n\n"
            "The user's question may be in Indonesian or English.\n\n"
            f"User question:\n{question}\n\n"
            f"First query (returned 0 rows):\n{first_sql}\n\n"
            "Now write ONE alternative SELECT query that is safer and more relaxed, "
            "but still trying to answer the same question."
        )

        sql = chat_completion(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=400,
        ).strip()

        sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
        sql = sql.replace("```", "").strip()

        return sql

    # ------------------------------------------------------------------
    # 4. Normalisasi SQL (buang ; di akhir, trim spasi)
    # ------------------------------------------------------------------
    def _normalize_sql(self, sql: str) -> str:
        sql = sql.strip()
        if sql.endswith(";"):
            sql = sql[:-1].strip()
        return sql

    # ------------------------------------------------------------------
    # 5. Validasi keamanan SQL
    # ------------------------------------------------------------------
    def _is_safe_sql(self, sql: str) -> bool:
        sql_lower = sql.lower()

        # wajib diawali SELECT
        if not sql_lower.startswith("select"):
            return False

        # blokir kata-kata berbahaya
        for bad in self.UNSAFE_WORDS:
            if bad in sql_lower:
                return False

        # blokir multi-statement
        if ";" in sql_lower:
            return False

        return True

    # ------------------------------------------------------------------
    # 6. Eksekusi ke SQLite
    # ------------------------------------------------------------------
    def _run_sql(self, sql: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)

        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        finally:
            conn.close()

        rows = rows[: self.top_n]

        return {"columns": cols, "rows": rows}

    # ------------------------------------------------------------------
    # 7. Ringkasan hasil dengan LLM (bahasa bisa dipilih)
    # ------------------------------------------------------------------
    def _summarize(
        self,
        question: str,
        sql: str,
        result: Dict[str, Any],
        answer_lang: str = "id",
    ) -> str:
        """
        answer_lang: 'id' (Bahasa Indonesia) atau 'en' (English)
        """
        if answer_lang == "en":
            system_prompt = (
                "You are a senior data analyst. Summarize the result of a SQLite "
                "query in clear and concise English."
            )
            lang_instruction = "Write the answer in English."
        else:
            system_prompt = (
                "Kamu adalah analis data senior. Ringkas hasil query SQLite "
                "dalam bahasa Indonesia yang jelas dan singkat."
            )
            lang_instruction = "Tulis jawaban dalam bahasa Indonesia."

        preview = f"Columns: {result['columns']}\nRows: {result['rows'][:10]}"

        user_msg = (
            f"User question:\n{question}\n\n"
            f"SQL executed:\n{sql}\n\n"
            f"Result (preview):\n{preview}\n\n"
            f"{lang_instruction} "
            "If there are no rows, clearly explain that there is no matching data."
        )

        return chat_completion(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=400,
        )

    # ------------------------------------------------------------------
    # 8. PUBLIC API (dengan fallback & pilihan bahasa jawaban)
    # ------------------------------------------------------------------
    def query(self, question: str, answer_lang: str = "id") -> Dict[str, Any]:
        """
        End-to-end:
        1) normalisasi pertanyaan (mapping kategori)
        2) generate SQL pertama (LLM)
        3) normalisasi & validasi keamanan
        4) eksekusi ke DB
        5) jika hasil kosong -> generate & jalankan fallback SQL
        6) ringkas hasil dengan LLM (bahasa ID/EN)
        """
        normalized_q = self._normalize_question_text(question)

        # --- SQL pertama ---
        sql_raw_1 = self._generate_sql(normalized_q)
        sql_1 = self._normalize_sql(sql_raw_1)

        if not self._is_safe_sql(sql_1):
            raise ValueError(f"Generated SQL dianggap tidak aman:\n{sql_raw_1}")

        result_1 = self._run_sql(sql_1)

        used_fallback = False
        sql_used = sql_1
        result_used = result_1
        sql_fallback = ""

        # --- fallback otomatis jika hasil kosong ---
        if len(result_1["rows"]) == 0:
            sql_raw_2 = self._generate_fallback_sql(normalized_q, sql_1)
            sql_2 = self._normalize_sql(sql_raw_2)

            if self._is_safe_sql(sql_2):
                result_2 = self._run_sql(sql_2)
                sql_used = sql_2
                result_used = result_2
                sql_fallback = sql_2
                used_fallback = True

        summary = self._summarize(question, sql_used, result_used, answer_lang=answer_lang)

        return {
            "question": question,
            "answer_lang": answer_lang,
            "sql_initial": sql_1,
            "sql_used": sql_used,
            "sql_fallback": sql_fallback,
            "used_fallback": used_fallback,
            "result": result_used,
            "summary": summary,
            "categories": self._categories,
        }

    # ------------------------------------------------------------------
    # 9. Implementasi BaseAgent.run (untuk multi-agent orchestrator)
    # ------------------------------------------------------------------
    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # default: jawaban dalam bahasa Indonesia
        output = self.query(task, answer_lang="id")

        new_ctx = context.copy()
        new_ctx["sql_query"] = output["sql_used"]
        new_ctx["sql_result"] = output["result"]
        new_ctx["sql_summary"] = output["summary"]

        return {
            "agent": self.name,
            "role": self.role,
            "summary": "SQL query executed (with fallback if needed) and summarized.",
            "sql_initial": output["sql_initial"],
            "sql_used": output["sql_used"],
            "used_fallback": output["used_fallback"],
            "table": output["result"],
            "llm_summary": output["summary"],
            "context": new_ctx,
        }
