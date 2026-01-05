import os
import sys

import pandas as pd
import streamlit as st

# --- setup import ke package app ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# (signature lama dipertahankan, tapi tidak dipakai lagi)
from app.agents.sql_agent import SQLAgent  # noqa: E402

from app.services.api_client import APIConfig, APIError, chat, health_check  # noqa: E402


# ---------- AGENT SINGLETON ----------
@st.cache_resource
def get_sql_agent() -> SQLAgent:
    # legacy: sengaja tidak dipakai (biar tidak menjalankan agent lokal)
    return SQLAgent()


def init_state():
    if "sql_state" not in st.session_state:
        st.session_state.sql_state = {}
    if "api_base" not in st.session_state:
        api_base = None
        try:
            api_base = st.secrets.get("API_BASE")  # type: ignore[attr-defined]
        except Exception:
            api_base = None
        if not api_base:
            api_base = os.getenv("API_BASE", "")
        st.session_state.api_base = api_base
    if "answer_lang" not in st.session_state:
        st.session_state.answer_lang = "id"
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = True  # untuk SQL biasanya berguna


def _history_for_backend_sql() -> list[dict]:
    # untuk SQL assistant, kita tetap bisa pakai history minimal
    # (atau kosong saja). Di sini kosong agar setiap query berdiri sendiri.
    return []


def _extract_sql_preview(out: dict):
    """
    Coba ambil preview SQL dari response ChatAgent kamu.
    Di output kamu sebelumnya, state menyimpan:
      last_sql, last_sql_columns, last_sql_preview_rows
    """
    state = out.get("state") or {}
    if not isinstance(state, dict):
        return None

    last_sql = state.get("last_sql")
    cols = state.get("last_sql_columns")
    rows = state.get("last_sql_preview_rows")

    if not last_sql or not cols or rows is None:
        return None

    if not isinstance(cols, list) or not isinstance(rows, list):
        return None

    return {"sql": last_sql, "columns": cols, "rows": rows}


def main():
    st.set_page_config(page_title="Olist SQL Assistant (Cloud Run)", layout="wide")
    init_state()

    st.title("🧮 Olist SQL Assistant (Cloud Run)")
    st.caption("Streamlit frontend → Cloud Run backend (`/chat`) untuk pertanyaan numerik/SQL.")

    # ---------- SIDEBAR ----------
    st.sidebar.header("Settings")
    st.session_state.api_base = st.sidebar.text_input(
        "Cloud Run Base URL",
        value=st.session_state.api_base,
        placeholder="https://<service>.run.app",
    )
    st.session_state.answer_lang = st.sidebar.selectbox(
        "Answer language", ["id", "en"], index=0 if st.session_state.answer_lang == "id" else 1
    )
    st.session_state.show_debug = st.sidebar.checkbox("Show debug", value=st.session_state.show_debug)

    cfg = APIConfig(base_url=st.session_state.api_base, timeout=120)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Check /health"):
            if not cfg.base_url:
                st.sidebar.error("API_BASE belum diisi.")
            else:
                code, body = health_check(cfg)
                st.sidebar.write("Status:", code)
                st.sidebar.write(body)
    with c2:
        if st.button("Clear state"):
            st.session_state.sql_state = {}
            st.rerun()

    # ---------- MAIN UI ----------
    st.subheader("Ask a question")
    q = st.text_area(
        "Contoh: 'berapa rata-rata review score untuk kategori health_beauty?'",
        height=90,
        placeholder="Tulis pertanyaan numerik/statistik...",
    )

    run = st.button("Run", type="primary")
    if run:
        if not cfg.base_url:
            st.error("API_BASE kosong. Isi dulu URL Cloud Run di sidebar.")
            st.stop()
        if not q.strip():
            st.warning("Pertanyaannya masih kosong.")
            st.stop()

        with st.spinner("Querying backend..."):
            try:
                out = chat(
                    cfg=cfg,
                    message=q.strip(),
                    history=_history_for_backend_sql(),
                    answer_lang=st.session_state.answer_lang,
                    show_debug=st.session_state.show_debug,
                    state=st.session_state.sql_state,
                )

                # simpan state
                if isinstance(out.get("state"), dict):
                    st.session_state.sql_state = out["state"]

                # tampilkan jawaban
                st.subheader("Answer")
                st.write(out.get("final_answer", "(no final_answer)"))

                # tampilkan SQL preview (kalau backend mengisi)
                preview = _extract_sql_preview(out)
                if preview:
                    st.subheader("Generated SQL")
                    st.code(preview["sql"], language="sql")

                    st.subheader("Query preview")
                    df = pd.DataFrame(preview["rows"], columns=preview["columns"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Backend tidak mengembalikan SQL preview untuk pertanyaan ini (mungkin bukan intent SQL).")

                # debug opsional
                if st.session_state.show_debug:
                    st.divider()
                    st.subheader("Debug")
                    st.json(
                        {
                            "used_tools": out.get("used_tools", []),
                            "debug": out.get("debug", {}),
                            "state": out.get("state", {}),
                            "tool_outputs": out.get("tool_outputs", {}),
                        }
                    )

            except APIError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)


if __name__ == "__main__":
    main()
