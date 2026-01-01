import os
import streamlit as st

from services.api_client import APIConfig, APIError, chat, health_check


# legacy placeholder (biar signature tidak bikin masalah kalau ada referensi)
@st.cache_resource
def get_chat_agent():
    return None


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "answer_lang" not in st.session_state:
        st.session_state.answer_lang = "id"
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {}

    if "api_base" not in st.session_state:
        api_base = None
        try:
            api_base = st.secrets.get("API_BASE")  # type: ignore[attr-defined]
        except Exception:
            api_base = None
        if not api_base:
            api_base = os.getenv("API_BASE", "")
        st.session_state.api_base = api_base


def _history_for_backend() -> list[dict]:
    history = []
    for m in st.session_state.messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            history.append({"role": m["role"], "content": m["content"]})
    return history


def main():
    st.set_page_config(page_title="Olist Chat (Cloud Run)", layout="wide")
    init_state()

    st.title("💬 Olist Chat (Cloud Run)")
    st.caption("Streamlit frontend → Cloud Run backend (`/chat`).")

    # sidebar
    st.sidebar.header("Settings")
    st.session_state.api_base = st.sidebar.text_input(
        "Cloud Run Base URL",
        value=st.session_state.api_base,
        placeholder="https://<service>.run.app",
    )
    st.session_state.answer_lang = st.sidebar.selectbox(
        "Answer language",
        ["id", "en"],
        index=0 if st.session_state.answer_lang == "id" else 1,
    )
    st.session_state.show_debug = st.sidebar.checkbox("Show debug", value=st.session_state.show_debug)

    cfg = APIConfig(base_url=st.session_state.api_base, timeout=90)

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
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.chat_state = {}
            st.rerun()

    # history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # input
    user_msg = st.chat_input("Ketik pertanyaanmu...")
    if user_msg:
        if not cfg.base_url:
            st.error("API_BASE kosong. Isi dulu URL Cloud Run di sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    out = chat(
                        cfg=cfg,
                        message=user_msg,
                        history=_history_for_backend(),
                        answer_lang=st.session_state.answer_lang,
                        show_debug=st.session_state.show_debug,
                        state=st.session_state.chat_state,
                    )

                    answer = out.get("final_answer", "(no final_answer)")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    if isinstance(out.get("state"), dict):
                        st.session_state.chat_state = out["state"]

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
