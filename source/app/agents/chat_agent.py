from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime, date

from .base_agent import BaseAgent
from .sql_agent import SQLAgent
from .rag_agent import RAGAgent
from .orchestrator import OrchestratorAgent
from app.core.llm import chat_completion

Intent = Literal["sql", "rag", "analytics", "hybrid", "general"]
Lang = Literal["id", "en"]


@dataclass
class ChatState:
    last_intent: Optional[Intent] = None
    last_sql: Optional[str] = None
    last_sql_columns: Optional[List[str]] = None
    last_sql_preview_rows: Optional[List[Any]] = None
    last_rag_top_sources: Optional[List[Dict[str, Any]]] = None


class ChatAgent(BaseAgent):
    def __init__(
        self,
        sql_agent: Optional[SQLAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        orchestrator: Optional[OrchestratorAgent] = None,
    ):
        super().__init__(name="ChatAgent", role="Main conversational router")
        self.sql_agent = sql_agent or SQLAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.orchestrator = orchestrator or OrchestratorAgent()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _safe_json_load(self, text: str) -> Optional[Dict[str, Any]]:
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.I)
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.S)
            return json.loads(m.group()) if m else None

    def _truncate(self, rows: List[Any], n: int = 5) -> List[Any]:
        return rows[:n]

    def _to_iso(self, x: Any) -> str:
        # datetime/date
        if isinstance(x, (datetime, date)):
            return x.isoformat()

        # pandas Timestamp / Timedelta
        try:
            import pandas as pd  # type: ignore
            if isinstance(x, pd.Timestamp):
                # safe even if tz-aware
                return x.isoformat()
            if isinstance(x, pd.Timedelta):
                return str(x)
        except Exception:
            pass

        # numpy datetime64 / timedelta64
        try:
            import numpy as np  # type: ignore
            if isinstance(x, np.datetime64):
                return str(x)
            if isinstance(x, np.timedelta64):
                return str(x)
        except Exception:
            pass

        return str(x)

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Convert non-JSON-serializable objects (DataFrame/Series/numpy/NaN/Timestamp)
        into JSON-friendly preview shapes.
        """
        # optional deps
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None  # type: ignore

        if obj is None:
            return None

        # datetime-like early
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        if pd is not None:
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, pd.Timedelta):
                return str(obj)

        if np is not None:
            if isinstance(obj, np.datetime64):
                return str(obj)
            if isinstance(obj, np.timedelta64):
                return str(obj)

        # numpy scalars
        if np is not None:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.floating,)):
                x = float(obj)
                if x != x or x in (float("inf"), float("-inf")):
                    return None
                return x

        # floats NaN/inf
        if isinstance(obj, float):
            if obj != obj or obj in (float("inf"), float("-inf")):
                return None
            return obj

        # pandas DataFrame/Series
        if pd is not None:
            if isinstance(obj, pd.DataFrame):
                preview = obj.head(30).copy()

                # replace NaN/inf -> None
                try:
                    preview = preview.replace([float("inf"), float("-inf")], None)
                    preview = preview.where(pd.notnull(preview), None)
                except Exception:
                    pass

                # convert timestamps inside DataFrame
                try:
                    for c in preview.columns:
                        if pd.api.types.is_datetime64_any_dtype(preview[c]):
                            preview[c] = preview[c].astype("datetime64[ns]").dt.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    pass

                return {
                    "_type": "dataframe",
                    "shape": [int(obj.shape[0]), int(obj.shape[1])],
                    "columns": [str(c) for c in preview.columns.tolist()],
                    "rows": preview.to_dict(orient="records"),
                }

            if isinstance(obj, pd.Series):
                s = obj.head(50)
                try:
                    s = s.replace([float("inf"), float("-inf")], None)
                    s = s.where(pd.notnull(s), None)
                except Exception:
                    pass

                # convert datetime series
                try:
                    if pd.api.types.is_datetime64_any_dtype(s):
                        s = s.dt.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    pass

                return {
                    "_type": "series",
                    "name": str(getattr(obj, "name", "")),
                    "values": s.tolist(),
                }

        # dict/list recursion
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(x) for x in obj]

        # last resort: try json, else string
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return self._to_iso(obj)

    def _safe_payload_text(self, payload: Dict[str, Any]) -> str:
        safe = self._sanitize_for_json(payload)
        # default=str is an extra safety net for weird objects
        return json.dumps(safe, ensure_ascii=False, default=str)

    # -----------------------------
    # Intent router
    # -----------------------------
    def _route_intent(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        state: Optional[ChatState],
        answer_lang: Lang,
    ) -> Dict[str, Any]:
        hist = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-6:]) or "(empty)"
        state_txt = json.dumps(asdict(state), ensure_ascii=False, default=str) if state else "{}"

        system = (
            "You are an intent router for a multi-agent analytics chatbot.\n"
            "Return ONLY valid JSON.\n\n"
            "Intents:\n"
            "- sql: aggregation/metrics\n"
            "- rag: descriptive/recommendation\n"
            "- analytics: dataset analysis/EDA/insights\n"
            "- hybrid: sql + explanation/context (SQL+RAG)\n"
            "- general: casual chat\n"
        )
        user = (
            f"Language: {answer_lang}\n\n"
            f"History:\n{hist}\n\n"
            f"State:\n{state_txt}\n\n"
            f"User message:\n{user_message}\n\n"
            "Return JSON with keys: intent, reason, need_followup, followup_question."
        )

        raw = chat_completion(system_prompt=system, messages=[{"role": "user", "content": user}], max_tokens=220)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"intent": "hybrid", "reason": "Fallback", "need_followup": False, "followup_question": ""}

        intent = str(parsed.get("intent", "hybrid")).strip().lower()
        if intent not in {"sql", "rag", "analytics", "hybrid", "general"}:
            intent = "hybrid"

        return {
            "intent": intent,
            "reason": str(parsed.get("reason", "")).strip(),
            "need_followup": bool(parsed.get("need_followup", False)),
            "followup_question": str(parsed.get("followup_question", "")).strip(),
        }

    # -----------------------------
    # Final answer composer
    # -----------------------------
    def _compose_answer(self, question: str, payload: Dict[str, Any], lang: Lang) -> str:
        if lang == "en":
            system = (
                "You are a senior data assistant.\n"
                "Write clear English.\n"
                "Do not dump raw JSON; only cite key results.\n"
                "Add 1-2 analytical implications.\n"
                "If analytics output exists, summarize 5 insights clearly."
            )
        else:
            system = (
                "Kamu adalah asisten data senior.\n"
                "Jawab dalam Bahasa Indonesia yang natural.\n"
                "Jangan dump JSON mentah; ambil poin penting saja.\n"
                "Tambahkan 1-2 implikasi analitis.\n"
                "Jika ada output analytics, rangkum 5 insight dengan jelas."
            )

        safe_payload = self._safe_payload_text(payload)
        user = (
            f"Question:\n{question}\n\n"
            f"Tool outputs (sanitized JSON preview):\n{safe_payload}\n\n"
            "Write the final answer."
        )
        return chat_completion(system_prompt=system, messages=[{"role": "user", "content": user}], max_tokens=900)

    # -----------------------------
    # Main chat
    # -----------------------------
    def chat(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        answer_lang: Lang = "id",
        state: Optional[Dict[str, Any]] = None,
        show_debug: bool = False,
    ) -> Dict[str, Any]:

        cur_state = ChatState(**state) if isinstance(state, dict) else ChatState()

        route = self._route_intent(user_message, history, cur_state, answer_lang)
        intent: Intent = route.get("intent", "hybrid")

        used_tools: List[str] = []
        tool_outputs: Dict[str, Any] = {}

        if intent == "sql":
            used_tools.append("SQLAgent")
            tool_outputs["sql"] = self.sql_agent.query(user_message, answer_lang)

        elif intent == "rag":
            used_tools.append("RAGAgent")
            tool_outputs["rag"] = self.rag_agent.answer(user_message)

        elif intent == "analytics":
            used_tools.append("OrchestratorAgent")
            tool_outputs["analytics"] = self.orchestrator.run(user_message)

        elif intent == "hybrid":
            used_tools.extend(["SQLAgent", "RAGAgent"])
            tool_outputs["sql"] = self.sql_agent.query(user_message, answer_lang)
            tool_outputs["rag"] = self.rag_agent.answer(user_message)

        else:
            tool_outputs["general"] = {"note": "No tool used."}

        final_answer = self._compose_answer(user_message, tool_outputs, answer_lang)

        # update state
        cur_state.last_intent = intent
        if isinstance(tool_outputs.get("sql"), dict):
            res = tool_outputs["sql"].get("result", {}) or {}
            cur_state.last_sql_columns = res.get("columns")
            cur_state.last_sql_preview_rows = self._truncate(res.get("rows", []))

        if isinstance(tool_outputs.get("rag"), dict):
            cur_state.last_rag_top_sources = (tool_outputs["rag"].get("sources") or [])[:3]

        out = {
            "final_answer": final_answer,
            "used_tools": used_tools,
            "tool_outputs": tool_outputs,
            "state": asdict(cur_state),
        }

        if show_debug:
            out["debug"] = {
                "intent": intent,
                "reason": route.get("reason", ""),
                "used_tools": used_tools,
            }

        return out

    # -----------------------------
    # BaseAgent abstract method
    # -----------------------------
    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        out = self.chat(
            user_message=task,
            history=context.get("history", []),
            answer_lang=context.get("answer_lang", "id"),
            state=context.get("state", {}),
            show_debug=context.get("show_debug", False),
        )
        new_ctx = dict(context)
        new_ctx["state"] = out.get("state", {})
        new_ctx["last_answer"] = out.get("final_answer", "")

        return {
            "agent": self.name,
            "role": self.role,
            "summary": "ChatAgent responded using routed tools.",
            "answer": out.get("final_answer", ""),
            "used_tools": out.get("used_tools", []),
            "tool_outputs": out.get("tool_outputs", {}),
            "context": new_ctx,
        }
