from __future__ import annotations

import math
import traceback
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from app.agents.chat_agent import ChatAgent

app = FastAPI(title="Olist Chatbot API", version="1.0.0")

# Lazy init: avoid crashing on import/startup due to missing env/files
_agent: Optional[ChatAgent] = None


def get_agent() -> ChatAgent:
    global _agent
    if _agent is None:
        _agent = ChatAgent()
    return _agent


def _is_bad_float(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)


def sanitize(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects (DataFrame, numpy scalars, NaN/Inf)
    into JSON-friendly shapes.
    """
    if obj is None:
        return None

    if isinstance(obj, float):
        return None if _is_bad_float(obj) else obj

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if _is_bad_float(x) else x

    if isinstance(obj, pd.DataFrame):
        preview = obj.head(50).copy()
        preview = preview.replace([np.nan, np.inf, -np.inf], None)
        return {
            "_type": "dataframe",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns": preview.columns.tolist(),
            "rows": preview.to_dict(orient="records"),
        }

    if isinstance(obj, pd.Series):
        s = obj.head(50).replace([np.nan, np.inf, -np.inf], None)
        return {"_type": "series", "name": obj.name, "values": s.tolist()}

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(x) for x in obj]

    return obj


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    answer_lang: Literal["id", "en"] = "id"
    show_debug: bool = False
    state: Dict[str, Any] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Chat endpoint (SAFE).
    We sanitize output to avoid DataFrame / NaN / numpy types issues.
    """
    try:
        history = [{"role": m.role, "content": m.content} for m in (req.history or [])]

        agent = get_agent()

        out = agent.chat(
            user_message=req.message,
            history=history,
            answer_lang=req.answer_lang,
            state=req.state,
            show_debug=req.show_debug,
        )

        resp: Dict[str, Any] = {
            "final_answer": str(out.get("final_answer", "")),
            "used_tools": out.get("used_tools") or [],
            "state": out.get("state") or {},
        }

        if req.show_debug:
            resp["tool_outputs"] = out.get("tool_outputs")
            resp["debug"] = out.get("debug") or {}
        
        return sanitize(resp)

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e}\n\n{tb}")


if __name__ == "__main__":
    # Runs the app instance directly
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
