from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class APIConfig:
    base_url: str
    timeout: int = 90


class APIError(RuntimeError):
    pass


def _join(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    p = path if path.startswith("/") else f"/{path}"
    return f"{base}{p}"


def health_check(cfg: APIConfig) -> Tuple[int, Dict[str, Any] | str]:
    """
    GET /health
    Return (status_code, json_or_text)
    """
    url = _join(cfg.base_url, "/health")
    r = requests.get(url, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def chat(
    cfg: APIConfig,
    message: str,
    history: list[dict],
    answer_lang: str = "id",
    show_debug: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    POST /chat
    Payload mengikuti API kamu:
      {
        "message": "...",
        "history": [...],
        "answer_lang": "id",
        "show_debug": false,
        "state": {}
      }
    """
    url = _join(cfg.base_url, "/chat")
    payload = {
        "message": message,
        "history": history or [],
        "answer_lang": answer_lang,
        "show_debug": bool(show_debug),
        "state": state or {},
    }

    r = requests.post(url, json=payload, timeout=cfg.timeout)

    # jika error, lempar detail supaya Streamlit bisa tampilkan
    if r.status_code != 200:
        # coba parse json error
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise APIError(f"HTTP {r.status_code}: {detail}")

    return r.json()
