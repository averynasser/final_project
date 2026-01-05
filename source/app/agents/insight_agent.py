from __future__ import annotations

from typing import Any, Dict, List

from .base_agent import BaseAgent
from app.core.llm import chat_completion


class InsightAgent(BaseAgent):
    """
    Turn EDA into 5 executive insights + evidence + next questions.

    Output is compact and UI-ready:
    analytics = {
      "headline": "...",
      "insights": [{"title":..,"finding":..,"evidence":..,"impact":..}, ...],
      "next_questions": [...]
    }
    """

    def __init__(self):
        super().__init__(name="InsightAgent", role="Generate executive insights")

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        eda = context.get("eda")
        if not eda:
            raise ValueError("EDA not found in context. Run EDAAgent first.")

        system = (
            "You are a senior data analyst.\n"
            "Create exactly 5 insights from the provided EDA summary.\n"
            "Each insight MUST include: title, finding (quantified if possible), evidence (what metric/table supports it), and impact (why it matters).\n"
            "Do not hallucinate numbers not present in the EDA.\n"
            "If the EDA lacks a metric, phrase it qualitatively and suggest next query.\n"
            "Return ONLY valid JSON."
        )

        user = (
            f"User request:\n{task}\n\n"
            f"EDA summary (JSON):\n{eda}\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "headline": str,\n'
            '  "insights": [\n'
            '    {"title": str, "finding": str, "evidence": str, "impact": str}\n'
            "  ],\n"
            '  "next_questions": [str, ...]\n'
            "}\n"
        )

        raw = chat_completion(system_prompt=system, messages=[{"role": "user", "content": user}], max_tokens=900)

        # best-effort parse JSON
        import json, re
        txt = raw.strip()
        txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.I).strip()
        try:
            analytics = json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.S)
            analytics = json.loads(m.group()) if m else {
                "headline": "Analytics summary",
                "insights": [],
                "next_questions": [],
            }

        # hard guard: ensure exactly 5 insights (pad/truncate)
        insights = analytics.get("insights") or []
        if len(insights) > 5:
            insights = insights[:5]
        while len(insights) < 5:
            insights.append({
                "title": "Insight tambahan (butuh data)",
                "finding": "EDA saat ini belum cukup untuk menyimpulkan poin ini secara kuantitatif.",
                "evidence": "Perlu query/EDA tambahan.",
                "impact": "Menentukan prioritas analisis lanjutan.",
            })
        analytics["insights"] = insights

        # store compact analytics only (no DataFrames)
        new_ctx = dict(context)
        new_ctx["analytics"] = analytics

        return {
            "agent": self.name,
            "role": self.role,
            "summary": "Generated 5 insights from EDA.",
            "analytics": analytics,
            "context": new_ctx,
        }
