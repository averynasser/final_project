from __future__ import annotations

from typing import Any, Dict

from .base_agent import BaseAgent
from .data_agent import DataAgent
from .eda_agent import EDAAgent
from .insight_agent import InsightAgent


class OrchestratorAgent(BaseAgent):
    """
    Analytics pipeline:
    1) DataAgent -> load/prepare merged dataset to context
    2) EDAAgent -> compute compact stats + small previews (no huge DataFrames)
    3) InsightAgent -> produce 5 insights + evidence + next questions
    """

    def __init__(self):
        super().__init__(name="OrchestratorAgent", role="Analytics pipeline orchestrator")
        self.data_agent = DataAgent()
        self.eda_agent = EDAAgent()
        self.insight_agent = InsightAgent()

    def run(self, task: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = dict(context or {})
        ctx["user_task"] = task

        # Step 1: load/merge
        step1 = self.data_agent.run(task, ctx)
        ctx = step1.get("context", ctx)

        # Step 2: EDA (compact)
        step2 = self.eda_agent.run(task, ctx)
        ctx = step2.get("context", ctx)

        # Step 3: Insights (final compact artifact)
        step3 = self.insight_agent.run(task, ctx)
        ctx = step3.get("context", ctx)

        # Final output: small & UI-ready
        return {
            "agent": self.name,
            "role": self.role,
            "summary": step3.get("summary", "Analytics completed."),
            "analytics": step3.get("analytics", {}),
            "context": ctx,
        }
