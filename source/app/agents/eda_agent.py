from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd

from .base_agent import BaseAgent


class EDAAgent(BaseAgent):
    """
    Compute compact EDA:
    - dataset shape
    - missingness summary (top columns)
    - numeric describe (compact)
    - category counts (top)
    - time range if datetime columns exist
    - NO giant DataFrames returned (only previews)
    """

    def __init__(self):
        super().__init__(name="EDAAgent", role="Compute compact EDA stats")

    def _top_missing(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        miss = df.isna().mean().sort_values(ascending=False).head(top_k)
        return [{"column": c, "missing_rate": float(v)} for c, v in miss.items() if v > 0]

    def _numeric_describe_compact(self, df: pd.DataFrame, cols: List[str], top_k: int = 12) -> Dict[str, Any]:
        cols = cols[:top_k]
        if not cols:
            return {"columns": [], "rows": []}
        desc = df[cols].describe().T.reset_index().rename(columns={"index": "column"})
        # keep only a few stats
        keep = ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        desc = desc[keep]
        rows = desc.to_dict(orient="records")
        # convert numpy types to python
        cleaned = []
        for r in rows:
            cleaned.append({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()})
        return {"columns": keep, "rows": cleaned}

    def _top_categories(self, df: pd.DataFrame, col: str, top_k: int = 10) -> Dict[str, Any]:
        vc = df[col].astype("object").fillna("NULL").value_counts().head(top_k)
        return {"column": col, "top": [{"value": str(k), "count": int(v)} for k, v in vc.items()]}

    def _detect_time_ranges(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                s = df[c].dropna()
                if len(s) == 0:
                    continue
                out.append(
                    {
                        "column": c,
                        "min": s.min().isoformat(),
                        "max": s.max().isoformat(),
                    }
                )
        return out

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if "orders_merged" not in context:
            raise ValueError("orders_merged not found in context. Run DataAgent first.")

        df: pd.DataFrame = context["orders_merged"]

        # Basic info
        shape = [int(df.shape[0]), int(df.shape[1])]
        cols = [str(c) for c in df.columns]

        # Datetime coercion (if any object columns look like timestamps, skip heavy parsing)
        # We rely on DataAgent for proper typing; this just safely detects.
        time_ranges = self._detect_time_ranges(df)

        # Missingness
        missing_top = self._top_missing(df, top_k=10)

        # Numeric columns
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric_desc = self._numeric_describe_compact(df, num_cols, top_k=12)

        # Category columns (choose a few common ones if exist)
        cat_candidates = []
        for candidate in ["product_category_name", "product_category_en", "seller_city", "customer_state", "order_status"]:
            if candidate in df.columns:
                cat_candidates.append(candidate)
        cat_candidates = cat_candidates[:4]
        category_summaries = [self._top_categories(df, c, top_k=10) for c in cat_candidates]

        eda = {
            "shape": shape,
            "columns": cols[:200],  # guard: don't dump huge col list
            "missing_top": missing_top,
            "numeric_describe": numeric_desc,
            "category_top": category_summaries,
            "time_ranges": time_ranges,
        }

        new_ctx = dict(context)
        new_ctx["eda"] = eda

        return {
            "agent": self.name,
            "role": self.role,
            "summary": "Computed compact EDA.",
            "eda": eda,
            "context": new_ctx,
        }
