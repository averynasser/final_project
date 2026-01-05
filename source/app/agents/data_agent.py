from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from .base_agent import BaseAgent
from app.core.data_loader import OlistDataLoader


class DataAgent(BaseAgent):
    """
    Agent untuk load dan menyiapkan subset data yang relevan.

    FIX UTAMA:
    - Jangan mengembalikan pandas DataFrame ke output/context public (akan bikin FastAPI/Pydantic 500).
    - Simpan DataFrame hanya di context "private" memakai key yang diawali underscore.
    """

    def __init__(self, loader: OlistDataLoader | None = None):
        super().__init__(name="DataAgent", role="Data loading and preprocessing")
        self.loader = loader or OlistDataLoader()

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Load dataset utama
        orders = self.loader.orders.copy()
        order_items = self.loader.order_items.copy()
        customers = self.loader.customers.copy()
        reviews = self.loader.order_reviews.copy()
        payments = self.loader.order_payments.copy()

        # Pastikan kolom tanggal berbentuk datetime (kalau masih string)
        orders = self._ensure_datetime(
            orders,
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        )

        # Agregasi order_items ke level order_id
        oi_agg = (
            order_items.groupby("order_id")
            .agg(
                total_items=("order_item_id", "count"),
                total_price=("price", "sum"),
                total_freight=("freight_value", "sum"),
            )
            .reset_index()
        )

        # Agregasi payments ke level order_id
        pay_agg = (
            payments.groupby("order_id")
            .agg(
                payment_value=("payment_value", "sum"),
                n_payments=("payment_sequential", "max"),
            )
            .reset_index()
        )

        # Ambil review score (1 row per order_id)
        reviews_simple = reviews[["order_id", "review_score"]].drop_duplicates()

        df = (
            orders.merge(oi_agg, on="order_id", how="left")
            .merge(pay_agg, on="order_id", how="left")
            .merge(reviews_simple, on="order_id", how="left")
            .merge(
                customers[["customer_id", "customer_city", "customer_state"]],
                on="customer_id",
                how="left",
            )
        )

        # Feature keterlambatan delivery
        if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
            df["delivery_delay"] = (
                df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
            ).dt.days
        else:
            df["delivery_delay"] = pd.NA

        # ====== Context handling (FIX UTAMA) ======
        # Private DF untuk chaining (EDAAgent butuh DF):
        result_context = context.copy()
        result_context["_orders_merged_df"] = df  # ⬅️ private, jangan dipublish ke API

        # Public/serializable preview untuk debug/UI/API:
        preview = df.head(20)
        result_context["orders_merged_preview"] = preview.to_dict(orient="records")
        result_context["orders_merged_columns"] = list(df.columns)
        result_context["orders_merged_shape"] = [int(df.shape[0]), int(df.shape[1])]

        return {
            "agent": self.name,
            "role": self.role,
            "summary": f"Loaded and merged datasets for {len(df):,} orders.",
            # ⚠️ Jangan return DF mentah. Context aman untuk API.
            "context": result_context,
        }
