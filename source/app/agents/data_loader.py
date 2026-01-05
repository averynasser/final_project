import pandas as pd
from typing import Dict, Any
from .base_agent import BaseAgent
from app.core.data_loader import OlistDataLoader


class DataLoaderAgent(BaseAgent):
    """
    Agent untuk memuat seluruh dataset Olist (tabel mentah).
    Biasanya tidak diperlukan kalau sudah memakai DataAgent,
    tapi tetap disediakan kalau ingin eksplorasi raw tables.
    """

    def __init__(self, loader: OlistDataLoader | None = None):
        super().__init__(name="DataLoaderAgent", role="Load Olist raw tables")
        self.loader = loader or OlistDataLoader()

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        customers = self.loader.customers
        orders = self.loader.orders
        order_items = self.loader.order_items
        products = self.loader.products
        sellers = self.loader.sellers
        payments = self.loader.order_payments
        reviews = self.loader.order_reviews

        result_context = context.copy()
        result_context.update(
            {
                "customers": customers,
                "orders": orders,
                "order_items": order_items,
                "products": products,
                "sellers": sellers,
                "payments": payments,
                "reviews": reviews,
            }
        )

        # Optional: build wide merged table (bisa besar dan duplikatif)
        orders_merged = (
            orders
            .merge(customers, on="customer_id", how="left")
            .merge(order_items, on="order_id", how="left")
            .merge(products, on="product_id", how="left")
            .merge(sellers, on="seller_id", how="left")
            .merge(payments, on="order_id", how="left")
            .merge(reviews, on="order_id", how="left")
        )
        result_context["orders_merged_raw"] = orders_merged

        return {
            "agent": self.name,
            "role": self.role,
            "summary": "Loaded all Olist raw tables (plus a wide merged table).",
            "context": result_context,
        }
