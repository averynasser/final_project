from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import DB_PATH


class OlistDataLoader:
    """
    Loader terpusat untuk semua dataset Olist dari SQLite (Mode B).
    API properties tetap sama seperti sebelumnya: customers, orders, order_items, dst.
    """
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self._cache: Dict[str, pd.DataFrame] = {}

        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")

    def _read_sql(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def _load_table(self, table_name: str) -> pd.DataFrame:
        if table_name in self._cache:
            return self._cache[table_name]

        df = self._read_sql(f"SELECT * FROM {table_name}")
        self._cache[table_name] = df
        return df

    @property
    def customers(self) -> pd.DataFrame:
        return self._load_table("customers")

    @property
    def geolocation(self) -> pd.DataFrame:
        return self._load_table("geolocation")

    @property
    def orders(self) -> pd.DataFrame:
        df = self._load_table("orders").copy()
        # parse datetime columns (kalau ada)
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    @property
    def order_items(self) -> pd.DataFrame:
        return self._load_table("order_items")

    @property
    def order_payments(self) -> pd.DataFrame:
        # di db kamu namanya "payments" atau "order_payments"?
        # build_sqlite kamu biasanya pakai "payments"
        # jadi kita coba keduanya secara aman.
        try:
            return self._load_table("payments")
        except Exception:
            return self._load_table("order_payments")

    @property
    def order_reviews(self) -> pd.DataFrame:
        try:
            return self._load_table("reviews")
        except Exception:
            return self._load_table("order_reviews")

    @property
    def products(self) -> pd.DataFrame:
        return self._load_table("products")

    @property
    def sellers(self) -> pd.DataFrame:
        return self._load_table("sellers")
