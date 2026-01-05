import os
from pathlib import Path
from dotenv import load_dotenv

# app/core/config.py -> parents[2] = project root (final_project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"[WARN] .env not found at: {ENV_PATH}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set. LLM calls will fail until you set it in .env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"  
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "app" / "db" / "olist.db"

# Bisa dioverride lewat env kalau mau
RAG_PRODUCTS_PATH = Path(os.getenv("RAG_PRODUCTS_PATH", str(DATA_PROCESSED_DIR / "rag_products.csv")))
FACT_ORDER_ITEMS_PATH = Path(os.getenv("FACT_ORDER_ITEMS_PATH", str(DATA_PROCESSED_DIR / "fact_order_items.csv")))
