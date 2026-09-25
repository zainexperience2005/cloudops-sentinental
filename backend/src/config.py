from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # Google Gemini AI & Embeddings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    # Pinecone Vector DB
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "cloudops-sentinel-gemini-self-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "incident-runbooks")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Internet search (Tavily Fallback)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Self-RAG controls
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_support_retries: int = int(os.getenv("MAX_SUPPORT_RETRIES", "2"))
    max_retrieval_rewrites: int = int(os.getenv("MAX_RETRIEVAL_REWRITES", "2"))
    max_web_rewrites: int = int(os.getenv("MAX_WEB_REWRITES", "2"))

    # Database Configuration (Neon PostgreSQL via SQLAlchemy)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://neondb_owner:npg_i5UvDM7ogQbj@ep-proud-thunder-b46b3rk4-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
    )
    database_path: str = os.getenv("DATABASE_PATH", "data/audit.db")

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()