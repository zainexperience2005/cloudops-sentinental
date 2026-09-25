"""
CloudOps Sentinel - Pinecone Vector Store & Gemini Embeddings Integration
==========================================================================

This module manages the vector search infrastructure for CloudOps Sentinel,
connecting Google Gemini embeddings (text-embedding-004) to Pinecone serverless vector indexes.
"""

from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import get_settings


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and caches the Google Generative AI embeddings client.

    Uses Google's state-of-the-art models/text-embedding-004 model configured
    with 768 vector dimensions for high-precision semantic retrieval.
    """
    s = get_settings()
    api_key = s.gemini_api_key or s.google_api_key
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured in environment or settings.")
    return GoogleGenerativeAIEmbeddings(
        model=s.embedding_model,
        google_api_key=api_key,
    )


def get_pinecone_client() -> Pinecone:
    """
    Initializes and returns the native Pinecone API client.
    """
    s = get_settings()
    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured in environment or settings.")
    return Pinecone(api_key=s.pinecone_api_key)


def ensure_index():
    """
    Ensures that the target Pinecone index exists with the correct metric and dimension.

    If the index does not exist, it creates a new serverless index on the configured
    cloud provider and region (defaults to AWS us-east-1) using cosine similarity.
    If the index exists, it verifies that the index dimension matches the embedding model.
    """
    s = get_settings()
    pc = get_pinecone_client()
    existing = {x.name for x in pc.list_indexes()}

    if s.pinecone_index_name not in existing:
        pc.create_index(
            name=s.pinecone_index_name,
            dimension=s.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )
    else:
        desc = pc.describe_index(s.pinecone_index_name)
        existing_dimension = getattr(desc, "dimension", None)
        if existing_dimension is None and isinstance(desc, dict):
            existing_dimension = desc.get("dimension")
        if existing_dimension and int(existing_dimension) != s.embedding_dimension:
            raise RuntimeError(
                f"Pinecone index '{s.pinecone_index_name}' has dimension {existing_dimension}, "
                f"but {s.embedding_model} is configured for {s.embedding_dimension}. "
                "Use a new index name or recreate the index with the correct dimension."
            )

    return pc.Index(s.pinecone_index_name)


def get_vector_store() -> PineconeVectorStore:
    """
    Instantiates the LangChain PineconeVectorStore client bound to the configured namespace.

    Namespaces allow logical separation of operational documents (e.g. 'incident-runbooks')
    within a shared cluster.
    """
    s = get_settings()
    index = ensure_index()
    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=s.pinecone_namespace,
    )


def get_retriever():
    """
    Returns a LangChain retriever interface configured with the configured top_k similarity limit.
    """
    s = get_settings()
    return get_vector_store().as_retriever(search_kwargs={"k": s.top_k})