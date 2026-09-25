"""
CloudOps Sentinel - API Request and Response Schemas
=====================================================

Pydantic models defining the REST contracts for chat queries,
file uploads, source citations, and graph diagnostic traces.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload sent by clients to submit an incident or troubleshooting question."""
    question: str = Field(
        ...,
        description="The technical or incident question submitted by the operator."
    )
    thread_id: str = Field(
        default="default",
        description="Session identifier enabling conversational memory across turns."
    )


class SourceItem(BaseModel):
    """Details of a single evidence source cited in the response."""
    type: str = Field(..., description="Source origin: 'internal' runbook or 'web' document.")
    title: str = Field(..., description="Document title or web page headline.")
    source: str = Field(..., description="File path or domain identifier.")
    url: Optional[str] = Field(default=None, description="External hyperlink when type is 'web'.")
    page: Optional[int] = Field(default=None, description="1-indexed page number for PDF/runbook sources.")


class ChatResponse(BaseModel):
    """Structured response returned by the Self-RAG engine."""
    answer: str = Field(..., description="Remediation guidance or answer synthesized by the LLM.")
    route: str = Field(..., description="Resolution channel (e.g. 'Private Runbooks', 'Internet Search').")
    used_web_search: bool = Field(default=False, description="Flag indicating if external web search was invoked.")
    support_status: str = Field(default="", description="Grounding status ('fully_supported', 'partially_supported', 'no_support').")
    usefulness: str = Field(default="", description="Usability verdict ('useful' or 'not_useful').")
    sources: List[SourceItem] = Field(default_factory=list, description="List of cited documents and URLs.")
    trace: List[str] = Field(default_factory=list, description="Diagnostic execution trace tracking graph steps.")
    thread_id: str = Field(..., description="Session thread ID.")
    memory_turns: int = Field(default=0, description="Total conversation exchanges retained in memory.")


class UploadResponse(BaseModel):
    """Response returned upon successful upload and indexing of a runbook file."""
    filename: str = Field(..., description="Name of the file uploaded.")
    chunks_indexed: int = Field(..., description="Number of vector chunks successfully indexed into Pinecone.")
    namespace: str = Field(..., description="Pinecone namespace containing the indexed vectors.")
