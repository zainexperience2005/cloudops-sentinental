"""
CloudOps Sentinel - Self-Reflective Retrieval-Augmented Generation (Self-RAG) Engine
=====================================================================================

This module implements an adaptive, self-reflective RAG pipeline tailored for
enterprise cloud operations, infrastructure incident response, and runbook automation.

Why Self-RAG for Cloud Operations?
----------------------------------
Standard RAG pipelines suffer from critical flaws when applied to production incidents:
1. Blind Retrieval: Querying the vector store for trivial or generic questions adds latency.
2. Irrelevant Context: Vector similarity matches noise, injecting irrelevant runbooks into prompts.
3. Hallucination / Unsupported Claims: Generating remediation commands not grounded in
   verified SOPs can cause catastrophic production outages (e.g., executing wrong rollback scripts).
4. Dead Ends: If internal documentation is missing or outdated, standard RAG fails instead of
   gracefully reformulating queries or falling back to verified external technical sources.

Self-RAG Solves This via a LangGraph State Machine:
---------------------------------------------------
- **Adaptive Retrieval**: Decides if internal runbooks/SOPs are actually required.
- **Contextualization**: Reformulates questions using multi-turn conversational memory.
- **Relevance Grading**: Evaluates each retrieved document chunk before feeding it to generation.
- **Query Rewriting**: Refines semantic search queries if initial chunk retrieval lacks relevance.
- **Internet Fallback (Tavily)**: Pivots to web search when internal runbooks have no answers.
- **Hallucination Verification (Grounding Check)**: Audits every factual claim against retrieved evidence.
- **Self-Correction (Answer Revision)**: Rewrites answers if ungrounded claims are detected.
- **Usefulness Audit**: Ensures the final remediation guidance directly and actionably solves the incident.
- **Multi-Turn Checkpointing**: Retains session state via SQLite checkpointer (SqliteSaver).

Architecture Flow:
------------------
  [START]
     │
     ▼
  contextualize_question (memory-aware query reformulation)
     │
     ▼
  decide_retrieval ──(No)──► generate_direct ──► commit_memory ──► [END]
     │ (Yes)
     ▼
  retrieve_internal (Pinecone vector search)
     │
     ▼
  grade_relevance ◄───────────────────────────┐
     │                                        │
     ├──(Relevant Docs Found)──► generate     │ (Fallback search)
     ├──(Needs Rewrite)──► rewrite_internal   │
     │                           │            │
     │                           ▼            │
     │                    retrieve_internal   │
     └──(Exhausted/No match)──► rewrite_web ──► web_search (Tavily)
                                  │
                                  ▼
     ┌────────────────────────────┘
     ▼
  generate_from_context (Grounded answer generation)
     │
     ▼
  check_support (Hallucination & evidence grounding check)
     │
     ├──(Not Grounded & retries left)──► revise_answer ──► (re-check support)
     │
     ▼
  check_usefulness (Addresses user question directly?)
     │
     ├──(Useful)──► commit_memory ──► [END]
     └──(Not Useful)──► rewrite_internal / rewrite_web / no_answer
"""

from typing import List, TypedDict, Literal, Annotated
import operator
import sqlite3
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import get_retriever


# ============================================================================
# Graph State Definition
# ============================================================================

class RAGState(TypedDict, total=False):
    """
    Central state dictionary passed between all nodes in the Self-RAG LangGraph.

    Attributes:
        user_question: The original raw question asked by the user or incident responder.
        question: The standalone, contextualized question rewritten to include conversation history.
        memory: Accumulated conversation turns, merged using operator.add for multi-turn history.
        retrieval_query: Optimized search query targeted at the Pinecone vector database.
        web_query: Optimized search query targeted at the Tavily internet search API.
        need_retrieval: Boolean flag indicating if knowledge retrieval is necessary.
        docs: Raw candidate Document chunks retrieved from internal storage or web search.
        relevant_docs: Filtered subset of Documents that passed the relevance grading check.
        context: Concatenated, formatted evidence string supplied to the generation prompt.
        answer: The synthesized answer or remediation guidance.
        support_status: Grounding audit status ('fully_supported', 'partially_supported', 'no_support').
        evidence: Extracted citations and text evidence supporting the claims in the answer.
        usefulness: Usability evaluation ('useful' or 'not_useful') indicating question resolution.
        use_reason: Justification explaining the usefulness grading decision.
        support_retries: Number of revision loops attempted to fix ungrounded claims.
        retrieval_rewrites: Number of query rewriting loops attempted for internal vector retrieval.
        web_rewrites: Number of query rewriting loops attempted for internet search.
        source_mode: Origin of evidence used ('internal', 'web', 'direct', or 'none').
        used_web_search: Flag indicating whether Tavily external search was invoked.
        trace: Step-by-step diagnostic breadcrumbs tracking node executions and routing decisions.
    """
    user_question: str
    question: str
    memory: Annotated[List[str], operator.add]
    retrieval_query: str
    web_query: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    support_status: Literal["fully_supported", "partially_supported", "no_support", ""]
    evidence: List[str]
    usefulness: Literal["useful", "not_useful", ""]
    use_reason: str
    support_retries: int
    retrieval_rewrites: int
    web_rewrites: int
    source_mode: Literal["internal", "web", "direct", "none"]
    used_web_search: bool
    trace: List[str]


# ============================================================================
# Pydantic Structured Output Schemas (LLM Decision Enforcers)
# ============================================================================

class RetrieveDecision(BaseModel):
    """Schema for the adaptive retrieval router: decides if documents are needed."""
    should_retrieve: bool = Field(
        description="True if question pertains to operations, runbooks, logs, incidents, or private infra; False for generic technical trivia."
    )


class RelevanceDecision(BaseModel):
    """Schema for document relevance grading."""
    is_relevant: bool = Field(
        description="True if document contains information helpful to answer the question, False otherwise."
    )


class SupportDecision(BaseModel):
    """Schema for hallucination / grounding verification."""
    status: Literal["fully_supported", "partially_supported", "no_support"] = Field(
        description="Grounding status: fully_supported if all claims match evidence; partially_supported if some do; no_support if claims are ungrounded."
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Concise excerpt strings extracted verbatim from the context supporting the claims."
    )


class UsefulnessDecision(BaseModel):
    """Schema for evaluating answer quality and actionable usefulness."""
    status: Literal["useful", "not_useful"] = Field(
        description="useful if answer directly addresses the question; not_useful if ambiguous or unresponsive."
    )
    reason: str = Field(
        description="One-sentence technical justification for the usefulness rating."
    )


class QueryRewrite(BaseModel):
    """Schema for query rewriting and search reformulation."""
    query: str = Field(
        description="The reformulated, keyword-dense search query."
    )


# ============================================================================
# Core Helper Utilities
# ============================================================================

def _llm() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the primary ChatGoogleGenerativeAI model instance.

    Uses temperature=0 to ensure deterministic, reproducible evaluations
    and minimize hallucination during incident response procedures.
    """
    s = get_settings()
    api_key = s.gemini_api_key or s.google_api_key
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured in environment or settings.")
    return ChatGoogleGenerativeAI(
        model=s.gemini_model,
        google_api_key=api_key,
        temperature=0,
    )


def _trace(state: RAGState, item: str) -> List[str]:
    """
    Appends a new diagnostic step event to the execution trace list.
    Provides complete observability for incident post-mortems and LangSmith tracking.
    """
    return [*(state.get("trace") or []), item]


def _format_context(docs: List[Document]) -> str:
    """
    Formats a list of retrieved Document objects into a structured, labeled evidence block.

    Separates internal runbooks from external web sources with explicit header tags,
    enabling the generation model to clearly differentiate authoritative private SOPs
    from third-party documentation.
    """
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        if meta.get("source_type") == "web":
            head = f"[WEB {i}] {meta.get('title','')} | {meta.get('url','')}"
        else:
            head = f"[INTERNAL {i}] {meta.get('title') or meta.get('document_name') or meta.get('source','')}"
            if meta.get("page") is not None:
                head += f" | page {int(meta['page']) + 1}"
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _memory_text(state: RAGState, limit: int = 4) -> str:
    """
    Retrieves the most recent conversation exchanges formatted for conversational prompts.
    Defaults to the last 4 turns to maintain context without overflowing token limits.
    """
    items = state.get("memory") or []
    return "\n\n".join(items[-limit:]) if items else "No previous conversation context."


# ============================================================================
# LangGraph Node Implementations
# ============================================================================

def contextualize_question(state: RAGState) -> dict:
    """
    Node 1: Question Contextualization.

    Analyzes multi-turn conversation memory and rewrites the incoming user message
    into a self-contained, standalone question. This guarantees that downstream vector
    similarity search can resolve pronouns ('it', 'that service', 'the error above')
    to specific cluster names, pod IDs, and microservices.
    """
    history = _memory_text(state)
    user_question = state.get("user_question") or state.get("question", "")

    # If there is no previous conversation history, use the question directly
    if not state.get("memory"):
        return {
            "question": user_question,
            "trace": _trace(state, "Memory: new incident session initialized"),
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the newest user message as a standalone cloud-operations question using "
            "the previous conversation only when needed. Preserve service names, symptoms, "
            "errors, and constraints. If the message already stands alone, return it unchanged. "
            "Do not answer the question.",
        ),
        ("human", "Previous conversation:\n{history}\n\nNewest message:\n{question}"),
    ])
    out = _llm().with_structured_output(QueryRewrite).invoke(
        prompt.format_messages(history=history, question=user_question)
    )
    return {
        "question": out.query,
        "trace": _trace(state, f"Memory contextualized question: {out.query}"),
    }


def commit_memory(state: RAGState) -> dict:
    """
    Terminal Node: Memory Commit.

    Persists the completed Q&A exchange to the state's memory channel.
    Because memory is defined with Annotated[List[str], operator.add], returning
    a single-element list appends it to the durable session state in SQLite.
    """
    user_question = state.get("user_question") or state.get("question", "")
    answer = state.get("answer", "")
    route = state.get("source_mode", "none")
    entry = f"User: {user_question}\nAssistant ({route}): {answer}"
    return {
        "memory": [entry],
        "trace": _trace(state, "SQLite memory checkpoint updated"),
    }


def decide_retrieval(state: RAGState) -> dict:
    """
    Node 2: Adaptive Retrieval Decision.

    Determines whether answering the query requires looking up operational runbooks,
    incident history, or infrastructure docs, versus answering directly from
    general foundational knowledge (e.g., 'What does HTTP 504 mean?').
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You decide whether the question needs retrieval. Choose true for cloud operations, "
            "production incidents, service runbooks, deployment procedures, infrastructure behavior, "
            "troubleshooting steps, specific/current technical facts, or whenever evidence is needed. "
            "Choose false only for generic technical explanations that can safely be answered from "
            "general knowledge. If unsure choose true.",
        ),
        ("human", "Question: {question}"),
    ])
    out = _llm().with_structured_output(RetrieveDecision).invoke(
        prompt.format_messages(question=state["question"])
    )
    return {
        "need_retrieval": out.should_retrieve,
        "trace": _trace(state, f"Retrieval decision: {out.should_retrieve}"),
    }


def route_after_decide(state: RAGState) -> Literal["direct", "retrieve"]:
    """
    Conditional Edge: Routes execution based on the retrieval decision.
    """
    return "retrieve" if state.get("need_retrieval", True) else "direct"


def generate_direct(state: RAGState) -> dict:
    """
    Direct Generation Node (Bypasses Vector Retrieval).

    Answers generic conceptual questions without querying the vector database.
    Explicitly guarded with instructions not to fabricate organization-specific
    credentials, internal hostnames, or production runbooks.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer briefly from general technical knowledge only. Do not invent organization-specific "
            "infrastructure, runbooks, credentials, incident history, or deployment procedures.",
        ),
        ("human", "{question}"),
    ])
    ans = _llm().invoke(prompt.format_messages(question=state["question"])).content
    return {
        "answer": ans,
        "source_mode": "direct",
        "trace": _trace(state, "Generated direct answer (General Technical Knowledge)"),
    }


def retrieve_internal(state: RAGState) -> dict:
    """
    Node 3: Internal Vector Retrieval.

    Performs vector similarity search against the Pinecone index using the
    active retrieval query (or standalone question). Attaches metadata tags
    indicating internal runbook provenance.
    """
    q = state.get("retrieval_query") or state["question"]
    docs = get_retriever().invoke(q)
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source_type": "internal"}
    return {
        "docs": docs,
        "relevant_docs": [],
        "source_mode": "internal",
        "trace": _trace(state, f"Internal Pinecone retrieval: {len(docs)} chunks matched"),
    }


def grade_relevance(state: RAGState) -> dict:
    """
    Node 4: Document Relevance Grading (Self-Reflection Filter).

    Iterates through each retrieved document chunk and grades whether it contains
    actionable context relevant to the user's specific problem. Chunks that fail
    the relevance filter are discarded to prevent prompt pollution and hallucination.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Judge relevance at the topic/evidence level. A document is relevant when it contains "
            "information useful for answering the user's question. Do not require the exact final answer. "
            "Be strict about unrelated content.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ])
    grader = _llm().with_structured_output(RelevanceDecision)
    relevant = []
    for d in state.get("docs", []):
        try:
            decision = grader.invoke(
                prompt.format_messages(
                    question=state["question"],
                    document=d.page_content[:7000]
                )
            )
            if decision.is_relevant:
                relevant.append(d)
        except Exception:
            continue

    mode = state.get("source_mode", "internal")
    return {
        "relevant_docs": relevant,
        "trace": _trace(
            state,
            f"Relevance grading ({mode}): {len(relevant)}/{len(state.get('docs', []))} chunks approved"
        ),
    }


def route_after_relevance(state: RAGState) -> Literal["generate", "rewrite_internal", "rewrite_web", "no_answer"]:
    """
    Conditional Edge: Evaluates relevance results and determines the next step.

    - If relevant chunks exist: Proceed to generation.
    - If no relevant internal chunks and rewrite quota remaining: Rewrite query and retry Pinecone.
    - If internal attempts exhausted: Pivot to external web search (Tavily).
    - If web attempts exhausted: Return graceful fallback.
    """
    if state.get("relevant_docs"):
        return "generate"
    s = get_settings()
    if state.get("source_mode") == "web":
        if state.get("web_rewrites", 0) < s.max_web_rewrites:
            return "rewrite_web"
        return "no_answer"
    if state.get("retrieval_rewrites", 0) < s.max_retrieval_rewrites:
        return "rewrite_internal"
    return "rewrite_web"


def rewrite_internal_query(state: RAGState) -> dict:
    """
    Node 5a: Internal Query Reformulation.

    When initial vector search fails to yield relevant runbook chunks,
    this node reformulates the query using operational synonyms, error codes,
    and system keywords optimized for dense vector embeddings.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the operations question for semantic vector retrieval over internal cloud "
            "runbooks, SOPs, postmortems, architecture notes, and troubleshooting documents. "
            "Use 6-18 words, preserve service names and error symptoms, add useful operations "
            "keywords, remove filler, and do not answer.",
        ),
        ("human", "Question: {question}\nPrevious query: {previous}"),
    ])
    out = _llm().with_structured_output(QueryRewrite).invoke(
        prompt.format_messages(
            question=state["question"],
            previous=state.get("retrieval_query", "")
        )
    )
    return {
        "retrieval_query": out.query,
        "retrieval_rewrites": state.get("retrieval_rewrites", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "trace": _trace(state, f"Rewrote internal query: {out.query}"),
    }


def rewrite_web_query(state: RAGState) -> dict:
    """
    Node 5b: External Web Query Formulation.

    When internal runbooks do not cover the incident, this node prepares
    a targeted, keyword-efficient web search query suitable for Tavily.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the question into a concise internet search query of 6-14 words. "
            "Preserve important entities. Add recency wording when the question asks "
            "for latest/current/today. Do not answer.",
        ),
        ("human", "Question: {question}\nPrevious web query: {previous}"),
    ])
    out = _llm().with_structured_output(QueryRewrite).invoke(
        prompt.format_messages(
            question=state["question"],
            previous=state.get("web_query", "")
        )
    )
    return {
        "web_query": out.query,
        "web_rewrites": state.get("web_rewrites", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "trace": _trace(state, f"Prepared internet search query: {out.query}"),
    }


def web_search(state: RAGState) -> dict:
    """
    Node 6: Fallback Web Search (Tavily Integration).

    Executes external internet search when internal knowledge is insufficient.
    Converts search results into LangChain Document objects with web metadata
    and tags the source mode as 'web'.
    """
    s = get_settings()
    if not s.tavily_api_key:
        return {
            "docs": [],
            "source_mode": "web",
            "used_web_search": True,
            "trace": _trace(state, "Internet search unavailable: TAVILY_API_KEY missing"),
        }
    client = TavilyClient(api_key=s.tavily_api_key)
    q = state.get("web_query") or state["question"]
    response = client.search(query=q, search_depth="advanced", max_results=5, include_answer=False)
    docs = []
    for r in response.get("results", []):
        content = r.get("content", "")
        docs.append(Document(
            page_content=content,
            metadata={
                "source_type": "web",
                "source": r.get("url", ""),
                "url": r.get("url", ""),
                "title": r.get("title", ""),
            },
        ))
    return {
        "docs": docs,
        "source_mode": "web",
        "used_web_search": True,
        "trace": _trace(state, f"Internet search completed: {len(docs)} results fetched"),
    }


def generate_from_context(state: RAGState) -> dict:
    """
    Node 7: Grounded Answer Synthesis.

    Generates actionable incident remediation instructions strictly based
    on the filtered, relevant context. If the evidence came from web search,
    it instructs the model to explicitly flag guidance as external best-practice.
    """
    context = _format_context(state.get("relevant_docs", []))
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are CloudOps Sentinel, an enterprise cloud operations and incident-response copilot. "
            "Answer using only the supplied evidence. Prefer private runbooks, SOPs, architecture "
            "notes, and postmortems when present. If the evidence comes from the web, clearly label it "
            "as external guidance and never present it as an organization-specific procedure. Do not "
            "invent infrastructure facts, credentials, commands, or incident history. Provide concise, "
            "actionable troubleshooting guidance and preserve any cautions contained in the evidence.",
        ),
        ("human", "Question:\n{question}\n\nEvidence:\n{context}"),
    ])
    ans = _llm().invoke(
        prompt.format_messages(question=state["question"], context=context)
    ).content
    return {
        "answer": ans,
        "context": context,
        "support_retries": 0,
        "trace": _trace(state, f"Generated answer from {state.get('source_mode','')} evidence"),
    }


def check_support(state: RAGState) -> dict:
    """
    Node 8: Hallucination & Grounding Audit (Self-Reflection Check).

    Audits the generated answer against the source context to ensure that every
    claim, diagnostic step, and remediation command is strictly supported by evidence.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Verify whether every meaningful claim in the answer is supported by the supplied evidence. "
            "Return fully_supported only when all important claims are grounded; partially_supported "
            "when some claims are grounded but some are not; no_support when key claims are unsupported. "
            "Evidence excerpts should be short.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{context}"),
    ])
    out = _llm().with_structured_output(SupportDecision).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )
    return {
        "support_status": out.status,
        "evidence": out.evidence,
        "trace": _trace(state, f"Support / Grounding audit: {out.status}"),
    }


def route_after_support(state: RAGState) -> Literal["usefulness", "revise"]:
    """
    Conditional Edge: Checks if answer is fully grounded or needs self-correction revision.
    """
    if state.get("support_status") == "fully_supported":
        return "usefulness"
    if state.get("support_retries", 0) >= get_settings().max_support_retries:
        return "usefulness"
    return "revise"


def revise_answer(state: RAGState) -> dict:
    """
    Node 9: Self-Correction Revision Loop.

    Rewrites the answer to eliminate ungrounded speculation or hallucinated commands,
    retaining only verifiable claims grounded in the provided runbook context.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the answer so every factual claim is directly supported by the provided evidence. "
            "Remove unsupported interpretation and speculation. Still answer the question naturally; "
            "do not mention this verification process.",
        ),
        ("human", "Question:\n{question}\n\nCurrent answer:\n{answer}\n\nEvidence:\n{context}"),
    ])
    ans = _llm().invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    ).content
    return {
        "answer": ans,
        "support_retries": state.get("support_retries", 0) + 1,
        "trace": _trace(state, "Revised answer to eliminate ungrounded claims (Self-Correction)"),
    }


def check_usefulness(state: RAGState) -> dict:
    """
    Node 10: Answer Usefulness Evaluation.

    Verifies that the generated response directly answers the user's operational
    question and provides actionable troubleshooting steps.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Judge only whether the answer directly addresses the user's question. Do not re-grade "
            "factual grounding. Return useful or not_useful and a one-line reason.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ])
    out = _llm().with_structured_output(UsefulnessDecision).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", "")
        )
    )
    return {
        "usefulness": out.status,
        "use_reason": out.reason,
        "trace": _trace(state, f"Usefulness evaluation: {out.status} ({out.reason})"),
    }


def route_after_usefulness(state: RAGState) -> Literal["end", "rewrite_internal", "rewrite_web", "no_answer"]:
    """
    Conditional Edge: Routes flow based on usefulness evaluation.

    - Useful: Save to memory and finish.
    - Not useful (internal source): Trigger query rewrite or fall back to web search.
    - Not useful (web source): Retry web query or exit cleanly.
    """
    if state.get("usefulness") == "useful":
        return "end"
    s = get_settings()
    if state.get("source_mode") == "internal":
        if state.get("retrieval_rewrites", 0) < s.max_retrieval_rewrites:
            return "rewrite_internal"
        return "rewrite_web"
    if state.get("source_mode") == "web" and state.get("web_rewrites", 0) < s.max_web_rewrites:
        return "rewrite_web"
    return "no_answer"


def no_answer(state: RAGState) -> dict:
    """
    Fallback Safety Node.

    Triggered when no reliable internal runbooks or web sources can safely answer
    the question. Prevents unsafe recommendations during high-severity outages.
    """
    return {
        "answer": (
            "I could not find enough reliable runbook or external evidence to recommend "
            "a safe troubleshooting action. Please consult primary engineering on-call."
        ),
        "source_mode": "none",
        "trace": _trace(state, "Safety stop: Insufficient reliable evidence to proceed"),
    }


# ============================================================================
# Graph Construction & Compilation
# ============================================================================

def build_graph():
    """
    Constructs and compiles the complete Self-RAG LangGraph state machine.

    Configures nodes, directed edges, conditional routing, and durable SQLite
    checkpointing for session persistence across multiple user messages.
    """
    g = StateGraph(RAGState)

    # 1. Register Nodes
    g.add_node("contextualize", contextualize_question)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("direct", generate_direct)
    g.add_node("retrieve", retrieve_internal)
    g.add_node("grade", grade_relevance)
    g.add_node("rewrite_internal", rewrite_internal_query)
    g.add_node("rewrite_web", rewrite_web_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate_from_context)
    g.add_node("support", check_support)
    g.add_node("revise", revise_answer)
    g.add_node("usefulness", check_usefulness)
    g.add_node("no_answer", no_answer)
    g.add_node("commit_memory", commit_memory)

    # 2. Wire Start and Retrieval Decision Flow
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"direct": "direct", "retrieve": "retrieve"}
    )
    g.add_edge("direct", "commit_memory")

    # 3. Wire Internal Retrieval & Relevance Grading
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_relevance,
        {
            "generate": "generate",
            "rewrite_internal": "rewrite_internal",
            "rewrite_web": "rewrite_web",
            "no_answer": "no_answer",
        }
    )
    g.add_edge("rewrite_internal", "retrieve")
    g.add_edge("rewrite_web", "web_search")
    g.add_edge("web_search", "grade")

    # 4. Wire Generation, Grounding Check, and Revision Loop
    g.add_edge("generate", "support")
    g.add_conditional_edges(
        "support",
        route_after_support,
        {"usefulness": "usefulness", "revise": "revise"}
    )
    g.add_edge("revise", "support")

    # 5. Wire Usefulness Audit and Final Termination
    g.add_conditional_edges(
        "usefulness",
        route_after_usefulness,
        {
            "end": "commit_memory",
            "rewrite_internal": "rewrite_internal",
            "rewrite_web": "rewrite_web",
            "no_answer": "no_answer",
        }
    )
    g.add_edge("no_answer", "commit_memory")
    g.add_edge("commit_memory", END)

    # 6. Attach Persistent SQLite Checkpointer
    db_path = Path(__file__).resolve().parents[1] / "data" / "langgraph_memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)


# ============================================================================
# Output Formatting & Invocation Interfaces
# ============================================================================

def _sources(docs: List[Document]) -> List[dict]:
    """
    Extracts deduplicated source citations from the retrieved document list.
    """
    seen, out = set(), []
    for d in docs or []:
        m = d.metadata or {}
        typ = "web" if m.get("source_type") == "web" else "internal"
        key = (typ, m.get("url") or m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        item = {
            "type": typ,
            "title": m.get("title") or m.get("document_name") or "",
            "source": m.get("source") or "",
            "url": m.get("url") if typ == "web" else None,
            "page": (int(m["page"]) + 1) if typ == "internal" and m.get("page") is not None else None,
        }
        out.append(item)
    return out


# Singleton instance of compiled LangGraph
_graph = None


def run_self_rag(question: str, thread_id: str) -> dict:
    """
    Synchronous Execution Wrapper for the Self-RAG engine.

    Args:
        question: Incident description or technical troubleshooting question.
        thread_id: Unique session identifier for tracking multi-turn memory.

    Returns:
        Dictionary containing answer, route mode, sources, execution trace, and metadata.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial: RAGState = {
        "user_question": question,
        "question": question,
        "memory": [],
        "retrieval_query": question,
        "web_query": "",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "support_status": "",
        "evidence": [],
        "usefulness": "",
        "use_reason": "",
        "support_retries": 0,
        "retrieval_rewrites": 0,
        "web_rewrites": 0,
        "source_mode": "internal",
        "used_web_search": False,
        "trace": [],
    }

    result = _graph.invoke(
        initial,
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    )

    mode = result.get("source_mode", "none")
    route = {
        "internal": "Private Runbooks",
        "web": "Internet Search",
        "direct": "General Knowledge",
        "none": "No Reliable Evidence",
    }.get(mode, mode)

    return {
        "answer": result.get("answer", ""),
        "route": route,
        "used_web_search": bool(result.get("used_web_search")),
        "support_status": result.get("support_status", ""),
        "usefulness": result.get("usefulness", ""),
        "sources": _sources(result.get("relevant_docs", [])),
        "trace": result.get("trace", []),
        "thread_id": thread_id,
        "memory_turns": len(result.get("memory", [])),
    }


def stream_self_rag(question: str, thread_id: str):
    """
    Streaming Generator for the Self-RAG engine.

    Yields intermediate graph step events, diagnostic trace items,
    word-level token chunks of the generated response, and the final result payload.
    Used by interactive dashboards and SSE (Server-Sent Events) streaming.

    Args:
        question: Incident description or technical troubleshooting question.
        thread_id: Unique session identifier for multi-turn conversational checkpointing.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial: RAGState = {
        "user_question": question,
        "question": question,
        "memory": [],
        "retrieval_query": question,
        "web_query": "",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "support_status": "",
        "evidence": [],
        "usefulness": "",
        "use_reason": "",
        "support_retries": 0,
        "retrieval_rewrites": 0,
        "web_rewrites": 0,
        "source_mode": "internal",
        "used_web_search": False,
        "trace": [],
    }

    accumulated_state = dict(initial)
    emitted_trace_count = 0

    # Stream graph updates step by step
    for step_output in _graph.stream(
        initial,
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 60},
        stream_mode="updates"
    ):
        for node_name, updates in step_output.items():
            if isinstance(updates, dict):
                accumulated_state.update(updates)
                current_trace = accumulated_state.get("trace", [])
                while emitted_trace_count < len(current_trace):
                    trace_item = current_trace[emitted_trace_count]
                    emitted_trace_count += 1
                    yield {
                        "type": "step",
                        "node": node_name,
                        "step": trace_item,
                        "trace": current_trace[:emitted_trace_count]
                    }

    mode = accumulated_state.get("source_mode", "none")
    route = {
        "internal": "Private Runbooks",
        "web": "Internet Search",
        "direct": "General Knowledge",
        "none": "No Reliable Evidence",
    }.get(mode, mode)

    final_result = {
        "answer": accumulated_state.get("answer", ""),
        "route": route,
        "used_web_search": bool(accumulated_state.get("used_web_search")),
        "support_status": accumulated_state.get("support_status", ""),
        "usefulness": accumulated_state.get("usefulness", ""),
        "sources": _sources(accumulated_state.get("relevant_docs", [])),
        "trace": accumulated_state.get("trace", []),
        "thread_id": thread_id,
        "memory_turns": len(accumulated_state.get("memory", [])),
    }

    # Stream tokens of the final synthesized answer
    answer_text = final_result["answer"]
    words = answer_text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield {
            "type": "token",
            "token": chunk
        }

    yield {
        "type": "done",
        "data": final_result
    }