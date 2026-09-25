"""
CloudOps Sentinel - Golden Evaluation Dataset Loader
=====================================================

Provides access to the 50-item golden test dataset for evaluating
Self-RAG and CRAG retrieval, generation, and self-reflection pipelines.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from deepeval.test_case import LLMTestCase

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "golden_dataset.json"


def load_raw_golden_dataset() -> List[Dict[str, Any]]:
    """Loads the raw 50-item golden dataset from JSON."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_golden_test_cases(
    limit: Optional[int] = None,
    category: Optional[str] = None,
    retrieval_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Returns filtered test cases from the golden dataset.
    
    Args:
        limit: Optional maximum number of test cases to return.
        category: Filter by specific category (e.g. 'checkout_api_runbook').
        retrieval_only: If True, only returns test cases that require vector retrieval.
    """
    data = load_raw_golden_dataset()
    if category:
        data = [d for d in data if d.get("category") == category]
    if retrieval_only:
        data = [d for d in data if d.get("retrieval_required", True)]
    if limit:
        data = data[:limit]
    return data


def create_llm_test_case(
    input_query: str,
    actual_output: str,
    retrieval_context: List[str],
    expected_output: Optional[str] = None,
    context: Optional[List[str]] = None
) -> LLMTestCase:
    """
    Constructs a DeepEval LLMTestCase configured with the required fields
    for RAG Triad and context retrieval metrics.
    
    Args:
        input_query: The question/prompt sent to the RAG pipeline.
        actual_output: The answer synthesized by the LLM.
        retrieval_context: List of text chunks retrieved by the vectorstore/retriever.
        expected_output: Ground truth answer (required for Contextual Recall).
        context: Ground truth reference context chunks (optional / fallback).
    """
    return LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context if retrieval_context else ["No context retrieved."],
        context=context if context else retrieval_context
    )
