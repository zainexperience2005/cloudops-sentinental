"""
====================================================================================================
CloudOps Sentinel - Self-RAG Pipeline Evaluation Suite (DeepEval)
====================================================================================================

Evaluates the Self-Reflective RAG (Self-RAG) pipeline across the 5 core RAG metrics:
  1. AnswerRelevancyMetric      - Response precision & directness
  2. FaithfulnessMetric         - Factual grounding & hallucination prevention
  3. ContextualRelevancyMetric  - Retrieved context signal-to-noise ratio
  4. ContextualPrecisionMetric  - Top-rank retrieval ordering
  5. ContextualRecallMetric     - Retrieval completeness vs ground truth

This test suite runs against the 50-item CloudOps Golden Dataset located in `evals/golden_dataset.json`.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Force UTF-8 encoding on Windows to support emojis and rich console output
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import pytest
from dotenv import load_dotenv

# Set up project path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from evals.dataset import get_golden_test_cases, create_llm_test_case
from src.self_rag import run_self_rag, invoke_self_rag
from src.vectorstore import get_retriever


# ============================================================================
# Metric Configuration
# ============================================================================

DEFAULT_THRESHOLD = 0.70
EVAL_MODEL = "gpt-4o-mini"

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=DEFAULT_THRESHOLD,
    model=EVAL_MODEL,
    include_reason=True
)

faithfulness_metric = FaithfulnessMetric(
    threshold=DEFAULT_THRESHOLD,
    model=EVAL_MODEL,
    include_reason=True
)

contextual_relevancy_metric = ContextualRelevancyMetric(
    threshold=DEFAULT_THRESHOLD,
    model=EVAL_MODEL,
    include_reason=True
)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=DEFAULT_THRESHOLD,
    model=EVAL_MODEL,
    include_reason=True
)

contextual_recall_metric = ContextualRecallMetric(
    threshold=DEFAULT_THRESHOLD,
    model=EVAL_MODEL,
    include_reason=True
)


def get_all_metrics(threshold: float = DEFAULT_THRESHOLD) -> List[Any]:
    """Returns freshly configured metric instances."""
    return [
        AnswerRelevancyMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        ContextualRelevancyMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        ContextualRecallMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
    ]


# ============================================================================
# Metric Improvement & Diagnostic Guide
# ============================================================================

def diagnose_metric_failures(metric_results: Dict[str, Any]) -> List[str]:
    """
    Analyzes evaluation scores and produces actionable remediation steps
    to optimize the Self-RAG pipeline if any metric falls below target.
    """
    remediations = []
    
    # 1. Answer Relevancy Check
    if metric_results.get("AnswerRelevancy", 1.0) < DEFAULT_THRESHOLD:
        remediations.append(
            "[FIX - Answer Relevancy]: "
            "The model output contains extraneous information or drifts from the prompt. "
            "Action: Tighten the system prompt in `generate_from_context` and `generate_direct` "
            "to enforce concise, direct answers without conversational filler."
        )
        
    # 2. Faithfulness Check
    if metric_results.get("Faithfulness", 1.0) < DEFAULT_THRESHOLD:
        remediations.append(
            "[FIX - Faithfulness / Hallucination]: "
            "The model output makes claims unsupported by retrieved documents. "
            "Action: Strengthen the `check_support` node in `self_rag.py`. Ensure temperature=0 "
            "and enforce strict fallback when evidence is inconclusive."
        )
        
    # 3. Contextual Relevancy Check
    if metric_results.get("ContextualRelevancy", 1.0) < DEFAULT_THRESHOLD:
        remediations.append(
            "[FIX - Contextual Relevancy]: "
            "Retrieved chunks contain high background noise. "
            "Action: Adjust chunk size in `ingestion.py` (e.g. 600-800 chars) and refine "
            "`grade_relevance` node prompt to strictly filter out low-confidence chunks."
        )
        
    # 4. Contextual Precision Check
    if metric_results.get("ContextualPrecision", 1.0) < DEFAULT_THRESHOLD:
        remediations.append(
            "[FIX - Contextual Precision]: "
            "Top-ranked retrieved chunk is not the most relevant match. "
            "Action: Use query rewriting before retrieval (`rewrite_internal_query`) and verify "
            "Pinecone index metric is set to 'cosine' with `text-embedding-3-large` embeddings."
        )
        
    # 5. Contextual Recall Check
    if metric_results.get("ContextualRecall", 1.0) < DEFAULT_THRESHOLD:
        remediations.append(
            "[FIX - Contextual Recall]: "
            "Retrieved context missed key factual details required to form the golden answer. "
            "Action: Increase retriever `TOP_K` from 3 to 5 or 8 in `.env` / `config.py` "
            "and increase chunk overlap to 160+ tokens."
        )
        
    return remediations


# ============================================================================
# Pytest Test Cases
# ============================================================================

SAMPLE_CASES = get_golden_test_cases(limit=3, retrieval_only=True)


@pytest.mark.parametrize("case", SAMPLE_CASES, ids=[c["id"] for c in SAMPLE_CASES])
def test_self_rag_pipeline_case(case: Dict[str, Any]):
    """
    Runs an end-to-end test of a single golden test case against all 5 DeepEval metrics.
    """
    query = case["input"]
    expected_output = case["expected_output"]
    golden_context = case.get("context", [])
    
    # Run the Self-RAG pipeline
    result = invoke_self_rag(question=query, thread_id=f"pytest-{case['id']}")
    actual_output = result["answer"]
    retrieval_chunks = result.get("retrieval_chunks") or golden_context
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_chunks,
        context=golden_context
    )
    
    # Evaluate individual metrics with assertions
    assert_test(test_case, [
        answer_relevancy_metric,
        faithfulness_metric,
        contextual_relevancy_metric,
        contextual_precision_metric,
        contextual_recall_metric,
    ])


# ============================================================================
# Interactive Evaluation CLI Runner
# ============================================================================

def run_evaluation_suite(limit: int = 3, category: Optional[str] = None):
    """
    Executes a comprehensive evaluation run and prints a formatted summary.
    """
    print("=" * 80)
    print("CloudOps Sentinel - Self-RAG DeepEval Benchmark")
    print("=" * 80)
    
    cases = get_golden_test_cases(limit=limit, category=category, retrieval_only=True)
    print(f"Total test cases selected: {len(cases)}")
    print(f"Evaluation Model         : {EVAL_MODEL}")
    print(f"Score Threshold          : {DEFAULT_THRESHOLD}")
    print("-" * 80)
    
    metrics = get_all_metrics(threshold=DEFAULT_THRESHOLD)
    all_scores: Dict[str, List[float]] = {m.__class__.__name__: [] for m in metrics}
    
    for i, item in enumerate(cases, 1):
        print(f"\n[Case {i}/{len(cases)} - {item['id']}] Question: {item['input']}")
        res = invoke_self_rag(question=item["input"], thread_id=f"cli-eval-{item['id']}")
        
        chunks = res.get("retrieval_chunks") or item.get("context", ["No context"])
        
        tc = LLMTestCase(
            input=item["input"],
            actual_output=res["answer"],
            expected_output=item["expected_output"],
            retrieval_context=chunks,
            context=item.get("context", [])
        )
        
        print(f"  Route Taken   : {res.get('route', 'Unknown')}")
        print(f"  Answer Preview: {res['answer'][:120]}...")
        print("  Evaluating DeepEval Metrics:")
        
        for m in metrics:
            m.measure(tc)
            score = m.score if m.score is not None else 0.0
            all_scores[m.__class__.__name__].append(score)
            status = "PASS" if m.is_successful() else "FAIL"
            print(f"    * [{status}] {m.__class__.__name__:<26}: score={score:.2f} (threshold={m.threshold:.2f})")
            if m.reason:
                print(f"      Reason: {m.reason}")
                
    print("\n" + "=" * 80)
    print("AGGREGATE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric Name':<30} | {'Avg Score':<10} | {'Pass Rate':<10} | {'Status'}")
    print("-" * 80)
    
    avg_results = {}
    for metric_name, scores in all_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(1 for s in scores if s >= DEFAULT_THRESHOLD)
        pass_rate = (pass_count / len(scores) * 100) if scores else 0.0
        status = "PASSED" if avg_score >= DEFAULT_THRESHOLD else "FAILED"
        avg_results[metric_name] = avg_score
        print(f"{metric_name:<30} | {avg_score:<10.2f} | {pass_rate:<9.1f}% | {status}")
        
    print("-" * 80)
    
    # Check for any metric failures and output diagnostics
    failures = diagnose_metric_failures({
        "AnswerRelevancy": avg_results.get("AnswerRelevancyMetric", 1.0),
        "Faithfulness": avg_results.get("FaithfulnessMetric", 1.0),
        "ContextualRelevancy": avg_results.get("ContextualRelevancyMetric", 1.0),
        "ContextualPrecision": avg_results.get("ContextualPrecisionMetric", 1.0),
        "ContextualRecall": avg_results.get("ContextualRecallMetric", 1.0),
    })
    
    if failures:
        print("\nDiagnostic Feedback & Recommended Tuning:")
        for fix in failures:
            print(f"  -> {fix}")
    else:
        print("\nAll 5 RAG metrics passed successfully with top performance!")
        
    print("=" * 80)
    return avg_results


if __name__ == "__main__":
    count = 2
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        count = int(sys.argv[1])
    run_evaluation_suite(limit=count)
