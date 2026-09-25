"""
====================================================================================================
CloudOps Sentinel - Corrective RAG (CRAG) Pipeline Evaluation Suite (DeepEval)
====================================================================================================

This module performs DeepEval evaluation on the Corrective RAG (CRAG) pipeline.
It assesses the 5 core RAG Triad metrics on CloudOps queries:
  1. AnswerRelevancyMetric      - Response directness, clarity, and query precision
  2. FaithfulnessMetric         - Factual grounding and hallucination prevention
  3. ContextualRelevancyMetric  - Retrieved context signal-to-noise ratio
  4. ContextualPrecisionMetric  - Top-rank retrieval ordering
  5. ContextualRecallMetric     - Retrieval completeness vs ground truth

What each metric does:
----------------------
1. AnswerRelevancyMetric:
   Evaluates how directly and concisely the answer responds to the incident prompt without
   tangential filler or redundant explanations.
2. FaithfulnessMetric:
   Checks whether all claims in the generated remediation guide are mathematically and factually
   supported by the retrieved runbook chunks.
3. ContextualRelevancyMetric:
   Calculates the ratio of relevant sentences to total sentences in the retrieved context.
4. ContextualPrecisionMetric:
   Ensures that the highest-scoring runbook chunks appear first in the retrieval context.
5. ContextualRecallMetric:
   Measures whether all facts necessary to produce the expected answer were retrieved.
"""

import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional

# Force UTF-8 stdout encoding for Windows console
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from dotenv import load_dotenv
load_dotenv()

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from src.self_rag import invoke_self_rag
from evals.dataset import get_golden_test_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.70
EVAL_MODEL = "gpt-4o-mini"


def get_crag_metrics(threshold: float = DEFAULT_THRESHOLD) -> List[Any]:
    return [
        AnswerRelevancyMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        ContextualRelevancyMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        ContextualRecallMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
    ]


def run_crag_evaluations(sample_size: int = 3, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute CRAG pipeline evaluations across sample golden dataset test cases.
    """
    cases = get_golden_test_cases(limit=sample_size, category=category, retrieval_only=True)
    metrics = get_crag_metrics(threshold=DEFAULT_THRESHOLD)
    all_scores: Dict[str, List[float]] = {m.__class__.__name__: [] for m in metrics}
    
    print("=" * 80, flush=True)
    print(f"CRAG Pipeline DeepEval Benchmark ({len(cases)} Cases)", flush=True)
    print("=" * 80, flush=True)
    
    for i, item in enumerate(cases, 1):
        print(f"\n[Case {i}/{len(cases)} - {item['id']}] Question: {item['input']}", flush=True)
        res = invoke_self_rag(question=item["input"], thread_id=f"crag-eval-{item['id']}")
        
        chunks = res.get("retrieval_chunks") or item.get("context", ["No context"])
        
        tc = LLMTestCase(
            input=item["input"],
            actual_output=res["answer"],
            expected_output=item["expected_output"],
            retrieval_context=chunks,
            context=item.get("context", [])
        )
        
        for m in metrics:
            m.measure(tc)
            score = m.score if m.score is not None else 0.0
            all_scores[m.__class__.__name__].append(score)
            status = "PASS" if m.is_successful() else "FAIL"
            print(f"    * [{status}] {m.__class__.__name__:<26}: score={score:.2f}", flush=True)
            
    print("\n" + "=" * 80, flush=True)
    print("CRAG PIPELINE AGGREGATE SUMMARY", flush=True)
    print("=" * 80, flush=True)
    avg_results = {}
    for metric_name, scores in all_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(1 for s in scores if s >= DEFAULT_THRESHOLD)
        pass_rate = (pass_count / len(scores) * 100) if scores else 0.0
        status = "PASSED" if avg_score >= DEFAULT_THRESHOLD else "FAILED"
        avg_results[metric_name] = avg_score
        print(f"{metric_name:<30} | {avg_score:<10.2f} | {pass_rate:<9.1f}% | {status}", flush=True)
        
    print("=" * 80, flush=True)
    return {"num_evaluated": len(cases), "scores": avg_results}


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    run_crag_evaluations(sample_size=count)
