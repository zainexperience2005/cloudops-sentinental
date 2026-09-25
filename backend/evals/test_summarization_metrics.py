"""
Summarization and Hallucination Metrics Test Suite
==================================================

This module assesses summarization quality, information retention, and factual grounding
for incident postmortems, alert digests, and SOP summaries using DeepEval.

Evaluated Metrics:
------------------
1. SummarizationMetric:
   - What it does: Quantifies how effectively an LLM-generated incident summary or alert digest
     captures key factual statements (truths) from the source runbook or incident log without
     omitting vital details or introducing extraneous claims.
   - Core Sub-Scores:
     * Alignment Score: Does the summary contain only facts asserted in the original document?
     * Inclusion / Coverage Score: Does the summary cover all key points requested in assessment questions?
   - Calculation: Harmonic / weighted combination of alignment and inclusion scores.
   - Production relevance: Essential for automated SRE handovers and executive incident summaries where
     missing a critical root cause or action item can lead to recurring outages.

2. HallucinationMetric:
   - What it does: Compares the generated response strictly against retrieved runbook context to
     detect fabricated commands, fictional configuration flags, hallucinated IP addresses, or ungrounded claims.
   - Calculation: Proportion of ungrounded sentences detected by LLM judge (Score = 1 - hallucination_rate).
   - Production relevance: Zero tolerance in infrastructure automation—a hallucinated kubectl flag or
     syntax error can crash production deployments or cause cascading failures.
"""

import sys
import os
import json
import logging
from typing import List, Dict, Any

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
    SummarizationMetric,
    HallucinationMetric,
)

from src.self_rag import invoke_self_rag
from evals.dataset import get_golden_test_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POSTMORTEM_SOURCE_DOCUMENTS = [
    {
        "title": "Incident INC-8492: Production Redis Memory Exhaustion Postmortem",
        "input": "Summarize the root cause, immediate mitigation, and long-term action items for Incident INC-8492.",
        "source_text": (
            "Incident INC-8492 occurred on 2026-09-15 at 04:12 UTC. "
            "Root Cause: High memory fragmentation ratio (1.82) in Redis Cluster node-01 due to rapid key churn "
            "and jemalloc memory allocation overhead without active defragmentation enabled. "
            "Immediate Mitigation: SRE on-call executed 'CONFIG SET activedefrag yes' and performed a graceful failover "
            "to replica-02, restoring write latency to sub-2ms within 8 minutes. "
            "Long-Term Action Items: 1) Enable 'activedefrag yes' permanently in Terraform Redis parameter group. "
            "2) Set maxmemory-policy to volatile-lru with a 75% memory alert threshold in CloudWatch."
        ),
        "assessment_questions": [
            "What was the root cause of the Redis memory exhaustion?",
            "What command was run for immediate mitigation?",
            "What are the long-term action items?",
        ],
    },
    {
        "title": "Incident INC-9104: Kubernetes Ingress 504 Gateway Timeout Outage",
        "input": "Summarize the failure cascade and remediation steps for Incident INC-9104.",
        "source_text": (
            "Incident INC-9104 occurred during a Canary deployment of Auth-Service v2.4.0. "
            "Root Cause: Missing backend keep-alive timeout configuration on NGINX Ingress controller caused "
            "premature connection termination under sustained load of 15,000 RPS, resulting in 504 Gateway Timeouts. "
            "Immediate Mitigation: Traffic was rolled back 100% to stable v2.3.9 via Argo Rollouts. "
            "Long-Term Action Items: 1) Standardize proxy-connect-timeout and proxy-read-timeout to 75s in Ingress ConfigMap. "
            "2) Implement automated load test gates in GitHub Actions before promoting canary releases."
        ),
        "assessment_questions": [
            "What caused the 504 Gateway Timeouts on Ingress?",
            "How was the outage immediately mitigated?",
            "What long-term changes were made to Ingress configuration?",
        ],
    }
]


def run_summarization_evaluations(sample_size: int = 2) -> Dict[str, Any]:
    """
    Execute Summarization and Hallucination metric evaluations.
    """
    print("=" * 80, flush=True)
    print("CloudOps Sentinel - Summarization & Hallucination Benchmark", flush=True)
    print("=" * 80, flush=True)
    
    summarization_metric = SummarizationMetric(
        threshold=0.70,
        model="gpt-4o-mini",
        n=5,
        include_reason=True,
    )
    hallucination_metric = HallucinationMetric(
        threshold=0.70,
        model="gpt-4o-mini",
        include_reason=True,
    )
    
    all_scores: Dict[str, List[float]] = {
        "SummarizationMetric": [],
        "HallucinationMetric": []
    }
    
    # 1. Test Summarization
    for doc in POSTMORTEM_SOURCE_DOCUMENTS[:sample_size]:
        prompt = f"Source Incident Report:\n{doc['source_text']}\n\nTask: {doc['input']}"
        print(f"\n[Summarization] Task: {doc['input']}", flush=True)
        res = invoke_self_rag(question=prompt, thread_id="eval-summary-doc")
        actual_output = res["answer"]
        
        tc = LLMTestCase(
            input=doc["source_text"],
            actual_output=actual_output,
            context=[doc["source_text"]],
            retrieval_context=[doc["source_text"]],
        )
        
        doc_metric = SummarizationMetric(
            threshold=0.70,
            model="gpt-4o-mini",
            assessment_questions=doc.get("assessment_questions"),
            include_reason=True,
        )
        doc_metric.measure(tc)
        sc = doc_metric.score if doc_metric.score is not None else 0.0
        all_scores["SummarizationMetric"].append(sc)
        status = "PASS" if doc_metric.is_successful() else "FAIL"
        print(f"    * [{status}] SummarizationMetric: score={sc:.2f}", flush=True)
        
    # 2. Test Hallucination on RAG knowledge
    cases = get_golden_test_cases(limit=sample_size, retrieval_only=True)
    for item in cases:
        print(f"\n[Hallucination Check] Question: {item['input']}", flush=True)
        res = invoke_self_rag(question=item["input"], thread_id=f"eval-hallucination-{item['id']}")
        actual_output = res["answer"]
        chunks = res.get("retrieval_chunks") or item.get("context", ["Default context"])
        
        tc = LLMTestCase(
            input=item["input"],
            actual_output=actual_output,
            expected_output=item["expected_output"],
            context=chunks,
            retrieval_context=chunks,
        )
        
        hallucination_metric.measure(tc)
        sc = hallucination_metric.score if hallucination_metric.score is not None else 0.0
        all_scores["HallucinationMetric"].append(sc)
        status = "PASS" if hallucination_metric.is_successful() else "FAIL"
        print(f"    * [{status}] HallucinationMetric: score={sc:.2f}", flush=True)
        
    print("\n" + "=" * 80, flush=True)
    print("SUMMARIZATION & FACTUAL GROUNDING SUMMARY", flush=True)
    print("=" * 80, flush=True)
    avg_results = {}
    for metric_name, scores in all_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(1 for s in scores if s >= 0.70)
        pass_rate = (pass_count / len(scores) * 100) if scores else 0.0
        status = "PASSED" if avg_score >= 0.70 else "FAILED"
        avg_results[metric_name] = avg_score
        print(f"{metric_name:<30} | {avg_score:<10.2f} | {pass_rate:<9.1f}% | {status}", flush=True)
        
    print("=" * 80, flush=True)
    return {"num_evaluated": len(POSTMORTEM_SOURCE_DOCUMENTS[:sample_size]) + len(cases), "scores": avg_results}


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
    run_summarization_evaluations(sample_size=count)
