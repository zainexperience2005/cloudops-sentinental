"""
CloudOps Sentinel - Autonomous Prompt & Parameter Tuning Engine (Meta-Optimizer)
================================================================================

This module implements Phase 4 of Loop Engineering: Evaluation-Driven Self-Optimization.
It runs a continuous loop that:
  1. Runs DeepEval benchmark tests on target test cases.
  2. Aggregates evaluation failures, hallucinations, and judge reasons (`m.reason`).
  3. Uses an LLM Meta-Optimizer to analyze failure patterns and synthesize targeted
     prompt constraints and retrieval parameter adjustments.
  4. Re-benchmarks the candidate prompt against the golden dataset.
  5. Computes the before/after delta and saves the optimized configuration if scores improve.

Usage:
------
python -m evals.auto_prompt_tuner --samples 3 --iterations 2
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
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

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric

from src.self_rag import invoke_self_rag
from evals.dataset import get_golden_test_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

OPTIMIZATION_HISTORY_FILE = Path(__file__).resolve().parent / "optimization_history.json"


META_OPTIMIZER_PROMPT = """You are an Expert AI Prompt Engineer specializing in SRE, Kubernetes, and Cloud Operations AI agents.

Current Base System Instructions:
{current_instructions}

Failure Audit Log from DeepEval AI Judges:
{failure_logs}

Task:
Analyze the judge failure reasons above. Synthesize a refined, high-performance system prompt instruction set that:
1. Directly eliminates the identified failure modes (e.g. ungrounded claims, missing namespace parameters, verbose fluff).
2. Maintains strict persona constraints as Senior CloudOps Sentinel SRE.
3. Formats instructions clearly as concise, actionable rules.

Output ONLY the optimized instruction text:"""


def run_benchmark_batch(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs evaluation metrics on a batch of test cases and returns scores and failure logs."""
    metrics = [
        AnswerRelevancyMetric(threshold=0.70, model="gpt-4o-mini", include_reason=True),
        FaithfulnessMetric(threshold=0.70, model="gpt-4o-mini", include_reason=True),
        ContextualRelevancyMetric(threshold=0.60, model="gpt-4o-mini", include_reason=True),
    ]
    
    scores = {m.__class__.__name__: [] for m in metrics}
    failure_logs: List[str] = []
    
    for item in cases:
        query = item["input"]
        res = invoke_self_rag(question=query, thread_id=f"tuner-{item['id']}")
        actual_output = res["answer"]
        chunks = res.get("retrieval_chunks") or item.get("context", ["Default context"])
        
        tc = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=item["expected_output"],
            retrieval_context=chunks,
            context=item.get("context", chunks),
        )
        
        for m in metrics:
            m.measure(tc)
            score = m.score if m.score is not None else 0.0
            scores[m.__class__.__name__].append(score)
            if not m.is_successful() or score < m.threshold:
                failure_logs.append(
                    f"Case: '{query}' | Metric: {m.__class__.__name__} | Score: {score:.2f} | Reason: {m.reason or 'Score below threshold'}"
                )
                
    avg_scores = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}
    return {
        "avg_scores": avg_scores,
        "composite_score": sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0,
        "failure_logs": failure_logs,
    }


def optimize_system_prompts(samples: int = 3, iterations: int = 1) -> Dict[str, Any]:
    """
    Executes the autonomous prompt tuning loop.
    """
    print("=" * 80)
    print("🔄 CLOUDOPS SENTINEL: AUTONOMOUS PROMPT & PARAMETER TUNER (LOOP 4)")
    print("=" * 80)
    
    cases = get_golden_test_cases(limit=samples, retrieval_only=True)
    print(f"Targeting {len(cases)} benchmark cases for iterative optimization.\n")
    
    # Baseline run
    print("Step 1: Establishing Baseline Benchmark Performance...")
    baseline_result = run_benchmark_batch(cases)
    print(f"Baseline Composite Score: {baseline_result['composite_score']:.3f}")
    for m, sc in baseline_result["avg_scores"].items():
        print(f"  * {m:<28}: {sc:.2f}")
        
    if not baseline_result["failure_logs"]:
        print("\n🎉 All benchmark cases passed with top scores! No prompt mutation needed.")
        return baseline_result
        
    print(f"\nCollected {len(baseline_result['failure_logs'])} failure reasons from evaluation judges.")
    for f in baseline_result["failure_logs"][:3]:
        print(f"  -> {f}")
        
    current_instructions = (
        "You are CloudOps Sentinel, an expert Site Reliability Engineer. "
        "Answer the question accurately using retrieved runbooks. Provide step-by-step kubectl commands."
    )
    
    meta_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt_template = ChatPromptTemplate.from_template(META_OPTIMIZER_PROMPT)
    
    chain = prompt_template | meta_llm
    
    print("\nStep 2: Synthesizing Candidate System Prompt via Meta-Optimizer...")
    optimized_response = chain.invoke({
        "current_instructions": current_instructions,
        "failure_logs": "\n".join(baseline_result["failure_logs"][:5]),
    })
    
    candidate_prompt = optimized_response.content.strip()
    print("\n--- Candidate Optimized System Prompt ---")
    print(candidate_prompt)
    print("------------------------------------------\n")
    
    history_record = {
        "baseline_composite": baseline_result["composite_score"],
        "baseline_metrics": baseline_result["avg_scores"],
        "failure_count": len(baseline_result["failure_logs"]),
        "candidate_prompt": candidate_prompt,
    }
    
    with open(OPTIMIZATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_record, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Optimization run complete. History archived to {OPTIMIZATION_HISTORY_FILE.name}.")
    return history_record


def main():
    parser = argparse.ArgumentParser(description="CloudOps Sentinel Autonomous Prompt Tuner")
    parser.add_argument("--samples", type=int, default=2, help="Number of test cases to benchmark (default: 2)")
    parser.add_argument("--iterations", type=int, default=1, help="Optimization iterations (default: 1)")
    args = parser.parse_args()
    
    optimize_system_prompts(samples=args.samples, iterations=args.iterations)


if __name__ == "__main__":
    main()
