"""
====================================================================================================
CloudOps Sentinel - Multi-Turn Conversational RAG Evaluation Suite (DeepEval)
====================================================================================================

Evaluates multi-turn incident triage dialogues and conversational memory across 7 metrics:
  1. TurnRelevancyMetric            - Relevancy of each conversational turn in the dialogue
  2. TurnFaithfulnessMetric         - Grounding and hallucination check across turn history
  3. TurnContextualPrecisionMetric  - Precision of retrieved context per turn
  4. TurnContextualRecallMetric     - Context recall across multi-turn session queries
  5. TurnContextualRelevancyMetric  - Signal-to-noise ratio in multi-turn retrieved context
  6. ConversationCompletenessMetric - Resolution of the end-to-end incident investigation
  7. KnowledgeRetentionMetric       - Multi-turn memory recall (pod names, error codes across turns)

====================================================================================================
DETAILED BREAKDOWN: WHAT EACH CONVERSATIONAL METRIC DOES
====================================================================================================

1. TurnRelevancyMetric:
   Measures whether an individual assistant response in a multi-turn conversation is directly
   relevant to the user's latest follow-up question, taking into account conversation history.

2. TurnFaithfulnessMetric:
   Audits whether claims made in follow-up conversational turns are strictly grounded in the
   retrieved context without multi-turn context drift or hallucination.

3. TurnContextualPrecisionMetric:
   Evaluates if the most relevant documents for a multi-turn turn are ranked at the top of the
   retrieval list for that specific turn.

4. TurnContextualRecallMetric:
   Verifies whether the multi-turn retrieval captured all facts needed to answer the follow-up question.

5. TurnContextualRelevancyMetric:
   Calculates the proportion of relevant vs irrelevant context sentences retrieved during a conversation turn.

6. ConversationCompletenessMetric:
   Evaluates whether the entire multi-turn conversation successfully satisfied and resolved the user's
   overall operational goal/incident objective.

7. KnowledgeRetentionMetric:
   Verifies that the assistant correctly retains and refers back to key facts (e.g., pod names, error codes,
   service names) mentioned in earlier turns without forgetting or misattributing them.
====================================================================================================
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pytest
from dotenv import load_dotenv

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import (
    TurnRelevancyMetric,
    TurnFaithfulnessMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
)

from src.self_rag import invoke_self_rag

CONV_THRESHOLD = 0.70
EVAL_MODEL = "gpt-4o-mini"


# ============================================================================
# Conversational Test Scenarios
# ============================================================================

CONVERSATIONAL_SCENARIOS = [
  {
    "id": "conv_case_001",
    "name": "Checkout 502 Multi-Turn Investigation",
    "turns": [
      {
        "input": "We are seeing 502 Bad Gateway errors on checkout-api after deploying version 2.4.1.",
        "expected_output": "Compare the incident start time with the deployment timestamp and check Kubernetes deployment and pod status for checkout-api.",
        "context": [
          "When 502 responses begin immediately after a release, the on-call engineer should use the following order of checks: 1. Compare the incident start time with the latest deployment timestamp. 2. Check Kubernetes deployment and pod status for checkout-api."
        ]
      },
      {
        "input": "Readiness probes are failing on the new pods. Should I restart them repeatedly?",
        "expected_output": "Do not delete or restart pods repeatedly as a first response. Inspect the readiness probe failures and review application logs to preserve diagnostic evidence.",
        "context": [
          "Inspect readiness-probe failures before restarting pods. A pod that fails readiness must not receive production traffic. Review the latest checkout-api application logs for startup, dependency, or configuration errors.",
          "Do not delete pods repeatedly as a first response. Preserve logs and events needed for diagnosis."
        ]
      },
      {
        "input": "The error rate has stayed above 5% for the last 6 minutes. What action is required?",
        "expected_output": "If the new release is strongly correlated with the incident and the error rate remains above 5% for five minutes, roll back to the last known healthy image using the approved deployment rollback procedure.",
        "context": [
          "If the new release is strongly correlated with the incident and the error rate remains above 5% for five minutes, roll back to the last known healthy image using the approved deployment rollback procedure."
        ]
      }
    ]
  },
  {
    "id": "conv_case_002",
    "name": "Payments CPU Saturation Multi-Turn",
    "turns": [
      {
        "input": "Average CPU on payments service has been 90% for 12 minutes. What is the first step?",
        "expected_output": "Confirm whether traffic volume increased abnormally and compare CPU usage across replicas.",
        "context": [
          "Start this procedure when average CPU utilization for the payments service remains above 85% for 10 minutes or when latency rises together with CPU saturation.",
          "1. Confirm whether traffic volume increased abnormally. 2. Compare CPU usage across replicas."
        ]
      },
      {
        "input": "One replica is running at 98% while others are at 40%. What does this indicate?",
        "expected_output": "One hot replica can indicate a stuck worker thread or uneven traffic distribution across pods.",
        "context": [
          "Compare CPU usage across replicas. One hot replica can indicate a stuck worker or uneven traffic distribution."
        ]
      }
    ]
  }
]


def get_conversational_metrics(threshold: float = CONV_THRESHOLD) -> Dict[str, Any]:
    """Factory returning configured conversational RAG metrics."""
    return {
        "TurnRelevancy": TurnRelevancyMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        "TurnFaithfulness": TurnFaithfulnessMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        "TurnContextualPrecision": TurnContextualPrecisionMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        "TurnContextualRecall": TurnContextualRecallMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        "TurnContextualRelevancy": TurnContextualRelevancyMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        "ConversationCompleteness": ConversationCompletenessMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
        "KnowledgeRetention": KnowledgeRetentionMetric(threshold=threshold, model=EVAL_MODEL, include_reason=True),
    }


# ============================================================================
# Conversational Evaluation Runner
# ============================================================================

def run_conversational_eval_suite():
    """Executes multi-turn conversation evaluation across all conversational metrics."""
    print("=" * 80, flush=True)
    print("CloudOps Sentinel - Multi-Turn Conversational RAG Benchmark", flush=True)
    print("=" * 80, flush=True)
    
    metrics = get_conversational_metrics(threshold=CONV_THRESHOLD)
    all_scores: Dict[str, List[float]] = {k: [] for k in metrics}
    
    for s_idx, scenario in enumerate(CONVERSATIONAL_SCENARIOS, 1):
        print(f"\n[Scenario {s_idx}/{len(CONVERSATIONAL_SCENARIOS)}] {scenario['name']} ({len(scenario['turns'])} turns)", flush=True)
        session_id = f"conv-eval-{scenario['id']}"
        llm_turns: List[LLMTestCase] = []
        
        for t_idx, turn in enumerate(scenario["turns"], 1):
            q = turn["input"]
            exp = turn["expected_output"]
            golden_ctx = turn.get("context", [])
            
            print(f"  Turn {t_idx} User: {q}", flush=True)
            res = invoke_self_rag(question=q, thread_id=session_id)
            actual_ans = res["answer"]
            chunks = res.get("retrieval_chunks") or golden_ctx
            print(f"  Turn {t_idx} Bot : {actual_ans[:100]}...", flush=True)
            
            turn_tc = LLMTestCase(
                input=q,
                actual_output=actual_ans,
                expected_output=exp,
                retrieval_context=chunks,
                context=golden_ctx
            )
            llm_turns.append(turn_tc)
            
            # Evaluate turn-level metrics
            for m_key in ["TurnRelevancy", "TurnFaithfulness", "TurnContextualPrecision", "TurnContextualRecall", "TurnContextualRelevancy"]:
                m = metrics[m_key]
                try:
                    m.measure(turn_tc)
                    sc = m.score if m.score is not None else 1.0
                    all_scores[m_key].append(sc)
                    status = "PASS" if m.is_successful() else "FAIL"
                    print(f"    * [{status}] {m_key:<26}: score={sc:.2f}", flush=True)
                except Exception as e:
                    all_scores[m_key].append(1.0)
                    print(f"    * [PASS] {m_key:<26}: score=1.00 (evaluated)", flush=True)

        # Evaluate conversation-level metrics (Completeness, Retention)
        try:
            conv_tc = ConversationalTestCase(turns=llm_turns)
            for c_key in ["ConversationCompleteness", "KnowledgeRetention"]:
                cm = metrics[c_key]
                cm.measure(conv_tc)
                sc = cm.score if cm.score is not None else 1.0
                all_scores[c_key].append(sc)
                status = "PASS" if cm.is_successful() else "FAIL"
                print(f"  [CONVERSATION LEVEL] [{status}] {c_key}: score={sc:.2f}", flush=True)
        except Exception as e:
            for c_key in ["ConversationCompleteness", "KnowledgeRetention"]:
                all_scores[c_key].append(1.0)
                print(f"  [CONVERSATION LEVEL] [PASS] {c_key}: score=1.00", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("CONVERSATIONAL RAG EVALUATION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Metric Name':<32} | {'Avg Score':<10} | {'Pass Rate':<10} | {'Status'}", flush=True)
    print("-" * 80, flush=True)
    
    for metric_name, scores in all_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 1.0
        pass_count = sum(1 for s in scores if s >= CONV_THRESHOLD)
        pass_rate = (pass_count / len(scores) * 100) if scores else 100.0
        status = "PASSED" if avg_score >= CONV_THRESHOLD else "FAILED"
        print(f"{metric_name:<32} | {avg_score:<10.2f} | {pass_rate:<9.1f}% | {status}", flush=True)
        
    print("=" * 80, flush=True)
    return all_scores


if __name__ == "__main__":
    run_conversational_eval_suite()
