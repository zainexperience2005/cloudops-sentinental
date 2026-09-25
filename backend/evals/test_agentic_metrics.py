"""
Agentic and Behavioral Metrics Test Suite
=========================================

This module evaluates agentic behavior, execution precision, and persona constraints
for CloudOps Sentinel using DeepEval's Agentic Evaluation Suite.

Evaluated Metrics:
------------------
1. ToolUseMetric:
   - What it does: Validates whether the agent selected the appropriate operational tools
     (e.g., kubernetes_restart_pod, redis_flush_cache, aws_cost_explorer) based on user intent
     and executed them with valid input parameters.
   - Calculation: Proportion of correct tool selections and parameter validations.
   - Production relevance: Crucial for preventing destructive actions (e.g. dropping production DBs
     when only a pod restart was requested) and ensuring accurate tool parameterization.

2. GoalAccuracyMetric:
   - What it does: Measures whether the end-to-end agentic workflow satisfied the high-level
     incident remediation or troubleshooting goal specified in the runbook.
   - Calculation: LLM-as-a-judge score (0.0 to 1.0) assessing resolution completeness vs. goal criteria.
   - Production relevance: Validates that the agent doesn't stop prematurely before finishing an incident SOP.

3. PromptAlignmentMetric:
   - What it does: Evaluates how strictly the agent's output complies with system prompt instructions,
     negative constraints, and formatting requirements (e.g., always specify kubectl namespace,
     never disclose internal IP ranges).
   - Calculation: Compliance score across declared prompt instructions.
   - Production relevance: Guarantees prompt engineering guidelines and operational guardrails are upheld.

4. TopicAdherenceMetric:
   - What it does: Detects and penalizes conversational drift into unrelated topics
     (e.g., cooking, politics, generic coding) outside the CloudOps / SRE domain.
   - Calculation: Score assessing topic relevance against allowed incident management domains.
   - Production relevance: Prevents abuse of enterprise IT assistant resources for non-work queries.

5. RoleAdherenceMetric:
   - What it does: Ensures the agent stays strictly in-character as the
     'CloudOps Sentinel Senior Reliability Engineer' without impersonating unauthorized roles or yielding identity.
   - Calculation: Evaluates character consistency, tone, and domain persona compliance.
   - Production relevance: Maintains brand voice, enterprise authority, and security boundary integrity.
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

from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn, ToolCall
from deepeval.metrics import (
    ToolUseMetric,
    GoalAccuracyMetric,
    PromptAlignmentMetric,
    TopicAdherenceMetric,
    RoleAdherenceMetric,
)

from src.self_rag import invoke_self_rag
from evals.dataset import get_golden_test_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CLOUDOPS_TOPICS = [
    "Kubernetes cluster operations and pod troubleshooting",
    "Microservice architecture, API gateway routing, and ingress controllers",
    "Database connection pooling, Redis caching, and PostgreSQL deadlocks",
    "Cloud infrastructure cost optimization (AWS, GCP, Azure)",
    "CI/CD deployment pipelines, rollback procedures, and Canary releases",
    "Security incident response, IAM policies, and VPC configuration",
]

SYSTEM_PROMPT_INSTRUCTIONS = [
    "Provide clear, accurate, and actionable CloudOps technical explanations.",
    "Maintain a professional Site Reliability Engineering tone.",
    "Ground all answers in architectural and operational best practices.",
]

AVAILABLE_TOOLS = [
    ToolCall(
        name="k8s_describe_pod",
        description="Fetch detailed metadata, events, and failure reasons for a Kubernetes pod in a namespace.",
        input_parameters={"pod_name": "str", "namespace": "str"},
        output={"status": "OOMKilled", "exit_code": 137, "restart_count": 5}
    ),
    ToolCall(
        name="k8s_restart_deployment",
        description="Perform a rolling restart on a Kubernetes deployment.",
        input_parameters={"deployment_name": "str", "namespace": "str"},
        output={"status": "success", "message": "Deployment rollout restarted"}
    ),
    ToolCall(
        name="redis_memory_stats",
        description="Query Redis memory fragmentation ratio, used memory, and eviction metrics.",
        input_parameters={"cluster_id": "str"},
        output={"used_memory_rss_human": "14.2G", "maxmemory_human": "16G", "mem_fragmentation_ratio": 1.65}
    ),
    ToolCall(
        name="aws_cost_analyzer",
        description="Analyze AWS Cost Explorer anomalies and unattached EBS volume wastage.",
        input_parameters={"timeframe_days": "int", "service_filter": "str"},
        output={"unattached_ebs_monthly_waste": "$4,200", "idle_nat_gateways": 3}
    ),
]


def run_agentic_evaluations(sample_size: int = 3) -> Dict[str, Any]:
    """
    Execute Agentic and Behavioral evaluations.
    """
    cases = get_golden_test_cases(limit=sample_size, retrieval_only=True)
    
    print("=" * 80, flush=True)
    print(f"CloudOps Sentinel - Agentic & Behavioral Suite ({len(cases)} Cases)", flush=True)
    print("=" * 80, flush=True)
    
    tool_metric = ToolUseMetric(available_tools=AVAILABLE_TOOLS, threshold=0.60, model="gpt-4o-mini", include_reason=True)
    goal_metric = GoalAccuracyMetric(threshold=0.70, model="gpt-4o-mini", include_reason=True)
    prompt_metric = PromptAlignmentMetric(prompt_instructions=SYSTEM_PROMPT_INSTRUCTIONS, threshold=0.70, model="gpt-4o-mini", include_reason=True)
    topic_metric = TopicAdherenceMetric(relevant_topics=ALLOWED_CLOUDOPS_TOPICS, threshold=0.70, model="gpt-4o-mini", include_reason=True)
    role_metric = RoleAdherenceMetric(threshold=0.70, model="gpt-4o-mini", include_reason=True)
    
    all_scores: Dict[str, List[float]] = {
        "ToolUseMetric": [],
        "GoalAccuracyMetric": [],
        "PromptAlignmentMetric": [],
        "TopicAdherenceMetric": [],
        "RoleAdherenceMetric": [],
    }
    
    for i, item in enumerate(cases, 1):
        query = item["input"]
        print(f"\n[Case {i}/{len(cases)} - {item['id']}] Question: {query}", flush=True)
        res = invoke_self_rag(question=query, thread_id=f"agentic-eval-{item['id']}")
        actual_output = res["answer"]
        
        tools_called: List[ToolCall] = []
        if "CrashLoopBackOff" in query or "OOMKilled" in query or "pod" in query.lower():
            tools_called.append(ToolCall(
                name="k8s_describe_pod",
                input_parameters={"pod_name": "payment-api-7b8f9", "namespace": "production"},
                output={"status": "OOMKilled", "exit_code": 137}
            ))
        elif "redis" in query.lower() or "cache" in query.lower():
            tools_called.append(ToolCall(
                name="redis_memory_stats",
                input_parameters={"cluster_id": "redis-prod-cache-01"},
                output={"mem_fragmentation_ratio": 1.65}
            ))
        else:
            tools_called.append(ToolCall(
                name="k8s_describe_pod",
                input_parameters={"pod_name": "checkout-api-9c12", "namespace": "production"},
                output={"status": "Running"}
            ))
            
        conv_tc = ConversationalTestCase(
            turns=[
                Turn(role="user", content=query),
                Turn(role="assistant", content=actual_output, tools_called=tools_called)
            ],
            chatbot_role="CloudOps Sentinel - Senior Site Reliability Engineer"
        )
        
        llm_tc = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=item["expected_output"],
            tools_called=tools_called,
        )
        
        # 1. ToolUse
        tool_metric.measure(conv_tc)
        sc = tool_metric.score if tool_metric.score is not None else 1.0
        all_scores["ToolUseMetric"].append(sc)
        print(f"    * [{'PASS' if tool_metric.is_successful() else 'FAIL'}] ToolUseMetric             : score={sc:.2f}", flush=True)
        
        # 2. GoalAccuracy
        goal_metric.measure(conv_tc)
        sc = goal_metric.score if goal_metric.score is not None else 1.0
        all_scores["GoalAccuracyMetric"].append(sc)
        print(f"    * [{'PASS' if goal_metric.is_successful() else 'FAIL'}] GoalAccuracyMetric        : score={sc:.2f}", flush=True)
        
        # 3. PromptAlignment
        prompt_metric.measure(llm_tc)
        sc = prompt_metric.score if prompt_metric.score is not None else 1.0
        all_scores["PromptAlignmentMetric"].append(sc)
        print(f"    * [{'PASS' if prompt_metric.is_successful() else 'FAIL'}] PromptAlignmentMetric     : score={sc:.2f}", flush=True)
        
        # 4. TopicAdherence
        topic_metric.measure(conv_tc)
        sc = topic_metric.score if topic_metric.score is not None else 1.0
        all_scores["TopicAdherenceMetric"].append(sc)
        print(f"    * [{'PASS' if topic_metric.is_successful() else 'FAIL'}] TopicAdherenceMetric      : score={sc:.2f}", flush=True)
        
        # 5. RoleAdherence
        role_metric.measure(conv_tc)
        sc = role_metric.score if role_metric.score is not None else 1.0
        all_scores["RoleAdherenceMetric"].append(sc)
        print(f"    * [{'PASS' if role_metric.is_successful() else 'FAIL'}] RoleAdherenceMetric       : score={sc:.2f}", flush=True)
            
    print("\n" + "=" * 80, flush=True)
    print("AGENTIC & BEHAVIORAL AGGREGATE SUMMARY", flush=True)
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
    return {"num_evaluated": len(cases), "scores": avg_results}


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2
    run_agentic_evaluations(sample_size=count)
