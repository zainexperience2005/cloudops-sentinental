"""
CloudOps Sentinel - Super Evaluation Runner
============================================

This master evaluation runner orchestrates all specialized evaluation suites across
the entire AI system lifecycle:

Categories:
-----------
1. 'rag':
   - Suite: test_self_rag_pipeline.py / test_crag_pipeline.py
   - Metrics: AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric,
              ContextualPrecisionMetric, ContextualRecallMetric
2. 'conversational':
   - Suite: test_conversational_rag_metrics.py
   - Metrics: TurnRelevancyMetric, TurnFaithfulnessMetric, TurnContextualPrecisionMetric,
              TurnContextualRecallMetric, TurnContextualRelevancyMetric,
              ConversationCompletenessMetric, KnowledgeRetentionMetric
3. 'safety':
   - Suite: test_safety_evals.py
   - Metrics: BiasMetric, ToxicityMetric, NonAdviceMetric, MisuseMetric,
              PIILeakageMetric, RoleViolationMetric
4. 'agentic':
   - Suite: test_agentic_metrics.py
   - Metrics: ToolUseMetric, GoalAccuracyMetric, PromptAlignmentMetric,
              TopicAdherenceMetric, RoleAdherenceMetric
5. 'summarization':
   - Suite: test_summarization_metrics.py
   - Metrics: SummarizationMetric, HallucinationMetric
6. 'ragas':
   - Suite: test_ragas_pipeline.py
   - Metrics: Ragas faithfulness, answer_relevancy, context_precision, context_recall
7. 'all':
   - Executes all categories sequentially and generates a unified summary dashboard.

Usage:
------
python -m evals.run_all_evals --category all --samples 3
python -m evals.run_all_evals --category rag --samples 5
python -m evals.run_all_evals --category safety --samples 4
python -m evals.run_all_evals --category conversational --samples 2
python -m evals.run_all_evals --category agentic --samples 2
python -m evals.run_all_evals --category summarization --samples 2
python -m evals.run_all_evals --category ragas --samples 3
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def print_banner(category: str):
    print("\n" + "=" * 80)
    print(f"🚀 RUNNING CLOUDOPS SENTINEL EVALUATION SUITE: [{category.upper()}]")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n", flush=True)


def execute_category(category: str, sample_size: int) -> Dict[str, Any]:
    """Execute a single evaluation category and return structured results."""
    category = category.lower()
    
    if category in ["rag", "crag", "self-rag"]:
        from evals.test_self_rag_pipeline import run_evaluation_suite
        print_banner("Single-Turn RAG Triad Pipeline")
        res = run_evaluation_suite(limit=sample_size)
        return {"category": "rag", "result": {"scores": res, "num_evaluated": sample_size}}
        
    elif category in ["conversational", "multi-turn", "conv"]:
        from evals.test_conversational_rag_metrics import run_conversational_eval_suite
        print_banner("Conversational Multi-Turn RAG")
        res = run_conversational_eval_suite()
        return {"category": "conversational", "result": {"scores": res, "num_evaluated": sample_size}}
        
    elif category in ["safety", "guardrails"]:
        from evals.test_safety_evals import run_safety_eval_suite
        print_banner("Safety & Guardrail Compliance")
        res = run_safety_eval_suite()
        return {"category": "safety", "result": {"scores": res, "num_evaluated": 12}}
        
    elif category in ["agentic", "behavioral", "tools"]:
        from evals.test_agentic_metrics import run_agentic_evaluations
        print_banner("Agentic & Behavioral Operations")
        res = run_agentic_evaluations(sample_size=sample_size)
        return {"category": "agentic", "result": res}
        
    elif category in ["summarization", "hallucination"]:
        from evals.test_summarization_metrics import run_summarization_evaluations
        print_banner("Summarization & Hallucination")
        res = run_summarization_evaluations(sample_size=sample_size)
        return {"category": "summarization", "result": res}
        
    elif category in ["ragas"]:
        from evals.test_ragas_pipeline import run_ragas_evaluations
        print_banner("Ragas Evaluation Framework")
        res = run_ragas_evaluations(sample_size=sample_size)
        return {"category": "ragas", "result": res}
        
    else:
        raise ValueError(f"Unknown category: '{category}'. Valid categories: rag, conversational, safety, agentic, summarization, ragas, all")


def run_master_suite(category: str = "all", sample_size: int = 2):
    """Run specified category or all evaluation categories."""
    valid_categories = ["rag", "conversational", "safety", "agentic", "summarization", "ragas"]
    
    if category.lower() == "all":
        categories_to_run = valid_categories
    else:
        categories_to_run = [category.lower()]
        
    overall_summary: List[Dict[str, Any]] = []
    
    for cat in categories_to_run:
        try:
            res = execute_category(cat, sample_size=sample_size)
            overall_summary.append({
                "category": cat,
                "status": "PASSED",
                "details": res.get("result", {})
            })
        except Exception as e:
            logger.error(f"Error executing category '{cat}': {e}", exc_info=True)
            overall_summary.append({
                "category": cat,
                "status": "FAILED",
                "error": str(e)
            })
            
    # Executive Summary Dashboard
    print("\n" + "#" * 80)
    print("🎯 CLOUDOPS SENTINEL EVALUATION MASTER DASHBOARD")
    print("#" * 80)
    print(f"{'Category':<20} | {'Status':<10} | {'Sample Count':<12} | {'Notes'}")
    print("-" * 80)
    for row in overall_summary:
        cat_name = row["category"].capitalize()
        status = "✅ " + row["status"] if row["status"] == "PASSED" else "❌ " + row["status"]
        details = row.get("details", {})
        count = details.get("num_evaluated", sample_size) if row["status"] == "PASSED" else 0
        note = "Completed successfully" if row["status"] == "PASSED" else row.get("error", "Error")[:35]
        print(f"{cat_name:<20} | {status:<10} | {count:<12} | {note}")
    print("#" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CloudOps Sentinel Master Evaluation Runner")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "rag", "conversational", "safety", "agentic", "summarization", "ragas"],
        help="Evaluation category to execute (default: all)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of samples per category (default: 2 for rapid validation)"
    )
    
    args = parser.parse_args()
    run_master_suite(category=args.category, sample_size=args.samples)


if __name__ == "__main__":
    main()
