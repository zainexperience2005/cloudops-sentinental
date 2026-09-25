"""
Ragas Pipeline Evaluation Test Suite
====================================

This module implements evaluation using the Ragas (Retrieval Augmented Generation Assessment)
framework to measure core RAG quality metrics on CloudOps Sentinel:
  1. Faithfulness: Factual alignment of generated response with retrieved context.
  2. Answer Relevancy: How directly and concisely the answer addresses the incident prompt.
  3. Context Precision: Signal-to-noise ratio of retrieved runbook context chunks.
  4. Context Recall: Extent to which retrieved context contains ground truth answers.

Framework: Ragas (ragas.io)
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

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.self_rag import invoke_self_rag
from evals.dataset import get_golden_test_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_ragas_evaluations(sample_size: int = 3) -> Dict[str, Any]:
    """
    Execute Ragas evaluation on CloudOps Sentinel pipeline.
    """
    cases = get_golden_test_cases(limit=sample_size, retrieval_only=True)
    
    print("=" * 80, flush=True)
    print(f"Ragas Evaluation Framework ({len(cases)} Cases)", flush=True)
    print("=" * 80, flush=True)
    
    user_inputs: List[str] = []
    responses: List[str] = []
    retrieved_contexts: List[List[str]] = []
    reference_answers: List[str] = []
    
    for i, item in enumerate(cases, 1):
        query = item["input"]
        print(f"\n[Ragas Case {i}/{len(cases)} - {item['id']}] Query: {query}", flush=True)
        res = invoke_self_rag(question=query, thread_id=f"ragas-eval-{item['id']}")
        actual_output = res["answer"]
        chunks = res.get("retrieval_chunks") or item.get("context", ["Default SRE context"])
        
        user_inputs.append(query)
        responses.append(actual_output)
        retrieved_contexts.append(chunks)
        reference_answers.append(item["expected_output"])
        
    dataset_dict = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference_answers,
    }
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    print("\nEvaluating dataset with Ragas metrics...", flush=True)
    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    
    print("\n" + "=" * 80, flush=True)
    print("RAGAS EVALUATION FRAMEWORK SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(results, flush=True)
    print("=" * 80, flush=True)
    
    return {
        "num_evaluated": len(cases),
        "scores": results,
    }


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2
    run_ragas_evaluations(sample_size=count)
