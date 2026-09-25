"""
CloudOps Sentinel - Feedback Flywheel & Golden Dataset Auto-Curation
=====================================================================

This module implements the Production Feedback Flywheel of Loop Engineering.
It ingests downvoted interactions and human SRE corrections from the audit/feedback database,
validates and formats them, and automatically enriches `evals/golden_dataset.json`.

Workflow:
---------
1. Query un-synced operator feedback from database (`user_feedback` table).
2. Filter for actionable corrections (`rating == 'downvote'` and `corrected_answer != ''`).
3. Generate synthetic ground-truth metadata and extract context references.
4. Append new test cases into `evals/golden_dataset.json`.
5. Mark processed records as synced to maintain state.

Usage:
------
python -m evals.feedback_flywheel
"""

import sys
import os
import json
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

from src.db import get_unprocessed_feedback, mark_feedback_synced
from evals.dataset import load_golden_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

DATASET_FILE = Path(__file__).resolve().parent / "golden_dataset.json"


def sync_feedback_to_golden_dataset() -> Dict[str, Any]:
    """
    Syncs verified user corrections into golden_dataset.json.
    """
    feedback_items = get_unprocessed_feedback()
    if not feedback_items:
        logger.info("No unprocessed feedback items found in queue.")
        return {"synced_count": 0, "message": "Queue is empty."}
        
    logger.info(f"Found {len(feedback_items)} unprocessed feedback items.")
    
    # Load existing dataset
    existing_data = load_golden_dataset()
    existing_questions = {item.get("input", "").strip().lower() for item in existing_data}
    
    new_cases: List[Dict[str, Any]] = []
    synced_ids: List[int] = []
    
    case_counter = len(existing_data) + 1
    
    for fb in feedback_items:
        fb_id = fb["id"]
        q = fb.get("question", "").strip()
        expected = fb.get("corrected_answer", "").strip() or fb.get("comments", "").strip()
        category = fb.get("category", "Production Incident Corrections")
        
        # Skip duplicates or empty corrections
        if not q or not expected:
            synced_ids.append(fb_id)
            continue
            
        if q.lower() in existing_questions:
            logger.info(f"Skipping duplicate query: {q[:50]}...")
            synced_ids.append(fb_id)
            continue
            
        case_id = f"case_flywheel_{case_counter:03d}"
        case_counter += 1
        
        new_case = {
            "id": case_id,
            "category": category,
            "input": q,
            "expected_output": expected,
            "context": [
                f"Source: Human SRE verified correction from Incident thread {fb.get('thread_id', 'prod-unknown')}."
            ],
            "metadata": {
                "source": "production_feedback_flywheel",
                "original_feedback_id": fb_id,
                "operator_comments": fb.get("comments", ""),
            }
        }
        
        new_cases.append(new_case)
        existing_questions.add(q.lower())
        synced_ids.append(fb_id)
        
    if new_cases:
        updated_dataset = existing_data + new_cases
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_dataset, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully added {len(new_cases)} new test cases to {DATASET_FILE.name}.")
        print(f"✅ Added {len(new_cases)} verified test cases from production feedback flywheel.")
    else:
        logger.info("No new actionable test cases generated from feedback.")
        print("ℹ️ No new test cases added (items were already present or lacked corrections).")
        
    # Mark records synced in database
    mark_feedback_synced(synced_ids)
    
    return {
        "synced_count": len(new_cases),
        "total_processed_feedback": len(synced_ids),
        "total_dataset_size": len(existing_data) + len(new_cases),
    }


if __name__ == "__main__":
    sync_feedback_to_golden_dataset()
