# 🎯 CloudOps Sentinel — Comprehensive AI Evaluation & Red-Teaming Suite

This directory contains the production evaluation framework for **CloudOps Sentinel**, benchmarking the Self-Reflective RAG (Self-RAG) pipeline across **20+ automated evaluation metrics** powered by **DeepEval** and **Ragas**.

---

## 📑 Table of Contents

- [Evaluation Categories & Metrics](#-evaluation-categories--metrics)
  - [1. Single-Turn RAG Triad (`test_self_rag_pipeline.py` & `test_crag_pipeline.py`)](#1-single-turn-rag-triad)
  - [2. Multi-Turn Conversational RAG (`test_conversational_rag_metrics.py`)](#2-multi-turn-conversational-rag)
  - [3. Safety & Guardrail Compliance (`test_safety_evals.py`)](#3-safety--guardrail-compliance)
  - [4. Agentic & Behavioral Suite (`test_agentic_metrics.py`)](#4-agentic--behavioral-suite)
  - [5. Summarization & Hallucination (`test_summarization_metrics.py`)](#5-summarization--hallucination)
  - [6. Ragas Framework Integration (`test_ragas_pipeline.py`)](#6-ragas-framework-integration)
- [Golden Dataset (`golden_dataset.json`)](#-golden-dataset)
- [Master Super Runner (`run_all_evals.py`)](#-master-super-runner)
- [Diagnostic Feedback & Remediation Engine](#-diagnostic-feedback--remediation-engine)

---

## 🔬 Evaluation Categories & Metrics

### 1. Single-Turn RAG Triad
**Files:** `evals/test_self_rag_pipeline.py`, `evals/test_crag_pipeline.py`  
**Purpose:** Assesses core retrieval precision, factual grounding, and response accuracy against CloudOps incident queries.

| Metric | What It Measures | Threshold |
| :--- | :--- | :--- |
| **`AnswerRelevancyMetric`** | Evaluates how directly and concisely the answer responds to the incident prompt without unnecessary filler. | `0.70` |
| **`FaithfulnessMetric`** | Checks whether all claims in the generated response are strictly grounded in retrieved runbook chunks. | `0.70` |
| **`ContextualRelevancyMetric`** | Measures signal-to-noise ratio in retrieved context chunks. | `0.60` |
| **`ContextualPrecisionMetric`** | Ensures the most relevant runbook chunks appear at the top of the retrieval rank list. | `0.60` |
| **`ContextualRecallMetric`** | Measures whether all facts necessary to produce the expected answer were retrieved. | `0.60` |

---

### 2. Multi-Turn Conversational RAG
**File:** `evals/test_conversational_rag_metrics.py`  
**Purpose:** Evaluates stateful incident triage dialogues, conversational memory retention, and follow-up query answering.

| Metric | What It Measures | Threshold |
| :--- | :--- | :--- |
| **`TurnRelevancyMetric`** | Verifies relevancy of individual assistant responses within the multi-turn dialogue context. | `0.70` |
| **`TurnFaithfulnessMetric`** | Audits factual grounding across multi-turn history without memory drift or hallucination. | `0.70` |
| **`TurnContextualPrecisionMetric`** | Evaluates chunk ranking precision for follow-up conversational queries. | `0.70` |
| **`TurnContextualRecallMetric`** | Verifies retrieval recall across interactive debugging sessions. | `0.70` |
| **`TurnContextualRelevancyMetric`** | Calculates the proportion of relevant vs irrelevant context sentences retrieved per turn. | `0.70` |
| **`ConversationCompletenessMetric`** | Evaluates whether the multi-turn conversation successfully satisfied the overall operational goal. | `0.70` |
| **`KnowledgeRetentionMetric`** | Verifies retention of pod names, cluster IDs, and error codes mentioned in earlier turns. | `0.70` |

---

### 3. Safety & Guardrail Compliance
**File:** `evals/test_safety_evals.py`  
**Purpose:** Red-teaming and adversarial evaluation across safety policies and jailbreak resistance.

| Metric | What It Measures | Target |
| :--- | :--- | :--- |
| **`BiasMetric`** | Audits gender, demographic, and regional bias in automated communications. | Pass (1.0) |
| **`ToxicityMetric`** | Detects abusive, hostile, or toxic language in incident logs and responses. | Pass (1.0) |
| **`NonAdviceMetric`** | Enforces refusal of financial, medical, or legal speculation outside CloudOps scope. | Pass (1.0) |
| **`MisuseMetric`** | Prevents malware generation, unauthorized credential harvesting, or destructive sabotage. | Pass (1.0) |
| **`PIILeakageMetric`** | Detects leakage of AWS secrets, production tokens, API keys, and personal data. | Pass (1.0) |
| **`RoleViolationMetric`** | Audits resistance against jailbreaks (DAN, sudo, DEV mode) and preserves SRE persona. | Pass (1.0) |

---

### 4. Agentic & Behavioral Suite
**File:** `evals/test_agentic_metrics.py`  
**Purpose:** Evaluates agentic decision-making, tool execution fidelity, and persona constraints.

| Metric | What It Measures | Threshold |
| :--- | :--- | :--- |
| **`ToolUseMetric`** | Validates correct tool selection (`k8s_describe_pod`, `redis_memory_stats`, `aws_cost_analyzer`) and input parameters. | `0.60` |
| **`GoalAccuracyMetric`** | Measures whether the agentic workflow satisfied the high-level incident remediation objective. | `0.70` |
| **`PromptAlignmentMetric`** | Evaluates compliance with system prompt instructions and operational guardrails. | `0.70` |
| **`TopicAdherenceMetric`** | Detects and penalizes conversational drift into non-CloudOps domains. | `0.70` |
| **`RoleAdherenceMetric`** | Ensures consistent SRE professional tone and authoritative technical communication. | `0.70` |

---

### 5. Summarization & Hallucination
**File:** `evals/test_summarization_metrics.py`  
**Purpose:** Assesses postmortem digest generation, information alignment, and zero-tolerance hallucination detection.

| Metric | What It Measures | Threshold |
| :--- | :--- | :--- |
| **`SummarizationMetric`** | Measures alignment (truthfulness) and inclusion (coverage of root causes and action items). | `0.70` |
| **`HallucinationMetric`** | Compares response against ground truth runbooks to detect fabricated CLI flags or parameters. | `0.70` |

---

### 6. Ragas Framework Integration
**File:** `evals/test_ragas_pipeline.py`  
**Purpose:** Native integration with Ragas evaluation library for industry-standard RAG metrics.

| Metric | Description |
| :--- | :--- |
| **`faithfulness`** | Factual consistency of the generated answer against the retrieved context. |
| **`answer_relevancy`** | Semantic alignment of the answer with the user's operational question. |
| **`context_precision`** | Signal-to-noise ratio and rank order of ground-truth chunks. |
| **`context_recall`** | Extent to which the retrieved context captures all reference answer facts. |

---

## 📚 Golden Dataset (`golden_dataset.json`)

The dataset comprises **62 enterprise-grade test cases**:
- **Cases 1–50 (Core CloudOps Scenarios):**
  - Kubernetes & Pod Lifecycle (CrashLoopBackOff, OOMKilled, Evicted, Pending).
  - Database & Caching (Redis maxmemory, PostgreSQL deadlock resolution, connection spikes).
  - Cloud Cost Optimization (AWS unattached EBS volumes, NAT Gateway idle costs, DynamoDB auto-scaling).
  - CI/CD & Deployments (Canary rollbacks, ArgoCD drift, Docker multi-stage builds).
- **Cases 51–62 (Safety & Red-Teaming):**
  - Bias & fairness audits across SRE hiring/engineering queries.
  - Toxicity and abuse detection in heated postmortems.
  - Non-advice compliance (cryptocurrency speculation, tax/legal advice).
  - Misuse prevention (ransomware scripts, destructive drop database commands).
  - PII and AWS secret exfiltration attempts.
  - Role violation and adversarial jailbreak prompts.

---

## 👑 Master Super Runner (`run_all_evals.py`)

Execute the unified CLI runner from the `backend/` directory:

```powershell
# Run all categories with a 3-sample batch
python -m evals.run_all_evals --category all --samples 3

# Run a specific evaluation category
python -m evals.run_all_evals --category rag --samples 5
python -m evals.run_all_evals --category conversational --samples 2
python -m evals.run_all_evals --category safety --samples 4
python -m evals.run_all_evals --category agentic --samples 3
python -m evals.run_all_evals --category summarization --samples 2
python -m evals.run_all_evals --category ragas --samples 3
```

---

## 🛠️ Diagnostic Feedback & Remediation Engine

When metrics drop below required thresholds, the diagnostic engine provides automatic remediation guidance:

```
Diagnostic Feedback & Recommended Tuning:
  -> [FIX - Contextual Relevancy]: Retrieved chunks contain high background noise.
     Action: Refine semantic chunk headers in ingestion.py and tighten relevance grader threshold.
  -> [FIX - Contextual Recall]: Key facts were missing from retrieved context.
     Action: Increase retriever TOP_K from 3 to 5 and increase chunk overlap to 100+ tokens.
  -> [FIX - Faithfulness]: Unsupported claims detected in generated remediation.
     Action: Refine generate_from_context system prompt with strict grounding constraints.
```
