# RAG Evaluation — Complete Reference

---

## 0. The Core Principle

> **You cannot improve what you cannot measure.**

RAG fails in specific, predictable ways. Evaluation exists to detect each failure mode:

| Failure Mode | Example | Evaluation Target |
|---|---|---|
| Wrong retrieval | Fetched irrelevant chunks | Retrieval metrics |
| Hallucination | Model added facts not in context | Faithfulness |
| Irrelevant answer | Retrieved correctly, answered wrong question | Answer relevance |
| Incomplete answer | Context had the answer, model missed it | Context recall |
| Language mixing | Arabic query → English answer | Language consistency |
| Ranking failure | Right doc at rank 50, noise at rank 1 | NDCG, MRR |

**Principle:** your metric hierarchy must match your failure mode hierarchy. For financial RAG, faithfulness is #1 — hallucination on banking data is the highest-risk failure.

---

## 1. The Three Layers of RAG Evaluation

```
RAG Pipeline:

  Query → [Retrieval] → [Augmentation] → [Generation] → Answer
               ↓               ↓               ↓
         Did we fetch      Did context      Did LLM use
         the right         fit properly?    it correctly?
         chunks?
```

Each layer is evaluated independently AND end-to-end.

---

## 2. Layer 1 — Retrieval Evaluation

### What to measure
Did the vector DB return the right documents in the right order?

---

### 2.1 Recall@K

$$Recall@K = \frac{|\text{relevant} \cap \text{retrieved}_K|}{|\text{total relevant}|}$$

- **What:** out of all relevant docs in the corpus, how many did we retrieve?
- **Failure it catches:** missing the answer entirely
- **When to use:** when coverage matters more than precision (open-domain QA)

**Example:**
```
Total relevant docs in corpus: 5
Retrieved in top-10: 4
Recall@10 = 4/5 = 0.80
```

---

### 2.2 Precision@K

$$Precision@K = \frac{|\text{relevant} \cap \text{retrieved}_K|}{K}$$

- **What:** of the K docs we retrieved, how many are actually relevant?
- **Failure it catches:** noise flooding the LLM context
- **When to use:** when context window is limited, noise is expensive

**Example:**
```
Retrieved top-10, 6 are relevant
Precision@10 = 6/10 = 0.60
```

---

### 2.3 MRR — Mean Reciprocal Rank

$$MRR = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

- **What:** how highly ranked is the FIRST relevant document?
- **Failure it catches:** correct answer buried deep in results
- **Only cares about the first relevant doc — ignores everything after**

**Example across 3 queries:**
```
Query 1: first relevant at rank 2 → 1/2
Query 2: first relevant at rank 1 → 1/1
Query 3: first relevant at rank 4 → 1/4

MRR = (0.5 + 1.0 + 0.25) / 3 = 0.583
```

**Use MRR when:** you care about finding ONE good answer fast (single-answer QA).

---

### 2.4 NDCG@K — Normalized Discounted Cumulative Gain

**Step 1 — DCG (Discounted Cumulative Gain):**

$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

- $rel_i$ = relevance score of doc at rank $i$ (binary 0/1, or graded 0/1/2/3)
- Position penalty: rank 1 is worth more than rank 10

**Step 2 — IDCG (Ideal DCG):**
DCG of perfect ranking (all relevant docs at top).

**Step 3 — NDCG:**

$$NDCG@K = \frac{DCG@K}{IDCG@K} \in [0, 1]$$

**Example:**
```
Ranked results:    [irrelevant, irrelevant, relevant, relevant, relevant]
rel scores:        [0,          0,          1,         1,         1       ]

DCG@5 = 0/log₂(2) + 0/log₂(3) + 1/log₂(4) + 1/log₂(5) + 1/log₂(6)
      = 0 + 0 + 0.5 + 0.43 + 0.39 = 1.32

IDCG@5 (perfect) = 1/log₂(2) + 1/log₂(3) + 1/log₂(4) + ...
                 = 1.0 + 0.63 + 0.5 + 0.43 + 0.39 = 2.95

NDCG@5 = 1.32 / 2.95 = 0.447
```

**Use NDCG when:** multiple relevant docs exist and their ranking order matters.

---

### MRR vs NDCG Summary

| | MRR | NDCG |
|---|---|---|
| Counts | First relevant only | All relevant docs |
| Graded relevance | No — binary | Yes |
| Best for | Single-answer retrieval | Multi-doc ranked retrieval |
| Your system | Validation judge (one answer) | Chunk retrieval quality |

---

## 3. Layer 2 — Generation Evaluation

### 3.1 Reference-Based Metrics (need ground truth)

| Metric | Formula | Measures | Weakness |
|---|---|---|---|
| **BLEU** | n-gram precision vs reference | Exact word overlap | Misses paraphrases |
| **ROUGE-L** | Longest common subsequence | Coverage | Surface-level only |
| **BERTScore** | Cosine sim of BERT embeddings | Semantic similarity | Slow, misses factual errors |

**When to use:** you have a golden dataset with reference answers.

**Limitation:** a model can score high BLEU while still hallucinating — it matched the surface but added extra false claims.

---

### 3.2 Reference-Free Metrics (no ground truth needed)

These are the critical metrics for production RAG.

---

#### Faithfulness

> Are all claims in the answer supported by the retrieved context?

**Process:**
1. Extract atomic claims from the answer
2. Check each claim against the context
3. Score = supported claims / total claims

$$Faithfulness = \frac{|\text{claims supported by context}|}{|\text{total claims in answer}|}$$

**Example:**
```
Answer: "Al Rajhi offers murabaha financing at 3.5% with 5-year term."
Context: mentions murabaha and 3.5% rate, no mention of term length.

Claims: [murabaha ✓, 3.5% ✓, 5-year term ✗]
Faithfulness = 2/3 = 0.67
```

**This is your #1 metric** — hallucination on financial data is highest risk.

---

#### Answer Relevance

> Does the answer actually address the question asked?

- High score: direct, complete response to the query
- Low score: technically correct but off-topic

$$Answer\_Relevance = \frac{1}{N}\sum_{i=1}^{N} sim(reverse\_question_i, original\_question)$$

Method: generate N questions from the answer, check if they match original query.

---

#### Context Precision

> Of the retrieved chunks, how many are actually relevant to the question?

$$Context\_Precision = \frac{|\text{relevant chunks retrieved}|}{|\text{total chunks retrieved}|}$$

Low context precision = noisy context = higher hallucination risk.

---

#### Context Recall

> Does the retrieved context contain all information needed to answer?

$$Context\_Recall = \frac{|\text{answer claims found in context}|}{|\text{total claims in ground truth answer}|}$$

Low context recall = answer is impossible from retrieved context.

---

## 4. Layer 3 — End-to-End: The RAG Triad

The most important evaluation framework for RAG systems.

```
           Query
          /     \
    Context ——— Answer

Each edge is a metric:
Query → Context    : Context Relevance
Context → Answer   : Groundedness (Faithfulness)
Query → Answer     : Answer Relevance
```

**All three must be high.** Common failure patterns:

| Context Relevance | Groundedness | Answer Relevance | Diagnosis |
|---|---|---|---|
| High | High | High | ✓ Working correctly |
| Low | High | Low | Wrong docs retrieved |
| High | Low | High | Hallucinating |
| High | High | Low | Model ignored the question |
| Low | Low | High | Lucky answer, broken pipeline |

---

## 5. Evaluation Methods

### Method 1 — LLM-as-Judge

Use a strong LLM to score outputs. Most flexible, handles Arabic natively.

```python
faithfulness_prompt = """
You are evaluating a RAG system answer.

Context: {context}
Question: {question}  
Answer: {answer}

Task: Check if every factual claim in the answer is supported by the context.
Extract each claim, verify it against context, return score.

Respond in JSON:
{
  "claims": ["claim1", "claim2", ...],
  "supported": [true, false, ...],
  "faithfulness_score": float,  # 0.0 - 1.0
  "reasoning": "..."
}
"""
```

**Pros:** flexible, no labeled data, handles Arabic, catches nuanced failures
**Cons:** expensive, biased toward longer answers, inconsistent across runs

**Tip:** use `temperature=0` for reproducibility. Run 3x and average for critical evaluations.

---

### Method 2 — RAGAS Framework

Open-source framework that automates all RAG metrics using LLM calls internally.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# Prepare dataset
data = {
    "question": ["What is murabaha?", ...],
    "answer": ["Murabaha is...", ...],
    "contexts": [["chunk1", "chunk2"], ...],
    "ground_truth": ["Reference answer...", ...]  # optional
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
# {'faithfulness': 0.87, 'answer_relevancy': 0.91, ...}
```

**RAGAS metric internals:**

| Metric | How RAGAS computes it |
|---|---|
| Faithfulness | LLM extracts claims → checks each vs context |
| Answer Relevancy | LLM generates reverse questions → embedding similarity |
| Context Precision | LLM judges each chunk relevance → weighted precision |
| Context Recall | LLM checks if ground truth claims appear in context |

---

### Method 3 — Human Evaluation (Golden Dataset)

Domain experts manually label query/answer pairs.

**Structure:**
```
golden_dataset.jsonl
{
  "query": "ما هي شروط التمويل العقاري؟",       # Arabic
  "relevant_doc_ids": ["doc_42", "doc_87"],
  "reference_answer": "...",
  "sharia_compliant": true,
  "difficulty": "hard"
}
```

**For Al Rajhi specifically:**
- Islamic finance experts validate Sharia compliance
- Arabic linguists check language quality
- Compliance team validates SAMA regulatory answers
- Minimum 500 queries covering all product categories

**Annotation guidelines:**
- Faithfulness: 1 (hallucination) → 5 (fully grounded)
- Completeness: 1 (missing key info) → 5 (complete)
- Language: 1 (wrong language/mixing) → 5 (correct Arabic/English)
- Tone: 1 (inappropriate) → 5 (professional, Islamic banking appropriate)

---

### Method 4 — A/B Testing (Online)

Compare two pipeline versions on real traffic.

```
50% traffic → Pipeline A (current)
50% traffic → Pipeline B (new retrieval/model)

Measure:
- Thumbs up/down rate
- Follow-up question rate (low = answer was complete)
- Session abandonment rate
- Query reformulation rate (user rephrasing = answer was bad)
```

---

## 6. Online vs Offline Evaluation

| Dimension | Offline | Online |
|---|---|---|
| **When** | Pre-deployment | Production |
| **Data** | Curated golden set | Real user queries |
| **Methods** | RAGAS, human eval, LLM judge | Implicit signals, A/B, real-time judge |
| **Latency constraint** | None | Must be fast |
| **Feedback loop** | Slow (batch) | Immediate |
| **Coverage** | Known failure modes | Unknown failure modes |

**Both are required.** Offline catches known failures. Online catches unknown ones.

---

## 7. Your System's Evaluation Stack

```
┌─────────────────────────────────────────────┐
│              OFFLINE EVALUATION             │
│                                             │
│  RAGAS → faithfulness, context recall,      │
│          answer relevancy on golden set     │
│                                             │
│  Retrieval → NDCG@10, Recall@10 on          │
│              labeled query-doc pairs        │
│                                             │
│  Human eval → Islamic finance experts,      │
│               Arabic linguists, compliance  │
└─────────────────────────────────────────────┘
                      ↓  deploy
┌─────────────────────────────────────────────┐
│              ONLINE EVALUATION              │
│                                             │
│  Llama 3.1 8B judge → per-request           │
│  faithfulness gate (12.5% overhead)         │
│                                             │
│  Implicit signals → thumbs up/down,         │
│  follow-up rate, reformulation rate         │
│                                             │
│  Logging → flag low-confidence answers      │
│  for async human review                     │
│                                             │
│  A/B testing → compare pipeline versions   │
└─────────────────────────────────────────────┘
                      ↓  feedback
┌─────────────────────────────────────────────┐
│           CONTINUOUS IMPROVEMENT            │
│                                             │
│  Failed queries → add to golden set         │
│  Low faithfulness → chunk size tuning       │
│  Low context recall → retrieval tuning      │
│  Low answer relevance → prompt tuning       │
└─────────────────────────────────────────────┘
```

---

## 8. Metric Thresholds (Production Targets)

| Metric | Minimum | Target | Critical Below |
|---|---|---|---|
| Faithfulness | 0.85 | 0.95 | 0.75 |
| Answer Relevance | 0.80 | 0.90 | 0.70 |
| Context Precision | 0.70 | 0.85 | 0.60 |
| Context Recall | 0.80 | 0.90 | 0.70 |
| NDCG@10 | 0.70 | 0.85 | 0.60 |
| Recall@10 | 0.80 | 0.92 | 0.70 |

**Faithfulness threshold is non-negotiable** — financial data hallucination is a compliance risk.

---

## 9. Failure Mode → Metric → Fix

| Symptom | Metric that catches it | Likely fix |
|---|---|---|
| Model adds facts not in context | Faithfulness ↓ | Stronger system prompt, lower temperature |
| Answer ignores the question | Answer Relevance ↓ | Prompt engineering, query rewriting |
| Right answer not in retrieved chunks | Context Recall ↓ | Increase K, fix chunking strategy |
| Too much noise in context | Context Precision ↓ | Better reranker, reduce K |
| Correct doc ranked too low | NDCG ↓ | Tune hybrid search alpha, reranker |
| Missing any relevant doc | Recall@K ↓ | Increase K, check embedding quality |
| Arabic query → English answer | Language check ↓ | Language detection + routing |

---

## 10. Interview One-Liners

**On evaluation philosophy:**
> "Evaluation must match failure modes. For financial RAG, faithfulness is #1 because hallucination on banking data is a compliance risk, not just a quality issue."

**On metrics choice:**
> "I use NDCG for retrieval — it penalizes relevant docs buried deep in rankings. MRR for single-answer scenarios. Faithfulness and answer relevance for generation. The RAG triad ties it end-to-end."

**On online vs offline:**
> "Offline with RAGAS catches known failure modes before deployment. Online with the 8B judge catches them per-request in production. Implicit signals catch unknown failure modes I didn't design tests for."

**On the 8B judge:**
> "The judge costs 12.5% compute overhead and catches hallucinations at request time. It's online evaluation baked into the serving path — not a separate monitoring system."

**On human eval:**
> "Automated metrics can't evaluate Sharia compliance or tone appropriateness for Islamic banking. That requires domain experts. Automation handles scale, humans handle nuance."
