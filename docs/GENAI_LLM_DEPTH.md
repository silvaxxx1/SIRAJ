# GEN AI / LLM DEPTH CHEAT SHEET
## Al Rajhi Bank — Senior AI Engineer Interview

---

# 1. TRANSFORMER ARCHITECTURE

## The Core: Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q = query matrix  (what am I looking for?)
K = key matrix    (what do I have?)
V = value matrix  (what do I return?)
d_k = key dimension (scaling prevents vanishing gradients in softmax)
```

Plain English: every token attends to every other token. The score tells the model how much to "look at" each token when producing an output.

## Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_o
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Why multiple heads: each head learns different relationships simultaneously.
- Head 1 might track subject-verb agreement
- Head 2 might track coreference (pronoun → noun)
- Head 3 might track positional proximity

## Full Transformer Block (Decoder-only, e.g. Llama)

```
Input tokens
     │
     ▼
Token Embedding + Positional Encoding
     │
     ▼
[× N layers]
  ├── RMSNorm
  ├── Multi-Head Causal Self-Attention (masked — can't see future tokens)
  ├── Residual connection
  ├── RMSNorm
  ├── Feed-Forward Network (FFN): 2 linear layers + activation
  └── Residual connection
     │
     ▼
RMSNorm → Linear (vocab projection) → Softmax → next token probability
```

---

# 2. KV CACHE — KNOW THIS COLD

## What it is

During autoregressive generation, the model recomputes K and V for ALL previous tokens at every step. This is O(n²) in compute.

KV cache: store the K and V tensors from previous tokens. On each new token, only compute K, V for the new token and append to cache.

```
Without KV cache:
  Token 1: compute attention over [t1]
  Token 2: compute attention over [t1, t2]        ← recomputes t1
  Token 3: compute attention over [t1, t2, t3]    ← recomputes t1, t2
  Cost: O(n²)

With KV cache:
  Token 1: compute [k1, v1] → store
  Token 2: compute [k2, v2] → append → attention over stored cache
  Token 3: compute [k3, v3] → append → attention over stored cache
  Cost: O(n) per new token
```

## Memory cost

```
KV cache size = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_bytes

Llama 3.1 70B, seq_len=4096, batch=1, fp16:
= 2 × 80 × 64 × 128 × 4096 × 1 × 2 bytes = ~10.7 GB
```

This is why long context is expensive — KV cache grows linearly with sequence length.

## PagedAttention (vLLM)

Problem: KV cache allocated contiguously → fragmentation → wasted GPU memory.
Solution: Store KV cache in non-contiguous pages (like OS virtual memory).
Result: near-zero waste → higher batch sizes → higher throughput.

## RadixAttention (SGLang)

Problem: PagedAttention caches per session. Two different users with the same system prompt both recompute it.
Solution: Cache KV states in a radix trie indexed by token prefix. Shared prefix = shared cache entry.
Result: system prompt computed once, cached for all requests with that prefix.

---

# 3. POSITIONAL ENCODING

| Method | How | Used by | Advantage |
|--------|-----|---------|-----------|
| **Absolute (sinusoidal)** | Fixed sin/cos at each position | Original Transformer | Simple, no params |
| **Learned absolute** | Trainable embedding per position | GPT-2, BERT | Learns task-specific position |
| **RoPE** (Rotary) | Rotate Q,K vectors by position angle | Llama, Mistral | Handles long context better, relative by nature |
| **ALiBi** | Subtract linear bias from attention score | MPT, BLOOM | Extrapolates beyond training length |

**Why RoPE matters:** Llama uses RoPE. It encodes relative distances naturally — token 5 and token 6 are always "1 apart" regardless of absolute position. Better generalization to contexts longer than training length.

---

# 4. CONTEXT WINDOW — PRACTICAL LIMITS

| Model | Context | Practical usable | Why not full |
|-------|---------|-----------------|--------------|
| Llama 3.1 8B | 128K | ~64K | Quality degrades past halfway |
| Llama 3.1 70B | 128K | ~64K | "Lost in the middle" effect |
| GPT-4 Turbo | 128K | ~96K | Better long-context training |
| Claude 3.5 | 200K | ~150K | Best long-context handling |

**"Lost in the middle" problem:** LLMs pay most attention to tokens at the beginning and end of context. Information in the middle gets underweighted. This is why reranking matters — put the most relevant chunks first.

---

# 5. TOKENIZATION

## BPE (Byte Pair Encoding) — used by most LLMs

```
Start: character-level vocabulary
Repeat:
  1. Count most frequent adjacent pair
  2. Merge into new token
  3. Add to vocabulary
Until vocabulary size reached (e.g. 32K tokens)
```

Common words → single token: "bank" = 1 token
Rare words → multiple tokens: "murabaha" = ["mur", "ab", "aha"] = 3 tokens

## Why Arabic tokenization is expensive

```
English: "The maximum loan limit" → 5 tokens
Arabic:  "الحد الأقصى للتمويل" → 8-12 tokens (Arabic words are morphologically complex)
```

Arabic is **2-3x more tokens** than equivalent English text.
Implication: same document in Arabic costs 2-3x more to process (context window, inference cost).
BGE-M3 was trained specifically to handle this efficiently.

---

# 6. SAMPLING STRATEGIES

| Strategy | How | Effect | Use when |
|----------|-----|--------|----------|
| **Greedy** | Always pick highest probability token | Deterministic, repetitive | Never in production |
| **Temperature** | Divide logits by T before softmax | T<1 = focused, T>1 = creative | T=0.1 for factual banking |
| **Top-K** | Sample from top K tokens only | Cuts off long tail | K=50 common default |
| **Top-P (nucleus)** | Sample from tokens summing to P probability | Dynamic cutoff | P=0.9 standard |
| **Beam search** | Keep top-B sequences, pick best | High quality, slow | Translation, summarization |

**For banking RAG:** Temperature=0.0-0.1 (near-deterministic), Top-P=0.9. You want consistent, factual answers — not creative ones.

```python
# SGLang generation config for banking
response = client.generate(
    prompt=assembled_prompt,
    sampling_params=SamplingParams(
        temperature=0.1,    # near-deterministic
        top_p=0.9,
        max_new_tokens=512,
        stop=["<|eot_id|>"]  # Llama 3 end token
    )
)
```

---

# 7. COMMON FAILURE MODES

| Failure | Cause | Fix |
|---------|-------|-----|
| **Hallucination** | Model generates beyond context | Faithfulness check, citation enforcement, lower temperature |
| **Repetition** | Model gets stuck in loop | Repetition penalty (1.1-1.3), diverse beam search |
| **Truncation** | Response cut off at max_tokens | Increase max_new_tokens, or detect mid-sentence and retry |
| **Refusal** | Overly cautious safety training | Tune system prompt, adjust guardrail thresholds |
| **Context overflow** | Input exceeds context window | Truncate retrieved chunks, reduce K, summarize |
| **Lost in middle** | LLM ignores middle chunks | Reranking puts most relevant first + last |
| **Language confusion** | Switches language mid-answer | Explicit language instruction in system prompt |
| **Sycophancy** | Agrees with wrong user claims | DPO training on non-sycophantic responses |

---

# 8. LLM EVALUATION

## Intrinsic Metrics (model quality)
- **Perplexity**: how surprised the model is by the next token. Lower = better language model. Not useful for task performance.
- **BLEU/ROUGE**: token overlap with reference. Good for translation/summarization, weak for open-ended generation.

## Extrinsic Metrics (task performance)
- **RAGAS** (faithfulness, relevancy, precision, recall) — for RAG
- **MMLU** — reasoning across 57 subjects (used to compare base models)
- **HumanEval** — code generation benchmark
- **MT-Bench** — multi-turn instruction following
- **LLM-as-judge** — use GPT-4 to rate outputs 1-5

## For production banking:
```
Faithfulness      > 0.85   (no hallucination)
Answer relevancy  > 0.80   (addresses the question)
Context precision > 0.75   (retrieved chunks are relevant)
Human escalation  < 2%     (system confident enough)
```

---

# 9. FLASH ATTENTION

**Problem:** Standard attention computes full N×N attention matrix and stores it in GPU HBM (slow memory). For seq_len=4096, that's 4096² = 16M floats per layer per head.

**FlashAttention solution:** Compute attention in tiles that fit in fast SRAM. Never materialize full N×N matrix.

Result:
- Memory: O(n) instead of O(n²)
- Speed: 2-4x faster on A100
- Exact same output — not approximate

FlashAttention v2: better parallelism, ~2x faster than v1.
All major serving frameworks (vLLM, SGLang, TGI) use it by default.

---

# 10. THE QUESTIONS THEY WILL ASK

**"Explain attention in plain English."**
> "Every token looks at every other token and decides how much to pay attention to it. That score is computed by taking the dot product of a query vector and a key vector — high dot product means high attention. The result is a weighted sum of value vectors. Multi-head attention runs this process in parallel with different learned projections, so the model simultaneously learns different types of relationships: syntax, semantics, coreference."

**"What is KV cache and why does it matter for production?"**
> "During generation, the model needs the key and value vectors for every previous token at every step. Without caching, that's quadratic recomputation. KV cache stores those vectors after computing them once and appends on each new token — linear cost instead of quadratic. The tradeoff is GPU memory: a 70B model at 4K context needs ~10GB just for the KV cache. This is why batch size is limited, and why PagedAttention and RadixAttention exist — they manage that memory more efficiently."

**"Why does temperature=0 not always give the same output?"**
> "Temperature=0 means greedy decoding — always pick the highest probability token. In theory deterministic. In practice, floating-point non-determinism across GPU kernels can produce slightly different logits, giving different greedy choices. For true determinism you need to fix the random seed AND use deterministic CUDA ops, which sacrifice some performance. Most production systems accept temperature=0.1 as 'close enough to deterministic' without the overhead."

**"What's the difference between top-K and top-P sampling?"**
> "Top-K samples from the K most likely tokens — a fixed number. Problem: on confident steps K=50 might include garbage tokens, on uncertain steps K=50 might cut off valid options. Top-P samples from the minimum set of tokens whose cumulative probability reaches P — dynamic cutoff. Confident step = fewer tokens needed to reach 0.9, uncertain step = more tokens included. Top-P is generally preferred because it adapts to the model's confidence."
