# REFRAG: Rethinking RAG based Decoding

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in leveraging extensive external knowledge to enhance responses in multi-turn and agentic applications, such as retrieval-augmented generation (RAG).  However, processing long-context inputs introduces significant system latency and demands substantial memory for the key-value cache, resulting in reduced throughput and a fundamental trade-off between knowledge enrichment and system efficiency. While minimizing latency for long-context inputs is a primary objective for LLMs, we contend that RAG systems require specialized consideration. In RAG, much of the LLM context consists of concatenated passages from retrieval, with only a small subset directly relevant to the query. These passages often exhibit low semantic similarity due to diversity or deduplication during re-ranking, leading to block-diagonal attention patterns that differ from those in standard LLM generation tasks. Based on this observation,  we argue that most computations over the RAG context during decoding are unnecessary and can be eliminated with minimal impact on performance. To this end, we propose REFRAG, an efficient decoding framework that compresses, senses, and expands to improve latency in RAG applications. By exploiting this attention sparsity structure, we demonstrate a $30.85\times$  the time-to-first-token acceleration ($3.75\times$ improvement to previous work) without loss in perplexity. In addition, our optimization framework for large context enables REFRAG to extend the context size of LLMs by $16\times$. We provide rigorous validation of REFRAG across diverse long-context tasks, including RAG, multi-turn conversations, and long document summarization, spanning a wide range of datasets. Experimental results confirm that REFRAG delivers substantial speedup with no loss in accuracy compared to LLaMA models and other state-of-the-art baselines across various context sizes. Additionally, our experiments establish that the expanded context window of REFRAG further enhances accuracy for popular applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes REFRAG, a decoding framework designed to improve latency and efficiency in retrieval-augmented generation (RAG) systems. The key idea is to replace most retrieved text tokens with pre-computed chunk embeddings, projected into the decoder’s token embedding space. During generation, these compressed embeddings are used instead of raw tokens, greatly reducing sequence length and attention cost. A lightweight reinforcement learning (RL) policy selectively expands important chunks back into tokens when needed. The authors further introduce a training recipe combining reconstruction pretraining, curriculum learning, continual pretraining, and instruction tuning to align the encoder and decoder.
Empirical results show up to 30× reduction in time-to-first-token (TTFT) while maintaining comparable perplexity and downstream task performance to full-context baselines. Experiments span multiple long-context tasks (RAG QA, multi-turn dialogue, summarization, etc.), and the method outperforms or matches several strong baselines such as CEPE, RePlug, and long-context LLaMA variants.

### Strengths
1. Practical motivation and strong real-world relevance: The paper targets the real bottleneck of RAG systems: high decoding latency due to long retrieved contexts. The problem is clearly stated, and the proposed solution directly addresses practical deployment concerns.

2. Modular and engineering-friendly design: REFRAG does not alter the core decoder architecture; it operates at the embedding level, allowing easy integration with existing retrieval pipelines and various encoders or projection layers.

3. Comprehensive experimental coverage: The authors evaluate on a wide range of tasks and models (LLaMA-2 7B/13B, LLaMA-3), and report detailed latency metrics (TTFT, throughput) along with standard quality metrics.

4. Significant empirical gains: The reported latency and throughput improvements are impressive, especially compared to CEPE and full-context models, demonstrating clear real-world impact.

5. Reasonably thorough ablation studies: The paper includes ablations on selective expansion, reconstruction loss, and curriculum learning, which strengthen the empirical credibility of the results.

### Weaknesses
1. Methodological novelty is limited: While effective, the proposed idea is relatively straightforward: replacing retrieved tokens with their pre-computed embeddings is a natural and somewhat trivial extension of existing RAG compression and prefix-caching ideas. The contribution feels more engineering-driven than conceptual.

2. Lack of theoretical grounding: The paper lacks formal or analytical justification for why the compressed embedding representation should preserve alignment and compositionality in the decoder’s hidden space. No theoretical analysis or representational study (e.g., probing, similarity metrics, mutual information) is presented to explain why the proposed projection preserves semantic fidelity.

3. Missing analysis on robustness to retrieval noise: Since the framework heavily relies on pre-computed embeddings, it’s unclear how it behaves under noisy or off-topic retrievals. There’s no sensitivity study on retriever quality or embedding corruption.

4. Evaluation focuses mainly on automatic metrics: Most results rely on perplexity or automated accuracy. For generation tasks (QA, summarization), human evaluations or factuality studies are necessary to ensure that compression and expansion do not degrade output faithfulness.

### Questions
1. How does REFRAG handle dynamic retrieval corpora where cached embeddings may become stale? Is there a cache invalidation or refresh mechanism?

2. Have you evaluated how retrieval quality affects REFRAG’s performance (e.g., varying top-k relevance or adding noise)?

3. Can you provide human evaluation results to assess factual consistency or fluency compared to full-context decoding?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes REFRAG, an efficient decoding framework for RAG systems that compresses context chunks into embeddings using a lightweight encoder, reducing input length and accelerating inference. It introduces a reinforcement learning policy for selective compression and demonstrates significant speedups in time-to-first-token (TTFT) without perplexity degradation across several long-context tasks.

### Strengths
- The idea is novel and well-motivated.
- The proposed method achieves impressive speedup while maintaining perplexity.

### Weaknesses
1. Clarity and Readability
Several parts of the paper are hard to follow, requiring readers to cross-reference multiple sections to understand the methodology. For example:
- On first reading, the description of "compress anywhere" in Section 1.1 and its relation to the "expand" operation in Figure 1 is unclear.
- In Line 167, the notation s + o = T is not clearly explained—whether chunks are processed jointly or separately. Similarly, Line 197 states "reconstructing s = k × L tokens from L chunk embeddings," but the reconstruction process is not adequately detailed.

2. Formatting and Presentation
- References to tables and figures are poorly placed. For instance, Table 1, Table 2, Figure 3, and especially Figure 4 are cited far from their actual positions, making it difficult to follow the discussion.

3. Experimental Setup
- The use of LLaMA-2-7B as the base model significantly weakens the credibility of the results. For example, the best performance on GSM8K is only 12.75 (Table 3), and NQ achieves only 26.08, which are far from convincing. Moreover, Table 14 shows that when using stronger models like LLaMA-3, perplexity becomes worse than the full-context baseline. It is unclear how much performance degradation actually occurs.
- The "Weak Retriever" setting (Section 5) is not well-justified. From my perspective, real-world systems can adopt strong retrievers like the one used in the paper.
- Do you compress the CoTs throughout the generation process? If not, including reasoning tasks, especially mathematical ones like GSM8K, raises concerns about the usefulness.
- Metrics: The paper relies heavily on perplexity (section 4), which is not always a reliable indicator of generation quality. Including human or LLM-based evaluation will improve the credibility.

4. Minor Issues
- Line 166: "data data point" → "data point".
- Line 233: "REFRAG" should be followed by a period.

### Questions
Please refer to Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces REFRAG, a decoding framework designed to accelerate inference for RAG applications by exploiting the sparse attention patterns inherent in RAG contexts. The core innovation is replacing token sequences from retrieved passages with pre-computed compressed chunk embeddings that are fed directly into the decoder. The method includes three main components: (1) a lightweight encoder  that processes context in fixed-size chunks to generate chunk embeddings, (2) a projection layer mapping these embeddings to the decoder's token space, and (3) an RL-trained policy for selective compression that determines which chunks require full token expansion.The authors train REFRAG through continual pre-training with curriculum learning on next-paragraph prediction tasks. With compression rates of k=16 and k=32, REFRAG achieves 16.53× and 30.85× TTFT acceleration respectively compared to LLaMA-2-7B, while maintaining comparable perplexity. The method is evaluated across RAG tasks, multi-turn conversations, and document summarization, demonstrating that compressed representations with extended context windows can match or exceed uncompressed baselines at equivalent latency budgets.

### Strengths
* The paper provides strong empirical evidence for attention sparsity in RAG contexts (Figure 7, Table 10), demonstrating block-diagonal patterns that justify the compression strategy. This RAG-specific observation differentiates the work from generic long-context methods.

* The proposed method achieves 30.85× TTFT acceleration while maintaining perplexity represents substantial practical improvement. It also extends context windows by 16× while preserving performance.

* The paper includes extensive ablations on compression rates (Figure 10), encoder-decoder combinations (Figure 11, Table 14), selective compression strategies (Figure 3, Table 13), and curriculum learning schedules (Table 8, Figure 6).

### Weaknesses
* The experimental comparison is not sufficient, missing a lot of new and important baselines, including:
    - No comparison with KV cache compression methods like SnapKV, H2O, and FastKV
    - Missing comparisons with prompt compression methods like LongLLMLingua and LLMLingua-2

* All main experiments use LLaMA-2-7B, which is pretty outdated. More recent models including LLaMA-3.1 should be included in all experiments (Brief ablations on LLaMA-3.1-8B  is insufficient). There is also no evaluation on larger models, which limits generalizability claims

* FlashAttention compatibility: The paper acknowledges requiring attention scores but doesn't compare against FlashAttention-enabled baselines, which is critical since FlashAttention is standard in real-world deployments.

* Missing efficiency metrics. Only theoretical analysis provided, but there is no actual peak memory consumption reported. Moreover, it would be great if author could give wall-time comparisons,

### Questions
* Can the authors include comparisons with important baselines mentioned in the weaknesses?

* The paper states the method requires attention scores and is incompatible with FlashAttention. Since FlashAttention is standard in real world  deployments, how does this affect practical applicability? Can you provide memory and latency comparisons against FlashAttention-enabled baselines?

* Why does CEPE show dramatically poor performance in your experiments (e.g., 0.03 EM on NQ in Table 3) compared to its original paper? This raises concerns about experimental setup. Can you clarify the differences?

* All main experiments use LLaMA-2-7B. Can you also includes comprehensive results on LLaMA-3.1-8B and also brief experiments on large-scale models to show if the proposed method scales?

* What is the total computational cost of continual pre-training and RL training? How does this compare to the inference savings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces REFRAG, a decoding-time framework for Retrieval-Augmented Generation (RAG). It first compresses retrieved context into precomputed chunk embeddings. Then, it detects which chunks must be expanded via a lightweight RL policy. Finally, it expands selected chunks back into token space when needed. Extensive experiments showing large time-to-first-token (TTFT) gains (up to ≈30×) with little/no perplexity loss across RAG, multi-turn conversations, and long-document summarization.

### Strengths
- The proposed method is promising, which shows very large TTFT accelerations while matching prior perplexity baselines.
- Evaluation across multiple datasets (ArXiv, Book, PG19, ProofPile), RAG tasks (many QA benchmarks), multi-turn convo datasets, and comparisons to recent long-context baselines (CEPE, LLAMA variants, REPLUG) strengthen the claims.

### Weaknesses
- While the RL selective expansion policy shows gains over heuristics, training such a policy (and using it online) introduces complexity. It lacks detailed analysis of RL training stability, wall-clock RL training cost, and sensitivity to reward design
- The paper shows experiments with RoBERTa variants, but it’s unclear how sensitive REFRAG is to encoder retraining mismatches, retrieval distribution shifts, or domain drift in large real-world corpora.

### Questions
- How is the end-to-end latency breakdown (retrieval time + embedding fetch + encoder recompute when needed + decoder prefill)
- How sensitive is RL selective expansion to miscalibrated reward signals?

### Soundness
3

### Presentation
3

### Contribution
3
