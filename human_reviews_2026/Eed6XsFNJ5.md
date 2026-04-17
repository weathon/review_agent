# CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Recent advances in large language models (LLMs) have significantly boosted long-context processing. However, the increasing key-value (KV) cache size poses critical challenges to memory and execution efficiency. Most KV cache compression methods rely on heuristic token eviction using all attention heads in Grouped Query Attention (GQA)-based LLMs. This method ignores the different functionalities of attention heads, leading to the eviction of critical tokens and thus degrades the performance of LLMs.

To address the issue above, instead of using all the attention heads in GQA-based LLMs to determine important tokens as in the previous work, we first identify the attention heads in each layer that are not only capable to retrieve the initial and final tokens of a prompt, but also capable of retrieving important tokens within the text and attending to their surrounding semantic context. Afterwards, we exploit such heads to determine the important tokens and retain their corresponding KV cache pairs. Furthermore, we analyze the cache eviction error of each layer individually and introduce a layer-adaptive KV cache allocation strategy. Experimental results demonstrate the the proposed framework CompressKV consistently outperforms state-of-the-art approaches under various memory budgets on LongBench and Needle-in-a-Haystack benchmarks. Notably, it retains over 97% of full‑cache performance using only 3% of KV cache on LongBench’s question‑answering tasks and achieves 90% of accuracy with just 0.7% of KV storage on Needle-in-a-Haystack benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the critical challenges of increased memory consumption and reduced execution efficiency caused by the expanding key-value (KV) cache size in large language models (LLMs) with advanced long-context processing capabilities. It notes that existing KV cache compression methods for Grouped Query Attention (GQA)-based LLMs typically use all attention heads for heuristic token eviction, which overlooks the functional differences among heads, often leading to the removal of critical tokens and thus degraded LLM performance. To solve this issue, the authors propose CompressKV: first, they identify specific attention heads in each layer of GQA-based LLMs—heads that can retrieve the initial and final tokens of prompts, important in-text tokens, and attend to the surrounding semantic context of these tokens—then use only these heads to determine and retain KV cache pairs corresponding to important tokens; additionally, they analyze cache eviction errors layer by layer and introduce a layer-adaptive KV cache allocation strategy. Experimental results on the LongBench and Needle-in-a-Haystack benchmarks show that CompressKV consistently outperforms state-of-the-art approaches under various memory budgets: notably, it retains over 97% of full-cache performance using just 3% of KV cache in LongBench’s question-answering tasks and achieves 90% accuracy with only 0.7% of KV storage in the Needle-in-a-Haystack benchmark, with the code provided in the supplementary material.

### Strengths
- *Introduces Semantic Retrieval Heads (SRHs) via span-aggregation (vs. peak-driven top-k) to fix mid-prompt token eviction from Streaming Head dominance in GQA models. Also proposes offline Frobenius-norm error analysis for layer-adaptive budgeting, avoiding online attention stats dependency.
- Outperforms 6 baselines on multiple GQA models (including 14B/32B scales) across LongBench (16 subtasks) and Needle-in-a-Haystack (2K–128K contexts). Key results: +2.5 points vs. HeadKV (256-token budget) and 90% NIAH accuracy with 0.7% cache. Ablations and integrations confirm robustness; A100 profiling validates latency/memory efficiency.
- Addresses KV cache bottlenecks for long-context LLM deployment (e.g., 99% LongBench performance with 19% cache), cuts memory/inference costs for RAG/dialogue. Compatible with other optimizations and informs future attention/model compression research.

### Weaknesses
**Methodological Limitations**: The identification of Semantic Retrieval Heads (SRHs) relies on Needle-in-a-Haystack-style prompts, which may bias toward retrieval tasks (e.g., QA) and limit generalization to non-QA scenarios like creative generation or multi-turn dialogues. For instance, the SRH scoring formula (Eq. 5) assumes clear answer spans but may degrade to top-1 criteria for short or ambiguous spans, reducing accuracy. Additionally, the offline Frobenius-norm error analysis uses LongBench with extreme compression (32 tokens/layer), but sensitivity to dataset choice or compression ratio is unexplored, potentially leading to suboptimal allocations in unseen domains. The fixed top-k SRH selection (top-4 per layer) lacks theoretical justification; ablations (Table 3) show performance plateaus after top-6, but adaptive k is not considered for better robustness. In GQA models, sharing token indices across heads in a layer assumes uniformity, yet per-group indices are not validated as potentially superior.

**Experimental Gaps**: While benchmarks like LongBench and Needle-in-a-Haystack are standard, comparisons omit recent methods such as KIVI (Liu et al., 2024) or D2O (Wan et al., 2025). Evaluations do not extend to real-world long-context applications, such as RAG systems or agentic tasks, where KV cache grows dynamically. Peak memory and latency profiling (Figure 7) on A100 GPUs lacks throughput metrics (e.g., tokens/sec) or multi-GPU scaling. Failure case analysis is absent, e.g., why performance dips in code completion tasks.

### Questions
Although Section 2.2 explicitly discusses D2O (Wan et al., 2025) as a recent dynamic cache allocation method, it is omitted from the experimental comparison. Since D2O shares a similar objective—dynamic, importance-aware cache budgeting—its exclusion leaves a gap in the empirical evaluation. The authors should either include D2O as a baseline or clarify why it is not directly comparable

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes CompressKV, a KV-cache compression framework that leverages attention head semantics to identify and discard less important tokens before generation. First, it proposes to locate “Semantic Retrieval Heads” (SRHs)—attention heads whose aggregated attention over answer spans correlates with semantic relevance—and use them to guide token eviction. Second, it introduces an error-aware, layer-adaptive cache allocation strategy to distribute compression budgets across layers according to estimated reconstruction error.

### Strengths
1) Robust Semantic Head Identification. The proposed identification of Semantic Retrieval Heads through answer-span attention aggregation effectively mitigates the limitations of conventional top-k single-token attention methods, which may overlook semantically distributed relevance.
2) Efficient Offline Layer-wise Importance Estimation. The offline computation of per-layer importance avoids the heavy online overhead faced by methods like CAKE or PyramidKV, enabling more efficient runtime compression without additional inference latency.
3) Strong Empirical Generalization. The proposed method is comprehensively evaluated across multiple large-scale LLMs and adapted with orthogonal methods, demonstrating robust and compatibility performance improvements.

### Weaknesses
1) Potential Circular Reasoning. If SRHs are identified using ground-truth answers and later evaluated on the same benchmark, the method may inadvertently benefit from prior exposure to the correct spans, leading to overly optimistic results. Although stages strength 1 and strength 2 appear effective in identifying semantic and streaming heads, the process relies on test-set analysis of head and layer behavior based on known answers.
2) Limited Generalization Beyond Retrieval-Oriented Tasks. The method is primarily validated on retrieval-style or short-answer tasks, whereas its effectiveness on complex reasoning or long-form generation tasks remains untested and uncertain.
3) Offline Dependency Restricts Plug-and-Play Usability. Since the offline analysis depends on the test dataset to compute head and layer characteristics, the current framework cannot yet serve as a fully plug-and-play compression module applicable across unseen domains or tasks.

### Questions
1) Dataset for Offline Evaluation. Which dataset was used for the offline analysis to compute head- and layer-level importance? Was it the same as the test set, or a separate held-out dataset?
2) Cross-Task Consistency. If different datasets or task types are used for the offline phase, do the identified retrieval heads and the resulting layer-wise allocation patterns remain consistent, or do they vary substantially?
3) Per-Head Token Selection. In Section 3.3, the paper states that “all the attention heads within a layer share a common set of selected token indices”. Why was a shared selection used instead of assigning distinct token indices for each head? Have the authors explored whether per-head token selection could yield better fine-grained compression or semantic fidelity?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The method define semantic retrieval heads for KV cache compression, by using the definition of semantic retrieval score to guide token importance and KV-cache eviction, therefore the method effectively balances the streaming-head and conditional retrieval-head. 

Experiments show that when applying semantic retrieval heads, the KV cache can be retained over 97% of full-cache performance using only 3% of KV cache on LongBench’s question-answering tasks and achieves 90% of accuracy with just 0.7% of KV storage on Needle-in-a-Haystack benchmark.

### Strengths
The attention heads are identified as semantic retrieval heads for a high ratio of KV Cache compression: 1) the Semantic Retrieval Score is defined over the entire answer span inserted into a long context; 2) then the score is averaged and ranked to determine the important tokens and evicted tokens, then the important token indexes are shared across different heads; 3) Although the compression budget are adaptive allocated for different heads by using the error-aware method.

Therefore the KV cache can be effectively compressed, while the performance of long-context scenarios is maintained, especially the long-input scenarios.

### Weaknesses
The concept of semantic retrieval heads is based on retrieval heads for important information storage and retrieval. And the token eviction may lead to significant information loss, especially for the long-cot or ReAT agent scenarios.

And the compression is based on the statistics of Semantic Retrieval Score, which relies on the Attention weights of tokens within the answer span, so the online overhead may be large, hindering the practical application.

### Questions
1. In the future, the native sparse attention may be popular, how can your method adapt to these methods?
2. How this method adapted to PageAttention, for system implementation?
3. How can this method used with quantization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the KV cache memory limitation in large language models by introducing an efficient KV cache eviction strategy. The authors first analyze attention heads across layers to identify those that can effectively retrieve not only the initial and final tokens of a prompt, but also key intermediate tokens and their relevant semantic context. These selected heads are then used to score token importance, ensuring that only the most informative tokens and their associated KV pairs are retained. The proposed method demonstrates strong effectiveness, preserving over 97% of full-cache performance while using only 3% of the KV cache on LongBench question-answering tasks. Additionally, the method achieves 90% accuracy with just 0.7% KV storage on the Needle-in-a-Haystack benchmark.

### Strengths
- The paper includes comprehensive experimental comparisons with multiple baseline methods, demonstrating improvements on LongBench and Needle-in-a-Haystack.
- The approach delivers strong end-to-end inference efficiency, highlighting its practical applicability for long-context scenarios.

### Weaknesses
Several claims require corrections or proper references
- The statement that prior work treats all attention heads equally (L154) is inaccurate. For example, DuoAttention (ICLR 2025) explicitly differentiates retrieval and streaming heads, contradicting this generalization.
- Claims in L169–170 regarding reliance on attention statistics (e.g., entropy, variance) lack citations or supporting references.
- (minor) L172, which states that methods adopt a fixed allocation strategy based on attention distributions, does not apply to recent approaches such as KVZip (2025).

Several claims require stronger justification
- The analysis in Section 3.1 (L187–210) is based only on a single illustrative example without statistical support. Important details such as the model, dataset, and generalizability of the observation across tasks and architectures are not clarified.
- The motivation for (L258-264) "Instead of relying on attention statistics as in the previous methods, this approach quantifies the compression error caused by KV cache compression" is not clearly supported by theoretical or empirical evidence. Also, there are no isolated empirical studies demonstrating the effectiveness of this proposal. 

Empirical analysis could be improved
- For real-world deployment, maintaining full-cache performance is a critical aspect of KV compression. However, direct comparisons to full-cache settings are currently missing.
- The evaluation mainly focuses on benchmarks that are becoming outdated (LongBench and NIAH). Incorporating more recent datasets such as LongBench V2 (2024) or SCBench (2025) would provide a more up-to-date and competitive evaluation.

Minor: 
- Including hyperlinks for all figure and table references would improve readability.

### Questions
Please refer to weakness section above.

### Soundness
1

### Presentation
1

### Contribution
2
