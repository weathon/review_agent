# Retrospective Sparse Attention for Efficient Long-Context Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Large Language Models (LLMs) are increasingly deployed in long-context tasks such as reasoning, code generation, and multi-turn dialogue. However, inference over extended contexts is bottlenecked by the Key-Value (KV) cache, whose memory footprint grows linearly with sequence length and dominates latency at each decoding step. While recent KV cache compression methods identify and load important few tokens, they focus predominantly on input contexts and fail to address the cumulative attention errors that arise during long decoding. In this paper, we introduce RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using newly arrived KV entries from subsequent decoding steps. By maintaining a lightweight output cache, RetroAttention enables past queries to be efficiently supplemented with more contexts, while incurring minimal latency overhead. This breaks the fixed-attention-output paradigm and allows continual correction of prior approximations. Extensive experiments on long-generation benchmarks show that RetroAttention consistently outperforms state-of-the-art (SOTA) KV compression methods, increasing effective KV exposure by up to 1.6$\times$ and accuracy by up to 21.9\%. We provide anonymized code in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper “Retrospective Sparse Attention” introduces an efficient attention mechanism designed to reduce the memory and computation cost of attention without sacrificing long-range dependency modeling. Instead of attending to all past tokens, the proposed Retrospective Attention mechanism selectively attends to a subset of important historical tokens determined by sparsity patterns guided by attention statistics from previous layers or steps. This retrospective sparsity is dynamically adapted based on contextual relevance, allowing the model to focus on informative past activations while maintaining global context coverage. The method is evaluated on long-context language modeling and sequence tasks, showing comparable performance to dense attention while significantly lowering memory and compute usage. The method achieves up to 21.9% and on average 5.6% accuracy gains over state-of-the-art baselines across long-context benchmarks

### Strengths
1. Effective expansion of effective KV budget: Through retrospective updates, each query effectively accesses 1.6× more KV entries than the allocated cache budget, without any increase in actual memory usage
2. Clear motivation and empirical validation: The paper convincingly identifies a progressive degradation problem in long-sequence generation due to frozen past attention and directly addresses it via backward refinement
3. Novel mechanism for long-context refinement: RetroAttention effectively mitigates cumulative attention degradation during long decoding, a limitation common in prior sparse attention methods

### Weaknesses
1. Weak ablation and sensitivity analysis: While the method introduces retrospective window size w as a key parameter, the results (Fig. 7) show saturation beyond w=8, but there is no discussion of why or how this affects efficiency or convergence
2. Ambiguity in the update mechanism: The process of overwriting old KV entries and how it interacts with downstream layers is conceptually underexplained
3. Limited scope of benchmarks: TThere are no experiments on code-completion settings where long-context dependencies differ.

### Questions
1. Is the proposed method only tested on multi-questions & multi-answers? Have you tested it on multi-turn conversations? It seems to be worthwhile to apply this method on multi-turn conversations
2. How sensitive is performance to w, and what governs the optimal trade-off between accuracy and computation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RetroAttention, a technique that retroactively updates past attention outputs using KV entries loaded at later decoding steps. The method keeps a lightweight output cache and, within a fixed retrospective window, recomputes supplementary attention for previously “unseen” KV pages, merging them with cached outputs to correct cumulative approximation errors in long generations. Claimed benefits include expanding the effective KV exposure without increasing the actual KV budget and adding only marginal latency. Experiments on LONGGENBENCH (GSM8K, MMLU, CSQA) and PG-19 are reported; the paper claims consistent gains over Quest under the same budget, and small, context-length–independent overheads.

### Strengths
+ The work introduces a new method for updating past KV caches using information from future tokens, allowing the model to retrospectively correct potential errors accumulated in earlier cache entries.

+ Concrete mechanism with system awareness. The design (supplementary attention + output cache + KV overwrite downstream) is explicit and connected to a memory-bound analysis; figures and pseudo-code aid clarity.

### Weaknesses
- Motivation not fully convincing relative to simpler practices. The paper assumes non-trivial benefits from recomputing within a small local window. However, in LServe-style serving (LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention), adjacent tokens often select the same pages—so neighboring queries tend to attend to similar KV slices. If adjacent queries already share top-k pages, retrospective supplementation within a small window may add little new signal while increasing complexity. The current empirical narrative does not isolate this effect or demonstrate cases where the new pages actually differ materially from what neighbors already saw.

- Limited breadth of models and tasks. The main results use LLAMA-3.1-8B-Instruct and swaps to DeepSeek-R1-Distill-Llama-8B for a subset of reasoning tasks. Absent are larger and diverse foundation models (eg. Qwen); conclusions about generality are therefore weak.

- Overhead and complexity vs. benefit. The method adds supplementary attention, output caching, mask bookkeeping, and KV overwrites across layers. The latency analysis argues the overhead is memory-bound and “negligible,” but even the authors’ figures indicate non-zero, window-dependent costs; moreover, these kernels complicate inference stacks and may reduce effective batch size , weakening throughput. The paper does not quantify batch-size impacts or serving-level throughput with and without the feature enabled.

- Insufficient ablations on design choices. For example, the choice of window size w, the policy for “unseen” page detection, and the exact merging schedule are only partially ablated; sensitivity to k, page size, and per-head selection policy is unclear beyond the limited plots.

- By the way, the paper states anonymized code is provided, but the link is not accessible on our side.

### Questions
- The core issue lies in the lack of convincing motivation. Within a block, the tokens are usually highly correlated, and in most sparse-attention implementations, the selected KV pages for neighboring tokens are largely identical. You should discuss in detail the perspective presented in the LServe paper, which shows that adjacent tokens typically attend to the same KV pages. To strengthen your argument, please provide additional experiments that clearly demonstrate the necessity of updating the KV cache within a sliding window and justify why such retrospective updates are beneficial.

- All other detailed comments can be found in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
RetroAttention targets the long-context inference bottleneck, the fact that the KV cache that grows linearly with sequence length, and the cumulative errors introduced by sparse/compressed attention during extended decoding. The method retrospectively revises past attention outputs using KV entries fetched at later decoding steps. It does this by computing the supplementary attention for earlier queries with currently loaded but previously unseen KV pages, and storing/updating these results in a lightweight output cache. Original and supplementary outputs are merged via a softmax-weighted linear combination, and the refined embeddings propagate upward by overwriting stale K/V entries in deeper layers.

### Strengths
1. RetroAttention has consistent gains over strong baselines while adding only marginal latency thanks to its memory-bound design.

2. The scheme expands each query's effective KV budget by up to 1.6x without increasing the actual KV budget or KV memory traffic. This helps improve contextual completeness under fixed memory. This is a very nice approach and I liked this part.

### Weaknesses
1. This method works only long-generation scenarios. This approach wont help if KV cache overheads are not a problem (such as cases using extreme quantization). 

2. An interesting fact is that the methods are not very simple (which would have been preferred). The benefits saturate as the retrospective window increases (nearly w=8). The approach introduces extra kernel steps (mask management, output merges, KV overwrites) that, while small, add complexity and non-zero overhead.

### Questions
1. Could RetroAttention be combined with eviction-style key selection (e.g., Keyformer) or MorphKV (Dialogue without limits paper) so that retrospective updates counteract the cumulative errors introduced by permanently discarded tokens, rather than conflicting with eviction's objectives?

2. Given the limited gains on long-input/short-output settings like LONGBENCH, would token-selection methods (Keyformer/MorphKV or similar strategies) fare better there? Also, could a hybrid (selection + retrospective updates) recover benefits across both regimes?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RetroAttention, a accuracy-refinement method for efficient long-context generation in large language models (LLMs). Unlike prior approaches (e.g., Quest, StreamingLLM, TOVA), which focus solely on compressing or sparsifying the current decoding step, RetroAttention retrospectively revises past attention outputs using newly arrived Key-Value (KV) entries from later decoding steps. Experiments across multiple long-generation benchmarks (LONGGENBENCH, AIME24, GPQA-DIAMOND, LIVECODEBENCH, and PG-19) show that RetroAttention improves effective KV exposure by up to 1.6× and yields up to 21.9% accuracy gains over state-of-the-art baselines, with small latency and memory overhead.

### Strengths
S1. This paper tackles an important problem of the refinement of sparse KV cache selection.

S2. In stead of proposing KV cache sparsification approaches, this paper proposes a refinement approach for the KV cache sparsificaitons, which has the potential to be applied to many backbone sparsifications.

S3. The proposed approach is a backed up with formal time and space overhead analysis.

### Weaknesses
W1. The experiments are limited to small scale models. Experiments on at least medium size, e.g., 32B models, would be useful.

W2. The comparison approaches are limited. More recent KV cache compression approaches, like ICECache, ArkVale, PQCache, MagicPIG, etc, should also be evaluated.

W3. I would suggest to also test larger cache reuse depth, e.g., w > 5, to show the effectiveness of reusage.

### Questions
Q1. It is not straightforward why the proposed approach is less effective on long context, which basically also suffers from the lower retrieval accuracy problem.

Q2. Would you also consider to use the retrieved context of the previous tokens to enhance the current tokens, since they are already in the GPU memory and they are expected to be in the top-k/2k/3k window? Why or why not?

Q3. Whether RetroAttention will work on distributed decoding? 

Q4. How would RetroAttention work on MoE models?

Q5. How would RetroAttention interplay with different KV cache retrieval strategies, like those listed in W3?

### Soundness
3

### Presentation
3

### Contribution
3
