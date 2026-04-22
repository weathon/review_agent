# ProxyAttn: Guided Sparse Attention via Representative Heads

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
The quadratic complexity of attention mechanisms limits the efficiency of Large Language Models (LLMs) on long-text tasks. Recently, methods that dynamically estimate block importance have enabled efficient block sparse attention, leading to significant acceleration in long-text pre-filling of LLMs. However, their block-level coarse-grained estimation inevitably leads to performance degradation at high sparsity ratios. In this work, we propose ProxyAttn, a training-free sparse attention algorithm that achieves token-level estimation by compressing the dimension of attention heads. Based on our observation of the similarity among multiple attention heads in long texts, we use the attention scores of pooled representative heads to approximate the scores for all heads. To account for the varying sparsity among heads, we also propose a block-aware dynamic budget estimation method. By combining the scores from a set of representative heads with a multi-head dynamic budget, we can achieve a more fine-grained block attention evaluation at a low computational cost. Experiments on a variety of mainstream models and extensive benchmarks confirm the underlying similarity among attention heads in long texts. Leveraging a token-level fine-grained estimation, the proposed method achieves substantial gains in performance and efficiency compared to existing methods. More precisely, ProxyAttn can achieve up to 10.3x attention acceleration and 2.4x prefilling acceleration without significant performance loss. Our code is available at https://github.com/wyxstriker/ProxyAttn.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ProxyAttn, a method for improving block selection in block-sparse attention. Instead of reducing along the sequence dimension to estimate high-attention regions, ProxyAttn computes a subset of attention heads and uses them as proxies to estimate which blocks to compute for the remaining heads. Experiments shows improved performance and efficiency when using the proposed sparse attention method.

### Strengths
1. The paper presents an interesting insight into the similarity of attention trends across different heads.
2. The design choices of the proxy attention mask and dynamic budget allocation are well-motivated by corresponding empirical observations and oracle analyses, enhancing the interpretability of the proposed method.
3. The proposed approach outperforms other sparse attention baselines on retrieval tasks (RULER benchmark), matching or even surpassing the dense attention baseline.

### Weaknesses
1. **Inconsistent performance on long-context understanding tasks:** Although the proposed method performs well on retrieval tasks, its advantage is less obvious on more realistic long-context benchmarks. This raises questions about whether the observed attention pattern similarity generalizes beyond retrieval settings, since real-world tasks often involve richer semantics and more diverse attention behaviors.

2. **Definition clarity:** Some core concepts, such as *attention score*, are ambiguously defined. For instance, does it refer to the $(i, j)$ entry of the attention matrix, or the aggregated attention value across columns?

3. **Figure clarity:** Several figures are not clearly explained. For example, the meaning of the x-axis in Figure 2(c) is unclear. How does it support the claim that the primary variation between heads lies not in the tokens they attend to?

### Questions
1. In the attention head consistency oracle experiment (Figure 2a), how much of the overlap is attributable to attention sinks? What happens if the influence of attention sinks is removed?
2. What does the term “average score” in Line 149 refer to? Is it the mean *attention score* across heads or something else?
3. Does the proxy head mechanism perform consistently across both retrieval and question-answering scenarios?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ProxyAttn, an attention approximation method designed to accelerate the pre-filling phase.

The method is based on the key observation that attention patterns across different heads within the same layer are highly similar, differing primarily in their sparsity. Therefore, ProxyAttn employs a pooling head to generate shared attention scores for top-K block retrieval.

Furthermore, the authors propose using the query from each head's final block to estimate its specific sparsity budget, which enables a more effective resource allocation.

### Strengths
1.  The paper's motivation is well-articulated. Through multi-faceted comparisons and visualizations, it substantiates the hypothesis that differences between attention heads lie not in their *patterns*, but rather in their *magnitude* and *sparsity*. This insight validates the use of a pooling head for important block selection as a reasonable approach.

2.  The experiments are thorough, conducted across the Ruler, LongBench-v2, and InfiniteBench benchmarks. The evaluation covers two mainstream models, Qwen and LLaMA, lending strong reliability to the results.

### Weaknesses
1.  The paper critiques prior methods for coarse-grained block retrieval. However, its own method still employs a stride, resulting in block-wise retrieval and top-k computations. This merely mitigates the coarse-granularity issue rather than fundamentally solving it—a true solution is likely infeasible due to prohibitive retrieval overhead.

2.  The method offers limited speedup, achieving only ~2x on a 70B model. The baseline selection seems insufficient, as current mainstream models often use 7x or 8x GQA ratios. In such high-ratio scenarios, FlashAttention's computational speed increases significantly, representing a more practical setting (however, not optimal for your end-to-end TTFT comparison). Moreover, with the rise of local + global attention models (e.g., Minimax, Llama 4), the significance of pre-filling acceleration is diminishing.

### Questions
1. Regarding line 142: "the score at position $(i,j)$ represents the cumulative attention score at head $j$ obtained by the top 1024 tokens from head $i$."This statement is unclear to me. Could you please provide the formula to explain this metric?

2. What is the TTFT of your method on the Qwen2.5-1.5B, 3B, and 7B models?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes ProxyAttn, a training-free sparse attention algorithm for long-context prefill. The key idea is to compress along the head dimension: group attention heads and use a small number of representative “proxy” heads to compute near token-level scores, then max-pool and aggregate them into fine-grained block importance estimates. To accommodate that different heads prefer different sparsity levels, the method adds a block-aware per-head dynamic budget based on a lightweight cumulative-probability criterion. This yields fine-grained, block-sparse masks at low estimation cost, delivering attention speedups and 2.4× prefill acceleration with no significant performance loss.

### Strengths
1. Novel sparse attention design. Instead of compressing along the sequence dimension (tends to be coarse), the paper compresses along the head dimension to obtain token-level importance at the cost of only a few proxy heads, which is conceptually clean and practically effective.  
2. No retraining or model changes; works as a drop-in prefill accelerator and ports across model families.
3. Strong speedups while maintaining full-attention accuracy on long context benchmarks.
4.  The approach is validated by the observation that heads share token focus on long context, justifying proxy-head sharing.

### Weaknesses
I don't see any substantive weakness.

### Questions
1. The method hinges on cross-head token-focus similarity; corner cases with highly specialized heads may degrade the shared ranking quality. Do authors observe tasks or head types where the shared ranking under-represents important blocks (e.g., rare long-range dependencies)?
2. How sensitive is performance to the group size and the choice of the representative head? Could learned or data-driven method further help beyond simple averaging (e.g., fine-tune somes head as proxyheads)? 
3. Could the proxy-heads be reused for decoding with query-aware sparse attention?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ProxyAttn, a training-free sparse attention method designed to speed up long-context Transformers by exploiting redundancy across attention heads. Instead of pruning tokens directly, the authors group similar heads into “proxy heads” and use their aggregated (max-pooled) attention scores to estimate token importance. Each real head then receives its own Top-K allocation based on its intrinsic sparsity. This approach effectively compresses along the head dimension rather than the sequence dimension, which is both novel and impactful. Experiments on large-scale LLMs (LLaMA 3.1, Qwen 2.5) show up to a 10× speedup with almost no drop in accuracy.

### Strengths
1.The method can plug into existing Transformer architectures without retraining or parameter tuning, making it practical for deployment.
2.Using proxy heads to estimate token-level importance leads to higher overall sparsity while preserving key contextual information—often matching or slightly outperforming full-attention baselines on long-context benchmarks.
3.The reported efficiency gains are impressive: up to a 10.3× reduction in kernel-level attention time and a 2.4× speedup in total prefilling latency, with accuracy losses staying under one percentage point—an excellent trade-off for latency-critical use cases.

### Weaknesses
1.The method is only evaluated during the prefill stage; it’s unclear whether similar benefits (or issues) would arise during autoregressive decoding, especially for long generations.
2.Important hyperparameters—like the sparsity threshold and minimum Top-K—are tuned via grid search. The lack of an adaptive or principled selection rule could make it harder to apply the method out-of-the-box to new models or domains.
3.While computational savings are detailed, the paper doesn’t discuss the memory overhead from storing proxy key–value tensors, which could be a concern on GPUs with limited memory.

### Questions
see weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
