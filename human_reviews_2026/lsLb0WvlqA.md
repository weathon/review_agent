# Efficient Attention via Pre-Scoring: Prioritizing Informative Keys in Transformers

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Recent advances in transformer architectures deeply enhanced long-context language modeling. Among them, HyperAttention achieves competitive efficiency by combining a single-level LSH-based clustering with uniform residual sampling. However, HyperAttention fails to find all significant keys, which in turn raises the overall perplexity. We propose a pre-scoring mechanism that prioritizes significant keys before applying HyperAttention. We introduce three scoring methods: $k$-means and kernel $k$-means clustering, $k$-median clustering, and leverage score-based ranking (inspired by LevAttention) to filter keys effectively. We further replace HyperAttention's original uniform residual sampling, relying exclusively on our pre-scoring mechanism. Experiments on ChatGLM2 (131k token context) reduce perplexity from 12 to 8.3, which outperforms standard HyperAttention. Moreover, when running on the Vision-Transformer (ViT), our method shows that it can guarantee similar accuracy compared with LevAttention, and will surpass LevAttention given specific parameters. Although this method introduces some computational overhead, its combination with HyperAttention remains 20 times faster than FlashAttention, providing a balanced trade-off between speed and modeling accuracy. Our results highlight the effectiveness of integrating pre-scoring into hierarchical attention mechanisms, significantly improving transformer efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes an extension of HyperAttention by introducing a pre-selection step using clustering methods (notably k-means) to prioritize informative keys before applying the LSH-based HyperAttention mechanism. The authors provide both theoretical analysis (via a planted-subspace model) and empirical results on long-context LLMs (ChatGLM2/3) and Vision Transformers.

### Strengths
- The planted-subspace model provides a useful lens for analyzing why clustering can recover heavy keys, giving the method some analytical backing
- The idea of preselecting the tokens is intuititive

### Weaknesses
- **Missing prior work:** The paper does not adequately discuss **Routing Transformer[1]**, which also introduced clustering for preselection of tokens. Furthermore, due to conceptual similarities, routing transformer should be one of the compared baselines. Moreover, apart from k-means clustering, **MoSA[2]** recently demonstrated the benefits of expert-choice routing for token preselection, and this should at least be discussed.
- **Algorithmic ambiguity:** The training procedure for k-means clustering is underspecified—does it employ EMA updates with top-s selection, or is clustering recomputed per step? This is important for reproducibility.
- **Autoregressivity concern:** The selection procedure relies on a **top-s operator**, which is inherently non-autoregressive (requiring access to future tokens). The implications for causal language modeling are not addressed.
- **Formatting issues:** Several citations are incorrectly formatted (missing parentheses), which detracts from the paper’s polish.
- **Weak gains over LevAttention:** The results do not demonstrate a significant gain over LevAttention baseline.
- **Convoluted writing:** The writing is often hard to follow, paragraphs seem disconnected, and it is hard to merge them into a cohesive narrative.

[1] - Efficient Content-Based Sparse Attention with Routing Transformers
[2] - Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

### Questions
- Why not **pre-select queries as well**, as done in Routing Transformer?
- Why are **HyperAttention baseline results missing** from Table 2? Without them, it is difficult to measure the incremental gain from pre-scoring.
- Under what conditions does the proposed method **outperform LevAttention**, given that LevAttention appears faster and in some cases more accurate?
- How is **k-means training implemented**—does it rely on EMA updates, online clustering, or recomputation per batch?
- How would the proposed method behave in a **fully autoregressive training regime**, where future tokens are not accessible for top-s selection?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes three methods to score keys before HyperAttention, enabling it to identify important keys: k-means, k-median, and leverage-score ranking. It then feeds the selected keys into HyperAttention, replacing its uniform residual sampling. Experiments on GLM2 show that the approach can be faster than FlashAttention. On ViT, the pre-scoring step captures heavy attention entries as well as, or better than, leverage scores.

### Strengths
+ Clear and practical idea: The paper provides a straightforward approach to enhance HyperAttention by pre-scoring and then attending. This directly addresses a known issue: HyperAttention’s hashing is not aware of which keys matter, and LevAttention’s “universal set” can get large. The bridge between them is simple and useful in practice.

+ Mix of theory and experiments: The paper offers proofs under a standard planted-subspace setup (to argue why the pre-scoring should work) and shows results on GLM2/GLM3 and ViT.

### Weaknesses
- Reason for PPL improvement: The best perplexity (~8.31) happens when pre-scoring is off (top-k = 0, sample_size = 0) and min_seq_len ≥ n_query is set. The paper itself says this gain comes from that configuration (forcing the faster block/tiled path), not from pre-scoring. A clean ablation is needed to separate the effects. 

- Unclear speedup claims: 
> Compared to the original HyperAttention, these methods can generate a mild acceleration, with performance becoming more remarkable starting at $2^{13}$ with a speedup factor of around 3 to 4 in Figure 1.

&nbsp;&nbsp;&nbsp;&nbsp; The abstract says “up to 20× faster than FlashAttention” (when combined with HyperAttention), but the text says around 3–4× at $2^{13}$ for the pre-scored variants, and often the reported gains are relative to HyperAttention. Since HyperAttention is not the paper’s main contribution, this framing can be misleading. Please clarify the exact conditions for 20× vs the 3–4× cases and state whether the 3–4× is typical across settings.

- Narrow baseline: Most comparisons are to HyperAttention and LevAttention. Adding Performer, Reformer, and newer retrieval/streaming methods would make the evaluation more complete and show the accuracy–speed trade-offs more clearly.

- Pre-scoring overhead: k-means and k-median add non-trivial compute and can increase memory traffic, which may shrink speed gains, especially in multi-head settings. Please verify this with profiling across many heads, different head dimensions, and batch sizes, and report how much overhead comes from the forward pass vs backward.

- Limited tasks and metrics: Using broader long-context tasks (e.g., QA, retrieval, summarization) and reporting task-level metrics (not just perplexity) would strengthen the paper and validate the method more fully.

### Minor
- Figure legend and axis text are small and hard to read.

### Questions
See the weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes an attention acceleration scheme that pre-scores keys before applying HyperAttention. The pre-scoring can be done via k-means or k-median clustering, or via approximate leverage scores. The retained set of “informative” keys is then fed to HyperAttention, replacing its uniform residual sampling. Empirically, on ChatGLM2 and ChatGLM3 with LongBench prompts, the method lowers perplexity relative to vanilla HyperAttention and, at certain top-k settings, reports a best PPL near 8.3 from a HyperAttention baseline of roughly 12. It also reports layer-level speedups over FlashAttention for sufficiently long sequences and applies a similar key selection idea to ViT, showing accuracy approaching softmax attention when sampling enough keys. Theoretically, the paper analyzes a planted-subspace model and proves that clustering with  𝑘=𝑑+1 separates “signal” from “noise” rows comparably to leverage-score selection, giving recovery guarantees of heavy keys under assumptions like row-norm regularity

### Strengths
1. Targeting the recall gap of HyperAttention by ranking keys beforehand is a clean, practical idea that directly addresses missed heavy scores. The algorithms are presented with simple wrappers over HyperAttention. 
2.The planted-subspace analysis and Theorems 1–2 formalize when clustering isolates heavy keys, matching the empirical intuition that important keys align with near-orthogonal directions. 
3. Results span LongBench perplexity on GLM2 and GLM3, speed comparisons vs FlashAttention, and a ViT “monkey-patch,” giving a multi-angle view of trade-offs. 
4. The paper breaks down where overhead appears, how it scales with  k and  d, and when speedups emerge, which is valuable for deployment decisions.

### Weaknesses
1. The strongest PPL ≈ 8.3 appears tied to the min_seq_len ≥ n_query configuration and sometimes even top-k set to zero, which partially credits an optimization switch rather than the proposed pre-scoring itself. The paper should isolate gains from pre-scoring vs implementation flags and report both. 
2.Speedups are reported per layer against FlashAttention and discussed asymptotically, but it is unclear how these translate to whole-model throughput and latency under realistic batch sizes and sequence distributions. Consolidated end-to-end metrics are needed. 
3. The paper notes a “corrected coupling” for GLM3 that changes behavior relative to GLM2. This suggests results are sensitive to integration choices. The exact coupling and ablations should be elevated from appendix to main text with code pointers. 
4. The baseline set focuses on HyperAttention, FlashAttention, and leverage-based selection. Given recent efficient attention methods, the empirical section would be stronger with a few additional modern query-aware or block-sparse baselines, or at least a rationale for exclusions. 
5. Guarantees rely on row-norm regularity and separability that may not hold uniformly across layers or modalities. Although LayerNorm helps, some layers can exhibit skewed norms and mixed subspaces. Sensitivity analyses to violations of these assumptions would strengthen the claims.

### Questions
1. How much of the perplexity gain remains when min_seq_len ≥ n_query is disabled and the exact same HyperAttention kernels and fallbacks are used for all methods, including top-k=0 settings? Please provide a clean ablation table. 
2. Can you report end-to-end speed, throughput, and memory vs FlashAttention and HyperAttention on GLM2 and GLM3 for realistic prompt length distributions and batch sizes, not just per layer? 
3. How stable are results to the choice of  k, number of clusters, and initialization of k-means or k-median? For example, do different random seeds flip the identity of retained keys and the downstream PPL curve? 
4. Could you quantify the additional FLOPs and memory of pre-scoring at inference time and show how they amortize with increasing sequence length, for each variant? 
5. Beyond ViT, have you tried audio or multimodal encoders where key distributions differ strongly from text? Any failure cases that violate the planted-subspace intuition or row-norm regularity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new method to efficiently approximate the attention mechanism, with the goal of reducing the computational cost of this operation. The paper heavily relies on previous works LevAttention and HyperAttention, trying to combine the best of both methods. In particular, HyperAttention works by grouping keys and queries into bucket, using locality sensitive hashing (LSH), and then compute the attention only for keys and queries that are in the same bucket. On the other hand, LevAttention selects a subset of the most important keys, and compute the attention over these only.

This paper proposes to select a subset of the keys, using a clustering algorithm such as k-means or k-median, and then to apply the HyperAttention method on the selected keys. The paper also states some theoretical results about the selection process of the keys with the clustering algorith. Finally, some experimental results are provided, replacing the standard self-attention mechanism with the proposed method in existing models such as the GLM language model or vision transformer (ViT) models. Here, the paper show that the method improve the results of LevAttention or HyperAttention.

### Strengths
The paper provides some theoretical analysis for the proposed method, but it is hard for me to understand what kind of guarantee it actually provides (see next section).

### Weaknesses
Overall, I have many concerns with the paper.

First, I found the paper very hard to read. One of the reason is that the authors assume that the reader are very familiar with previous works LevAttention and HyperAttention. For example, many concepts are not introduced in the paper ("heavy attention scores" line 59, "statistical leverage scores" line 99, "polynomial based attention" line 100, "positional locality" line 103, "planted model" line 135, etc...). Similarly, the different theorems or assumptions are stated without motivation or explanation, making it hard to understand how these relate to the performance of the method. Similarly, the algorithm proposed in the paper is never stated clearly, relying on previous paper instead. These different factors made it very hard for me to understand the method and theoretical claims.

Second, the paper mostly discusses LevAttention and HyperAttention as previous work to improve the efficiency of the self-attention. These two works are from 2024 and 2023 respectively, while there exists a wide body of earlier literature addressing this problem, and which are not discussed in the paper. Particularly relevant are the Reformer paper (Kitaev et al, 2020) which also proposes to use LSH to group keys and queries and restrict the self-attention between similar keys and queries, or Routing transformer (Roy et al, 2020) which proposes a similar approach based on k-means clustering.

Third, I found the experimetal results to be unconvincing. The paper only compared the proposed approach to LevAttention and HyperAttention, and not earlier works which led to strong results. Second, the performance of the method seems to be quite poor. For example, on the language modeling experiments, the perplexity obtained with the different approximation techniques considered is above 10, while I believe that the perplexity of the original model is around 6. The performance of the original model should actually be included in Figure 3. Similarly, in Figure 4, the reported results for the considered methods are significantly worse than the original models, showing that the method is probably not useful in practice.

### Questions
What is the perplexity of the original GLM 2 language model (Fig. 3)?

**Missing references**

Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier. 2020. Efficient Content-Based Sparse Attention with Routing Transformers. 

Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya. 2020. Reformer: The Efficient Transformer

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap. 2019. Compressive Transformers for Long-Range Sequence Modelling

Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, Hongsheng Li. 2018. Efficient Attention: Attention with Linear Complexities

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma. 2020. Linformer: Self-Attention with Linear Complexity

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. 2020. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

Manzil Zaheer et al. 2020. Big Bird: Transformers for Longer Sequences.

Iz Beltagy, Matthew E. Peters, Arman Cohan. 2020. Longformer: The Long-Document Transformer

Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever. 2019. Generating Long Sequences with Sparse Transformers

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh. 2021. Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention

### Soundness
2

### Presentation
1

### Contribution
1
