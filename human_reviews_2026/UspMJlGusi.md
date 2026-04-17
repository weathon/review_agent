# Scaling Attention via Feature Sparsity

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Scaling Transformers to ultra-long contexts is bottlenecked by the $O(n^2 d)$ cost of self-attention. Existing methods reduce this cost along the sequence axis through local windows, kernel approximations, or token-level sparsity, but these approaches consistently degrade accuracy. In this paper, we instead explore an orthogonal axis: \emph{feature sparsity}. We propose \textbf{Sparse Feature Attention (SFA)}, where queries and keys are represented as $k$-sparse codes that preserve high-dimensional expressivity while reducing the cost of attention from $\Theta(n^2 d)$ to $\Theta(n^2 k^2/d)$. To make this efficient at scale, we introduce \textbf{FlashSFA}, an IO-aware kernel that extends FlashAttention to operate directly on sparse overlaps without materializing dense score matrices. Across GPT-2 and Qwen3 pretraining, SFA matches dense baselines while improving speed by up to $2.5\times$ and reducing FLOPs and KV-cache by nearly 50\%. On synthetic and downstream benchmarks, SFA preserves retrieval accuracy and robustness at long contexts, outperforming short-embedding baselines that collapse feature diversity. These results establish feature-level sparsity as a complementary and underexplored axis for efficient attention, enabling Transformers to scale to orders-of-magnitude longer contexts with minimal quality loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Sparse Feature Attention (SFA), which enforces k-sparse query/key vectors and computes attention only on overlapping active feature coordinates, reducing the arithmetic of QKᵀ from Θ(n²d) to Θ(n²k²/d). To make this practical, the authors propose FlashSFA, an IO-aware kernel that integrates sparse overlaps into the FlashAttention tiling/online-softmax pipeline without materializing n×n score matrices. Experiments on GPT-2 and Qwen3 pretraining, synthetic long-context retrieval (NIAH), and several downstream tasks claim to match dense-attention quality while yielding up to ~2.5× speedups, ~49% FLOPs, and ~41% KV-cache savings, particularly at long contexts.

### Strengths
- Clear, orthogonal idea: Moves the sparsity lever from tokens to feature coordinates, preserving token coverage while lowering per-interaction cost (Θ(n²k²/d)). The complexity argument is clean and intuitive.
- Kernel contribution: FlashSFA marries sparse overlaps with FlashAttention’s online softmax/tiling, avoiding n² materialization and targeting the actual memory bottleneck.
- Evidence of efficiency/quality trade-off: At fixed models, SFA often matches dense perplexity/accuracy and beats “short embeddings” that shrink d, with up to ~2.5× speed, ~49% FLOPs, and ~41% KV-cache reductions reported.
- Long-context relevance: On synthetic NIAH and long-sequence latency/KV scaling, SFA’s advantages grow with context length/head dimension, aligning with the paper’s motivation.
- Compatibility: Conceptually composes with token sparsity/paging; could multiply gains in real systems.

### Weaknesses
- Missing or thin comparisons to modern long-context/efficient baselines beyond “short embeddings”: e.g., token-sparse (Longformer/BigBird/Routing), KV-pruning/paging (H2O, Quest, SnapKV), low-rank keys (LoKI), and kernel methods (Performer/Nyström). SFA should be combined and compared head-to-head under the same training/inference regimes.
- The k-sparsification uses row-wise Top-k with a straight-through estimator; stability/variance, gradient bias, and sensitivity to k (and to head dim d) are not adequately analyzed.
- “Short-embedding” baselines may be overly weak (simple linear downsizing). Stronger compression baselines (e.g., low-rank Q/K, grouped-query attention variants, learned projections) should be included.
- Reported GSM-8K lags for SFA vs dense after finetuning suggest arithmetic reasoning sensitivity to feature pruning; this deserves deeper diagnosis and mitigation (e.g., adaptive k, soft-top-k).

### Questions
- Can you add head-to-head comparisons with H2O, Quest, SnapKV, LoKI, Routing/BigBird/Longformer, Performer/Nyström at the same training/inference configs (same context windows, same datasets), and report both quality and speed/KV?
- GSM-8K drops relative to dense FT—can adaptive k or hybrid (dense for a subset of heads/layers) recover this?

### Soundness
1

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
3

### Summary
This paper proposes a novel method of introducing sparsity along the feature dimension of attention computation. Specifically, instead of directly introducing sparsity on the attention matrix, the paper proposes to instead introduce in the key and query matrices by retaining only the top-$k$ largest magnitude entries along the feature dimension. Naively, the full attention matrix requires $O(n^2)$ memory to store and process, but the authors combined this sparsity idea with FlashAttention, which processes attention weights by a tiling mechanism that only requires a compact score buffer.

Experiments show that the proposed method achieves significant latency reduction while retaining most of the performance of dense models with only minimal degradations, measured both in terms of PPL (pre-training setup) and NIAH/Math/QA performance (post-training setup). Additionally, the paper also shows that sparsity patterns can be easily introduced by fine-tuning, thus opening the door of converting dense pre-trained models into sparse models for more efficient inference.

### Strengths
1. The simplicity of the method is quite impressive considering how significant of an efficiency gain it was able to achieve.
2. Performance preservation has been verified thoroughly, covering both LLM pre-training and post-training measures.

### Weaknesses
1. One main drawback of the paper is the lack of discussion and comparison with the DeepSeek Multi-Head Latent Attention, which computes a low-rank compression of the key/value pairs. To me, there is a pretty significant resemblance between those two ideas, even though MLA did not explicitly explore sparsity. Some form of detailed discussion on the differences of those two ideas and some empirical comparison would further highlight the benefit of the proposal.
2. Another minor complaint is that the experiments are mostly on toy tasks and are of pretty small model scale. I know the authors might be limited by the compute they can access, but it would be great if there's at least one experiment on a medium-scale model (7B-14B) which verifies that the latency benefit is able to extrapolate over model scale.

### Questions
1. For Table 1, how is the "Best" in "Best results highlighted" defined? I'm confused because SFA numbers are highlighted on the OWT/Pile scale even though dense models achieved lower numbers.
2. For Table 2, your dense model has a dimension size of 2. Is that really the case? That looks like a typo to me because there is an experiment with $k=8$ below.
3. Can you clarify on the comment on L376 -- "sparse kernels incur lookup overhead"? What overhead are you referring to?

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the $O(n^2d)$ computational bottleneck of self-attention in Transformers. While existing methods focus on sparsity along the sequence axis, this paper explores an "orthogonal axis: feature sparsity". The authors propose Sparse Feature Attention (SFA). SFA represents queries (Q) and keys (K) as "k-sparse codes", claiming to reduce the attention cost from $\Theta(n^2d)$ to $\Theta(n^2k^2/d)$. To implement this efficiently, the paper introduces FlashSFA, an "IO-aware kernel that extends FlashAttention"  to operate on sparse overlaps without materializing dense score matrices. Experiments on GPT-2 and Qwen3 pretraining show SFA matches dense baselines while improving speed and reducing FLOPs and KV-cache.

### Strengths
1. **Originality**: The paper's primary strength is shifting the optimization focus from the token axis to the "feature axis". This approach is "orthogonal to token-level sparsity and paging".

2. **Theoretical Idea**: The concept of using k-sparse codes to reduce the computational complexity of the $QK^\top$ interaction is novel.

3. **Empirical Quality** (as reported): The reported results, if verifiable, suggest that SFA can maintain "comparable quality to dense attention" (Table 1 , Table 3 ) in pretraining and fine-tuning scenarios.

### Weaknesses
* 1. **Reproducibility**: The paper's central empirical results (e.g., "up to 2.5×" speedup , Figure 3 , Figure 5 ) are entirely dependent on the "IO-aware kernel" FlashSFA. This kernel is positioned as a primary contribution. However, the paper provides only high-level pseudocode (Algorithm 1 ) and a brief description (Appendix C). The kernel code itself, which is essential for reproducing any of the speedup benchmarks, is not provided.
  * Parallelism Model: How are CUDA threads, warps, and blocks mapped to the tile computations? How are the nested loops (Line 2, Line 4) parallelized?
  * Atomic Operations: How are the "scatter-adds" to the `scores` buffer (Line 11) handled? Multiple threads processing different features will inevitably write to the same `scores[r, c]` location, requiring atomic operations which are not mentioned but are critical for correctness and performance.
  * Memory Management: The algorithm hides all memory access patterns. How are reads from the sparse `Q` and `K` structures (e.g., `Q_indices`, `Kf_indices`) coalesced for efficient HBM bandwidth usage? How is the `scores` buffer  truly managed in SRAM (shared memory) across threads?
  * Core Data Structures: The algorithm relies on a non-standard $CSC_{feat}(\overline{K})$ format. The precise memory layout of this "feature-wise" CSC matrix is not defined.
  * Search Implementation: The efficiency of `BINARY_SEARCH_RANGE` (Line 7) is crucial. Its implementation details in a parallel CUDA environment are non-trivial and un-discussed.

* 2. **Complexity Analysis that Ignores the P@V Bottleneck**: The paper's headline cost analysis, $\Theta(n^2k^2/d)$ (Abstract, Section 3.1), is confusing. This analysis only accounts for the $QK^\top$ computation. The authors explicitly state they are "keeping V dense". The authors admit in their own Appendix B.2 that "a large proportion of the floating-point operations in the sparse version come from matrix multiplication in the P@V stage". This non-sparsified $O(n^2 d_v)$ operation remains a quadratic bottleneck. The paper's main analysis in Section 3.1 ignores this, and the resulting speculative claim of a "reduction of more than 1000x" is unsubstantiated as it omits a dominant, quadratic term that the authors themselves acknowledge.

* 3. **Core Operational Cost (Top-k) in Analysis**: The paper's efficiency analysis in Section 3.1 omits the computational cost of the $Topk_k$ operation (Eq. 3). This operation must be performed on both Q and K matrices (size $n \times d$) for every forward and backward pass. While the authors mention using an "RTopK kernel" and "lookup overhead", they never incorporate this non-trivial (e.g., $O(ndk)$ or $O(nd \log d)$) cost into their formal complexity model, making that analysis incomplete.

* 4. **Limited Applicability and Inherent Distribution Shift**: The authors' own experiments in Section 5 demonstrate that applying SFA to pretrained models "introduces a severe distribution shift". The authors' solution is a multi-stage, regularized fine-tuning process, which requires "an additional MSE loss" and even pre-finetuning on another dataset. This implies the method's high adaptation cost.

### Questions
1. Can the authors provide the corresponding implementation details for the FlashSFA kernel, wrt the above questions in W1?

2. Can the authors provide a revised complexity analysis for the entire SFA block, including the $O(ndk)$ or $O(nd \log d)$ cost of Top-k and the $O(n^2 d_v)$ cost of the P@V multiplication? How does this revised, complete analysis affect the scaling claims?

3. Given that P@V is a "large proportion" of the FLOPs, why was V not sparsified? Did the authors experiment with this, and if so, what was the impact on performance?

4. The $k^2/d^2$ scaling analysis (Eq. 7) relies on an assumption that "supports are balanced across dimensions". Is there empirical evidence for this? What happens to performance and cost if the feature selection is unbalanced?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a feature sparse attention to improve the prefill time efficiency. They provide a FlashSFA kernel which integrates their idea into flash attention kernel. They show that models can be trained from scratch to get reasonably close perplexity and downstream accuracy as compared to dense models.

### Strengths
1. Feature sparsity is not well explored in literature to the best of my knowledge. This paper does that
2. Opensource FlashSFA kernel that be used by the community.

### Weaknesses
1. Lack of baselines.
2. experiments are not convincing. There is degradation in quality when compared to baseline. Requires separate training for retrieval task.

### Questions
1. Does FlashSFA also provide improvement in time to next token (TTNT)? From what i understand it can only provide improvement in time to first token(TTFT). In table 2 the speedups refer to TTFT or TTNT?

2.Related to above, given a $k$, what is the memory footprint of storing $k$ indices for one row in CSR format isnt it $3k$? So if we had d=64, $k < 64/3$ is required to get any memory gains.  is that correct understanding?

3. While the paper dismisses sparsity at sequence level as degrading accuracy, we do not see any comparison with such methods in experiments. sparsity in training (NSA / DSA) would be good baselines in my understanding. Can Authors elaborate more on the what are good baslines for FlashSFA in sparse attention techniques. 

4. Related to above, the FlashSFA also shows degradation across the board compared to baseline. So it is reasonable to compare against other sparse attention techniques at training time. 

5. The fact that we had to train for retreival tasks indepdnently from scratch is a bit worrying. It might have something to do with the information bottleneck in SFA. In principle we are looking for a model that can do all of the tasks. training for specific task is against the foundational model paradigm. 

6. How does top-k coordinate distribution look like in practice. Equation 7 assumes that it is balanced. But in practice (atleast in the case of dense models), some channels can show outlier distribution (See AWQ quantization).

7. Also, authors mention that the benefits are multiplicative and their technique can be combined with sequence-sparse attention methods.  Is there some evidence to this effect in authors' experiments / literature.

### Soundness
2

### Presentation
2

### Contribution
2
