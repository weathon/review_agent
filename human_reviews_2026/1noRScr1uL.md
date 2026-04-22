# MARché: Fast Masked Autoregressive Image Generation with Cache-Aware Attention

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Masked autoregressive (MAR) models unify the strengths of masked and autoregressive generation by predicting tokens in a fixed order using bidirectional attention for image generation. While effective, MAR models suffer from significant computational overhead, as they recompute attention and feed-forward representations for all tokens at every decoding step, despite most tokens remaining semantically stable across steps. We propose a training-free generation framework MARché to address this inefficiency through two key components: cache-aware attention and selective KV refresh. Cache-aware attention partitions tokens into active and cached sets, enabling separate computation paths that allow efficient reuse of previously computed key/value projections without compromising full-context modeling. But a cached token cannot be used indefinitely without recomputation due to the changing contextual information over multiple steps. MARché recognizes this challenge and applies a technique called selective KV refresh. Selective KV refresh identifies contextually relevant tokens based on attention scores from newly generated tokens and updates only those tokens that require recomputation, while preserving image generation quality. MARché significantly reduces redundant computation in MAR without modifying the underlying architecture. Empirically, MARché achieves up to 1.7x speedup with negligible impact on image quality, offering a scalable and broadly applicable solution for efficient masked transformer generation. The code is available at https://anonymous.4open.science/r/MARche-26F0.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript introduces "MARché," a training-free inference acceleration framework specifically designed to optimize the image generation process of Masked Autoregressive (MAR) models. MAR models exhibit significant computational redundancy during inference: attention and feed-forward network (FFN) computations are recalculated for all tokens at each step, even though most tokens remain unchanged. MARché addresses this by leveraging KV Cache through two key components:

1. Cache-aware Attention: Partitions tokens into "Active" and "Cached" sets, designing distinct computation paths (Active tokens recompute QKV and pass through FFN; Cached tokens only provide KV and skip FFN) while maintaining mathematical equivalence.

2. Selective KV Refresh: Recognizes that caches cannot be used indefinitely. This mechanism analyzes attention scores from the 2nd decoder layer to identify and select the Top-K cached tokens most relevant to the currently generating tokens as "Refreshing Tokens," adding them to the active set for forced updates. It also incorporates periodic full refresh (default: every 3 steps) to prevent error accumulation.

Without modifying the MAR model structure or requiring retraining, MARché significantly reduces inference computation. Experiments on the ImageNet 256x256 dataset show that MARché provides up to 1.7x (approaching 1.8x for MAR-H) inference speedup for MAR models, with minimal impact on generation quality (FID/IS).

### Strengths
1. **Inference Acceleration**: The core contribution is practical and impactful, achieving speedups of 1.57x to nearly 1.8x for MAR models (Table 1).

2. **Training-Free**: As a pure inference-time optimization, MARché is easy to adopt and applicable to existing pretrained MAR models without costly retraining.

3. **Preservation of Generation Quality**: The method maintains high fidelity, with only slight increases in FID while significantly reducing latency (Table 1, Fig 9).

4. **Clever Mechanism Design**: The combination of cache-aware attention (with efficient kernel implementation) and attention-guided selective KV refresh is well-motivated and technically sound.

5. **Clear Exposition and Thorough Experimentation**: The paper effectively explains the redundancy problem in MAR models and clearly details the MARché design. It provides comprehensive experimental validation, including extensive ablation studies, strongly supporting the methodology and the role of each component.

### Weaknesses
1. **Dependence on MAR Architecture**: MARché is specifically optimized for MAR models (with fixed generation order and bidirectional attention). It's unclear if this optimization can be extended to other types of masked generative models (like MaskGIT with potentially different generation strategies) or models without a fixed order. Its generality might be limited by the MAR architecture itself.

2. **Hyperparameter Sensitivity**: The effectiveness of selective KV refresh seems dependent on several key hyperparameters, such as the layer chosen for selecting refresh tokens, the number K of refresh tokens (dynamic strategy vs. fixed value), and the frequency of periodic full refresh. While ablations are provided, choosing these hyperparameters might still require careful tuning for specific applications.

3. **Insufficient Comparison with Similar Methods**: The paper mentions another MAR acceleration work, LazyMAR (Yan et al., 2025), and notes differences (reusing hidden states vs. KV cache, changing vs. preserving generation order), but lacks a direct performance comparison (speedup vs. FID). This makes it difficult for readers to assess the relative advantages of MARché among similar approaches.

### Questions
**Generality of the Refresh Token Selection Layer:**

Question: The paper chooses the 2nd layer's attention scores to determine refresh tokens, showing it's a good speed-quality balance (Fig 5). Is this choice optimal across different model scales (MAR-L, MAR-H) and potentially other datasets or tasks? For instance, might deeper models benefit from scores derived from a deeper layer (e.g., Layer 3, whose Top-K selection aligns more closely with even deeper layers, see Fig 4) to maintain quality? Discussion or experiments on generality are suggested.

**Comparison of Dynamic K vs. Fixed K Refresh Strategies:**

Question: The current strategy dynamically adjusts the number of refresh tokens (K) to fill a batch size of 64. How does performance (speed vs. FID) change if a strategy with a fixed K (e.g., always refreshing the Top-50 tokens) is adopted? Does the dynamic K strategy lead to significant fluctuations in computation per step? Which strategy is preferable for practical deployment?

**Potential for Adaptive Periodic Full Refresh Frequency:**

Question: The current approach uses a fixed 3-step cycle for full refresh. Considering that token changes might differ across generation stages (e.g., early stages generating outlines vs. late stages filling details), could an adaptive full refresh strategy (e.g., dynamically adjusting frequency based on token change rate or generation stage) further optimize performance?

**Applicability of MARché to Other Bidirectional Attention Models:**

Question: The core idea of MARché (identifying and caching less-changing KVs) doesn't seem strictly limited to MAR models with a fixed order. To what extent do the authors believe this method could be applied to other iterative generation models employing bidirectional attention (e.g., certain types of image editing models, non-autoregressive iterative refinement for language models, or MaskGIT models with non-fixed generation orders)? What would be the main challenges?

### Soundness
3

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
4

### Summary
This paper presents MARché a training-free acceleration framework for Masked Autoregressive (MAR) image generation models. The method addresses computational redundancy in MAR models through two core components: (1) cache-aware attention that partitions tokens into active and cached sets with independent computational paths, and (2) selective KV refresh that identifies context-relevant tokens for recomputation based on attention scores. Experimental results demonstrate that MARch achieves 1.7× inference speedup on ImageNet 256×256 with negligible impact on image quality.

### Strengths
1. The KV caching mechanism is innovatively adapted from autoregressive language models for masked generation tasks, effectively solving the cache staleness issue via selective refresh. The approach is elegantly designed, with its implementation details clearly articulated.
2. The method is architecture-agnostic and training-free, making it easily integrable into existing MAR frameworks.

### Weaknesses
1. The method incorporates multiple empirical hyperparameters (e.g., fixed active set size of 64, full refresh every 3 steps, use of Layer 2 attention scores). Although ablation studies demonstrate their effectiveness, deeper theoretical analysis or principled design guidance is lacking. For instance, is a fixed size of 64 always optimal across varying image complexities or generation steps? It is suggested to discuss the robustness of these hyperparameters in different scenarios or explore an adaptive mechanism for determining the active set size.
2. Missing analysis of memory overhead: While KV caching accelerates computation, it inevitably increases memory usage. The paper does not quantify or discuss the additional memory overhead introduced by MARché.
3. Experiments are conducted only on ImageNet 256×256, lacking validation on higher-resolution datasets such as 512×512 or 1024×1024.
4. Insufficient comparison with recent works: The paper mentions LazyMAR in the related work section but does not include it as a baseline in the experiments. A direct performance (speed/quality) comparison with LazyMAR is needed.Additionally, while the proposed method achieves a 1.7× speedup, the IS metric decreases significantly, which is not observed with LazyMAR. Please explain the reason for this discrepancy.
5. The discussion of limitations in the conclusion is relatively brief. For example, during stages where the generated content changes abruptly (e.g., transitioning from background to foreground objects), the stability of KV projections may decrease. Could this affect the efficiency and quality of MARché?

### Questions
1. Although ablation studies demonstrate their effectiveness, deeper theoretical analysis or principled design guidance is lacking. For instance, is a fixed size of 64 always optimal across varying image complexities or generation steps?
2. The discussion of limitations in the conclusion is relatively brief. For example, during stages where the generated content changes abruptly (e.g., transitioning from background to foreground objects), the stability of KV projections may decrease. Could this affect the efficiency and quality of MARché?

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
4

### Summary
The paper introduces MARché, a training-free decoding scheme that speeds up masked autoregressive (MAR) image generation by avoiding redundant recomputation. It splits tokens each step into an active set (newly generated + a few contextually relevant “refresh” tokens) and a cached set whose key/value projections are reused via cache-aware attention; relevance is decided by attention from newly generated tokens (selective KV refresh). This preserves full-context modeling (using an online-softmax merge that’s equivalent to standard attention) while skipping unnecessary FFN work. On ImageNet 256×256, MARché reports up to ~1.7× latency speedup vs. MAR with only minor FID/IS changes, and requires no architecture changes or retraining.

### Strengths
Training-Free Speedup: MARché offers a significant speedup in masked autoregressive (MAR) image generation without requiring any retraining, making it highly efficient for real-time applications.

Efficient Use of Cache: By reusing key/value projections from previous steps through cache-aware attention, the method avoids redundant calculations, which reduces computational load while preserving high-quality generation.

No Architecture Modifications: It doesn't require any changes to the underlying model architecture, making it easy to integrate into existing systems.

Performance Gains: The paper reports a 1.7× latency reduction with only minor decreases in image quality metrics like FID (Fréchet Inception Distance) and IS (Inception Score), showing that it delivers efficiency without significantly compromising performance.

Broad Applicability: The approach can be applied to other autoregressive models without much modification, offering potential for wide adoption in various image generation tasks.

### Weaknesses
1 The acceleration gains from the proposed MARché method are modest. Despite achieving a 1.7× speedup, the improvement may not be significant enough for certain applications.

2 The method was primarily tested on the ImageNet 256×256 dataset. Its performance and acceleration effect on larger-scale tasks, such as high-resolution images, video generation, or multi-modal data, remain unclear. If the method does not perform well in these scenarios, its general applicability could be limited.

3 The method lacks significant novelty, as it primarily builds on existing techniques like cache management and selective KV refresh, and its scope is limited to optimizing a specific approach in the rapidly advancing field of autoregressive image generation.

### Questions
see above

### Soundness
3

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
This paper proposes MARCHÉ, a training-free acceleration method for masked autoregressive (MAR) image generation. The key idea is to reuse KV representations that remain stable across decoding steps by maintaining cached tokens and selectively refreshing active ones via attention signals and online softmax. The method reports up to ~1.7× speedup on MAR models with minor quality changes and requires no model modification.

### Strengths
- Training-free & plug-and-play: No architectural change, no retraining, practical for deployment.
- Clear and practical idea: Efficient KV reuse tailored to masked autoregressive generation.

### Weaknesses
1. Lack of evaluation across different MAR schedules
  - The KV-reuse assumption is tested under a single MAR step schedule.
  - It is unclear whether the method generalizes to different decoding steps.

2. KV-cache strategy generality remains uncertain
  - Only evaluated on MAR.
  - It is unknown how well the approach applies to other iterative masked Autoregressive generative frameworks.

3. Comparison to LazyMAR needs clarification
  - LazyMAR is also a training-free MAR acceleration method. However, the paper does not clearly demonstrate superior performance over LazyMAR (e.g., matched-speed or matched-quality comparisons). A direct quality-speed Pareto comparison would be necessary to substantiate claims of advantage.

### Questions
1. KV Memory Layout Question
Have you explored pre-allocating a single contiguous KV buffer and directly writing both cached tokens and newly active tokens into it during loading, instead of handling them in separate buffers and fusing the output? In principle, constructing a contiguous KV block upfront could eliminate the concatenate step and enable one attention kernel call on a regularized sequence length, which may yield higher efficiency.

2. Question on the “generation order” claim
I am not a domain expert, but from reading LazyMAR, my understanding is that it keeps the predefined MAR decoding schedule (i.e., which tokens are decoded at step t) and only dynamically decides which tokens can reuse features within that step to avoid recomputation. In other words, LazyMAR seems to reuse features across layers but does not change which tokens are revealed at each decoding step.
Given this, I am not fully understanding the statement that LazyMAR “departs from MAR’s predefined generation order.” Could you clarify in what sense the generation order is altered? If LazyMAR truly modifies the token decoding order, it would be helpful to reference the specific mechanism or evidence.

### Soundness
2

### Presentation
3

### Contribution
2
