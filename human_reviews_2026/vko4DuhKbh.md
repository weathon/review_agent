# Flow Caching for Autoregressive Video Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames—an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps.
In this paper, we present \textbf{FlowCache}, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance–redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality.
Our method achieves remarkable speedups of $\textbf{2.38}\times$ on MAGI-1 and $\textbf{6.7}\times$ on SkyReels-V2, with negligible quality degradation (VBench: $0.87\uparrow$ and $0.79\downarrow$ respectively). These results demonstrate that FlowCache, successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation—establishing a new benchmark for efficient video synthesis at scale. The code is available at https://github.com/mikeallen39/FlowCache.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a **cache compression method** designed to significantly reduce the KV cache size in autoregressive (AR) video generation. The proposed approach effectively improves generation speed while largely maintaining output quality.

### Strengths
- The paper is **well-organized**, **logically structured**, and clearly highlights its main contributions.  
- The proposed method introduces an **adaptive criterion** for KV cache compression, enabling dynamic adjustment of the compression ratio.  
- This adaptive mechanism effectively improves cache compression efficiency and leads to faster video generation without significant quality degradation.

### Weaknesses
1. **Equation (8)** appears potentially ambiguous. It seems that the Softmax operation should be applied along the \( L_k \) dimension, but the current formulation places Softmax directly outside \( q_i \) and \( k_j \), which may be mathematically inconsistent.  
2. It is unclear **when compression begins**—does it occur before reaching the cache budget, or only after exceeding it? If the method only keeps top-B tokens, then no compression happens before reaching the budget. The paper should clarify that the reported speedup comparisons are made **after the cache reaches the specified budget**.  
3. Although the proposed approach achieves high compression rates, each compression step requires a **pre-computation of attention scores** to determine which tokens to keep, followed by another computation of the true attention scores for the retained tokens. This effectively adds an extra forward pass. The paper should report the **computational overhead ratio** introduced by this additional calculation.

### Questions
See weakness above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces FlowCache, a novel caching and compression framework for autoregressive video generation that addresses the unique challenge of heterogeneous chunk denoising states. The method is theoretically grounded, empirically strong, and significantly advances the state of the art in efficient long-video synthesis.

### Strengths
1.  The paper's primary strength is its clear identification and empirical validation of *why* existing caching methods fail for autoregressive models: the "heterogeneous denoising states." This is a sharp and important insight.
2. The "chunkwise caching" policy is a direct, logical, and highly effective solution to the problem identified. The ablation study decisively proves that this chunk-specific strategy is the main reason for the method's success in preserving quality.
3. Additionally, the KV cache compression method is also well-motivated. It correctly identifies the high-redundancy problem in video data and proposes a solution that intelligently balances both importance and redundancy, which is a clear improvement over importance-only methods from language modeling. From my perspective, this is an excellent work overall.

### Weaknesses
1. There is an internal contradiction between the paper's theory and its empirical results. **Theorem 1** (and its proof in Appendix B) is used to establish that the relative L1 distance is a *monotonically decreasing* function of time (as $t$ goes from 0 to T). However, the paper's *own empirical plot* (Figure 2) and *main text* (e.g., "relative L1 distance monotonically increases as denoising progresses" in line 302) show the exact opposite. This contradiction undermines the stated theoretical foundation.
2. The ablation in Table 2 suggests that the complex KV cache compression adds very little performance on top of the main chunkwise policy. On MAGI-1, the chunkwise policy *alone* achieves a 77.66% VBench score, while the *full* method with compression gets 77.93%. This small benefit may not justify the added complexity of the importance-redundancy scoring mechanism; more discussion is required.

### Questions
How sensitive is the KV cache compression performance to the choice of λ and the granularity settings? Is there a recommended configuration for different video types?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies training-free acceleration for autoregressive video diffusion. The authors empirically demonstrate that different chunks should have independent feature caching policies rather than a single global caching policy. A redundancy-aware KV-cache compression scheme is also adopted for long-video generation.
Experiments on MAGI-1 and SkyReels-V2 demonstrate a noticeable speedup with a minor drop in VBench score.

### Strengths
- The paper proposed training-free, plug-in acceleration for causal video diffusion. Treating each video chunk as its own, with an independent reuse policy, is well motivated by the observed heterogeneity across chunks at the same timesteps.
- Experiment isolates the benefit of chunkwise feature reuse over full reuse and shows that kv-compression has a small impact.

### Weaknesses
- Memory claims lack evidence. Memory usage is stated to be fixed, but no benchmark on memory usage is exhibited in the paper.
- Lack of quality comparison. Only a few images are displayed in the paper. No video clip was provided, making it hard to evaluate the visual quality.
- Lack of evaluation on long-video benchmark, e.g., VBench-long, since the method is claimed to be helpful for long-video generation.
- MAGI-1 already applied window attention (8-second preceding video content). Weakening the motivation for KV-cache compression.
- MAGI-1 has a shortcut step-distill version. The proposed method does not compare with it, nor apply the feature reuse to it.
- Feature reuse necessarily increases memory consumption because the cache has to be retained. This seems to conflict with the stated motivation of KVcache compression, which is to reduce memory usage.

### Questions
- line 99-101: Please clarify the experimental/testing setup for this result (e.g., GPU type, memory, batch size, video length). Without this, it’s hard to judge how generalizable the observation is.
- The paper claims to “provide insights into memory–quality trade-offs,” but the experiments do not actually show memory vs. quality curves/tables (e.g., VBench vs. cache size/compression ratio). This weakens the contribution.
- How is the chunkwise caching policy obtained in practice? Is it derived offline from a calibration set, or learned/heuristic? When generating videos of different lengths or motion patterns, does the policy need to be recomputed or adapted?
- Do you have numerical benchmarks (peak memory, KV size per frame/chunk, vs. baseline) to substantiate the claim of reducing the memory with KV-cache compression?
- misc:
  - The citation of [1] is not about diffusion and seems unrelated to the context in which it is cited.
  - VBench is a scaled score, not a percentage

[1] Mengwei Xu, et la, Deepcache: Principled cache for mobile deep vision. 2018

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a caching framework for autoregressive video generation. The key idea is to change existing uniform caching strategies used in common video diffusion models. Firstly, it applies dynamic caching strategies to different noise levels. Higher noise level is more likely to reuse cached feature, while lower noise level is more likely to recompute. It also introduce a compression mechanism to kv cache due to large memory consumption based on importance and redundancy.

### Strengths
- The dynamic assignment of calculate or reuse significant improves the efficiency of the model i.e., more than 2x comparing to TeaCache-fast.
- The proposed KV cache compression method balances both past token visual importance and redundancy. It is specifically designed for video token characteristics.

### Weaknesses
- The idea of reuse or recompute based on L1 similarity  of L1rel is clearly stated and proved, but the detail of how to decide to reuse or recompute the cache is not clear. Is there any threshold or decision making module for this part?
- It is not clear why the proposed method both out perform the baseline method, TeaCache both in terms of speed and video quality. The reuse operation accelerates the speed, but why it also achieve better frame quality. It would better to have more detailed comparison and discussion between the baseline method.

### Questions
- For the dynamic chunk caching and reuse mechanism, is it adaptively applied across different videos based on the degree of motion dynamics? As mentioned in Weakness 1, the decision-making process remains unclear — it would be helpful to provide a concrete inference example illustrating how this adaptation works.
- Are there any known failure cases for the two proposed designs? For instance, how do they perform on videos with large or complex motion? In such cases, is the speed up improvement less significant?

### Soundness
3

### Presentation
3

### Contribution
3
