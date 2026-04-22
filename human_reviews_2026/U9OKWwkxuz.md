# Make Your LVLM KV Cache More Lightweight

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Key-Value (**KV**) cache has become a _de facto_ component of modern Large Vision-Language Models (**LVLM**s) for inference.
While it enhances decoding efficiency in Large Language Models (**LLMs**), its direct adoption in LVLMs introduces substantial GPU memory overhead due to the large number of vision tokens processed during the prefill stage.
To tackle this problem, we propose LightKV, a novel approach that reduces KV cache size by exploiting the redundancy among vision-token embeddings.
Guided by text prompts, LightKV employs cross-modality message passing to aggregate informative messages across vision tokens and progressively compress them during prefill.
This prompt-aware guidance distinguishes our method from prior vision-only compression strategies.
We evaluate LightKV on eight open-source LVLMs across eight public benchmarks, such as MME and SeedBench.
Experimental results demonstrate that with only 50% of the original vision tokens, LightKV (i) halves KV cache size, (ii) reduces computation by up to 40%, and (iii) preserves general-purpose performance while significantly outperforming existing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper propose a training-free method that reduces the KV cache size in MLLMs by compressing redundant vision tokens during the prefill stage. It uses prompt-guided graph message passing to aggregate important visual information under text-aware attention. This approach cuts GPU memory and computation by up to 50% and 40%, respectively, while maintaining or even improving model performance across multiple benchmarks.

### Strengths
Works consistently across multiple MLLM architectures (e.g., LLaVA, InternVL, Qwen-VL, EVE) and benchmarks.

### Weaknesses
1 The motivation of this paper is rather conventional and lacks insight. Moreover, it is not well aligned with the experimental setup. The authors claim that MLLMs suffer from much longer context sequences than LLMs, yet their experiments are conducted only on a few image-level tasks—far from scenarios that truly reflect the claimed motivation. Additionally, the issue of modality discrepancy has already been extensively studied. Prior works (e.g., FastV) have clearly shown that text tokens dominate most of the attention distribution, and many methods have already been designed to address this imbalance, e.g., mminference. Therefore, highlighting this as a new motivation is unconvincing.The authors should discuss the issue of existing methods. Besides, unless the paper explicitly targets long-text or long-video settings, it is expected that most approaches will preserve text tokens while compressing visual tokens. In this context, the paper’s emphasis on modality discrepancy feels misplaced and lacks clear justification.

2 While the paper claims novelty in using prompt-guided mechanisms to select or aggregate visual tokens, this idea is not new. A large body of prior work — even before MLLMs — has explored text-guided visual filtering or attention modulation (e.g., CLIP). The authors do not clearly explain what new insights or advantages their approach brings beyond these established practices. Without a deeper theoretical or architectural difference, the method appears to be a repackaging of existing cross-modal attention strategies rather than a genuine innovation. Moreover, I cant understand why previous attention-based token pruning is not prompt-guided. They also include  cross-modality attention score.

3 The proposed graph-based token compression method raises concerns about scalability. The approach has not been tested on long-context scenarios (e.g., 20k+ tokens), where the computational and memory cost of graph message passing could become prohibitive.
Furthermore, the baselines are outdated — recent efficient inference frameworks such as MMInference or DynamicCache should be included for a fair comparison.
In addition, the reported speedup is modest, suggesting that the method may not translate well to real-world efficiency gains.
Finally, the current design is not well-suited for video or multi-frame inputs, where the graph structure and local aggregation may not scale temporally.

### Questions
Please follow the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces LightKV, a framework designed to accelerate the prefill stage and reduce the KV cache size in Large Vision-Language Models (LVLMs) by compressing and merging visual tokens through text-guided graph message passing and aggregation. Experimental evaluations on multiple image benchmarks validate the efficacy of the proposed approach.

### Strengths
1. The technical contribution is well-motivated.
2. The paper is well-structured and clearly written.

### Weaknesses
1. The motivation of the paper appears to be overstated. The authors claim in lines 61-62 that "current LVLMs are limited by significantly heavier GPU memory usage than their LLM counterparts during the prefill stage." However, this is not always the case in real-world applications, where LLMs commonly handle contexts of millions of tokens.
2. Furthermore, the KV cache becomes a bottleneck primarily under extremely long token sequences. The experiments, however, are conducted only on image benchmarks, where the number of visual tokens is at most in the thousands—far from reaching a memory-bound scenario. Consequently, the experimental results do not sufficiently support the paper's central claim.
3. Several experimental details are omitted. For instance, in Table 4 and Table 10, the length of the token sequence used during the prefill stage is not specified.
4. The paper reports only the end-to-end latency of the proposed method, without breaking down the overhead of individual operations within the pipeline.
5. A major limitation of the method is its requirement to compute full attention scores in certain layers. In long-context scenarios, an eager implementation of attention becomes impractical due to the excessive memory needed to store large attention score matrices. Previous studies, such as [1,2], have introduced probing strategies to mitigate this issue. The authors should consider adopting a similar strategy to enhance the practicality of their method in real-world applications.

[1] Zipcache: Accurate and efficient kv cache quantization with salient token identification. NeurIPS 2024.

[2] Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. NeurIPS 2024.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces LightKV, a training-free approach for reducing the KV cache size in Large Vision-Language Models (LVLMs). The study identifies that vision tokens are the primary source of GPU memory overhead during the prefill stage. To mitigate this issue, LightKV employs a prompt-guided cross-modal compression strategy, which aggregates redundant vision tokens via graph-based message passing guided by text-prompt attention. The method performs hierarchical compression across layers and within local spatial windows, progressively minimizing KV cache size while retaining visually and semantically important information. Extensive experiments on eight LVLMs (including LLaVA-v1.5, LLaVA-NeXT, InternVL2, and Qwen2.5-VL) demonstrate that LightKV can halve KV cache memory usage and reduce computation by up to 40%, all while maintaining—or even slightly improving—accuracy on diverse benchmarks such as COCO, MME, SeedBench, and VizWiz.

### Strengths
This paper addresses a practical and important efficiency bottleneck in LVLMs—the excessive KV cache size from vision token. The proposed LightKV is training-free and can be seamlessly applied to various existing LVLM architectures without retraining, demonstrating strong generality.

### Weaknesses
1. The paper is somewhat difficult to follow, and the motivation behind the proposed method is not clearly articulated. It remains unclear why graph message passing is selected as the core mechanism for vision token compression. The authors should provide clearer intuition or justification for this design choice, explaining how it specifically contributes to improving efficiency or preserving cross-modal information compared with alternative strategies. In addition, the paper introduces an excessive number of notations, which overcomplicates the presentation and makes the method harder to understand. 

2. The main limitation of this paper is that it relies on attention weights to estimate token importance, which is incompatible with I/O-optimized implementations such as FlashAttention (Dao et al., 2022) that are widely adopted in long-context LVLMs. Although the authors note that attention scores are computed only in a small subset of layers, this still incurs additional computational overhead and may limit the scalability and practicality of the approach, especially for long-context inference or large-scale deployments where memory efficiency is critical.

### Questions
1. In Line 203, it is unclear what FD stands for. It would be clearer and more reader-friendly to spell out the full term (e.g., Feature Divergence) the first time it appears, rather than introducing it directly as an abbreviation. 

2. Many implementation details of the proposed method are unclear. For example, during graph construction, it is not specified how the set of nodes is divided into two subsets. Is there any criterion or heuristic guiding this split, or does the method simply follow the same strategy as ToMe (Bolya et al., 2023)? Providing clarification on this point would improve the method’s reproducibility and clarity. 

3. The proposed method adopts a bipartite matching strategy, similar to ToMe (Bolya et al., 2023), and demonstrates improved performance. However, the paper does not clearly discuss the computational complexity of the proposed approach relative to ToMe. Providing a detailed comparison or analysis of the computational cost would help clarify whether the performance gains come at the expense of higher complexity. 

4. The paper only evaluates on image-based benchmarks. It would be valuable to assess the method on video understanding datasets (e.g., Video-MME [1], MVBench [2]) to verify whether LightKV can handle temporal redundancy and long-context multimodal inputs effectively. 

5. While the paper claims notable efficiency improvements, it lacks a detailed runtime or wall-clock latency analysis, particularly comparing the overhead introduced by graph construction and message passing with that of baseline token-pruning methods. Providing such a breakdown would offer a clearer picture of the true computational trade-offs and strengthen the paper’s efficiency claims. 

 

Reference: 

[1] Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. CVPR 2025. 

[2] Mvbench: A comprehensive multi-modal video understanding benchmark. CVPR 2024.

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
5

### Summary
This paper proposes LightKV, a training-free method that compresses the KV cache by exploiting redundancy among vision-token embeddings. LightKV uses cross-modal (text-prompt-guided) message passing to aggregate and compress vision tokens. The method is evaluated on open-source LVLMs across multiple benchmarks, demonstrating that LightKV can halve the KV cache size and maintain or even improve model performance compared to existing baselines.

### Strengths
1. The prompt-aware, training-free approach for KV cache compression is both novel and practical.
2. LightKV achieves substantial reductions in GPU memory usage and computational cost, which is critical for deploying LVLMs in resource-constrained environments.
3. Across most benchmarks, LightKV matches or even surpasses the performance of uncompressed models, and outperforms other SOTA compression baselines.

### Weaknesses
1. It would be helpful to add the motivation for using graph message passing. And for the middle to deep layer, it is not clear that spatial information still persist for the visual token.
2. Previous work [VLCache](https://arxiv.org/pdf/2410.23317?) has also explored using text prompts to guide visual token compression. It would be valuable to include a discussion.
3. Since this method requires explicitly materializing the attention scores for several layers, it still incurs O(N²) complexity. Including a throughput analysis in the experiments would provide a clearer picture of the practical efficiency.
4. In addition to KV cache compression baselines, it would be more comprehensive to compare against visual token compression methods under the same memory budget, such as[PruneViD](https://arxiv.org/abs/2412.16117) and [Prumerge](https://arxiv.org/abs/2403.15388). 
5. In Table 1, the Avg % improvement for LLaVA-v1.5-13B and 7B is quite limited compared to FastV, whereas the improvement is more pronounced for the LLaVA-NeXT series models. It would be helpful to expand the discussion to explain the reasons behind this difference.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
3
