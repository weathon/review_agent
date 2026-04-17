# Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
The quadratic computational complexity of self-attention mechanisms pose a critical challenge for transformer-based video generation in synthesizing ultra-long sequences. 
 Current sparse approaches with fixed patterns fail to fully exploit the inherent spatio-temporal redundancies in video data. Through systematic analysis of video diffusion transformers (DiT), we observed that critical information in attention matrices can be stably covered in a set of structured, yet heterogeneous, sparse patterns, including a cross-shaped attention pattern and patterns' dynamic behavior over time. To fully exploit the redundancies, we propose Compact Attention, a hardware-aware acceleration framework featuring three innovations: 1) Adaptive tiling strategies that approximate diverse spatial interaction patterns via dynamic tile grouping, 2) Temporally varying windows that adjust sparsity levels based on frame proximity, and 3) a recall-driven, offline search algorithm that automatically optimizes sparse masks while preserving critical attention pathways. 
 Our method achieves up to 3x acceleration in Hunyuan and 2.25x in 14B Wan2.1 model in attention computation on single-GPU setups while maintaining comparable visual quality with full-attention baselines. By grounding acceleration in the empirical discovery of fundamental attention structures, this work provides a principled approach to efficient long video generation through structured sparsity exploitation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a training-free sparse attention framework that accelerates video diffusion transformers by exploiting stable spatiotemporal attention patterns. Through analysis of models like HunyuanVideo and Wan 2.1, the authors identify recurring cross-shaped spatial and time-dependent patterns and design a tile-based deformable mask with dual rectangular windows and frame-group-wise sparsity. An offline greedy search algorithm then optimizes these masks for efficiency while maintaining recall. Implemented on FlashAttention-3, Compact Attention achieves up to 3× faster attention computation and 2× end-to-end speedup with minimal quality loss, outperforming prior sparse attention methods such as STA, SVG, and SpargeAttention

### Strengths
1. The observation of cross-shaped attention patterns (horizontal and vertical spatial structures) is novel and complements prior findings that emphasized 3D cube (STA) or spatial-temporal periodic (SVG) patterns. This adds a valuable new perspective to understanding video transformer sparsity.
2. The iterative shrinking and auto-search mechanism is well-motivated and effectively designed. The greedy mask search algorithm demonstrates clear advantages over prior heuristic approaches such as STA’s manually pre-defined masks.
3. I appreciate that the authors report full-attention baselines using FlashAttention-3. Many previous works (and even other submissions I reviewed) play the trick of comparing against FA-2 on Hopper GPUs, which can exaggerate speedups. This transparency adds credibility.
4. The authors explicitly analyze the effect of keeping the first few denoising steps dense. This is an important implementation trick that several previous papers quietly adopt without quantifying its impact.

Overall, the paper presents solid results and comprehensive experiments with clear visual and quantitative evidence

### Weaknesses
1. The term “attention vocabularies” may be unnecessary. Maybe avoid constructing new terms?
2. The proposed method requires an offline greedy search per head and per frame-distance group, followed by a union across prompts. It would be nice if the paper quantify: a. he total GPU hours or wall-clock time required for this search b.how often the search must be repeated (e.g., for a new  resolution or different finetunes of the same base Wan 2.1 model).

### Questions
1. What would happen if one searched for a very high-sparsity mask and then finetuned the model to recover quality? In that case, might the first few steps no longer need to remain dense? (This is not a required experiment but would be interesting to discuss.)
2. How exactly is the sparsity in Tables 1 and 2 computed? Does it account for the early full-attention steps, or is it measured only over the sparse steps?

### Soundness
4

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
5

### Summary
The paper proposes Compact Attention, a training-free video generation acceleration method based on the spatial and temporal redundancy. It mainly consists of three components:

1. Adaptive sparse tiles, which allows different sparsity levels for near and distant frames with the proposed spatial tiles.

2. Offline search for mask: the authors analyze the attention maps during the video generation for different prompts, and provide a static sparse strategy for the future inference usage.

This work achieves better performance than several baselines and achieves up to 3x acceleration.

### Strengths
1. This work shows that the proposed method achieves better performance than several baselines.

2. This work shows that they can achieve up to 3x acceleration.

### Weaknesses
1. The novelty of spatial and temporal redundancy focus is limited, which is similar as SVG [1] . SVG also partitions tokens into local tiles, and employs cross-frame attention masks.

2. The offline search for the optimal static pruning strategy is empirically unconvincing. The sparse patterns always related to the input prompt, denoising step, layer depth, and seed. The author also show that the similarity is over 0.8, which means the static strategy may not be the optimal one and may lead to crash in some special case.

3. In Table 1, the experiments are not clear, the sparsity is wired. The sparsity used for different methods are different, and the similarity metrics including SSIM, PSNR, and LPIPS are not reported.

4. In Table 2, again, the sparsity is not equal, it is a unfair comparison, and the SVG results are not included.

----
[1] Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity

### Questions
1. Please provide the clear experimental results compared to other baselines, the results should include the similarity metrics.

2. Why the proposed method claims for the dynamic sparsity? the off-line search is adopted to get a static pruning strategy.

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
This paper proposes Compact Attention. The authors observe that for the same DiT model, the attention patterns remain similar across different inputs. Based on this, they precompute the attention patterns offline and apply them during video generation. Specifically, the method represents sparse masks using Dual Attention Windows and employs a greedy algorithm for the offline precomputation. Experiments show that Compact Attention can achieve up to 3× attention speedup in video generation while maintaining video quality.

### Strengths
The proposed approach is novel, particularly the idea of offline precomputation of sparse masks and the use of dual attention windows to represent attention masks.

The kernel implementation is tailored for their method.

### Weaknesses
The paper’s presentation is rather unclear in several places. For example:

- The description of the greedy algorithm suggests a progressive contraction, but the pseudocode shows progressive expansion.
- In Figure 4, there is a “Flag” term that is never defined in the text.
- Sections such as Reuse Masks across Denoising Steps lack sufficient details.


The experimental evaluation is limited. For instance, in Table 2, on Wan 2.1, the STA method achieves twice the sparsity of Compact Attention, which makes the performance comparison less convincing.

The paper statistically verifies that the sparse patterns in DiT models are independent of the prompt and random seed. This finding is somewhat counterintuitive and lacks sufficient analysis and discussion.

### Questions
The authors focus on three spatial patterns and fit them using the dual attention windows. However, this design choice lacks justification. Could the authors provide quantitative evidence showing what proportion of attention maps actually exhibit these three spatial patterns?

What is the computational cost of the offline greedy algorithm, and how much memory is required to store the precomputed attention patterns?

In the Quality Evaluation section, the authors argue that video similarity should not be compared, yet in the comparison with STA, only similarity metrics are reported. Why not use the quality evaluation metrics introduced in that section to compare against STA?

Sparse video gen has a 2nd version; I suggest that you use it as a baseline.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Compact Attention, a framework designed to accelerate transformer-based video generation by exploiting structured spatio-temporal sparsity in attention mechanisms. The authors identify recurring sparse attention patterns in video diffusion transformers—such as cross-shaped and time-variant structure and propose a sparse attention mechanism based on adaptive tile-based sparse masks and a greedy auto-search algorithm that precomputes optimal attention masks. Compact Attention achieves up to 3× acceleration on large-scale models like Hunyuan and Wan2.1 while maintaining comparable visual quality and temporal consistency.

### Strengths
1. The authors demonstrates significant acceleration (up to 3×) with negligible quality loss, validated on strong baselines and realistic benchmarks.

2. The method is training-free and hardware-aware, making it practical for real-world applications and compatible with existing transformer architectures.

3. The study on the effectiveness on "delaying sparse attention" is good. Although it is well-observed in previous studies, none of them plot this trend.

### Weaknesses
1. The offline mask search, though effective, could be computationally heavy for deployment across diverse configurations. The authors should add experiments to discuss how the final quality changes with respect to the computation spent based on this calibration process.

2. On highly dynamic or non-redundant video scenes, is the sparsity pattern still highly similar to the offline searched sparse attention pattern? It seems that in Figure 3(b) some similarity score is as low as 0.7 (based on color).

3. The authors should faithfully report the speedup number for Wan 2.1 and HunyuanVideo separately, rather than only reporting "up to 3x".

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
