# DSA: Efficient Inference For Video Generation Models via Distributed Sparse Attention

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Diffusion Transformer models have driven the rapid advances in video generation, achieving state-of-the-art quality and flexibility. However, their attention mechanism remains a major performance bottleneck, as its dense computation scales quadratically with the sequence length. To overcome this limitation and reduce the generation latency, we propose DSA, a novel attention mechanism that integrates sparse attention with distributed inference for diffusion-based video generation. By leveraging carefully-designed parallelism strategies and scheduling, DSA significantly reduces redundant computation while preserving global context. Extensive experiments on benchmark datasets demonstrate that, when deployed on 8 GPUs, DSA achieves up to 1.43× inference speedup than the existing distributed method and 10.79× faster than single-GPU inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DSA, a system that combines sparse attention mechanisms with distributed inference to accelerate video generation using Diffusion Transformer models. The approach introduces two key components: (1) Mixed Parallelism (MP) that applies different parallelism strategies (spatial sequence parallel vs. temporal sequence parallel) based on attention patterns, and (2) Dynamic Attention Scheduling (DAS) that optimizes computation-communication overlap. Experiments on Wan and Hunyuan-Video models show up to 10.79× speedup over single-GPU inference while maintaining video quality.

### Strengths
The paper makes a solid contribution by integrating sparse attention with distributed inference for video generation. The key insight of matching different parallelism strategies to spatial versus temporal attention patterns is well-motivated and novel. The experimental evaluation covers multiple models with comprehensive quality and system performance metrics, demonstrating impressive super-linear scaling for larger models. The training-free nature makes it immediately applicable to existing models.

### Weaknesses
1.The paper only reports PSNR, SSIM, and LPIPS metrics without perceptual quality metrics like VBench scores that evaluate specific video generation dimensions such as subject consistency, temporal style, spatial relationships, and overall consistency. These metrics are crucial for assessing whether sparse patterns preserve the semantic and temporal coherence of generated videos.

2.The evaluation does not compare against other sparse attention methods for video generation such as DiTFastAttn, MInference applied to video models, or cache-based methods like PAB. 

3.Critical design decisions lack ablation studies. The paper does not validate whether the mixed parallelism strategy outperforms a uniform strategy. The impact of dynamic scheduling versus naive scheduling is mentioned briefly but not thoroughly evaluated across different models and workloads. There is no sensitivity analysis on the sparsity hyperparameters (cs and ct) to show robustness across different efficiency-accuracy trade-offs.

4.The paper provides only two visual examples in Figure 6. More extensive qualitative comparisons across diverse scenarios (minor vs. significant scene changes, rare vs. frequent object interactions) would strengthen the quality claims, especially given the method's high PSNR but potential for subtle temporal artifacts.

5.While the paper claims hardware efficiency through layout transformation, it does not provide detailed kernel-level benchmarks comparing the proposed approach against naive sparse attention implementations across different sparsity levels.

### Questions
See weakness

### Soundness
3

### Presentation
4

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
This paper proposes Dynamic Selective Attention (DSA) to accelerate video generation.

DSA adopts the dynamic sparse patterns based on the fixed spatial and temporal sparse patterns.

This work mainly focus on the contribution on the system level scheduling design.

The experimental results show that this work achieves good results.

### Strengths
1. This work shows how to schedule the attention computation for sparse attention.

### Weaknesses
1. This work does not provide the detailed dynamic sparse pattern online generation methods. In Section 4.1, this work only shows that they adopted the spatial and temporal kernels from the work [1], which shows the limited novelty. The dynamic pattern looks like the token-importance based pruning, which is not novel.

2. This work claims dynamic sparse pattern, while it compares to the fixed sparse pattern works like [1]. This work does not compare to dynamic sparse pattern works like [2] 


---
[1] Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity

[2] DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

### Questions
1. Please provide the details of the dynamic sparse pattern generation method.

2. Please provide the overhead (like latency) brought by the dynamic sparse pattern generation.

3. Please provide the comparison to dynamic sparse pattern works in video generation like [1].

---
[1] DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

### Soundness
2

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
The paper presents an interesting approach to improving the efficiency of video generation models using distributed sparse attention. However, the lack of supplementary materials, incomplete evaluation, and inconsistent reporting of results hinder the ability to fully assess the effectiveness of the proposed method. Addressing these issues will significantly enhance the paper's quality and credibility.

### Strengths
The paper presents an interesting approach to improving the efficiency of video generation models using distributed sparse attention.

### Weaknesses
1. The authors have not provided the MP4 files for the proposed method and comparison methods in the supplementary materials. This makes it very difficult to assess the actual visual quality and temporal consistency of the generated videos.

2. Even without the MP4 files, the authors could have provided code or other means to reproduce the results and evaluate the visual quality.

3. The authors have not tested the full range of metrics from VBench or VBench2.0. Comprehensive evaluation is crucial to understand the strengths and weaknesses of the proposed method.

4. Table 2 includes timing results for USP, but Table 1 lacks corresponding quality metrics for USP. This inconsistency makes it difficult to compare the proposed method with USP comprehensively.

5. The related work section could benefit from a more comprehensive review of pre-trained models for video generation acceleration and sparse/linear attention methods.

If the authors address the above concerns effectively, particularly by providing supplementary materials and a more comprehensive evaluation, I would be willing to reconsider my assessment and potentially give a more positive score.

### Questions
no

### Soundness
2

### Presentation
3

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
This paper proposed distributed sparse attention (DSA) for DiT-based video generation model and achieved $1.43 \times$ speed up than the unified sequence parallelism (USP).

### Strengths
1. By combining sparse attention with distributed strategies, the video generation model achieves improved multi-GPU efficiency while preserving the original generation quality as much as possible.

2. Evaluations were conducted on several mainstream models, including Wan2.1-1.3B, Wan2.1-14B, and Hunyuan-Video, covering video quality metrics (PSNR, SSIM, LPIPS, VBench) as well as system performance metrics (latency and speedup).

### Weaknesses
1. The overall approach resembles a combination of the SVG method and the USP method, with optimized attention scheduling. 

typo: Line 152, "xx tokens"

### Questions
1. In Table 2, the results of USP + SVG can be provided.

### Soundness
2

### Presentation
2

### Contribution
2
