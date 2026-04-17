# Q&C: When Quantization Meets Cache in Efficient Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Quantization and cache mechanisms are typically applied individually in efficient generation tasks, each showing notable potential for acceleration. However, their joint effect on efficiency remains under-explored. Through both empirical investigation and theoretical analysis, we find that that combining quantization with caching is non-trivial, as it introduces two major challenges that severely degrade performance:
(i) the sample efficacy of calibration datasets in post-training quantization (PTQ) is significantly eliminated by cache operation; (ii) the joint use of the two mechanisms exacerbates exposure bias in the sampling distribution, leading to amplified error accumulation during generation. In this work, we take advantage of these two acceleration mechanisms and propose a hybrid acceleration method by tackling the above challenges, aiming to further improve the efficiency of tasks while maintaining excellent generation capability.
Concretely, a temporal-aware parallel clustering (TAP) is designed to dynamically improve the sample selection efficacy for the calibration within PTQ for different diffusion steps. A variance compensation (VC) strategy is derived to correct the sampling distribution. It mitigates exposure bias through an adaptive correction factor generation. Extensive experiments demonstrate that our method is broadly applicable to diverse generation tasks, achieving up to 12.7x acceleration while preserving competitive generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to jointly use quantization and cache mechanisms in image generation process. The authors identify two main challenges when integrating quantization and cache and propose the TAP and VC methods to address this. Experimental results show that the proposed method accelerates the image generation process by up to 12.7× without compromising quality.

### Strengths
This paper is well-motivated. This paper aims to combine quantization and cache techniques  simultaneously to further accelerate the image generation process. In this process, the authors identified two key challenges: (1) Amplification of Exposure Bias (2) Degradation in Calibration Dataset Effectiveness. To this end, the authors specifically propose TAP and VC as solutions. The authors also provide comprehensive comparative experiments and ablation study results to demonstrate the effectiveness of the proposed method.

### Weaknesses
Add a new column in Table 2 to include latency data for each configuration. I would like to see the impact of TAP and VC on latency.

### Questions
Please refer to weakness.

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
This paper tackles a very practical problem in accelerating generative models, specifically diffusion transformers (like DiT). The goal is to combine two standard speedup techniques: quantization (using fewer bits for weights and activations) and caching (like the K-V cache in transformers, which saves past computations).
The authors' key finding is that just "turning both on" doesn't work—in fact, it severely degrades performance. The paper then diagnoses why this happens (they identify two major challenges) and proposes two novel methods TAP and VC, that allow these two techniques to work together effectively. The end result is a massive speedup (up to 12.7x) on ImageNet generation while maintaining the quality of the original, slow model.

### Strengths
1.	This is a strong, high-quality paper. The originality doesn't come from inventing a brand-new algorithm, but from investigating a subtle, negative interaction between two known techniques that everyone assumed would just work together. Identifying, analyzing, and solving this kind of interaction problem is a very valuable contribution.

2.	The significance is crystal clear. Diffusion models are notoriously slow. A 12.7x speedup on a state-of-the-art DiT model running on ImageNet is a massive practical win. It's the kind of work that could be immediately adopted by anyone trying to deploy these models in the real world.

### Weaknesses
1.	A potential weakness is the focus on Post-Training Quantization (PTQ). PTQ is fast, but it's often outperformed by Quantization-Aware Training (QAT). My question is: Why not just use QAT? It's possible QAT would automatically learn to be robust to the cache interactions, making this complex PTQ-specific solution unnecessary.

2.	Could the authors include speed metric In Table 2 as well for a clear comparison?

Minors:
1.	Line 411 ‘The results, presented in Table 6’ should be Table 2.

### Questions
1.	Why the quantized model in Figure 2 does not show too much higher accumulated error than the original one? It’s inconsistent to the results in [1].

Reference:

[1] Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, and Baochang Zhang. Q-dm: An efficient low-bit quantized diffusion model. Advances in Neural Information Processing Systems, 36, 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of enabling both caching and quantization in DiT models.  The authors first identify two key obstacles: exposure bias amplification and reduced effectiveness of calibration datasets.  They then address these issues by proposing Temporal-Aware Parallel Clustering (TAP) and a variance compensation technique. Extensive experiments and ablation studies demonstrate that the proposed approach achieves notable speed improvements with only minimal degradation in downstream task performance.

### Strengths
•	The paper tackles a well-motivated problem, clearly isolating the issues and addressing them with thoughtful solutions.

•	Comprehensive experimental evaluation across multiple tasks, including detailed ablation studies, supports the claims of improved efficiency.

### Weaknesses
•	The results lack statistical rigor. Without confidence intervals or similar measures, it is difficult to assess the significance of the reported differences.

•	Compared to techniques that employ only caching or quantization, the observed speed-up is relatively modest and, in some cases, comes at the cost of performance degradation.

### Questions
•	How is the speed-up calculated?

•	Why is the combined approach not achieving greater speed-up compared to individual techniques? For example, PTQ4DiT reports a 10x speed-up and Learn-to-Cache achieves 6.3x. Why doesn’t the combined method exceed 12.7x?

### Soundness
3

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
The paper find two key challenges that degrade performance when combining quantization and cache. To address these issues, the authors propose a temporal-aware parallel clustering and a variance compensation strategy. Higher speedup is achieved with image quality slightly hurted.

### Strengths
1. First paper I'm aware of to combine quantization and cache for image generation.
2. The proposed Temporal-Aware Parallel Clustering is interesting. 
3. The ablation study is detailed and validate the effectiveness of proposed approaches.

### Weaknesses
1. Section 3.2 (Variance Compensation) lacks novelty. The approach appears almost identical to that in [1] (which the authors cite), making it difficult to count as a separate contribution.

2. Why not perform quantization calibration before applying the cache? This way, the cache could better utilize the output of the quantized layers to decide what to store, and it would also avoid the calibration data issues mentioned.

3. The chosen quantization method is relatively old. Combining the proposed cache mechanism with newer DiT quantization methods, such as SVDQuant[2] or ViDiT-Q[3], would strengthen the paper's impact.

4. Experiments were only conducted on DiT-XL/2, showing a lack of generalizability. Experiments on more recent models like FLUX or PixArt would further enhance the paper's persuasiveness.

5. The reported speedup ratios lack empirical validation. The authors didn't describe how the speedup is measured. Previous quantization works like Q-diffusion and PTQD did not provide latency results. Clearly, the actual performance often falls short of theoretical claims—for example, MixDQ[4]'s tests show that W4A8 reduces VRAM usage by 3x but only achieves a 1.45x speedup. Yet, without any explanation, the authors claim that W8A8 can deliver a 2x speedup and W4A8 a 2.5x speedup. Therefore, I have reasonable doubts about the reported speedup of the proposed approach.

[1] Timestep-Aware Correction for Quantized Diffusion Models

[2] Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models 

[3] ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation 

[4] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization

### Questions
Please see the weaknesses above, and:

1. Can the authors explicitly clarify the difference between their approach in Section 3.2 and the one presented in [1]?

2. Can the authors provide results for the "quantize-first, then-cache" pipeline? (Perhaps also showing its performance when combined with Variance Compensation).

### Soundness
3

### Presentation
2

### Contribution
2
