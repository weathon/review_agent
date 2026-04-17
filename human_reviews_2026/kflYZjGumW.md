# DiCache: Let Diffusion Model Determine Its Own Cache

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent years have witnessed the rapid development of acceleration techniques for diffusion models, especially caching-based acceleration methods. These studies seek to answer two fundamental questions: _"When to cache"_ and _"How to use cache"_, typically relying on predefined empirical laws or dataset-level priors to determine caching timings and adopting handcrafted rules for multi-step cache utilization. However, given the highly dynamic nature of the diffusion process, they often exhibit limited generalizability and fail to cope with diverse samples. In this paper, a strong sample-specific correlation is revealed between the variation patterns of the shallow-layer feature differences in the diffusion model and those of deep-layer features. Moreover, we have observed that the features from different model layers form similar trajectories. Based on these observations, we present **DiCache**, a novel training-free adaptive caching strategy for accelerating diffusion models at runtime, answering both when and how to cache within a unified framework. Specifically, DiCache is composed of two principal components: (1) _Online Probe Profiling Scheme_ leverages a shallow-layer online probe to obtain an on-the-fly indicator for the caching error in real time, enabling the model to dynamically customize the caching schedule for each sample. (2) _Dynamic Cache Trajectory Alignment_ adaptively approximates the deep-layer feature output from multi-step historical caches based on the shallow-layer feature trajectory, facilitating higher visual quality. Extensive experiments validate DiCache’s capability in achieving higher efficiency and improved fidelity over state-of-the-art approaches on various leading diffusion models including WAN 2.1, HunyuanVideo and Flux.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a caching mechanism for Diffusion Transformers (DiT). The core idea is to use shallow-layer residual probing to estimate deep-layer feature caching. The author verified DiCache on both text-to-image and text-to-video tasks and obtained superior results compared with the baseline method. In summary, I would say this work introduces a simple yet effective DiT caching method that is easy to use for the community. However, more experimental analysis is expected to make the work more solid and convincing.

### Strengths
1. The motivation of this paper is clear. Empirical findings are quite convincing and interesting.
2. This method is easy to use and can be integrated with other intra-step diffusion acceleration methods (such as the sparse-attention-based method). 
3. On one hand, the speedup is non-trivial which is especially useful for video diffusion model acceleration. On the other hand, the quality loss is marginal and it can maintain high consistency with the vanilla samples.

### Weaknesses
The main weakness of this paper lies in the presentation and insufficient experimental analysis. Please see the questions below.

Some advice on the presentation:
1. I strongly recommend redesigning Figure 2 as it increases the understanding cost.
2. Axes numbers in Figure 3 are too small.
3. Abbreviations are better introduced at the first time they appear, such as DCTA.

### Questions
1. Since Wan2.1 and HunyuanVideo are tested, I suggest that the authors provide VBench results on the t2v task. Evaluating generated video quality using image-oriented metrics is not convincing enough.
2. Compared with baseline methods such as TaylorSeer, DiCache uses fewer text-to-image metrics, making the evaluation not comprehensive enough. For example, how does DiCache affect the text-image alignment?
3. Since m is set to 1 for all experiments and the author claimed $m\in [1,2,3] $ is sufficient, I suggest that the author show corresponding diagrams in Figures 3 and 4, instead of only showing m=5.
4.  What is the recomputation rate under different parameter settings? In other words, is $\delta$ hard to tune for different DiT models? It seems this value differs across the DiT models tested.
5. For DCTA, does it mean the current residual is continuously computed from the two most recent recomputed timesteps? How much additional overhead (in terms of FLOPS and latency) will this cause?
6. How does DiCache perform on distilled models?
7. What's the source of text-to-video prompts?

I will raise my score when concerns are mostly addressed.

### Soundness
2

### Presentation
2

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
This paper proposes DiCache, a unified probe-driven framework that adaptively schedules and utilizes cache during diffusion inference, achieving efficient acceleration without additional training.

### Strengths
1. The paper addresses the two fundamental challenges of cache-based acceleration through a unified probe-driven framework, which reduces reliance on empirical heuristics and offline calibration.

2. DiCache can be further combined with Sparse VideoGen to achieve additional acceleration, demonstrating its complementarity with sparse attention techniques.

3. The authors empirically observe a strong correlation between shallow-layer feature differences and deep-layer residuals, and find that features across different DiT blocks exhibit similar trajectories during the sampling process, providing the foundation for the probe-based error estimation and trajectory-based cache blending.

### Weaknesses
1. The coverage of baselines is somewhat limited. Although the experimental tables include TeaCache, EasyCache, TaylorSeer, and ToCa, the Related Work section also discusses other comparable methods such as FasterCache, FORA, and Δ-DiT. Incorporating these methods into the quantitative comparison tables would make the empirical positioning of the proposed approach more complete.

2. The paper primarily evaluates performance using similarity metrics (LPIPS, SSIM, and PSNR) with respect to the outputs of the original model, along with the inference speedup ratio. However, it provides limited analysis of broader perceptual quality or downstream task metrics. Since DiCache does not demonstrate particularly strong fidelity compared to the base model, including perceptual quality evaluations such as VBench scores and user studies would be necessary to provide a more comprehensive assessment.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents DiCache, a training-free, runtime-adaptive caching framework for diffusion models.
In DiCache, the Online Probe Profiling Scheme runs the first few layers to extract shallow probe features and estimate a per-sample caching error, which serves as the threshold for cache reuse.
Dynamic Cache Trajectory Alignment then uses the probe’s estimated progress to select and align deep features from nearby cached steps, reconstructing the current step without recomputing the heavy layers.
As a result, DiCache achieves high speedups with stronger fidelity across WAN 2.1, HunyuanVideo, and Flux.

### Strengths
Novel idea on dynamic cache trajectory alignment that effectively and efficiently adapts the cached value to the current layer that reuses it.

training-free and plug-and-play, requiring no model fine-tuning; it works at inference by wrapping around any DiT models.

DiCache consistently achieves faster inference without sacrificing output quality, outperforming prior caching methods on both image and video diffusion models

Clear analysis and ablations with generally smooth writing and informative figures; easy to follow overall.

### Weaknesses
1.	**Reliance on Threshold Hyperparameter:** Although DiCache demonstrates effective runtime caching under the reported experimental settings, the chosen probe depth (m) and accumulated caching error threshold (δ) should ideally generalize across different models. Alternatively, the authors could justify that the method’s effectiveness is not sensitive to these hyperparameters to substantiate the “calibration-free” claim. However, the current analysis of both hyperparameters lacks evidence of such generalization. For example, Spearman correlation analyses (e.g., Fig. 3 (d)) across multiple architectures would strengthen this point.
2.	**Hyperparameter Trade-offs:** There exists a strong trade-off among probe depth, reuse threshold, and achieved speedup, as well as between the reuse threshold and output accuracy. This implies that achieving the optimal quality–efficiency balance may require manual hyperparameter tuning, making the approach functionally similar to other calibration-based caching methods rather than being fully self-adaptive.
3.	**Overlap with Prior Works on Adaptive Caching and Probing:** DiCache’s adaptive cache decision, which is based on accumulated probe scores and thresholding, follows the same high-level “accumulate + threshold” mechanism used in TeaCache, and its distance-based error proxy conceptually resembles AdaCache’s feature-change metric. Furthermore, shallow-layer probing and caching is a common technique in transformer models [1, 2]. The authors should clarify the DiT-specific novelty in their cache-layer determination, beyond the incremental combination or adaptation of these existing ideas.
4.	Effectiveness of DCTA: Although DCTA is presented as a major contribution of this work, the breakdown of DiCache’s accuracy in Table 2 raises concern about its actual impact, as the observed improvement appears marginal. It would be helpful to clarify whether the effectiveness of DCTA, while seemingly limited in magnitude, remains consistent across different models.
5.	**Memory Usage Analysis:** The paper lacks quantitative analysis of memory usage, even though higher-order or multi-step caching inevitably trades off between memory consumption and accuracy. Reporting VRAM or feature-map memory profiles would clarify the practical scalability of the proposed method.

[1] LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding, Mostafa Elhoushi, Akshat Shrivastava, et al., ACL 2024.

[2] Reducing Transformer Key-Value Cache Size with Cross-Layer Attention, William Brandon, Mayank Mishra et al., NeurIPS 2024.

### Questions
Mainly listed in the weakness. Below are the additional questions.

1.	**Reasoning on probing:** Explaination on why the first few layers can be so informative for predicting global caching error—e.g., what structural or representational property of DiTs enables such strong shallow–deep feature correlation would strenghten the paper.

2.	**Minor:** (i) Typo: missing space in “…probe feature trajectory,which…” (line 108). (ii) Naming inconsistency between “Online Probe Profiling Scheme” (main text) and “Online Probe Profiling Strategy” (Fig. 2 caption).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DiCache, a training-free, adaptive caching strategy to accelerate diffusion models. DiCache addresses two core challenges in caching-based acceleration: "when to cache" and "how to use cache." The method introduces two components: (1) an Online Probe Profiling Scheme to dynamically determine when to reuse cached outputs, and (2) Dynamic Cache Trajectory Alignment to improve multi-step feature reuse through shallow-to-deep trajectory consistency. Extensive experiments on WAN 2.1, HunyuanVideo, and Flux validate that DiCache achieves significant speedup while maintaining high visual fidelity.

### Strengths
* The motivation behind the proposed method is well articulated. The method's design is strongly supported by empirical evidence presented in the paper.

* The method is completely training-free, making it highly practical and broadly applicable across different diffusion models.

### Weaknesses
* The proposed reuse threshold δ appears to require manual, per-model tuning (δ = 0.2 for WAN 2.1, δ = 0.1 for HunyuanVideo, δ = 0.4 for Flux), which may reduce generality and increase tuning effort for new architectures.

* While the probe is “shallow,” it is still computed at every single timestep to accumulate caching error. It remains unclear how much speedup is offset by repeated probing on large backbones.

### Questions
1.Given the memory and dynamic variability challenges, is there a possibility to automatically determine δ or adaptively calibrate it without per-model tuning?

2.The probe is shallow but still executed at every timestep. Could the authors provide a detailed comparison of the probe cost across different architectures.

3.The probe uses L1 relative distance on shallow features. Have alternative feature distance metrics been considered, especially for semantic coherence?

### Soundness
3

### Presentation
3

### Contribution
3
