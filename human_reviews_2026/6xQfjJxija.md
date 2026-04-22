# DiffNCL: Diffusion-Driven Weakly-Noisy Correspondence Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Current noisy correspondence learning (NCL) pipelines typically treat correspondence quality as a binary variable, neglecting the abundant category of weakly-noisy correspondences. Two persistent issues are introduced: (i) over-exclusion, where partially informative pairs are discarded as negatives, shrinking the effective data manifold, and (ii) under-alignment, where residual noise from weakly mismatched pairs propagates through gradient updates, degrading representation fidelity. To address these challenges, this work pioneers a unified forward–reverse diffusion framework called "DiffNCL" to explicitly amplify and subsequently purify weakly noisy correspondences for robust noisy correspondence learning. In the forward diffusion, synchronized stochastic perturbations inject Gaussian noise into paired visual–textual embeddings, and step-wise similarities are aggregated to highlight the diffusion discrepancy of weakly noisy mismatches. During reverse diffusion, two complementary consistency objectives, i.e., intra‑modal structural consistency and cross‑modal semantic consistency, progressively purify and reconstruct weakly noisy correspondences into high-quality pairs for subsequent training cycles. Extensive experiments on benchmark datasets, including Flickr30K, MS-COCO, and CC152K, are conducted to demonstrate the superiority of DiffNCL over state-of-the-art baselines for cross-modal retrieval against noisy correspondences.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces DiffNCL, a novel framework that tackles weakly-noisy correspondences in cross-modal retrieval—partially misaligned image-text pairs that are often discarded by existing methods. DiffNCL employs a forward diffusion process to identify these pairs by analyzing their sensitivity to noise, and a reverse diffusion process to purify them into high-quality "pseudo-clean" features using consistency constraints. Extensive experiments show that DiffNCL outperforms baseline methods across various noisy benchmarks.

### Strengths
1. The paper is well-motivated. The idea of leveraging a forward-reverse diffusion process not for generation but for noise robustness in cross-modal retrieval is interesting.
2. Comprehensive experiments are reported to verify the proposed method.

### Weaknesses
1. Strictly speaking, the experiments were conducted on the CC152K subset of CC, not CC. This statement in the abstract should be rigorous.
2. In line 41, the paper should focus on the false positive problem.
3. The discussions and citations of related work in 2025 are lacking.
4. According to Definition 3, what is the approximate proportion of each type during training? (For example, 20% noise) Whether this division is reasonable requires clearer indicators to support the conclusion of the analysis.
5. Is the proposed method also applicable to other backbones, such as VSE$\infty$ and CLIP?
6. How tolerant is the proposed method to extreme conditions, such as 80% noise?

### Questions
See Weaknesses.

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
3

### Summary
This paper investigates the noisy correspondence problem in the context of cross-modal retrieval. Unlike existing works that primarily adopt discriminative-model-based approaches, this study explores a generative perspective by leveraging the diffusion process. Specifically, the proposed method exploits diffusion discrepancies to identify noisy pairs and denoises them during the reverse diffusion process. Experiments conducted on three widely used noisy correspondence benchmarks demonstrate the effectiveness of the proposed approach.

### Strengths
+ The paper focuses on the noisy correspondence problem, which is a recently emerged yet practical and important challenge, especially in the multimodal community where data typically appear in paired form.

+ The proposed method is technically sound and provides fresh insights to the community. In particular, it reveals that, beyond conventional discriminative models, generative diffusion models can serve as a promising paradigm for enhancing robustness against noisy correspondences.

### Weaknesses
- Figure 1 does not clearly illustrate the key idea and motivation. Moreover, it is confusing why high-similarity pairs are labeled as noisy while low-similarity ones are treated as clean.
- In fact, noisy correspondence learning is not limited to cross-modal retrieval. As summarized in “Noise-robust Vision-Language Pre-training with Positive-Negative Learning”, related research also includes tasks such as vision-language pretraining, person re-identification, and others. It is encouraged to provide a more comprehensive review of noisy correspondence learning so that readers can gain a clearer and broader understanding of this research field.
- The authors claim that existing noisy correspondence learning methods ignore weakly noisy pairs. However, I notice that [A] also considers this issue. The authors need to further discuss and clarify their claim.
[A] Cross-modal Retrieval with Noisy Correspondence via Consistency Refining and Mining, TIP 2024.
- It is unclear why computing the so-called semantic alignment (Eq. 6) during the forward process can effectively distinguish clean, partial, and noisy pairs. More discussion or analytical experiments are needed.
- The authors claim that incorporating diffusion discrepancy achieves better identification performance compared to relying solely on the memorization effect. However, no experiments are provided to support this claim.
- The notations are overly complex and sometimes unclear—for example, the up and down arrows in Eq. (12) are not properly explained.
- The two loss terms in Eqs. (14) and (15) are not analyzed through ablation studies.
- Many implementation details are missing, raising concerns about reproducibility. For example, is an off-the-shelf diffusion model used? If so, which one?
- Since the proposed method is implemented based on SGR, the results of SGR should also be reported in the comparison.
- In the Flickr dataset with 20% noise, the 2021 NCR method outperforms several later noise-robust approaches. Was this experiment conducted correctly?
- It is unclear how the Base variants in the ablation study are constructed. Moreover, the performance improvement over the base model is relatively marginal.
- Possible typo: “The comparative results are summarized in Table 2.” — it seems this should refer to Table 1 instead.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DiffNCL, a diffusion-driven framework for weakly-noisy correspondence learning in cross-modal retrieval. Unlike prior approaches that treat correspondences as clean vs. noisy in a binary way, DiffNCL argues that many real-world pairs are weakly noisy—partially aligned but informative. The method includes (1) a forward diffusion process that injects synchronized noise into image-text embeddings and computes diffusion discrepancy to distinguish clean, weakly-noisy, and noisy pairs, and (2) a reverse diffusion process with intra-modal structural consistency and cross-modal semantic consistency to reconstruct pseudo-clean embeddings.

### Strengths
1. The work insightfully identifies weakly-noisy correspondences as an overlooked but practically significant regime between clean and noisy data.

2. Integrates forward discrepancy modeling and reverse refinement in a principled manner, offering a new methodological lens for noisy cross-modal learning.

### Weaknesses
1. The paper relies on diffusion discrepancy $\Psi$ to distinguish clean, weakly-noisy, and noisy pairs, but lacks empirical evidence of its behavior. I suggest that authors visualize similarity trajectories across diffusion steps for representative samples of each category, along with their  $\Psi$ values.
2. Since  $\Psi$ is already designed to separate correspondence types, it is unclear why it still needs to be combined with the per-sample loss $l$. An ablation analysis of partitioning accuracy before and after incorporating $l$ would clarify the necessity of using both signals.
3. In line 41, the paper claims that noisy correspondences inject false negatives, whereas mislabeled noisy positives should instead induce false positives.
4. In Figure 1, the pseudo-clean pair generated by reverse diffusion exhibits lower similarity than the weakly-noisy pair. This appears counter-intuitive and requires clarification
5. Since embeddings already encode abstract semantics, it is not obvious that progressively injecting perturbations into the representation space meaningfully simulates semantic corruption. I suggest that authors visualize the embedding trajectories or distributions during forward and reverse diffusion to validate this procedure.
6. Computing similarities and their gradients at every diffusion step may introduce non-trivial overhead. A comparison of computational cost versus small-loss-based methods would clarify its scalability.
7. Equation (13) leverages cross-modal attention over original embeddings during denoising. Yet weakly-noisy pairs contain partially incorrect semantics, raising the question of how cross-modal attention avoids propagating the erroneous associations.
8. In Table 1, the Flickr30K results appear lower than those under the synthetic 60% noise setting, but the configuration for such case is not described.
9. In Table 2, some traditional noisy-correspondence methods [1-3] show competitive or even superior results in certain metrics.
10. Weakly-noisy samples are simulated by random word replacement, which may not reflect realistic semantic drift. When key cross-modal semantics are altered (e.g., replacing “dog” in a dog-related caption), the pair should be considered fully mismatched rather than weakly aligned.

[1] Cross-modal Active Complementary Learning with Self-refining Correspondence

[2] Enhancing True Correspondence Discrimination through Relation Consistency for Robust Noisy Correspondence Learning

[3] Mitigating Noisy Correspondence by Geometrical Structure Consistency Learning

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

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
This paper introduces DiffNCL, a framework for robust cross-modal retrieval under noisy correspondences. The method focuses on identifying and utilizing weakly-noisy (partially misaligned) image-text pairs that previous studies often ignored or overtrusted. DiffNCL applies a forward-reverse diffusion mechanism. In forward diffusion, modality-specific noise is injected to estimate diffusion discrepancies and classify data into clean, weakly-noisy, and noisy groups. In reverse diffusion, weakly-noisy samples are denoised under consistency constraints, producing pseudo-clean pairs for robust retrieval training. Experiments on Flickr30K, MS-COCO, and Conceptual Captions show clear improvements over strong baselines in both synthetic and real noisy conditions. Extensive ablation studies and case analyses further validate the approach.

### Strengths
s1. DiffNCL is the first to use a diffusion process to explicitly separate and purify weakly-noisy correspondences in cross-modal retrieval. It goes beyond the simple binary separation of clean vs. noisy data.

### Weaknesses
w1 The discussion of the similarity gradient spectral norm in the forward diffusion stage is mainly heuristic. It lacks formal mathematical proofs or a clear analysis of computational complexity.

w2 Although Tables 4 and 5 show that the additional cost is small, the scalability under large-scale settings remains unclear. The impact of the diffusion step count, parameter sharing, and memory usage should be further quantified.

w3 The weak-noise concept based on atomic semantic units is theoretically precise but difficult to compute in practice. In experiments, weak noise is only simulated by word-level perturbations, which may not accurately represent real-world weakly-noisy correspondences.

### Questions
refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
