# Deep Global-sense Hard-negative Discriminative Generation Hashing for Cross-modal Retrieval

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Hard negative generation (HNG) provides valuable signals for deep learning, but existing methods mostly rely on local correlations while neglecting the global geometry of the embedding space. This limitation often leads to weak discrimination, particularly in cross-modal hashing, which obtains compact binary codes.
We propose Deep Global-sense Hard-negative Discriminative Generation Hashing (DGHDGH), a framework that constructs a structured graph with dual-iterative message propagation to capture global correlations, and then performs difficulty-adaptive, channel-wise interpolation to synthesize semantically consistent hard negatives aligned with global Hamming geometry.
Our approach yields more informative negatives, sharpens semantic boundaries in the Hamming co-space, and substantially enhances cross-modal retrieval. Experiments on multiple benchmarks consistently demonstrate improvements in retrieval accuracy, verifying the discriminative advantages brought by global-sense HNG in cross-modal hashing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of discriminative cross-modal hashing through a global-sense perspective, constructing adaptive hard negatives that reflect global semantics rather than local feature proximity. The approach builds upon two main components: a graph propagation network (RGP) for capturing higher-order semantic dependencies and a discriminative synthesis unit (DGS) that regulates interpolation difficulty via channel-wise weighting. Conceptually, this work reframes hard-negative generation as an optimization over a semantic manifold.

### Strengths
1. Introduces a coherent, theoretically inspired motivation for rethinking negative sampling as a global consistency problem.
2. The loss design reflects an interesting interplay between semantic preservation, interpolation similarity, and coefficient diversity.
3. Empirical results validate the conceptual claims with strong mAP improvements.

### Weaknesses
1. In Figure 3, the comparison between DGHDGH and DHaPH shows large performance gains, but it is unclear whether both models use identical backbones and training setups. A controlled experiment would be necessary to ensure fairness.
2. The RGP module remains largely intuitive, while its empirical benefits are evident, there is no theoretical analysis of how the propagation maintains information fidelity or prevents over-smoothing.
3. The DGS module’s channel-wise λ weights perform well, but their dynamics are not visualized. A λ-distribution plot or feature-space interpolation visualization would clarify how “difficulty” is being modulated.
4. The method is validated only for image-text retrieval. Can DDGSH extend to audio/video modalities?

### Questions
1. Could the authors mathematically relate the RGP operation to spectral diffusion or Laplacian smoothing?
2. How sensitive is the overall model to λ initialization?
3. Discussing the future work like audio/video retrieval in Conclusion section would strengthen the impact.

### Soundness
3

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
5

### Summary
DGHDGH presents a technically elegant yet computationally efficient solution for enhancing cross-modal hashing. The paper integrates a lightweight graph propagation (RGP) and an adaptive interpolation module (DGS) into a CLIP-based framework, showing measurable improvements with minimal resource overhead. From an implementation viewpoint, the proposed pipeline is easy to reproduce and could serve as a plug-and-play enhancement to existing retrieval systems.

### Strengths
1.The RGP–DGS pipeline is an elegant architectural contribution combining graph-based correlation learning with adaptive synthesis.
2.Provides a principled treatment of difficulty adaptation, moving beyond heuristic sampling.
3.The experiments are extensive, statistically robust, and demonstrate consistent gains over diverse baselines.
4.The approach is efficient (no extra generator) and generalizable to existing hashing frameworks.

### Weaknesses
1.The paper introduces λ as a channel-wise coefficient but does not explicitly state whether it is a fixed hyperparameter or a learned variable. Clarifying whether λ is shared between modalities would aid implementation.
2.While the paper claims not to use additional generators, RGP is still a GNN-based component. It would be useful to show FLOPs or parameter comparisons between DGHDGH and baselines to substantiate the claim of efficiency.
3.The RGP module resembles self-attention. Could the authors comment on whether it could be replaced by a lightweight transformer encoder?
4.The paper alternates between the terms “Global-sense” and “Global correlation”, which may confuse readers. Unifying terminology at the beginning of Section 3 would make the exposition cleaner.

### Questions
1.Is λ initialized randomly or via prior heuristics?
2.Could the authors quantify the overhead (Params / FLOPs) of RGP relative to a standard self-attention layer?
3.For deployment, have the authors explored quantizing RGP parameters to further reduce inference cost? Some discussion on the theme matched with hashing retrieval would be welcome.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes DGHDGH, a new framework introducing hard negative generation into cross-modal hashing retrieval. The key idea is to model global semantic correlations among heterogeneous samples via a Relevance Global Propagation graph transformer, and synthesize channel-wise adaptive hard negatives using the Discriminative Global-sense Synthesis module. The method avoids relying solely on local pairwise interpolation, thereby maintaining semantic consistency in Hamming space. Extensive experiments across MIRFLICKR-25K, NUS-WIDE, and MS-COCO show state-of-the-art results.

### Strengths
1. The RGP–DGS pipeline is an elegant architectural contribution combining graph-based correlation learning with adaptive synthesis.

2. Provides a principled treatment of difficulty adaptation, moving beyond heuristic sampling.

3. The experiments are extensive, statistically robust, and demonstrate consistent gains over diverse baselines.

4. The approach is efficient and generalizable to existing hashing frameworks.

### Weaknesses
1. The three loss components (L_sp, L_is, L_cd) are optimized in parallel, yet the paper does not clarify their relative weights or potential gradient interactions. A short sensitivity analysis would strengthen the presentation.

2. The experiments mainly use CLIP-ViT backbones; limited tests with other vision–language models (e.g., BLIP, SigLIP, ALIGN) make it difficult to judge generalization across architectures.

3. The radar plot visualizing parameter sensitivity (Fig. 7) is not clearly described — axis meaning, normalization range, and metric selection should be elaborated to help readers interpret the results.

### Questions
1. Are the loss weights fixed throughout training or tuned per dataset? Could the authors report whether the optimization of three loss terms exhibits any instability during early training stages?

2. How are the radar-plot axes normalized, by relative gain or absolute metric value?

### Soundness
4

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
The authors propose DGHDGH, a cross-modal hashing framework coupling a global propagation network (RGP) with an adaptive negative synthesis module (DGS). The paper delivers strong quantitative results and clear empirical validation, but certain visualizations and terminological inconsistencies slightly hinder comprehension.

### Strengths
1.	Demonstrates strong performance on multiple datasets and provides meaningful ablation studies.
2.	The framework is innovative in combining semantic propagation with synthetic negative mining. The method is technically coherent and easily interpretable when fully understood.
3.	The framework appears extendable to other multi-modal applications.

### Weaknesses
1.	Some statistical figures, such as the radar chart summarizing multiple metrics, lack sufficient description. It is unclear what normalization or metrics were used for each axis.
2.	How effectively do the Fisher Ratio and PH2 verify the discrimination of Hamming spaces? It is necessary to add more experimental analysis in Section 4.
3.	It would be beneficial to test whether the global propagation remains stable under noisy or partially corrupted modalities—for instance, when the embedding graphs contain random noise—to verify robustness.

### Questions
1.	Future work might explore coupling DGHDGH with pre-trained large multi-modal models (e.g., BLIP-2) to test transferability.
2.	It might also be fruitful to explore hybrid discrete–continuous codes instead of pure binary hashing, leveraging the same hard-negative generation principle.
3.	The figures could use larger fonts and more contrast; are the authors planning visual revisions for the camera-ready version?

### Soundness
3

### Presentation
3

### Contribution
2
