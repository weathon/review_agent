# PatchRefiner V2: Fast and Lightweight Real-Domain High-Resolution Metric Depth Estimation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
While current high-resolution depth estimation methods achieve strong results, they often suffer from computational inefficiencies due to reliance on heavyweight models and multiple inference steps, increasing inference time. To address this, we introduce PatchRefiner V2 (PRV2), which replaces heavy refiner models with lightweight encoders. This reduces model size and inference time but introduces noisy features. To overcome this, we propose a Coarse-to-Fine (C2F) module with a Guided Denoising Unit for refining and denoising the refiner features and a Noisy Pretraining strategy to pretrain the refiner branch to fully exploit the potential of the lightweight refiner branch. Additionally, we propose to adopt the Scale-and-Shift Invariant Gradient Matching (SSIGM) loss within local windows to enhance synthetic-to-real domain transfer. PRV2 outperforms state-of-the-art depth estimation methods on UnrealStereo4K in both accuracy and speed, using fewer parameters and faster inference. It also shows improved depth boundary delineation on real-world datasets like CityScapes, demonstrating its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper extends PatchRefiner (ECCV 2024) for high-resolution monocular metric depth estimation. PatchRefiner V2 replaces the heavy refinement encoder with a lightweight alternative to significantly reduce runtime and memory usage during both training and inference. To compensate for the reduced capacity, the method introduces a coarse-to-fine module that injects coarse-level features to guide and denoise refinement features, along with a noisy pretraining strategy to improve the refiner's robustness. The paper also adopts an improved local-window SSIGM loss to enhance synthetic-to-real transfer. Experiments on UnrealStereo4K and Cityscapes demonstrate improved accuracy and efficiency compared to the original PatchRefiner.

### Strengths
- The proposed PatchRefiner V2 achieves impressive improvements in runtime and memory efficiency over the original PatchRefiner, while also achieving equal or better accuracy. This makes high-resolution monocular depth estimation more feasible for practical use.

- The coarse-to-fine module is well-motivated and effectively compensates for the reduced capacity of the lightweight refiner. The guided denoising design is clearly presented, and its contribution is supported by thorough ablation studies.

- The noisy pretraining strategy is simple yet empirically effective. Pretraining the refinement branch with randomized coarse features leads to a more robust model without requiring additional external data or complex setups.

- The local SSIGM loss provides more precise supervision. Enforcing scale-and-shift consistency at a local spatial level results in sharper depth boundaries and avoids introducing scale error.

- The paper is well-written and easy to follow, with clear figures and architectural diagrams that effectively explain the design choices.

- The visual annotations (e.g., the snails and lightning bolts in Fig. 1) clearly and intuitively highlight the performance and efficiency differences. The visualization of features in Fig. 2 clearly presents the motivation and effectiveness of the fusion model. These figures improved readability and made the narrative smoother and more engaging.

### Weaknesses
- Discussion of Related Work Could Be Expanded.
The motivation and effectiveness of the proposed coarse-to-fine module are clearly presented, and the design is well-justified. However, there are existing two-branch feature fusion strategies in related areas (e.g., [1] and [2]). While these works focus on different tasks and modalities, a brief discussion comparing the design philosophy or fusion flow direction could further clarify the novelty of the proposed C2F module and situate the contribution more explicitly in the broader literature.

[1] Bi-SSC: Geometric-Semantic Bidirectional Fusion for 3D Scene Completion

[2] FFB6D: Full Flow Bidirectional Fusion for 6D Pose Estimation

- Clarification of PRV2’s Advantage Over High-Resolution Backbone Models.
The improvement of PRV2 over DepthPro is quite substantial, which strongly supports the value of the refinement design. Since DepthPro is already a high-resolution metric depth model, it would be helpful for the paper to provide a bit more insight into why PRV2 achieves such notable gains when refining DepthPro outputs. 

- While PRV2 is not intended to be a general-purpose “zero-shot” depth refiner, a short discussion of the expected generalization behavior could help guide future follow-up work aiming toward more foundational refinement pipelines.


- There is a small typo in Table 2: in the caption, “GM and wins.” can be removed for clarity.

- Since the updated local-window SSIGM loss is one of the key improvements, adding a brief pseudo-code snippet in the supplement would make re-implementation easier. This would improve the usability.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents PatchRefiner V2 (PRV2), an enhanced framework for high-resolution monocular depth estimation that aims to address the computational inefficiencies of its predecessor, PatchRefiner (PRV1). The core contributions are a lightweight refiner branch, a novel Coarse-to-Fine (C2F) module with Guided Denoising Units (GDUs), a Noisy Pretraining (NP) strategy, and a local Scale-and-Shift Invariant Gradient Matching (local SSIGM) loss. The paper is well-structured, the problem is clearly motivated, and the experimental evaluation is comprehensive.

### Strengths
1. The paper effectively identifies the critical bottlenecks of PRV1—high inference time, large memory footprint, and the inability for end-to-end training—due to using a heavyweight base model for patch-level refinement. The motivation for replacing it with a lightweight encoder is well-justified and addresses a practical need for real-world applications.
2.The idea of using coarse features to "denoise" the features from a lightweight encoder is intuitive and effective. The GDU mechanism is clearly explained and visualized showing a tangible improvement in feature quality.
3. Noisy Pretraining is a simple yet clever strategy to force the refiner branch to learn robust, depth-relevant features from the high-resolution input itself. The ablation studies strongly validate its importance.
4. Extending the SSI loss to the gradient domain and applying it within local windows is a thoughtful approach to improve boundary accuracy without compromising global scale. The significant improvement in the boundary F1 score on CityScapes is a key result.
5. The paper provides thorough quantitative and qualitative evidence. The results on UnrealStereo4K are impressive, demonstrating that PRV2 can achieve state-of-the-art or comparable accuracy with a massive reduction in parameters (up to 9.2x) and inference time (up to 10.7x faster). The ablation studies are systematic and clearly demonstrate the contribution of each proposed component (C2F, NP, E2E training, local SSIGM).
6. The inclusion of experiments on a real-world dataset (CityScapes) and the analysis of boundary quality are highly valuable and demonstrate the method's practical utility.
7. The method is described in sufficient detail, with clear diagrams and mathematical formulations for the GDU and local SSIGM loss. The implementation details provided in Section 4.2 are adequate for reproduction.

### Weaknesses
1The GDU is a central component, but the ablation only compares it to one alternative. A more detailed analysis, for instance, comparing the proposed sigmoid-based gating to an additive fusion or an attention-based mechanism, would provide deeper insights into why the current design is optimal.
2.  While the overall framework is much faster, the specific computational cost introduced by the C2F module and the local SSIGM loss (during training) is not discussed. A brief note on their relative overhead would be useful for readers considering implementation.
3. The experiments are focused on UnrealStereo4K and CityScapes. While the results on CityScapes show good synthetic-to-real transfer, a brief zero-shot evaluation on other standard depth benchmarks (e.g., KITTI, NYUv2) would more strongly demonstrate the generalizability and robustness of the learned representations, especially given the use of the local SSIGM loss.
4. The field of efficient high-resolution vision is rapidly evolving. A discussion of how PRV2 compares to other contemporary lightweight or patch-based refinement approaches (beyond PRV1 and PatchFusion) would better situate its contributions within the current research landscape.
5. Some arxiv papers are actually published on important conferences, and please cite the published information, not arxiv.

### Questions
While the term noisy features is used to describe the output of the lightweight encoder, a more precise characterization would strengthen the argument. Is this noise in the traditional sense (random, high-frequency artifacts), or is it a lack of depth-specific semantic structure? A brief quantitative analysis (e.g., using feature similarity metrics) could complement the visual evidence.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a refined version of PatchRefiner, aiming to accelerate inference by replacing the original patch refiner with a lightweight network. The proposed approach incorporates a coarse-to-fine module, a guided denoising unit, and a noisy pre-training strategy. With these enhancements, PatchRefiner V2 achieves substantial speed improvements while delivering superior performance compared to its predecessor.

### Strengths
- The paper is well written and clearly organized.
- The proposed improvements to PatchRefiner demonstrate both significant acceleration in inference speed and enhanced performance.

### Weaknesses
1. At Line 377, the authors state that the Cityscapes dataset is used for synthetic-to-real transfer evaluation. However, quantitative comparisons with other methods are missing in both the main text and the supplementary material, which limits the completeness of the evaluation.
2. Only quantitative results on the in-domain UnrealStereo4K dataset are reported. Considering that the base model, ZoeDepth, is a generalizable depth estimator, it would be valuable to include experiments under cross-dataset settings to provide a more comprehensive assessment of the proposed method. 
3. At Line 365, the authors claim that local SSIGM performs better than matching gradients over the entire map. However, as shown in Table 3, the variant with zero windows does not exhibit significant performance degradation compared to local variants, and the influence of window size and number of windows on performance appears minimal. This observation reduces the perceived effectiveness of the proposed local SSIGM loss.

### Questions
1. Regarding the noisy pre-training strategy, it would be helpful to clarify whether the type of noise used during training influences the final performance. For instance, how would the results change if Gaussian noise were replaced with uniform noise?

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
2

### Summary
This paper proposed PatchRefiner V2(PRV2) for real-domain high-resolution metric depth estimation. PRV2 replaces a heavyweight refiner  with lightweight encoder and adds a coarse-to-fine (C2F) block with a guided denoising unit. The paper also present noisy pretraining. Experiments on UnrealStreao4K  (synthetic) and Cityscapes (real) report the efficiency of PRV2.

### Strengths
1. Experimental results
On UnrealStereo4K, PRV2 delivers state-of-the-art accuracy while using fewer parameters and achieving faster inference than strong baselines.

2. Simple design & recipe
A modular architecture and a minimal noisy-pretraining scheme make the method easy to reproduce and extend into existing pipelines.

### Weaknesses
1. Benchmark coverage is narrow (mainly UnrealStereo4K)
The main SOTA claims are substantiated primarily on one synthetic dataset, while the real-domain evidence is limited to Cityscapes.

2. Incomplete comparators for 2024–2025 SOTA
UnrealStereo4K comparisons largely focus on ZoeDepth/ZoeDepth+PF/ZoeDepth+PRV1. Please add or discuss comparisons (or a justified protocol mismatch) against strong recent depth estimation methods (e.g., Marigold, SharpDepth, ...) matched resolution/compute, and clarify where a fair comparison is infeasible.

3. Ablations split across datasets create interpretation friction
Core architectural ablations (C2F/NP) are on UnrealStereo4K, while loss/boundary analyses are on Cityscapes, which makes it hard to see how each module contributes on the same data. Please add a unified ablation table on one dataset (preferably a real set) so readers can read row-wise improvements coherently

4. Contribution novelty leans engineering rather than conceptual
GDU-style gating and the NP strategy are practical and effective, but feel incremental relative to prior guided fusion paradigms

5. Timing definition under-reports full pipeline cost
The paper defines T as the refiner-branch time per image; please also report end-to-end wall-clock (coarse + all patch refinements) and memory for a fair, reproducible comparison for follow-up research

### Questions
No questions

### Soundness
3

### Presentation
3

### Contribution
2
