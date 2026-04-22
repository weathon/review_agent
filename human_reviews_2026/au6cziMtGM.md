# Interp3D: Correspondence-aware Interpolation for Generative Textured 3D Morphing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Textured 3D morphing seeks to generate smooth and plausible transitions between two 3D assets, preserving both structural coherence and fine-grained appearance. This ability is crucial not only for advancing 3D generation research but also for practical applications in animation, editing, and digital content creation. Existing approaches either operate directly on geometry, limiting them to shape-only morphing while neglecting textures, or extend 2D interpolation strategies into 3D, which often causes semantic ambiguity, structural misalignment, and texture blurring. These challenges underscore the necessity to jointly preserve geometric consistency, texture alignment, and robustness throughout the transition process. To address this, we propose Interp3D, a novel training-free framework for textured 3D morphing. It harnesses generative priors and adopts a progressive alignment principle to ensure both geometric fidelity and texture coherence. Starting from semantically aligned interpolation in condition space, Interp3D enforces structural consistency via SLAT (Structured Latent)-guided structure interpolation, and finally transfers appearance details through fine-grained texture fusion. For comprehensive evaluations, we construct a dedicated dataset, Interp3DData, with graded difficulty levels and assess generation results from fidelity, transition smoothness, and plausibility. Both quantitative metrics and human studies demonstrate the significant advantages of our proposed approach over previous methods. Source code is available at https://github.com/xiaolul2/Interp3D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Interp3D, a novel and training-free framework for generative textured 3D morphing. The key insight is to address the critical challenge of 3D correspondence misalignments, a root cause of semantic ambiguity, structural distortion, and texture blurring in prior works. To this end, the authors introduce a progressive alignment framework that performs correspondence-aware interpolation across three levels: semantic (in condition space), structural (guided by SLAT features), and texture (via fine-grained fusion). Experiments on a newly constructed benchmark, Interp3DData, demonstrate that Interp3D outperforms existing baselines in terms of structural fidelity, transition smoothness, and visual plausibility.

### Strengths
1. Practical training-free formulation. The work presents a training-free method that effectively leverages pre-trained generative priors. This practical approach demonstrates strong performance, achieving superior results over existing baselines without the need for costly fine-tuning.
2. Contribution to the field via benchmarking. A key strength is the introduction of the Interp3DData benchmark. By categorizing the 3D morphing problem into distinct difficulty levels, the paper provides a valuable and standardized framework for evaluation, which facilitates future comparisons and advances in the field.
3. Clear motivation and well-structured methodology. The paper is clearly structured, with a well-articulated motivation that pinpoints the problem of 3D correspondence misalignments. The proposed three-stage progressive alignment framework is a logical and direct response to these identified challenges, making the technical contributions easy to follow.

### Weaknesses
1. Limited generality and model dependency. The entire proposed pipeline is intrinsically tied to the specific architecture and latent representations of the Trellis model. While this is a valid choice for a proof-of-concept, it raises questions about the generalizability of the method. The approach's strong dependency on Trellis-specific components (e.g., the SLAT representation) makes it non-trivial to adapt Interp3D to other 3D generative frameworks, thereby limiting its immediate applicability and scope.
2. Insufficient analysis of module interactions. The paper introduces three progressive alignment components but does not thoroughly investigate the interplay between them. For instance, it remains unclear whether the semantic alignment in the condition space and the subsequent SLAT-guided structure alignment always work in harmony or could potentially introduce conflicting signals in some cases. A more detailed ablation study, perhaps with counterexamples, is needed to demonstrate that this progressive strategy is consistently synergistic and does not lead to contradictory guidance during the generation process.

### Questions
1. In the semantic-aligned condition interpolation, the target condition tokens are permuted to match the source's semantic layout. While this benefits intermediate frames, could this permutation inadvertently harm the fidelity of the final generated target (when α=1)? At this extreme, one would expect an unaltered target output, but the permuted condition might distort it. Could the authors discuss or provide analysis on this potential issue and whether the alignment strategy is adjusted near the endpoints?
2. Given the potential for conflicting guidance between the different alignment modules (e.g., semantic vs. structural), could the authors provide further ablation studies that not only add modules progressively but also examine the performance of individual modules and other combinations? This would more directly validate the synergy of the proposed pipeline.

I would like to raise my score if above concerns are properly solved.

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
The paper proposes Interp3D, a training-free framework for textured 3D morphing that leverages the generative prior of TRELLIS. The method enforces correspondence progressively at three levels: (1) semantic-aligned condition interpolation in 2D token space, (2) SLAT-guided structure interpolation during 3D structure generation with fused attention and dynamic patch matching, and (3) fine-grained texture fusion for appearance transfer. The authors also curate Interp3DData and evaluate with FID, PPL, LPIPS plus a user study, showing consistent improvements over previous methods.

### Strengths
1. The method cleanly exploits TRELLIS to realize training-free 3D morphing, turning a strong generative prior into a controllable morph pipeline.

2. A dedicated benchmark (Interp3DData) and quantitative comparisons demonstrate that Interp3D surpasses prior SOTA on multiple metrics.

3. The paper is clearly structured. The presentation is clear and easy to follow.

4. Results are visually compelling—fewer geometric failures and significantly less texture blur than baselines.

5. User study included: Human preference indicates better fidelity, smoothness, and overall plausibility for Interp3D.

6. The paper computes alignment between condition tokens and generation-time tokens and then performs correspondence-aware attention interpolation, which is principled and well motivated.

### Weaknesses
Actually, I do not think this paper has an obvious weakness. My concern is about the manual hyperparameter sensitivity: The pipeline exposes numerous hand-tuned knobs (e.g., grid patch size and step-wise schedules in SLAT-Guided Structure Interpolation). It is unclear how robust these settings are across diverse source–target pairs or whether per-pair tuning is required.

### Questions
Failure cases: Do you observe failures when the source and target are semantically or structurally very far apart, or when TRELLIS reconstruction itself fails? Please include concrete examples, diagnostics, and whether the morph collapses or exhibits flickering/blur.

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
3

### Summary
Interp3D addresses this by coupling generative priors with reliable 3D correspondences through a progressive alignment principle.

SLAT-Guided Structure Interpolation: Enforces geometric consistency by maintaining structural correspondences using SLAT features from a 3D foundation model.

Fine-Grained Texture Fusion: Transfers appearance details by retrieving and fusing source and target features at corresponding locations, ensuring coherent and realistic surface appearance.

### Strengths
The paper presents a clear and detailed description of the Interp3D framework, including the three stages of alignment and the specific techniques used at each stage. The pseudocode provided in the appendix further enhances the clarity of the implementation.

The creation of the Interp3DData dataset provides a valuable resource for the research community.


The method is shown to be robust across a wide range of source–target pairs, including those with geometric and textural differences.

### Weaknesses
Current metrics (FID, PPL, LPIPS) focus on visual quality but lack assessment of semantic coherence and structural fidelity. 

The method heavily relies on attention interpolation of an existing method Trellis3D, which lacks novelty. It's rather than revealing the findings of  Trellis3D.

### Questions
Will the Interp3DData data be open-sourced?

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
4

### Summary
This paper tackles textured 3D morphing, aiming for smooth and plausible transitions between 3D assets that preserve both structure and appearance. The authors argue existing methods fail by either ignoring texture or by extending 2D strategies, which causes semantic and structural errors.

The paper proposes Interp3D, a training-free framework built on a "progressive alignment principle." This 3-stage process involves:

1. Semantic-Aligned Condition Interpolation: Matches semantic patches using DINOv2 features.

2. SLAT-Guided Structure Interpolation: Uses structured latents (SLAT) from a 3D model (TRELLIS) to guide geometric correspondence.

4. Fine-Grained Texture Fusion: Aggregates texture features to preserve detail.

A new dataset, Interp3DData, was built for evaluation. Results show Interp3D outperforms baselines in fidelity, smoothness, and plausibility.

### Strengths
1. The paper clearly identifies a specific technical problem (artifacts in 3D morphing) and proposes a logical, multi-stage framework that effectively solves it.

2. The training-free nature is a significant practical advantage, showing how to guide a generative prior using feature-space manipulation.

3. The evaluation is comprehensive, using quantitative metrics (FID, PPL, LPIPS) and a user study (Table 2) to prove its superiority over baselines.

4. The creation of Interp3DData is a useful, albeit minor, contribution to this specific sub-field.

### Weaknesses
1. The primary weakness is the limited scope and perceived significance of the task itself. 3D morphing is a relatively niche problem. The proposed solution, while effective, feels more like a clever engineering trick or application built on top of TRELLIS, rather than a novel, generalizable research contribution. The work is highly incremental.

2. The method seems tightly coupled to the TRELLIS model's SLAT representation. It's unclear if this principle generalizes to other 3D models.

3. The paper admits failure when semantic gaps are huge (Fig. 10), suggesting the initial DINOv2 patch matching is a bottleneck.

4. The "dynamic patch correspondence" (Sec 4.2) is vague. The paper lacks sensitivity analysis for the patch size $s_t$ schedule and the similarity threshold $\tau_0$.

### Questions
1. How generalizable is Interp3D? How dependent is it on the TRELLIS SLAT representation?

2. The failure case (Fig. 10) points to the DINOv2 matching as a bottleneck. Could this be improved with different 2D features (e.g., CLIP) or a 3D-native correspondence search?

3. Regarding Sec 4.2: What is the specific value for the threshold $\tau_0$ and the schedule used for decreasing the patch size $s_t$?

4. What is the performance gain from using the Beta(5, 5) distribution for sampling compared to standard linear interpolation?

### Soundness
3

### Presentation
3

### Contribution
2
