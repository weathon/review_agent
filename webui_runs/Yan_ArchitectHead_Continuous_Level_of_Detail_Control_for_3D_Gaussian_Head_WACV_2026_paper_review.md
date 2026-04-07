
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the core contribution: continuous LOD control for 3D Gaussian head avatars. The abstract is well-structured, stating the problem, the proposed solution (ArchitectHead), and key results. However, the claim of being **"the first framework"** for continuous LOD control in this domain is a strong novelty claim that must be carefully scrutinized against the related work discussed later (e.g., Milef et al.'s "continuous LOD method" for static scenes [26], LoDAvatar [4]). The abstract should more precisely position its novelty relative to these works. The quantitative results (e.g., "L1 Loss +7.9%") are stated but the baseline for this percentage change is ambiguous (is it relative to the model's own highest LOD performance?).

### Introduction & Motivation
The motivation is solid: balancing rendering efficiency and quality for scalable applications (VR, telepresence) is a genuine need. The gap identification—that existing 3DGS avatars use a fixed number of Gaussians—is accurate. The introduction of continuous vs. discrete LOD concepts is clear. The high-level description of the UV-based strategy and the two remaining challenges (insufficient local information, balancing resolutions) effectively sets the stage for the method. The contributions are clearly listed. A minor point: the claim "first head avatar creation framework that supports continuous adjustment" in the text should be tempered with the acknowledgment of prior continuous LOD work in *static* 3DGS, clarifying the novelty lies in applying it to *dynamic, animatable head avatars*.

### Method / Approach
The method is generally well-explained and appears reproducible.
*   **Section 3.2 (Framework Overview):** The use of FLAME and UV rasterization is standard. The definition of LOD `l` and the resolution calculation (Eq. 3) is clear. The introduction of a learnable multi-level UV feature field is the core innovation.
*   **Section 3.3 (Multi-Level UV Feature Field):** The rationale for the multi-level field (avoiding smoothing of critical information when resizing a single map) is logical. The weighted blending strategy (Eq. 4, 5) using a softmax over log-resolutions is a reasonable design for smooth interpolation. However, the choice of temperature `τ=0.35` and the specific set of three resolutions (256, 128, 64) seem arbitrary. An ablation on the number of levels and the `τ` parameter would strengthen the justification. Why not more levels for finer control?
*   **Section 3.4 (Loss Functions):** The losses are standard. The weighting of the scale regularization `L_s` more heavily for smaller `l` (higher LOD) is noted but not theoretically justified. It seems intuitive (denser Gaussians might need stricter scale control), but a brief explanation would be helpful.
*   **Section 3.5 (Training Scheme):** The two-stage training (first high LOD, then random LOD sampling) is sensible. A question arises: why fix the mapping network `M` in the second stage? The paper states the first stage stabilizes learning; does unfreezing `M` lead to instability? This is a minor implementation detail.

**Overall Assessment of Method:** The technical approach is sound and novel for the head avatar domain. The primary concern is the depth of ablation studies to validate architectural choices (e.g., number of feature map levels, interpolation scheme).

### Experiments & Results
This section is extensive but has several critical weaknesses that must be addressed for ICLR.
*   **Self-Reenactment (Tables 1, 2, Fig. 3, 4):** Quantitative results show SOTA or near-SOTA performance at LOD 0.0, which is excellent. The qualitative figures (Fig. 3, 4) support the claim of high quality. The inclusion of fp16 results is a good practical note.
*   **Cross-Identity Reenactment (Fig. 5):** Only qualitative results are shown. This is a **significant omission**. For a rigorous evaluation, quantitative metrics on cross-identity reenactment are necessary, especially since the method conditions on FLAME expression codes which should generalize.
*   **LOD Control (Fig. 6, Tables 3, 4):** Table 3 effectively shows the rendering speed-up with lower LODs. Table 4 and Figure 6 are the core of the contribution. However:
    *   The quality degradation metrics in the abstract and text (e.g., L1 +7.9% at LOD 1.0) are relative to the method's own LOD 0.0 performance. It is more informative to compare the *absolute* quality at low LODs against *other methods* at their fixed (high) Gaussian count. The paper shows it maintains "near SOTA" performance at lower LODs, but this needs clearer presentation. A plot of PSNR/SSIM/LPIPS vs. #Gaussians (or FPS) comparing ArchitectHead at various LODs against baselines would be more compelling than Table 4 alone.
    *   Figure 6 shows artifacts (red arrows) at low LODs, which is expected. However, the claim of "smooth transitions" is qualitative. A quantitative analysis of the smoothness (e.g., measuring the difference in rendered frames as LOD changes continuously) would strengthen the "continuous" claim.
    *   **Major Missing Baseline:** The paper rightly cites LoDAvatar [4] as a related discrete LOD method for avatars. A direct comparison with LoDAvatar (or a discussion of why it's not feasible, e.g., requirement for multi-view data) is **crucial** to establish the advantage of *continuous* LOD. Without this, the benefit over simply training 3-4 discrete models is not fully quantified.
*   **Ablation Studies (Table 4, Fig. 7):** These are good, showing the importance of the learnable feature field and the multi-level design over a single map. The observation that a single high-res map (`fmap (256)`) degrades high-LOD quality is interesting and supports the multi-level design. However, as noted in the Method section, ablations on the number of levels and the interpolation strategy are missing.

### Writing & Clarity
The paper is generally well-written. The pipeline figure (Fig. 2) is clear. Some minor points:
*   Figure 1 caption: "Red arrow indicates visible artifacts (sparse Gaussians)" – the arrow seems to point to a region, but it's not immediately clear what the specific artifact is. Labeling might help.
*   Section 4.4: "We also evaluated the trained avatar in half-precision (fp16), and the results are comparable..." This statement is vague. It should reference the specific quantitative results in Tables 1 and 2 (the `Ours (fp16)` rows).
*   The limitations section is cursory.

### Limitations & Broader Impact
The stated limitations are valid (dependence on accurate FLAME tracking, overfitting to rare expression/pose combinations) but are common to most prior work. A more thorough discussion is needed for ICLR:
*   **Technical Limitations:** What happens at very low LODs (e.g., <64 resolution)? Does the model break down? Is there a minimum usable resolution? The method is UV-based; how does it handle topological changes or extreme expressions that distort the UV parametrization?
*   **Broader Impact:** This section is **entirely missing**. ICLR expects a discussion of potential societal impacts, both positive (e.g., enabling more efficient telepresence) and negative (e.g., misuse for deepfakes, energy consumption of training). This must be added.

### Overall Assessment
ArchitectHead presents a novel and technically sound approach to a meaningful problem: enabling continuous Level of Detail control for 3D Gaussian head avatars. The core idea of a multi-level UV feature field with weighted resampling is elegant and well-motivated. The method achieves impressive results, matching or exceeding SOTA quality at high detail while offering a tunable efficiency-quality trade-off. However, the paper in its current form has significant gaps that lower its readiness for ICLR. The most critical issues are: (1) the lack of quantitative results for cross-identity reenactment, (2) an insufficient comparison to prior LOD methods (especially discrete ones like LoDAvatar) to concretely justify the need for *continuous* control, and (3) the complete absence of a broader impact statement. Furthermore, the ablation studies could be deepened. If the authors can thoroughly address these points—particularly by adding missing quantitative evaluations and a compelling comparison to discrete LOD alternatives—the contribution would be strong and likely meet ICLR's bar. As it stands, it is a promising but incomplete submission.

────────────────────────────────────────
NEUTRAL REVIEWER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper proposes ArchitectHead, a framework for creating 3D Gaussian head avatars that support real-time, continuous control over the Level of Detail (LOD). The core idea is to parameterize Gaussians in a 2D UV space, using a learnable multi-level UV feature field. A neural decoder generates Gaussian attributes from features resampled at a target resolution, enabling smooth LOD adjustment by simply changing the UV map resolution without retraining. The method achieves state-of-the-art reconstruction quality at the highest LOD while allowing a significant reduction in Gaussian count (to 6.2%) for faster rendering with moderate quality degradation.

### Strengths
1. **Novel and Well-Motivated Contribution**: The paper clearly identifies a gap—the lack of continuous LOD control for dynamic 3DGS head avatars—and proposes the first framework to address it. The concept of controlling Gaussian count via UV map resolution is intuitive and well-explained (Sections 1, 3.2).
2. **Strong Empirical Results**: The method achieves SOTA or near-SOTA metrics (L1, PSNR, SSIM, LPIPS) on standard self- and cross-identity reenactment tasks (Tables 1, 2, Figure 3). The LOD control is demonstrated effectively, showing a near doubling of FPS at the lowest LOD with a relatively small quality drop (Table 4, Table 3).
3. **Effective Technical Design**: The multi-level UV feature field with the softmax-based weighted resampling strategy (Eq. 4, 5) is a clever solution to enable smooth transitions between LODs. The ablation study (Table 4, Figure 7) provides solid evidence that this design is superior to using a single-resolution feature map.
4. **Practical Utility and Clarity**: The framework is practical, enabling real-time rendering with fp16 support. The pipeline is clearly depicted (Figure 2), and the two-stage training scheme is sensible. The project page and planned code release support reproducibility.

### Weaknesses
1. **Heavy Reliance on FLAME Priors**: The method is contingent on accurate FLAME tracking for mesh generation and UV rasterization. The paper acknowledges that tracking failures or overfitting to rare poses/expressions can lead to artifacts (Section 5). This dependence limits applicability to data where such a prior is unavailable or unreliable.
2. **Limited Demonstration of Generalization**: While cross-identity reenactment is shown (Figure 5), the evaluation is primarily quantitative on a limited set of subjects (3 from PointAvatar, 9 from INSTA). A more rigorous analysis of generalization to unseen identities, extreme expressions, or in-the-wild monocular videos is lacking, which is important for ICLR's focus on learning.
3. **Incomplete Baseline Comparison for LOD Core Contribution**: For the central claim of continuous LOD control, the comparison is primarily against methods with *fixed* Gaussian counts. A direct comparison to the most relevant prior, **LoDAvatar** [4] (discrete LOD for body avatars), is only discussed qualitatively. Quantitative results (e.g., rendering speed vs. quality trade-off curve) comparing against a discrete LOD adaptation of a baseline would strengthen the claim of the advantage of *continuous* control.
4. **Superficial Discussion of Limitations**: The limitations section is brief. It does not discuss potential issues like the fixed bounds of the UV feature field (S_max=256, S_min=64), how the method might handle topological changes (e.g., mouth opening), or the memory overhead of storing the multi-level feature field.

### Novelty & Significance
**Novelty:** High. To the best of the reviewer's knowledge, this is the first work to enable **continuous** LOD control for animatable 3D Gaussian head avatars. The formulation of LOD control as a UV-space resampling problem and the introduction of the trainable multi-level UV feature field are novel contributions.
**Significance:** Moderately High. The work addresses a practical need for scalable avatar rendering in real-time applications. The ability to dynamically balance quality and performance is crucial for VR/telepresence with multiple avatars. The core idea of a resamplable feature representation could influence other dynamic 3DGS tasks.
**Clarity:** Good. The paper is generally well-written, with clear figures and a logical flow.
**Reproducibility:** Likely High. The method is described in sufficient detail, with training stages, hyperparameters, and loss functions provided. The promise of code release further supports this.

### Suggestions for Improvement
1. **Strengthen the LOD-Specific Analysis**: Add a quantitative comparison modeling a "discrete LOD" baseline (e.g., training multiple fixed-resolution FlashAvatar models). Plot a curve of performance (PSNR/LPIPS) vs. rendering speed (FPS) or #Gaussians to visually demonstrate the smoothness and advantage of your continuous approach over a discrete one.
2. **Deepen the Generalization Evaluation**: Conduct a more challenging cross-identity experiment, perhaps on a held-out subject from a different dataset, to test the limits of the method's generalization. Also, analyze failure cases due to FLAME tracking errors more concretely.
3. **Expand Ablation and Analysis**: Ablate the choice of the number of levels in the feature field (why 3?). Analyze the performance and visual quality at intermediate LOD values (e.g., LOD=0.25, 0.75) in more detail, not just 0.0, 0.5, 1.0.
4. **Discuss Broader Impact and Future Work More Concretely**: The conclusion mentions multi-view video and training efficiency. Elaborate on the specific challenges (e.g., feature field consistency across views) and potential solutions. A brief discussion on extending the continuous LOD principle to other avatar properties (e.g., material detail) could be insightful.

────────────────────────────────────────
SPARK FINDER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No direct comparison to existing LOD methods for 3DGS.** The paper claims to be the first for head avatars but does not compare to LOD-GS, CityGaussian, or FLoD, even in an adapted or simplified form. Without this, the novelty and effectiveness of the proposed LOD mechanism are not properly benchmarked.
2. **No rigorous evaluation of "continuous" LOD.** The paper shows results at discrete levels (0.0, 0.5, 1.0). To substantiate the claim of *continuous* control, an experiment showing smooth interpolation of a metric (e.g., PSNR) across many closely spaced LOD values (e.g., 0.0, 0.05, 0.10, ... 1.0) is missing. This is essential for the core claim.
3. **Insufficient analysis of rendering speed/quality trade-off.** Table 3 shows FPS but does not report the corresponding image quality metrics (PSNR, LPIPS) at those same LODs on the same hardware. A plot of FPS vs. quality (e.g., LPIPS) across the LOD range is needed to validate the practical utility of the method.
4. **Ablation on the two-stage training scheme is incomplete.** The paper states stage two uses randomly sampled LODs but does not ablate against other strategies (e.g., only stage one, or a joint training from the start). The necessity and design of the two-stage process for enabling continuous LOD are not proven.

### Deeper Analysis Needed (top 3-5 only)
1. **No analysis of what the UV feature field learns.** The paper claims the feature map captures "local information" but provides no analysis (e.g., via visualization, clustering, or sensitivity analysis) of what these features represent (e.g., identity, expression, texture). Without this, it's unclear if the method is learning a meaningful latent space or just overfitting.
2. **Lack of failure mode analysis for cross-identity reenactment.** The qualitative results (Fig. 5) are shown but with no quantitative metrics or analysis of where and why the method fails (e.g., with extreme expressions or mismatched identity geometry). This is critical for assessing generalization.
3. **The impact of FLAME tracking accuracy is not quantified.** The paper states reliance on accurate FLAME tracking as a limitation but does not show how performance degrades with noisy or inaccurate tracking estimates (e.g., by injecting synthetic noise). This is a major point of fragility for the entire pipeline.

### Visualizations & Case Studies
1. **Visualize the Gaussian point clouds at different LODs.** Figure 1 shows rendered images with grey dots, but clear visualizations of the 3D Gaussian distributions (point count, spatial distribution) at high, medium, and low LOD are missing. This would directly show if the resampling creates meaningful geometric simplifications or just sparse, artifact-prone point clouds.
2. **Show a real-time video demo of continuous LOD transition.** The claim of smooth, continuous control is best validated with a supplementary video that smoothly interpolates the LOD parameter during rendering, showing the visual transition. A static paper cannot demonstrate this key feature.
3. **Case studies on practical application scenarios.** The paper motivates LOD for multi-avatar rendering. A case study showing 2-4 avatars rendered simultaneously at different LODs, with a breakdown of total FPS and per-avatar quality, would powerfully demonstrate the claimed scalability.

### Obvious Next Steps
1. **Implement and compare against a strong discrete LOD baseline.** A logical next step is to train multiple fixed-resolution models (e.g., at resolutions 256, 128, 64) and compare their performance (quality, speed) to the proposed continuous method at equivalent Gaussian counts. This would ground the benefit of continuity.
2. **Explore view-dependent LOD.** The introduction mentions view-dependent LOD as a category. A natural extension is to make the LOD parameter view-dependent (e.g., based on distance from camera), which is a standard next step for a graphics-oriented paper.
3. **Extend evaluation to a standardized, larger-scale dataset.** The experiments are on a small, combined set of 12 identities. To be convincing for ICLR, evaluation on a larger, standard benchmark (like HDTF or a subset of VoxCeleb) with more diverse motions and identities is necessary.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (deepseek/deepseek-v3.2:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (deepseek/deepseek-v3.2 via OpenRouter)
========================================================================

## Summary
ArchitectHead introduces a framework for 3D Gaussian head avatars that enables continuous level-of-detail (LOD) control by parameterizing Gaussians in a UV space and using a multi-level feature field with weighted resampling. It achieves state-of-the-art quality at high LOD while allowing dynamic adjustment to reduce Gaussian count for faster rendering with moderate quality degradation.

## Strengths
- **First continuous LOD control for 3D Gaussian head avatars**: The paper clearly identifies a gap in existing methods—all fixed Gaussian counts—and proposes a novel solution tailored to dynamic, animatable avatars. Evidence: The abstract and introduction state this, and related work (Section 2.3) confirms prior LOD methods focus on static scenes or discrete control for avatars.
- **Strong empirical performance**: ArchitectHead matches or exceeds SOTA metrics (L1, PSNR, SSIM, LPIPS) on self-reenactment tasks at the highest LOD, and maintains near-SOTA quality at lower LODs with significant Gaussian reduction (6.2% at lowest LOD) and faster rendering. Evidence: Tables 1 and 2 show top results; Table 4 and Table 3 demonstrate the efficiency-quality trade-off.
- **Effective technical design**: The multi-level UV feature field with softmax-based weighted resampling enables smooth LOD transitions without retraining, and ablation studies validate its superiority over single-resolution feature maps. Evidence: Section 3.3 details the design; Table 4 and Figure 7 show improved performance with the full method.

## Weaknesses
- **Lack of quantitative cross-identity reenactment evaluation**: Only qualitative results are provided (Figure 5), missing metrics like PSNR or LPIPS for cross-identity driving. This undermines assessment of generalization ability, which is crucial for learning-based avatars.
- **Incomplete comparison to discrete LOD methods**: While the paper cites LoDAvatar and other LOD works, it does not quantitatively compare against a discrete LOD baseline (e.g., training multiple fixed-resolution models) to concretely demonstrate the advantage of continuous control. This leaves the practical benefit of continuity insufficiently justified.
- **Insufficient demonstration of continuity**: The core claim of "continuous" LOD is supported only by discrete samples (e.g., LOD 0.0, 0.5, 1.0). No quantitative analysis of smooth transitions (e.g., interpolating metrics across many LOD values) is provided, weakening the claim.
- **Limited ablation studies on architectural choices**: Key design decisions, such as the number of levels in the feature field (fixed at three) and the temperature parameter for resampling, are not ablated, leaving their justification incomplete.
- **Missing broader impact discussion**: The paper omits a section on societal implications, which is standard for ICLR submissions to address potential positive (e.g., efficient telepresence) and negative (e.g., deepfake misuse) impacts.

## Nice-to-Haves
- Deeper analysis of what the UV feature field learns, e.g., via visualizations or sensitivity analysis.
- Case studies on multi-avatar rendering scenarios to practically demonstrate scalability.
- Exploration of view-dependent LOD as a natural extension for future work.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Formatting nitpicks**: Criticisms about minor clarity issues in figure captions (e.g., arrow labels in Figure 1) are stylistic and do not affect scientific merit.
- **Overstated novelty claim**: The paper correctly positions itself as the first for head avatars, given prior LOD works focus on static scenes or discrete control; thus, challenges to this claim are unfounded.
- **Demand for theoretical justification of loss weights**: The empirical setup and results reasonably justify the loss design; requiring theoretical backing is unnecessary for this empirical contribution.
- **Request for extensive dataset expansion**: The current datasets (PointAvatar and INSTA) are standard in the field; demanding larger benchmarks is a generic improvement not core to the contribution.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add quantitative metrics (e.g., PSNR, LPIPS) for cross-identity reenactment in the experiments section.
- Include a comparison to a discrete LOD baseline, such as training fixed-resolution versions of a baseline method, and plot quality vs. rendering speed trade-offs to highlight the advantage of continuous control.
- Conduct ablation studies on the number of levels in the feature field and the temperature parameter to strengthen design choices.
- Incorporate a broader impact section discussing societal implications, including potential benefits and risks.

========================================================================
PREDICTED SCORE
========================================================================

Score: 5.2
Decision: N/A
Total Cost: $0.0212
