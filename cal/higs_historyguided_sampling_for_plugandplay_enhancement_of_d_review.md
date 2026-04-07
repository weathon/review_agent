=== CALIBRATION EXAMPLE 62 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:**
The title accurately reflects the core contribution: a history-guided sampling technique. The abstract clearly states the problem (unrealistic outputs with low NFEs/guidance), the proposed solution (HiGS), its key attributes (training-free, plug-and-play), and the main quantitative result (SOTA FID 1.61 on ImageNet 256x256 with 30 steps). The claims are specific and appear supported by the paper's content.

**Introduction & Motivation:**
The introduction effectively motivates the problem: diffusion models can produce blurry or unrealistic outputs when sampling efficiency is prioritized (fewer NFEs) or when using lower guidance scales to avoid oversaturation/diversity loss. The need for training-free methods to improve quality in these regimes is well-argued. The contributions are stated: a momentum-based, training-free sampling method that integrates past predictions. One minor weakness is that the related concept of "momentum" in sampling (e.g., in some ODE solvers) is not directly contrasted, which could sharpen the novelty claim.

**Method / Approach (Section 4):**
This is the core section. The motivation linking Euler steps to SGD on a time-varying energy function is insightful. However, the subsequent connection to STORM and the derivation of the momentum term feels slightly stretched and could be presented more as intuitive inspiration rather than a formal derivation. The strength of the paper lies in the empirical design choices (EMA history, square-root scheduling, orthogonal projection, DCT filtering) which are well-explained and justified via ablations. The error analysis in Appendix B is a valuable theoretical addition, showing how HiGS can reduce truncation error, though its practical correspondence to the full HiGS algorithm (which includes DCT and projection) is not fully established. The method is reproducible, with pseudocode and detailed hyperparameters provided. A logical gap: the initial motivation uses the score/Euler update, but the final algorithm applies the history correction to the *denoised prediction* `D_CFG(z_t)` rather than directly to the score/update term `u(z_t, t)`. The connection between these two applications could be clarified.

**Experiments & Results (Section 5):**
Extensive and generally convincing. The paper evaluates across multiple model families (Stable Diffusion variants, DiT, SiT), tasks (text-to-image, class-conditional), and regimes (varying NFE, CFG scale). The use of multiple human-preference metrics (HPSv2, ImageReward) alongside FID/IS/Precision/Recall is appropriate. The key claim of improved quality across budgets and scales is supported by Figures 5a and 5b. The SOTA FID result on ImageNet (1.61 with 30 steps, unguided) is significant.
*Concerns*:
1. **Baseline Clarity:** For the main text-to-image comparisons (Tables 1, 2), it is stated that the same NFE and CFG scale are used for baseline and HiGS. However, HiGS introduces its own scale `w_HiGS`. The improvement could partly stem from effectively increasing the total "guidance" magnitude. A more controlled ablation would compare HiGS to a baseline where the CFG scale `w_CFG` is increased to match the perceptual "strength" or computational cost (though HiGS adds no extra NFEs). The authors partially address this by showing HiGS helps even when combined with high-CFG (Figure 2), but the low-NFE/low-CFG regime needs this control.
2. **Statistical Significance & Reporting:** Win rates are reported (e.g., 0.93), but the number of pairwise comparisons (N) used to compute these rates is not stated. For robust conclusions, N should be provided, and confidence intervals or significance tests would strengthen the claims. The FID scores in Table 3 are reported without standard errors (common but still a limitation).
3. **Missing Baseline:** For the low-NFE regime, a relevant baseline is other training-free sampling accelerators like DPM-Solver++ or UniPC. Table 6 shows HiGS improves upon these when added, but it's not shown if HiGS alone (on a simple Euler solver) outperforms these advanced solvers at the same step count. A direct comparison would better establish HiGS's value.
4. **Ablations (Appendix E):** Very thorough and a major strength. They validate design choices (EMA vs. average, DCT filtering necessity, schedule thresholds). The compatibility with distilled models (Table 4) and other guidance methods (APG, Figure 6) is excellent and broadens the impact.

**Writing & Clarity:**
The paper is generally well-written and structured. The figures effectively illustrate qualitative improvements. Some parts of the method section (Section 4.2) are dense due to the sequential introduction of multiple components (scheduling, projection, DCT). A summarizing schematic or a more streamlined high-level description before diving into details could improve readability. The pseudocode and algorithms in the appendix are clear and aid reproducibility.

**Limitations & Broader Impact:**
The discussion section briefly states that HiGS "inherits, albeit to a lesser extent, the biases and some limitations of the underlying diffusion models," which is appropriate but vague. A more concrete discussion of observed failure modes or edge cases (e.g., does HiGS ever degrade quality? Under what hyperparameters?) would be valuable. The broader impact statement is standard and adequate, noting the dual-use nature of generative model improvements. The reproducibility statement is good, linking to code and providing hyperparameter tables.

### Overall Assessment
This paper presents a simple, training-free, and empirically powerful modification to diffusion sampling that consistently improves perceptual quality across a wide range of models, step counts, and guidance scales. Its plug-and-play nature and compatibility with existing techniques (distilled models, other samplers, other guidance methods) make it highly practical. The most significant weakness is the somewhat tenuous theoretical motivation, which could be reframed more as inspired intuition. The empirical evaluation is extensive but would be strengthened by more direct comparisons to advanced solvers at low NFE and a clearer analysis of whether gains are purely additive or involve an effective guidance scale increase. Despite these concerns, the core contribution—a method that delivers substantial quality gains with zero extra training or forward passes—is compelling, well-validated, and likely to meet ICLR's bar for a solid incremental contribution with clear practical utility.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes History-Guided Sampling (HiGS), a training-free, momentum-based modification to diffusion model inference that leverages a weighted average of past model predictions to steer the sampling trajectory toward higher-quality outputs. HiGS is designed to enhance image quality, particularly when using fewer sampling steps (low NFEs) or lower classifier-free guidance (CFG) scales, introducing negligible computational overhead. The method is shown to improve various quality metrics across multiple diffusion architectures and, when applied to a pretrained SiT model, achieves a state-of-the-art FID of 1.61 for unguided ImageNet-256 generation using only 30 steps.

### Strengths
1.  **Strong Empirical Validation:** The paper provides extensive quantitative and qualitative experiments across a diverse set of models (Stable Diffusion variants, DiT, SiT, and distilled models), sampling budgets, and guidance scales. The consistent improvements in metrics like FID, HPSv2, and ImageReward, along with the SOTA result on ImageNet, constitute compelling evidence for the method's effectiveness (Tables 1, 2, 3; Figures 2-5).
2.  **Practical and Efficient Contribution:** HiGS is a plug-and-play method that requires no retraining or additional neural function evaluations (NFEs). The computational overhead is minimal, consisting only of lightweight operations on past predictions (DCT, projection), which is convincingly demonstrated by identical iteration-per-second measurements compared to the baseline (Section D).
3.  **Theoretical Motivation and Analysis:** The paper provides a solid conceptual foundation by framing the Euler sampler as SGD on a time-varying energy function and linking HiGS to variance-reduction techniques like STORM. The error analysis in Appendix B is a notable strength, theoretically showing how the history term can reduce the local truncation error of the ODE solver.
4.  **Comprehensive Ablation and Design Study:** The authors thoroughly justify their design choices through systematic ablations (Appendix E), testing different history inputs, averaging functions, weight schedules, and the impact of DCT filtering and projection. This strengthens the methodological contributions and provides a clear recipe for implementation.

### Weaknesses
1.  **Hyperparameter Sensitivity and Tuning:** While the ablations show robustness within ranges, the method introduces several new hyperparameters (e.g., EMA decay `α`, DCT cutoff `R_c`, schedule bounds `t_min`/`t_max`, guidance weight `w_HiGS`, projection weight `η`). The need to tune these for different models or tasks (as seen in Tables 10-12) could hinder out-of-the-box adoption and raises questions about generalizability to entirely new architectures.
2.  **Limited Exploration Beyond Standard Image Generation:** The evaluation is comprehensive but primarily focused on class-conditional (ImageNet) and text-to-image generation. The applicability and benefits of HiGS for other modalities (e.g., video, audio, 3D) or more complex conditional tasks (e.g., inpainting, compositional generation) remain unverified, limiting the claim of being a "universal enhancement."
3.  **Insufficient Comparison to Closely Related Samplers:** The related work section covers the field broadly but could more directly compare HiGS to other multi-step or predictor-corrector ODE solvers (e.g., detailed comparison to DPM-Solver++ or UniPC) that also utilize past information. A deeper discussion on how HiGS's momentum-based guidance conceptually differs from or complements the error correction in higher-order ODE solvers would clarify its novelty.
4.  **Potential Over-Engineering:** The final method incorporates multiple components (EMA history, scheduled weight, orthogonal projection, DCT filtering). While each is justified via ablation, the cumulative complexity might obscure the core, most impactful insight. A simpler variant (e.g., without projection or DCT) might suffice for many use cases, and the paper could better delineate which components are essential vs. supplementary.

### Novelty & Significance
**Novelty:** The core idea of using an EMA of past predictions as a momentum term to guide diffusion sampling is novel in this specific formulation and context. While momentum in optimization and multi-step ODE solvers are established concepts, their integration into the CFG update rule with the proposed refinements (frequency filtering, projection) represents a clear and non-trivial advancement.
**Clarity:** The paper is generally well-written. The motivation, method derivation, and algorithm description (supported by pseudocode in the appendix) are clear. The connection to SGD and STORM provides an intuitive hook.
**Reproducibility:** High. The paper includes a detailed reproducibility statement, pseudocode (Algorithms 1-3), and specific hyperparameter tables for key experiments. Building upon official model implementations further aids reproducibility.
**Significance:** The significance is high. Improving sample quality in the low-NFE and low-CFG regime is a critical problem for making diffusion models more efficient and practical. A training-free method that delivers consistent gains across models and achieves a new SOTA result on a standard benchmark is a valuable contribution to the community.

### Suggestions for Improvement
1.  **Conduct a "Simplicity Ablation":** Present results for a minimally viable version of HiGS (e.g., just the EMA history term with a constant weight) versus the full, tuned version. This would help the community understand the performance-complexity trade-off and identify the most critical component.
2.  **Explore Hyperparameter Generalization:** Test the hyperparameter sets from Tables 10-12 on a held-out model (e.g., a different text-to-image model not used in development) to evaluate how much tuning is genuinely required for a new model. Proposing a heuristic or adaptive method for setting key parameters would greatly enhance practicality.
3.  **Expand the Scope of Evaluation:** Demonstrate HiGS on at least one additional task or modality (e.g., latent video generation, audio synthesis, or image editing) to substantiate the claim of being a broadly applicable plug-and-play enhancement.
4.  **Strengthen the Related Work Comparison:** Add a dedicated quantitative comparison or discussion section contrasting HiGS with other advanced sampling techniques that use past steps (e.g., higher-order ODE solvers). This would better position HiGS within the existing sampler landscape.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to state-of-the-art fast samplers and distillation techniques.** The paper only compares HiGS to standard CFG sampling. To claim a universal plug-and-play enhancement, it must be benchmarked against other advanced, low-NFE samplers (e.g., DPM-Solver++, UniPC, recent ODE solvers) and training-free enhancement methods (e.g., FreeU, SVD). Without this, it's unclear if HiGS offers unique benefits.
2. **Systematic low-NFE evaluation on high-resolution text-to-image models.** The primary claim is enhancement with fewer steps, but quantitative results (Tables 1, 2) are shown at relatively moderate steps (e.g., 20 for SDXL). Critical evaluation at very low NFEs (e.g., 4-8 steps) on models like SD3/Flux is missing. Does HiGS prevent complete collapse or just mildly improve quality?
3. **Ablation on the necessity of DCT filtering and projection.** The method has several heuristic components (DCT filter, projection, EMA schedule). The ablation in Appendix E shows they help, but a crucial experiment is missing: does HiGS *without* these complex post-processing steps still beat the baseline? If not, the core "momentum" idea is insufficient.
4. **Evaluation on a broader set of tasks and architectures.** Experiments are limited to class-conditional ImageNet and text-to-image. Testing on other modalities (e.g., latent vs. pixel-space models, video diffusion, depth/segmentation conditioning) is needed to substantiate the "plug-and-play" claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical analysis of why "momentum" works for diffusion sampling.** The provided error analysis (Appendix B) is generic ODE analysis, not specific to diffusion models. A deeper analysis linking the empirical improvement to the *learned score function's properties* (e.g., reducing bias/variance in score estimation) is required. Without it, the method appears as an unprincipled heuristic.
2. **Analysis of trade-offs: quality vs. diversity vs. alignment.** The paper shows improved HPS/FID but does not thoroughly analyze potential downsides. Does HiGS reduce sample diversity (Recall) or affect prompt alignment (CLIP score) despite claims? A quantitative analysis of the diversity-quality trade-off across CFG scales is missing.
3. **Diagnosis of failure modes and limitations.** The paper states HiGS works "across all settings," but no systematic analysis of when it fails is provided. Does it amplify artifacts in certain prompt categories? Does it interact negatively with specific sampler discretizations? A dedicated failure case study is essential for trust.

### Visualizations & Case Studies
1. **Side-by-side comparison of sampling trajectories.** Visualizing the latent trajectory (e.g., in a 2D PCA projection) for baseline CFG vs. HiGS would reveal if the "guidance" actually steers the path meaningfully or just adds noise. This is direct evidence for the core claim.
2. **Visualization of the DCT-filtered update term ∆D_t.** Showing what the high-pass filtered "guidance signal" looks like across timesteps would clarify what structural information HiGS is injecting. Is it primarily edges, textures, or something else?
3. **Grid of samples showing diversity within a single prompt.** Current visuals are single cherry-picked images. To assess diversity preservation, a grid of 10+ samples per prompt for baseline and HiGS is necessary.

### Obvious Next Steps
1. **Hyperparameter sensitivity study.** The method has many hyperparameters (`w_HiGS`, `t_min`, `t_max`, `α`, `η`, `R_c`, `λ`). A systematic sensitivity analysis (beyond the limited ablations) should be in the main paper to show robustness and provide reliable defaults.
2. **Compute and memory overhead profiling.** The claim of "practically no additional computation" needs verification. A proper table comparing exact runtime/FLOPs/memory for baseline vs. HiGS across different batch sizes and resolutions is required, as DCT/projection operations do have non-zero cost.
3. **Direct comparison with a simple moving average baseline.** A critical ablation is to replace the HiGS update with a simple update like `z_t = z_t + γ*(D_CFG(z_t) - D_CFG(z_{t-1}))`. If this performs similarly, it undermines the novelty of the history buffer and EMA design.

# Final Consolidated Review
## Summary
This paper proposes History-Guided Sampling (HiGS), a training-free, plug-and-play method to enhance diffusion model sampling. HiGS incorporates a momentum term derived from an exponential moving average of past denoised predictions to steer the sampling trajectory. It improves image quality, especially when using fewer sampling steps or lower guidance scales, without additional neural network evaluations. The method is extensively validated across multiple model families and achieves a state-of-the-art FID of 1.61 on unguided ImageNet-256 generation with only 30 steps.

## Strengths
- **Effective and efficient plug-and-play enhancement:** HiGS provides consistent quality improvements across diverse models (Stable Diffusion variants, DiT, SiT) and regimes (varying NFEs and CFG scales) while introducing negligible computational overhead—runtime and memory usage are identical to the standard CFG baseline, as confirmed in Section D.
- **Comprehensive empirical validation and ablation:** The paper demonstrates quantitative gains on multiple metrics (FID, HPSv2, ImageReward) and includes thorough ablation studies (Appendix E) that justify each design choice (EMA history, scheduled weight, DCT filtering, projection). Compatibility with distilled models and other guidance techniques (e.g., APG) further broadens its applicability.
- **Theoretical motivation and error analysis:** The paper provides a clear conceptual link by interpreting the Euler sampler as SGD on a time-varying energy and connecting HiGS to variance-reduction techniques. Appendix B offers a rigorous error analysis, showing HiGS can reduce the local truncation error of the ODE solver from O(h²) to O(h³).

## Weaknesses
- **Hyperparameter sensitivity and tuning burden:** While ablations show robustness within ranges, HiGS introduces several new hyperparameters (e.g., `w_HiGS`, `t_min`, `t_max`, EMA decay `α`, DCT cutoff `R_c`). Tables 10–12 indicate these are tuned per model, which may hinder out-of-the-box adoption and raises questions about generalizability to novel architectures without tuning.
- **Insufficient direct comparison to advanced low-NFE samplers:** The paper shows HiGS improves upon default samplers (Euler) and is compatible with others (Table 6). However, a direct, controlled comparison against state-of-the-art, low-NFE dedicated samplers (e.g., DPM-Solver++, UniPC) at the same step count is missing. This makes it difficult to assess whether HiGS’s gains are unique or if similar benefits could be achieved by simply using a more advanced solver.
- **Reliance on heuristic post-processing components:** The final algorithm incorporates several empirically motivated components (DCT high-pass filtering, orthogonal projection). While ablations show they help, their necessity suggests the core momentum idea alone is insufficient for robust improvements. A deeper analysis linking these heuristics to specific failure modes (e.g., color shifts) would strengthen the method’s foundation.

## Nice-to-Haves
- **Visual analysis of the guidance signal:** Visualizing the DCT-filtered update term ∆D_t across timesteps could clarify what structural information HiGS injects (e.g., edges, textures) and provide more intuitive evidence for its operation.
- **"Simplicity ablation" presentation:** Showing results for a minimal version of HiGS (e.g., just the EMA history with a constant weight) versus the full tuned version would help users understand the performance-complexity trade-off and identify the most critical component.
- **Exploration in additional modalities:** A brief demonstration on one additional task or modality (e.g., latent video generation or image editing) would further support the "plug-and-play" claim beyond class-conditional and text-to-image generation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness about statistical significance of win rates:** The paper reports win rates (e.g., 0.93) with clear margins; providing exact N and confidence intervals is not standard practice in the field for such metrics, and the results are sufficiently convincing without them.
- **Weakness about HiGS effectively increasing total guidance magnitude:** The paper shows HiGS improves quality across all CFG scales (Figure 5a), including high scales where an increase in effective guidance would likely cause oversaturation. The gains are thus not merely due to an increased guidance magnitude.
- **Weakness about missing comparison to simple momentum baselines:** The paper includes ablations comparing different history functions (Table 9, including simple averaging), which shows the EMA design is effective but not uniquely superior. A simpler moving average baseline is conceptually similar and its performance is indirectly addressed.
- **Weakness about need for evaluation at extremely low NFEs (4–8 steps):** The paper focuses on practical step counts (e.g., 10–30) where quality is still meaningful; demanding evaluation at near-collapse regimes is outside the stated scope of providing a "plug-and-play enhancement" for realistic efficiency gains.
- **Strength about the paper being "well-written" or "topic is important":** These are generic and do not highlight what this specific paper does better than others.

## Novel Insights
The paper’s core novel insight is framing the diffusion sampling update as a form of stochastic gradient descent on a time-varying energy function, which then naturally motivates the use of a momentum term from optimization (inspired by STORM) to reduce variance. This perspective allows the authors to reinterpret the difference between the current denoised prediction and a weighted history of past predictions as a guidance direction that steers sampling toward higher-probability regions. The insight that this history term can act as an implicit “weaker model” signal—similar in spirit to autoguidance but requiring no extra training—is clever and well-supported by the consistent improvements in low-NFE and low-CFG regimes.

## Suggestions
- **Add a direct comparison to advanced samplers:** In the main experiments, include a quantitative comparison between HiGS (on a simple Euler solver) and other state-of-the-art low-NFE samplers (e.g., DPM-Solver++, UniPC) at the same step count, using the same base model. This will clearly delineate HiGS’s unique contribution relative to existing sampling advances.
- **Provide clearer guidance on hyperparameter transfer:** Based on the observed robustness ranges (e.g., `α` ∈ [0.5,0.75], `t_min` ∈ [0.3,0.5]), propose a recommended default set or a simple heuristic for initializing these parameters on a new model, reducing the tuning burden for adopters.
- **Expand the limitation discussion:** Briefly discuss observed failure modes or edge cases (e.g., does HiGS ever degrade quality? Under what extreme hyperparameters?). This would help users avoid pitfalls and better understand the method’s boundaries.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
