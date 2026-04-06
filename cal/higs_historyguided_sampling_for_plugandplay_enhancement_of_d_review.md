=== CALIBRATION EXAMPLE 64 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title accurately reflects the core contribution: a history-guided sampling technique. The abstract clearly states the problem (unrealistic outputs with low NFEs/low CFG), the solution (HiGS as a plug-and-play, momentum-based method), and the key result (SOTA FID of 1.61 on unguided ImageNet-256 with 30 steps). Claims are specific and seem supported by the body.

**Introduction & Motivation:** The problem is well-motivated: the trade-off between sampling efficiency (NFEs) or guidance scale and output quality is a recognized practical bottleneck. The introduction effectively sets up the need for training-free methods that work in low-NFE/low-CFG regimes. The contributions (a plug-and-play method, consistent improvements, SOTA results) are clearly stated.

**Background (Section 3):** Standard and correct. It sets up the necessary notation and concepts (CFG) clearly.

**Method / Approach (Section 4):**
*   **Motivation & Intuition (4.1):** The connection of the Euler step to gradient descent on a time-varying energy function is insightful and provides a useful lens. The link to STORM-like momentum is plausible as an inspiration. However, the theoretical motivation feels somewhat *post-hoc*. The final algorithm is a complex, engineered solution (EMA history, scheduled weight, orthogonal projection, DCT filtering) that is several steps removed from the initial SGD/momentum analogy. The derivation feels more like an intuitive justification than a strict guide for the design.
*   **Algorithm Design (4.2):** The description is detailed, but the method has **many hyperparameters**: `w_HiGS`, scheduling bounds `t_min`/`t_max`, EMA parameter `α`, projection weight `η`, DCT cutoff `R_c` and sharpness `λ`. While the ablations in Appendix E show robustness within ranges, the complexity is high. The necessity of *all* components (especially DCT filtering for color correctness) is demonstrated via ablation, but it raises the question of whether this is an elegant principle or a bundle of heuristics tuned for good results. The claim of "practically no additional computation" is true regarding NFEs, but the added operations (DCT/iDCT, projection) are non-trivial overheads, even if small compared to model evaluation.
*   **Theoretical Analysis (Appendix B):** Theorem B.1 claims that HiGS can improve the local truncation error of the Euler solver from O(h²) to O(h³) with a specific weight choice `w_k = h_k / (2h_{k-1})`. This is a non-trivial and valuable claim. However, **this specific, time-dependent weight `w_k` is not the fixed or scheduled `w_HiGS` used in the actual algorithm (Eq. 9)**. The theorem shows a *potential* error reduction for a *specific* instantiation of the history idea, but the implemented HiGS method uses a different, more complex update rule. This weakens the direct theoretical support for the final algorithm. The analysis should either be extended to justify the implemented schedule or its limitations should be clearly acknowledged.

**Experiments & Results (Section 5):**
*   **Scope and Models:** The evaluation is extensive and impressive, covering multiple model families (Stable Diffusion 1/2/3, DiT, SiT), conditional and text-to-image tasks, and distilled models. This strongly supports the "plug-and-play" claim.
*   **Metrics:** Appropriate metrics are used (FID/IS/Precision/Recall for class-conditional, HPSv2/ImageReward for text-to-image). The use of HPSv2 win rate is particularly convincing for human preference.
*   **Main Results:** Tables 1, 2, and 3 show consistent improvements across the board. The ImageNet result (FID 1.61 in 30 steps unguided) is a strong quantitative result if reproducible.
*   **Baselines:** The primary baseline is standard sampling with CFG. This is correct, as HiGS is a modification to the sampling loop, not a new sampler. The paper shows HiGS is complementary to different ODE solvers (Table 6) and other guidance methods (APG, Fig 6).
*   **Ablations (Appendix E):** This section is thorough and necessary, validating design choices like using CFG output for history, the need for DCT filtering, and the effect of hyperparameters. It demonstrates empirical robustness.
*   **Major Concerns:**
    1.  **Statistical Significance & Reporting:** For the key SOTA claim (FID 1.61), is this a single run? FID can have non-trivial variance. The paper should report the standard error or multiple seeds. This is crucial for an ICLR submission.
    2.  **Comparative Baselines:** While HiGS is compared to the base sampler, it is **not compared to other advanced, training-free sampling enhancements** (e.g., UniPC, DPM-Solver++(2M), or other predictor-corrector methods). A comparison showing HiGS's improvement *on top of* these advanced solvers, or versus them, is needed to fully establish its utility. The claim of being a "universal enhancement" requires testing integration with other high-performance samplers.
    3.  **Failure Modes / Limitations:** The experiments overwhelmingly show success. A dedicated discussion or experiment on when HiGS might fail or degrade performance (e.g., with very small step counts like 4-5, or on specific image categories) would strengthen the paper. The method's sensitivity to its hyperparameters, while ablated, should be explicitly stated as a limitation.

**Writing & Clarity:** The paper is generally well-written. The algorithm (Algorithms 1-3) is clearly presented. Some parts of Section 4 are heavy with notation but manageable. A higher-level, intuitive summary of the final update rule (Eq. 9) before diving into the components would improve readability.

**Limitations & Broader Impact:**
*   **Limitations:** The current "Limitations" paragraph in Section 6 is too vague ("inherits... biases"). Specific limitations should be stated: (1) The method introduces several new hyperparameters that may require mild tuning per model. (2) Its performance relative to other advanced solvers is not benchmarked. (3) The theoretical error reduction (Appendix B) uses a different formulation than the final algorithm. (4) Potential failure modes with extremely low NFEs are not explored.
*   **Broader Impact:** The statement is standard and appropriate for ICLR.

**Reproducibility:** The pseudocode and detailed hyperparameter tables (10-12) are excellent and should enable reproduction. The dependency on the DCT library is noted.

### Overall Assessment
HiGS presents a clever, empirically powerful, and extensively validated plug-and-play technique for improving diffusion sampling. Its ability to boost performance across diverse models and regimes, especially in low-NFE/low-CFG settings, is a strong practical contribution. However, the paper is weakened by: (1) a disconnect between the motivating theory and the final engineered algorithm, (2) a lack of comparison to other advanced sampling methods, and (3) insufficient discussion of statistical significance for its SOTA claim and potential limitations. Addressing these issues is essential for meeting ICLR's high bar. The core idea and empirical results are promising, but the analysis and positioning need refinement to solidify the contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces History-Guided Sampling (HiGS), a training-free, momentum-inspired modification to diffusion model sampling that leverages a weighted average of past model predictions to steer the sampling trajectory toward higher-quality outputs. The method aims to address common issues like blurriness and lack of detail when using fewer sampling steps or lower classifier-free guidance scales. HiGS requires no extra neural function evaluations, integrates seamlessly into existing samplers, and demonstrates consistent improvements across multiple models and settings, including achieving a new state-of-the-art FID of 1.61 for unguided ImageNet 256×256 generation with only 30 steps.

### Strengths
1. **Strong empirical results across diverse settings:** The paper provides extensive quantitative and qualitative evidence that HiGS improves image quality (measured by FID, HPSv2, ImageReward, etc.) across various diffusion models (Stable Diffusion variants, DiT, SiT), sampling budgets (low to high NFEs), and guidance scales (low to high CFG). The achievement of a new SOTA FID on ImageNet with very few steps (30 instead of 250) is a notable result.
2. **Practical and efficient design:** HiGS is training-free, adds negligible computational overhead (no extra forward passes), and is compatible with existing samplers and architectures, including distilled models. The paper confirms identical runtime and memory usage compared to baseline CFG (Section D), making it an attractive plug-and-play enhancement.
3. **Theoretical motivation and analysis:** The paper grounds HiGS in a stochastic gradient descent interpretation of diffusion sampling and provides an error analysis (Appendix B) showing that the method reduces the local truncation error of the Euler solver from O(h²) to O(h³), offering a principled justification for the improved convergence with fewer steps.

### Weaknesses
1. **Hyperparameter complexity and sensitivity:** HiGS introduces multiple hyperparameters (e.g., \(w_{\text{HiGS}}\), \(t_{\min}\), \(t_{\max}\), \(\alpha\), \(\eta\), \(R_c\), \(\lambda\)) and design choices (EMA averaging, orthogonal projection, DCT filtering). While ablation studies in Appendix E show robustness within ranges, the need to tune these parameters for optimal performance on new models or tasks could hinder its plug-and-play claim. The paper would benefit from more explicit guidelines for setting these parameters.
2. **Limited analysis of limitations and failure modes:** The paper primarily highlights successes, but does not thoroughly discuss scenarios where HiGS might fail or degrade performance (e.g., with very high CFG scales, certain architectures, or specific image types). A discussion of the method's boundaries would strengthen the critical assessment.
3. **Methodological complexity:** The full HiGS algorithm involves several components (history buffering, projection, DCT filtering) that, while individually justified, collectively make the method somewhat intricate. This complexity may obscure the core insight and could be a barrier to adoption and understanding.

### Novelty & Significance
HiGS introduces a novel perspective on improving diffusion sampling by leveraging past predictions in a momentum-like fashion, distinct from prior work on better ODE solvers or distillation. The connection to variance reduction in SGD and the error analysis provide fresh theoretical grounding. The practical significance is high: a training-free method that consistently boosts quality across models and settings, particularly in low-NFE and low-CFG regimes, addresses a key efficiency-quality trade-off in diffusion models. The SOTA ImageNet result with few steps is a clear indicator of impact.

### Suggestions for Improvement
1. **Simplify and consolidate hyperparameters:** Consider proposing a simplified version with fewer tunable parameters (e.g., fixing \(\eta=1\), using a default DCT threshold) for broader usability, while keeping advanced options for specialists. Provide clearer heuristics or a tuning protocol for new models.
2. **Add a failure mode analysis:** Include a subsection or appendix discussing cases where HiGS does not help or underperforms, along with potential reasons. This would provide a more complete picture of the method's applicability.
3. **Strengthen the plug-and-play claim:** While compatibility is shown with many models, explicitly test HiGS on a wider range of open-source diffusion architectures (e.g., latent vs. pixel-based, different noise schedules) and provide a simple, unified code snippet that can be dropped into any sampler with minimal adjustments.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to advanced, training-free samplers.** The paper lacks direct comparison against other sophisticated, training-free sampling enhancements like higher-order ODE solvers (e.g., Heun's method) or predictor-corrector methods (e.g., UniPC). Without this, it's unclear if the observed gains are simply due to using a more accurate integrator rather than the novel "history" mechanism.
2. **Empirical validation of the error analysis.** Appendix B provides a theoretical error reduction claim. This must be validated empirically by measuring the actual truncation error or the distance to a high-precision reference trajectory across steps for Euler vs. HiGS. Without this, the theoretical motivation is unsupported.
3. **Comprehensive hyperparameter sensitivity across architectures.** The ablations (e.g., for \( \alpha \), \( t_{min} \), \( w_{\text{HiGS}} \)) are primarily on Stable Diffusion XL. For a "plug-and-play" claim, a rigorous sensitivity analysis across diverse model families (DiT, SiT, latent vs. pixel-based) and tasks (text-to-image, class-conditional) is required to show robustness.
4. **Evaluation with proper diversity metrics.** The claim that HiGS avoids the diversity loss of high CFG is weakly supported only by visual examples. This must be quantified using standard diversity metrics (e.g., recall, pairwise perceptual distance) across a large batch of samples for low CFG+HiGS vs. high CFG.

### Deeper Analysis Needed (top 3-5 only)
1. **Mechanistic analysis of why HiGS works.** The link to variance reduction in SGD is speculative. The authors should analyze the *direction* of the update term \(\Delta D_{t_k}\): does it correlate with an estimate of the score function error or gradient variance? A simple analysis of its magnitude/norm over time would provide evidence.
2. **Justification for the frequency-domain filtering.** The DCT high-pass filter is a critical, non-intuitive component. The paper must analyze what spectral components are removed/attenuated and show direct evidence (e.g., visualizations of the filtered signal) that these correspond to unrealistic color artifacts. Without this, the filter seems like an unprincipled hack.
3. **Investigation of trajectory steering.** The core claim is that HiGS "steers the sampling trajectory." This should be visualized or quantified, for example by plotting the latent space trajectory with/without HiGS and measuring its alignment with directions of increasing likelihood or quality (e.g., via a separately trained critic).

### Visualizations & Case Studies
1. **Systematic failure cases.** The paper shows only successful examples. A dedicated analysis of when and how HiGS fails (e.g., on certain prompt types, with very low step counts (<5), or with specific model architectures) is necessary to understand its limitations and boundary conditions.
2. **Visualization of the update signal.** Showing the spatial and frequency components of the \(\Delta D_{t_k}\) term before and after DCT filtering, across time steps, would make the method's operation transparent and justify the design choices (projection, filtering).

### Obvious Next Steps
1. **Human evaluation study.** For a method claiming improved perceptual quality, reliance on automated metrics (HPSv2, ImageReward) is insufficient for an ICLR paper. A forced-choice human preference study (e.g., on MTurk) comparing HiGS vs. baseline across a diverse prompt set is essential to validate the perceptual claims.
2. **Test on more challenging and diverse generation tasks.** To solidify the "plug-and-play" claim, the method should be demonstrated on tasks beyond standard text-to-image and class-conditional ImageNet, such as inpainting, editing, or video generation, which would test its generalizability.
3. **Precise reporting of added computational cost.** While claiming "practically no additional computation," operations like DCT/iDCT and projection have non-zero cost. A clear breakdown of the added latency (ms per step) and memory overhead on different hardware should be provided.

# Final Consolidated Review
## Summary
This paper introduces History-Guided Sampling (HiGS), a training-free, momentum-based modification to diffusion model sampling that leverages a weighted average of past predictions to steer the sampling trajectory. HiGS aims to improve image quality, especially when using fewer sampling steps or lower classifier-free guidance scales, without requiring extra neural function evaluations. The method is shown to boost performance across diverse models and settings, including achieving a state-of-the-art FID of 1.61 for unguided ImageNet 256×256 generation with only 30 steps.

## Strengths
- **Strong and extensive empirical validation:** HiGS consistently improves image quality across multiple model families (Stable Diffusion variants, DiT, SiT), sampling budgets, and guidance scales, as measured by standard metrics (FID, HPSv2, ImageReward). The reported SOTA FID on ImageNet with very few steps is a notable result.
- **Practical and efficient design:** The method requires no extra training or forward passes, adds negligible runtime and memory overhead, and integrates seamlessly into existing samplers and architectures, including distilled models. Its plug‑and‑play nature is well‑demonstrated.
- **Theoretical motivation and analysis:** The paper provides a novel perspective by linking the Euler step in diffusion sampling to gradient descent on a time‑varying energy function and draws inspiration from momentum‑based variance reduction. An error analysis in Appendix B shows that a history‑based update can reduce the local truncation error of the Euler solver.

## Weaknesses
- **Lack of comparison to advanced, training‑free samplers:** The paper does not compare HiGS to other state‑of‑the‑art, training‑free sampling methods (e.g., higher‑order ODE solvers like DPM‑Solver++). Without such a comparison, it is unclear whether the observed gains are due to the novel history mechanism or simply from using a more accurate integrator, limiting the claim that HiGS is a “universal enhancement.”
- **Statistical significance of the SOTA claim:** The key SOTA result (FID 1.61 on ImageNet with 30 steps) is reported without any measure of variance (e.g., standard error over multiple runs). For a strong claim that may influence future work, demonstrating statistical reliability is important.
- **Disconnect between theory and algorithm:** The theoretical error analysis (Appendix B) derives a specific weight schedule \(w_k = h_k/(2h_{k-1})\) to achieve an \(O(h^3)\) local truncation error, but the implemented algorithm uses a different, fixed or scheduled \(w_{\text{HiGS}}\). This gap weakens the direct theoretical support for the final design.

## Nice-to-Haves
- A more detailed discussion of failure modes or limitations (e.g., performance with extremely low step counts or on certain image categories) would provide a more complete understanding of the method’s boundaries.
- Human preference studies could complement the automated metrics (HPSv2, ImageReward) to further validate the perceptual improvements claimed.

## Novel Insights
The paper offers a novel perspective on improving diffusion sampling by treating the Euler step as gradient descent on a time‑varying energy and incorporating a momentum‑like term from past predictions. This approach is distinct from prior work on better ODE solvers or distillation. The connection to variance reduction in stochastic optimization and the accompanying error analysis provide a fresh theoretical lens for understanding sampling efficiency.

## Suggestions
- Conduct a comparison between HiGS and at least one state‑of‑the‑art, training‑free sampler (e.g., DPM‑Solver++) to better contextualize the improvement offered by HiGS.
- Report the standard error or multiple runs for the key SOTA FID result to establish its statistical significance.
- Either adjust the theoretical analysis to more closely align with the implemented algorithm or clearly discuss the discrepancy and its implications.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
