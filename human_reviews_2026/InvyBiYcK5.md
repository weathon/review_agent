# ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Diffusion models suffer from substantial computational overhead due to their inherently iterative inference process. While feature caching offers a promising acceleration strategy by reusing intermediate outputs across timesteps, naive reuse often incurs noticeable quality degradation. 
In this work, we formally analyze the cumulative error introduced by caching and decompose it into two principal components: feature shift error, caused by inaccuracies in cached outputs, and step amplification error, which arises from error propagation under fixed timestep schedules.
To address these issues, we propose ERTACache a principled caching framework that jointly rectifies both error types. Our method employs an offline residual profiling stage to identify reusable steps, dynamically adjusts integration intervals via a trajectory-aware correction coefficient, and analytically approximates cache-induced errors through a closed-form residual linearization model. Together, these components enable accurate and efficient sampling under aggressive cache reuse.
Extensive experiments across standard image and video generation benchmarks show that ERTACache achieves up to 2x inference speedup while consistently preserving or even improving visual quality. Notably, on the state-of-the-art Wan 2.1 video diffusion model, ERTACache delivers 2x acceleration with minimal VBench degradation, effectively maintaining baseline fidelity while significantly improving efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors theoretically evaluate the error accumulation from cache-based acceleration for diffusion model in video generation, and summarize the error as a summed product of step amplification error and feature shift error. Targeting the two error components, the authors propose ERTACache that performs (1) offline policy calibration that runs full inference on a calibration set, finding optimal steps to cache based on threshold; (2) adaptive timestep adjustment to dynamically control the diffusion trajectory, and (3) explicit error rectification that uses a linear model to approximate the error for a small extracted dataset, then use the same model to correct in inference.

### Strengths
1. The theoretical analysis clearly pinpoints and characterizes the error sources in cache-based acceleration. 
2. The methods proposed are general.

### Weaknesses
1. After pointing out the feature shift and step amplification errors, they are no longer mentioned in the methods. 
2. The methods are relatively simple with limited effectiveness. 
3. Minor writing clarity issues. e.g., "Figure 2" is missing in the figure caption, and TACache in the figure is not defined. There are other typos, too.

### Questions
1. How specifically are the three methods (offline policy calibration, trajectory-aware timestep adjustment, and explicit error rectification) related to the two error sources (feature shift and step amplification)? I understand that the methods are to reduce error, but how are the two error sources specifically targeted? 
2. In Table 3, is Offline Policy built above Uniform Cache? Looks like the Offline Policy contributes the most to generation quality. 
3. Probing using a small dataset looks important in this work. Does it potentially lead to overfitting? 
4. Could you also provide speedup evaluations in ablation study?

### Soundness
3

### Presentation
2

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
This paper addresses the computational inefficiency of diffusion models and the quality degradation issue of naive feature caching, proposing ERTACache, a caching framework that decomposes cache-induced errors into feature shift and step amplification errors, and mitigates them via offline residual profiling, trajectory-aware timestep adjustment, and closed-form residual linearization.
Extensive experiments on Flux-dev 1.0 and Open-Sora 1.2, CogVideoX, Wan2.1 diffusion models show ERTACache achieves up to 2.47× inference speedup while preserving or improving visual quality.

### Strengths
1. Formalizes and addresses two core cache-induced errors in diffusion models (feature shift error and step amplification error) via a targeted dual-correction strategy, which differentiates from prior caching methods that lack explicit error decomposition and mitigation.
2. Integrates three training-free components (offline residual profiling for reusable steps, trajectory-aware timestep adjustment, closed-form residual linearization) to enable both high efficiency and fidelity, avoiding the trade-off of either high memory cost (e.g., AdaCache) or quality degradation in existing works.
3. Achieves superior performance balance on video diffusion models (e.g., 2.17× speedup with 80.73% VBench on Wan2.1 vs. TeaCache’s 2× speedup and 76.04% VBench) while maintaining comparable memory to baselines.

### Weaknesses
1. Insufficient details on key implementation aspects: e.g., no clarity on the calibration dataset’s specifics (size, domain) for offline policy tuning, or the exact logic for selecting the optimal threshold λ across models.

2. The paper did not verify performance under extreme conditions: no tests on long-duration videos (e.g., >100 frames), ultra-high resolutions (e.g., 1080P/4K).

3. Lacks human subjective evaluation to complement objective metrics (e.g., VBench, PSNR), which may lead to inconsistencies between reported numerical results and actual perceptual quality.

### Questions
1. Regarding novelty: Since the core components of ERTACache (offline calibration, dynamic timestep adjustment, closed-form error correction) are derived from adapting existing techniques (e.g., precomputed steps in DPM-Solver, error compensation in numerical methods), what unique mathematical properties of diffusion model cache errors does this framework reveal to distinguish it from "technical migration" rather than "fundamental innovation"?

2. On method implementation: In the trajectory-aware timestep adjustment, the correction coefficient uses the denominator ||v_i - v_{i+1}||₁. How is numerical instability avoided when this denominator approaches zero (e.g., to prevent division by a near-zero value)?

3. For method robustness: The offline policy calibration relies on a fixed threshold λ (e.g., 0.08 for Wan2.1, 0.6 for Flux-dev). Is there an adaptive mechanism to determine λ automatically across models, and how stable is ERTACache’s performance when λ fluctuates by 20%?

### Soundness
3

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
4

### Summary
This paper proposes ERTACache, a training-free caching framework for diffusion models. It decomposes cache-induced degradation into feature shift and step amplification errors, and introduces three modules to mitigate them: (1) Offline residual profiling for selecting reusable steps (λ-based policy), (2) Trajectory-aware timestep adjustment using a scaling factor ϕt, and (3) Linearized residual correction with closed-form parameters Ki and Bi. Experiments on Open-Sora, CogVideoX, Wan2.1, and Flux-dev show about 2× speedup while maintaining comparable visual quality.

### Strengths
1.The paper provides a clear and interpretable error decomposition, distinguishing the two main error sources of caching in diffusion models. This theoretical framing is well-motivated and guides the method design.

2.Comprehensive experiments across multiple video and image backbones demonstrate the generality of the approach. The comparisons with strong baselines (TeaCache, FasterCache, PAB, etc.) show consistent efficiency–quality improvements.

3.The ablation study clearly isolates contributions from each component, verifying that the offline policy, timestep adjustment, and error rectification complement one another.

### Weaknesses
The method relies on an offline calibration set and a threshold λ to select reusable steps, but the paper does not analyze how the calibration set size, diversity, or λ values affect the performance across different domains. A sensitivity study on λ and calibration configuration would strengthen the empirical validation.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents ERTACache, an error-rectified caching framework designed to improve the efficiency of diffusion-based inference. The method introduces a modular strategy that combines: (1) offline cache calibration to identify redundant denoising steps; (2) trajectory-aware time-step adjustment using a residual-driven coefficient $\phi_i$ to suppress drift accumulation; and (3) explicit error rectification that analytically re-injects residuals to maintain stability. Across several diffusion benchmarks, ERTACache achieves roughly $2×$ acceleration with negligible loss in visual quality, requiring no retraining or architectural modification. The approach provides a practical trade-off between efficiency and fidelity, complementing prior acceleration techniques such as dynamic skipping, pruning, and distillation.

### Strengths
ERTACache demonstrates strong advantages in **error modeling depth**, **cross-task adaptability**, and **quality–efficiency balance**, providing a theoretically grounded solution to the long-standing problem of *error accumulation* overlooked by prior caching methods.

1. Existing caching methods (e.g., TeaCache, PAB) rely on *heuristic thresholds* for redundancy detection and lack theoretical understanding of how caching degrades output quality, often resulting in the phenomenon “higher speed → worse quality.”  
   **ERTACache’s core innovation** lies in its formal *dual-source decomposition* of caching error (Eq. 11):  
   - **Feature Shift Error (εᵢ = ṽᵢ − vᵢ):** deviation between cached and true features, originating from inaccurate reuse.  
   - **Step Amplification Error (δᵢ₋ₘ = Σₖ₌₀^{m−1} Δtᵢ₋ₖ ⋅ εᵢ₋ₖ):** cumulative error propagation under fixed step size via ODE integration, amplifying earlier deviations.  
   This theoretical decomposition reveals, for the first time, the *root causes of quality loss in caching*, paving the way for principled correction strategies.  
   **Closed-loop error correction mechanism:** ERTACache jointly addresses both error types through (1) offline step selection to minimize low-error reuse, (2) dynamic timestep adjustment to reduce amplification, and (3) residual linearization to directly correct feature shifts — forming a full *“error identification → prevention → correction”* loop rather than a one-shot caching heuristic.

2. ERTACache achieves *high acceleration with near-lossless quality* in both video and image generation without retraining:  
   - **Video Generation:**  
     - *Open-Sora 1.2 (51f, 480P):* ERTACache-fast attains 2.47× speed-up (18.04 s vs 44.56 s), VBench 78.64% (−0.58%), LPIPS 0.1659 (vs TeaCache-fast 0.2511), PSNR 22.34.  
     - *CogVideoX (48f, 480P):* 2.93× speed-up with LPIPS 0.1012, SSIM 0.8702, PSNR 26.44 (TeaCache 20.97).  
     - *Wan2.1-1.3B (81f, 480P):* 2.17× speed-up, VBench 80.73% (vs 81.30%), LPIPS 0.1095, SSIM 0.8200 — far better than TeaCache (LPIPS 0.2913, SSIM 0.5685).  
   - **Image Generation:**  
     On *Flux-dev 1.0*, 1.86× speed-up yields CLIP 0.9534 (vs TeaCache 0.9065), PSNR 20.51 (+4.34), preserving finer visual details (Table 2, Fig. 3).

3. ERTACache balances theoretical rigor with practical deployment:  
   - **Offline Policy Calibration:** A small calibration set (no extra data) precomputes *true residuals* using relative $L_1$ error $\ell_{1rel}$, forming a reusable step set $S$ for inference without runtime computation. Even with only 20 prompts, performance matches a 1000-sample calibration (Appendix A.2).  
   - **Trajectory-Aware Timestep Adjustment:** Defines correction factor  
     $\phi_i = \text{clip}(1 - \frac{\|v_i - v_{i+1}\|_1}{\|ṽ_i - v_i\|_1}, 0, 1)$  
     to dynamically adapt step size $Δt_i = Δt_c ⋅ φ_i$, aligning sampling trajectories with true ODE paths. On Flux-dev 1.0, this boosts PSNR by +0.95 (Table 3).  
   - **Low sensitivity to hyperparameters:** With $\lambda$ ranging 0.08–0.3, VBench varies only 0.3–1.2% (Table 1, 3), requiring no per-model retuning.

4. **Strong Theoretical Foundation — Precise Error Correction via Residual Linearization**  
   Instead of heuristic compensation, ERTACache employs a *closed-form linearized residual model* for direct error correction:  
   - **Lightweight model:** Approximates feature error as $ε_i ≈ σ(K_i ṽ_i + B_i)$ , deriving closed-form solutions for $K_i, B_i$ via Taylor expansion (Appendix A.3), adding <0.5% latency.  
   - **Effective correction:** On Wan2.1-1.3B, correction reduces LPIPS from 0.1267 → 0.1095 and increases PSNR 22.94 → 23.77 (Table 3), eliminating frame jitter and detail blur common in TeaCache (Fig. 3–4).

### Weaknesses
1. **Reduced Correction Accuracy in Highly Dynamic Scenes**  
   The “universal error model” assumption holds for static scenes (e.g., still objects) but struggles under dynamic motion:  
   - The closed-form coefficients $(K_i, B_i)$are calibrated offline and assume stationary error distribution, which breaks under rapidly changing dynamics. Fig. 6 shows distinctly different error-channel distributions between dynamic and static prompts, yet no adaptive adjustment is made.  
   - Experiments use mostly *low- to medium-motion* prompts (“washing clothes,” “airplane flight”) without validating performance on highly dynamic scenes.

2. **Offline Calibration Generalization Depends on Dataset Quality**  
   Though small calibration sets suffice, the method lacks analysis of *domain shift* effects:  
   - If calibration data are “natural landscapes” but deployment involves “abstract art,” the precomputed reusable-step set S may mispredict high-error steps.  
   - No guidance on calibration-set composition or prompt diversity is provided, increasing deployment trial cost.

### Questions
1. How well does the linear residual approximation hold under highly non-linear diffusion dynamics or alternative noise schedules?

2. Could ERTACache be combined with adaptive ODE solvers, and how might local error control interact with the correction term?

### Soundness
3

### Presentation
3

### Contribution
3
