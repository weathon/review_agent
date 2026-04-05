=== CALIBRATION EXAMPLE 64 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately captures the core contribution: a history-guided sampling mechanism designed as a training-free enhancement for diffusion models.
- The abstract clearly states the problem (quality degradation at low NFEs/guidance scales), the proposed method (momentum-style integration of past predictions), and key results (consistent quality gains, SOTA FID of 1.61 at 30 steps).
- The claim "practically no additional computation" is slightly imprecise. While NFEs remain identical, the DCT/IDCT filtering and orthogonal projections introduce non-negligible tensor operations, particularly at higher spatial resolutions. This does not invalidate the claim but warrants a minor qualification. The abstract's claims are otherwise well-supported by the experiments.

### Introduction & Motivation
- The problem is well-motivated: reducing inference cost (NFEs) and mitigating the diversity loss/oversaturation associated with high CFG scales are practically important challenges. The gap is clearly identified as a lack of effective *training-free* methods that maintain quality under aggressive step/guidance reductions.
- Contributions are explicitly listed and accurately reflect the paper's content. The introduction avoids significant overclaiming.
- One minor concern: The introduction positions advanced ODE solvers (e.g., DPM variants) primarily as high-step or training-dependent approaches, which slightly undersells the existing state-of-the-art in low-step, training-free sampling. This should be refined to better contextualize HiGS against modern multistep ODE solvers that already excel at ~20 steps.

### Method / Approach
- The method is clearly described, and pseudocode (Algorithms 1–3) ensures strong reproducibility. Key hyperparameters (EMA decay, weight schedule, DCT threshold) are explicitly defined.
- **Logical/Theoretical Gap:** Section 4.1 and Appendix B derive a theoretical justification by connecting the Euler update to STORM-style momentum and proving an improved local truncation error of $O(h_k^3)$ (Theorem B.1). However, this proof relies on two key assumptions that diverge from the implemented method:
  1. The optimal weight $w_k$ is derived as step-size dependent: $w_k = 2h_k / h_{k-1}$. In practice, $w_{\text{HiGS}}(t)$ uses a fixed square-root time schedule (Eq. 6) independent of actual solver step sizes.
  2. The proof only analyzes a 1-step difference ($u(z_{tk}) - u(z_{tk-1})$). The actual implementation uses an exponential moving average over multiple past predictions ($g(H_k)$ in Eq. 5) within a window $W$.
  The theoretical analysis does not actually bound or justify the EMA-based heuristic that is deployed. The authors should clearly separate the theoretical motivation (error reduction insight) from the empirical engineering choices (EMA, DCT, projection) and either extend the theory to cover the EMA case or reframe the derivation as an intuitive starting point rather than a rigorous guarantee.
- The DCT high-pass filtering (Section 4.2) and orthogonal projection (Eq. 7) are introduced as empirical fixes for color artifacts and oversaturation. This is acceptable for a methods paper, but their presentation should explicitly acknowledge their heuristic nature rather than implying they flow directly from the momentum derivation.
- Edge cases (e.g., early-stage instability) are reasonably handled via the $t_{\min}$ schedule cutoff.

### Experiments & Results
- The experiments systematically test the paper's claims across step budgets (Figs. 3, 5b), guidance scales (Figs. 4, 5a), and architectures/models (Tables 1–4). The qualitative results strongly support the claim of improved sharpness and structure.
- **Baselines & Positioning:** The primary baseline is standard sampling (usually Euler/DDIM) with CFG. For a paper claiming to solve the "fewer NFEs" challenge, head-to-head comparisons against modern low-step ODE solvers (e.g., DPM-Solver++, UniPC, DEIS) at matched NFEs are missing. Table 6 shows HiGS *improves* various samplers, but does not demonstrate how `HiGS + Euler(20 steps)` compares to `UniPC(20 steps)` or `DPM-Solver++(20 steps)`. Without this, it is unclear whether HiGS provides additive value over the current SOTA in few-step sampling, or merely brings a basic Euler solver up to par.
- **Statistical Significance & Variance:** Generative metrics (FID, HPSv2, ImageReward) are reported without variance estimates, confidence intervals, or multiple random seed evaluations. Given the substantial FID reductions reported (e.g., SiT-XL FID from 12.08 to 4.86) and near-deterministic win rates (90%+), reporting standard deviation across at least 3 seeds is necessary to satisfy ICLR's rigor standards and rule out evaluation noise or seed dependency.
- **Ablations:** Appendix E provides thorough ablations on EMA $\alpha$, DCT thresholds, schedules, and projection. This strongly supports the design choices.
- Datasets (ImageNet, COCO prompts) and metrics (FID, HPSv2, IS, Precision/Recall) are appropriate and standard for the field.

### Writing & Clarity
- The paper is generally well-structured and accessible. The progression from motivation -> formulation -> empirical refinements is logical.
- A clarity improvement would be to explicitly decouple the theoretical derivation (Section 4.1, Appendix B) from the practical implementation pipeline. As currently written, the transition to DCT filtering and orthogonal projection feels abrupt and theoretically unmotivated. A brief paragraph clarifying that these are empirical corrections for known failure modes of momentum-style residuals in latent space would improve scientific transparency.
- Figures and tables are highly informative. Visual comparisons (Figs 2, 3, 4, 7, 8) effectively demonstrate the qualitative improvements. Table layouts are clean and metrics are consistent.

### Limitations & Broader Impact
- The stated limitations in the conclusion ("inherits biases... of underlying models") are superficial and miss several practical constraints inherent to the method:
  1. **Hyperparameter Sensitivity:** Tables 10–12 show that optimal $w_{\text{HiGS}}$, $\alpha$, and $\eta$ vary across architectures (e.g., SiT vs. DiT vs. SDXL). While ranges are suggested, the "plug-and-play" claim is slightly weakened by the need for model-specific tuning to achieve peak performance.
  2. **High-Resolution Compute/Memory:** The DCT/IDCT filtering operates on full spatial dimensions. While negligible for 256×256 or 1024×1024, it may become a bottleneck for 4K+ generation or video diffusion where tensor sizes are large. This should be acknowledged.
  3. **Theoretical-Empirical Disconnect:** As detailed above, the gap between the $O(h^3)$ proof and the actual EMA+schedule implementation is a technical limitation of the current contribution.
- Broader impact follows standard conventions and adequately addresses synthetic media concerns.

### Overall Assessment
This paper presents a simple, effective, and practically valuable training-free enhancement for diffusion sampling. The core idea of leveraging prediction residuals as a corrective signal is intuitive, and the empirical results are robust across multiple modern architectures, demonstrating clear improvements in both quantitative metrics and visual quality, particularly in low-NFE and low-CFG regimes. The contribution is strong and highly relevant to ICLR's focus on efficient and improved generative modeling. However, the paper has two notable shortcomings that must be addressed for full alignment with ICLR's standards: (1) a disconnect between the theoretical error-reduction derivation in Appendix B and the actual EMA-based implementation, which needs clearer framing or theoretical extension; and (2) missing head-to-head baselines against state-of-the-art low-step ODE solvers (e.g., DPM-Solver++, UniPC) to conclusively position HiGS's value relative to existing few-step methods. Additionally, reporting variance across seeds for generative metrics would strengthen the empirical claims. If the authors clarify the theory-method relationship, add the requested baseline comparisons, and report statistical variance, this work stands as a solid and impactful contribution to diffusion sampling methodology.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes HiGS (History-Guided Sampling), a training-free, momentum-based modification to diffusion model inference that integrates an EMA-weighted history of past predictions to guide each sampling step. Coupled with a time-dependent weight schedule, optional orthogonal projection, and DCT-based high-pass filtering, HiGS aims to improve sample sharpness and structural coherence without requiring additional neural forward passes. Extensive experiments demonstrate consistent gains across multiple architectures (SDXL, SD3, SiT), samplers, distilled models, and low-CFG/low-NFE regimes, achieving a state-of-the-art FID of 1.61 on unguided ImageNet-256 with only 30 steps.

### Strengths
1. **Practical Zero-Cost Design:** HiGS explicitly reuses cached model outputs rather than introducing additional forward passes, aligning with ICLR's emphasis on computational efficiency. Section D.1 empirically confirms identical iteration throughput and memory footprint compared to standard CFG.
2. **Comprehensive Empirical Validation:** The method is rigorously tested across a diverse set of models (text-to-image, class-conditional, distilled) and samplers (DDIM, DPM++, PLMS, UniPC), with results reported in Tables 1–4 and 6. The consistent improvement across HPSv2, ImageReward, and FID highlights robust cross-domain applicability.
3. **Strong Theoretical & Analytical Motivation:** Section 4.1 and Appendix B establish a clear link between Euler-based diffusion sampling and gradient descent, drawing inspiration from STORM's momentum-based variance reduction. The truncation error analysis (Theorem B.1) formally justifies why history terms can accelerate convergence.
4. **Thorough Ablation & Reproducibility:** Appendix E systematically ablates key components (buffer input choice, scheduling, projection, DCT filtering, EMA $\alpha$, history function variants). Full hyperparameters per experiment are provided in Tables 10–12, alongside clear pseudocode (Algorithms 1–3) and PyTorch snippets, ensuring high reproducibility.

### Weaknesses
1. **Theory-Practice Misalignment:** The theoretical error reduction in Appendix B assumes a specific weight $w_k = 2h_k/h_{k-1}$ to cancel the $O(h_k^2)$ term. The practical implementation instead uses an EMA history buffer and a square-root time schedule with a tunable $w_{\text{HiGS}}$. The paper does not mathematically reconcile how the practical scheduling preserves the $O(h^3)$ local error bound, creating a gap between the theoretical claims and the deployed heuristic.
2. **Hyperparameter Tuning Burden Undermines "Plug-and-Play" Claim:** While described as universal, optimal performance requires careful selection of 6+ new hyperparameters ($w_{\text{HiGS}}$, $t_{\min}$, $t_{\max}$, $\alpha$, $\eta$, $R_c$). Tables 10–12 reveal that values like $\eta$ (0 vs. 1) and $w_{\text{HiGS}}$ (0.75 vs. 2.5) vary significantly across models, contradicting the claim of seamless, zero-tuning integration.
3. **Incomplete Computational Overhead Profiling:** The paper states "practically no additional computation," but the DCT/iDCT filtering on high-dimensional latent tensors adds non-trivial CPU/GPU cycles. Section D.1 reports aggregate sampling speed but does not isolate or profile the latency of the DCT masking operation, which could become a bottleneck in high-throughput or video generation pipelines.
4. **Limited Comparison to Advanced Multistep/Correction Samplers:** Beyond APG, the paper lacks direct comparison to recent ODE solver corrections, predictor-corrector schemes, or adaptive guidance schedulers that also target low-NFE regimes. This makes it difficult to position HiGS relative to the state-of-the-art in sampling algorithm design.

### Novelty & Significance
**Novelty:** Moderate to High. The concept of leveraging past predictions in ODE solvers and momentum optimization is well-established, but adapting it to diffusion sampling with the specific triad of EMA history, orthogonal projection, and frequency-domain filtering is a novel engineering synthesis. The primary novelty lies in the practical pipeline rather than a fundamentally new sampling paradigm.
**Clarity:** High. The paper is well-structured, motivationally clear, and supported by high-quality qualitative figures and comprehensive appendices.
**Reproducibility:** Excellent. Detailed hyperparameter tables, evaluation protocols (ADM framework, fixed sample counts), and open algorithmic pseudocode make it straightforward to replicate.
**Significance:** High. Addressing the trade-off between NFE, CFG scale, and output fidelity is highly relevant for both research and production deployments. A training-free, forward-pass-free quality boost that delivers SOTA ImageNet metrics is practically impactful and likely to see rapid adoption.

### Suggestions for Improvement
1. **Bridge the Theory-Practice Gap:** Extend Appendix B to analyze the truncation error under the actual EMA + square-root scheduling formulation, or explicitly acknowledge the discrepancy and provide empirical justification for why the heuristic still yields stability and convergence improvements.
2. **Streamline Hyperparameter Recommendations:** Propose a single "default" hyperparameter configuration that works robustly across all tested models without per-model tuning. If this isn't possible, introduce a lightweight, compute-free heuristic to auto-scale $w_{\text{HiGS}}$ and $t_{\min}/t_{\max}$ based on the noise schedule or model architecture.
3. **Profile DCT Overhead Explicitly:** Add a latency breakdown table or figure showing the milliseconds consumed by DCT/iDCT filtering across different resolutions/latent dimensions. This will validate the "zero overhead" claim with transparent, hardware-agnostic metrics.
4. **Strengthen Baselines & Statistical Rigor:** Include quantitative comparisons against recent predictor-corrector and multistep ODE solvers that explicitly correct sampling trajectories at low NFEs. Additionally, report confidence intervals or statistical significance tests (e.g., p-values for win rates) over multiple random seeds to strengthen the empirical claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Benchmark against modern multi-step ODE solvers (e.g., DPM-Solver++2M/3M, Heun) at 10-30 steps using identical models. Beating a vanilla Euler sampler is insufficient for ICLR; you must demonstrate HiGS outperforms established low-NFE baselines under the exact same computational budgets to justify efficiency claims.
2. Conduct a strict compute-matched ablation comparing HiGS to advanced guidance scheduling techniques (e.g., guidance interval, dynamic CFG scaling). This is required to isolate whether performance gains stem from the history mechanism itself or merely coincidentally mimic the effective annealing behavior of standard schedule tweaks.
3. Evaluate HiGS on high-resolution synthesis ($\ge 512 \times 512$) and early video generation setups. Claims of "universal plug-and-play enhancement" are not convincing without evidence that the history and DCT operations scale without introducing spatial incoherence or temporal flickering across larger or higher-dimensional latent spaces.

### Deeper Analysis Needed (top 3-5 only)
1. Rigorously validate the theoretical link between STORM momentum and diffusion ODEs. STORM is designed for optimizing a fixed non-convex objective, whereas diffusion involves a strictly time-varying energy field $E_t(z)$; you must mathematically justify why a history term from $t_{k-1}$ remains a valid descent direction at $t_k$ or explicitly reframe the method as an empirical heuristic.
2. Analyze the theoretical and empirical bias introduced by the DCT high-pass filter on global scene coherence. Since your method actively suppresses low-frequency update signals to fix color shifts, you must quantify why this suppression does not actively degrade prompts requiring specific ambient lighting, shadows, or large uniform gradients.
3. Quantify error propagation dynamics in high-noise regimes. Investigate whether the history term amplifies early-stage denoising hallucinations when initialized from pure Gaussian noise, and explain how your weight schedule $w_{\text{HiGS}}(t)$ theoretically prevents trajectory divergence before the signal-to-noise ratio stabilizes.

### Visualizations & Case Studies
1. Plot sampling trajectories with and without HiGS in a reduced latent space or via t-SNE/PCA. This visualization is necessary to verify the core claim that history guidance actually steers the reverse process toward higher-probability data regions, rather than just arbitrarily perturbing the baseline path.
2. Visualize the raw history update term $\Delta D_{t_k}$ across different timesteps with and without the DCT filter applied. This exposes whether HiGS is generating meaningful semantic correction vectors or injecting high-frequency noise that your post-hoc filters are aggressively forced to strip out.
3. Showcase explicit, unfiltered failure cases where the DCT filtering or orthogonal projection causes visible degradation. Demonstrating scenarios where low-frequency suppression breaks large coherent structures or smooth gradients is critical to trusting the method's robustness in real-world deployment.

### Obvious Next Steps
1. Provide a precise wall-clock latency benchmark that isolates the DCT/IDCT, projection, and EMA overheads from the neural forward pass. The claim of "practically no additional computation" is scientifically inaccurate without quantifying the exact millisecond and memory costs, especially at high resolutions.
2. Compare HiGS directly against other inference-time compute methods like denoiser ensembling, test-time scaling, or self-consistency sampling. This comparison is essential to clarify whether HiGS offers a uniquely efficient scaling law for test-time compute or if you are simply shifting the compute bottleneck to less standard operations.
3. Formulate a principled or adaptive strategy for the DCT threshold $R_c$ and projection weight $\eta$. Relying on manual, dataset-specific tuning to prevent severe color artifacts undermines the "plug-and-play" claim; proposing an auto-tuning mechanism based on intermediate image statistics is a required step for practical adoption.

# Final Consolidated Review
## Summary
The paper introduces HiGS, a training-free enhancement for diffusion model sampling that leverages an EMA-weighted history of past predictions to guide each denoising step without additional network evaluations. By combining a time-dependent weight schedule with orthogonal projection and DCT-based high-pass filtering, the method consistently sharpens details and improves structural coherence, especially under low NFE and low CFG regimes. Extensive experiments across text-to-image and class-conditional architectures demonstrate robust quality gains, culminating in a state-of-the-art unguided FID of 1.61 on ImageNet-256 using only 30 steps.

## Strengths
- **Zero-Forward-Pass Efficiency:** HiGS intelligently reuses cached model outputs rather than introducing extra NFEs, maintaining identical memory footprint and iteration throughput compared to standard CFG (verified in Appendix D). This aligns well with practical deployment constraints.
- **Comprehensive Empirical Validation:** The method is rigorously tested across a diverse matrix of settings: varying step budgets, guidance scales, distilled models, and multiple ODE/EDM samplers (Tables 1–6). The reported SOTA FID on unguided ImageNet-256 and consistent high win rates (>80%) across human preference metrics provide strong quantitative and qualitative evidence of efficacy.
- **Systematic Engineering & Ablation:** Appendix E thoroughly dissects the design pipeline, validating each heuristic component (EMA $\alpha$, square-root scheduling, DCT filtering, projection) and demonstrating robustness to parameter variations. Full hyperparameter tables and pseudocode ensure high reproducibility.

## Weaknesses
- **Theory-Practice Decoupling:** The theoretical motivation (Section 4.1, Appendix B) derives an $O(h^3)$ local truncation error bound for a modified Euler step with a *step-size-dependent* weight $w_k = 2h_k/h_{k-1}$ and a *1-step* momentum difference. However, the deployed algorithm replaces this with a multi-step EMA buffer and a fixed, time-based square-root schedule independent of $h_k$. The paper fails to mathematically justify why this practical heuristic preserves the theoretical stability properties, leaving a significant gap between the stated guarantees and the actual implementation.
- **Hyperparameter Sensitivity Undermines "Plug-and-Play" Claim:** While touted as a universal enhancement, optimal performance requires model-specific tuning of at least six hyperparameters ($w_{\text{HiGS}}, t_{\min}, t_{\max}, \alpha, \eta, R_c$). Tables 10–12 reveal substantial variation in optimal values (e.g., $\eta$ toggles between 0 and 1, $w_{\text{HiGS}}$ ranges from 0.75 to 2.5 across SiT vs. SDXL). Without a principled, architecture-agnostic heuristic to auto-scale these parameters, the method demands non-trivial search effort, contradicting the zero-tuning narrative.
- **Insufficient Head-to-Head Few-Step Baselines:** Table 6 demonstrates that HiGS *improves* existing samplers, but the core results primarily benchmark `Default+HiGS` against `Default+CFG`. To conclusively prove HiGS's value in the few-step regime, direct comparisons against modern low-NFE solvers (e.g., DPM-Solver++2M/3M, DEIS) *at strictly matched step counts and without HiGS* are missing. It remains unclear whether HiGS provides additive algorithmic value or merely elevates first-order integrators to the performance floor of established multistep methods.
- **Lack of Statistical Rigor in Reporting:** Given the substantial metric improvements (e.g., SiT-XL FID dropping from 12.08 to 4.86) and high preference win rates, the absence of standard deviations, confidence intervals, or multi-seed evaluations across key tables weakens the empirical claims. Generative metrics are known to be sensitive to random seeds; reporting variance over at least 3 runs is expected for a contribution of this scope.

## Nice-to-Haves
- Provide a wall-clock latency breakdown isolating the DCT/IDCT and projection overheads relative to the forward pass across varying resolutions, which would validate the "zero overhead" claim for high-throughput or video pipelines.
- Visualize sampling trajectories (e.g., via PCA/t-SNE in latent space) and the raw unfiltered $\Delta D_{t_k}$ vectors to empirically demonstrate how history guidance alters the reverse process dynamics.
- Analyze whether aggressive high-pass DCT filtering inadvertently degrades prompts requiring large-scale coherent lighting, gradients, or atmospheric effects, as suppressing low-frequency update signals could theoretically harm global scene structure.

## Novel Insights
HiGS reframes diffusion sampling from a purely state-to-state integration problem into a *frequency-aware consistency correction* task. By treating the drift between current and historically smoothed predictions as a meaningful directional signal—rather than mere discretization error—and strategically isolating its high-frequency semantic components via spectral filtering, the method reveals that implicit momentum can recover structural fidelity lost in aggressive step reduction. This bridges optimization theory (variance reduction) with generative dynamics, suggesting that temporal prediction residuals encode a latent corrective gradient that operates independently of explicit guidance mechanisms.

## Potentially Missed Related Work
- **Advanced Multistep ODE Solvers** (e.g., DPM-Solver++ [1], DEIS [2], UniPC [3]): These represent the current standard for training-free, low-NFE sampling and should be used as baseline comparators to position HiGS's absolute efficiency gains.
- **Predictor-Corrector & Test-Time Refinement Methods:** Works leveraging iterative self-correction or Langevin dynamics at inference time are relevant for contextualizing HiGS as a history-driven refinement signal.

## Suggestions
1. **Reframe Theoretical Claims:** Explicitly state in Section 4.1 that Theorem B.1 provides intuitive motivation for incorporating history terms into the ODE discretization, but does not strictly bound the implemented EMA + fixed-schedule pipeline. Add a brief discussion or empirical study showing why the practical heuristic maintains numerical stability despite deviating from the theoretical $w_k$.
2. **Introduce Auto-Scaling Heuristic:** To substantiate the "plug-and-play" claim, propose a simple, compute-free rule to auto-calibrate $w_{\text{HiGS}}$ and $t_{\min}/t_{\max}$ based on intrinsic noise schedule properties (e.g., $\nabla_t \sigma(t)$) or model architecture size, reducing the need for per-model grid searches.
3. **Expand Comparative Baselines & Statistics:** Conduct direct head-to-head evaluations against SOTA few-step samplers at matched NFEs (10, 20, 30 steps) using identical backbones. Crucially, report $\pm$ standard deviation over at least 3 random seeds for all primary FID/IS/HPSv2 results to establish statistical reliability and rule out seed dependency.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
