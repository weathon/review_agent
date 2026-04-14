=== CALIBRATION EXAMPLE 26 ===

# Final Consolidated Review
## Summary
This paper introduces Multidimensional Trajectory Optimization (MTO), which generalizes scalar time-coefficients in flow and diffusion models to per-dimension vectors (γ ∈ ℝ^d), allowing each spatial dimension to follow its own interpolation schedule. The core argument is that trajectory optimality should be determined by end-to-end sample quality under fixed solver configurations, rather than pre-defined geometric properties like straightness. The authors propose a two-stage training scheme: (1) pre-training the diffusion model H_θ with randomly sampled multidimensional coefficients, and (2) jointly optimizing H_θ and a parameterized coefficient network γ_φ via adversarial training. On CIFAR-10 conditional generation, the method achieves FID 1.37 at 5 NFE, which is state-of-the-art among entries in their comparison table, including 1-NFE distillation methods.

---

## Strengths

- **Genuine conceptual contribution: straightness ≠ optimality.** The 2D experiments (Table 1, Figure 5) provide clean, quantitative evidence that OT-optimal (straight) trajectories are suboptimal for actual transportation quality as measured by W₂. This directly challenges the dominant rectified-flow/OT paradigm and is a substantive empirical insight, not merely a theoretical claim.

- **CIFAR-10 conditional SOTA is real.** At FID 1.37 (5 NFE), EDM-MTO outperforms GDD-1 (1.44, 1 NFE), CTM (1.73–2.04, 1–6 NFE), and StyleGAN-XL (1.85, 1 NFE) on conditional CIFAR-10. This holds even against 1-NFE methods, making the SOTA claim legitimate rather than an artifact of extra compute.

- **Training efficiency advantage.** Table 6 shows substantially lower kimg requirements versus GDD-I (e.g., 1382 vs. 5000 on CIFAR-10), and training was conducted on 48GB VRAM hardware vs. the 80GB setups common in the literature. This is a concrete practical advantage.

- **Interpretable coefficient analysis.** The t-SNE visualization of trained sinusoidal weights (Figure 9) reveals that the optimized γ_φ diverges from the pre-training distribution, clusters by class label in conditional generation, and produces sparser outputs in conditional vs. unconditional settings. This provides mechanistic insight into *why* the method outperforms more in conditional settings.

- **Confirmed non-linearity of optimized trajectories.** Figure 8 quantifies the deviation of the optimized trajectory from straight-line interpolation across all datasets, corroborating that MTO genuinely discovers non-trivial paths.

---

## Weaknesses

### Fatal
None.

### Major

- **The multidimensional coefficient explains only a fraction of the empirical gain, but the paper's narrative underemphasizes this.** Table 5 is the most important ablation in the paper and is not given adequate discussion:
  - EDM-γ + Adv.**θ** (adversarial fine-tuning only, no φ optimization): FID 2.28 uncond / 2.14 cond
  - EDM-γ + Adv.**θ,φ** (full MTO): FID 1.81 uncond / 1.42 cond

  The incremental gain from adding the multidimensional φ on top of adversarial θ fine-tuning is 0.47 FID (uncond) and 0.72 FID (cond). These are non-trivial but modest improvements over simply applying adversarial training to θ. Meanwhile, the conceptual framing of the paper positions the multidimensional coefficient as the primary innovation, and the substantial machinery around Γ_h, pre-training, and U_φ is presented as necessary and impactful. The disconnect between the narrative and the ablation numbers should be directly addressed: what specifically does per-dimension scheduling provide that a unidimensional adaptive schedule would not? (See also the Spark Finder's request for a learnable scalar baseline, which is absent and would sharpen this analysis considerably.)

- **Performance on FFHQ and AFHQv2 is a genuine scalability concern.** On FFHQ-64×64, EDM-MTO achieves FID 2.27 (5 NFE) versus GDD-1's 0.85 (1 NFE) — a factor of ~2.7× worse. On AFHQv2, 2.04 vs. 1.23. The paper's limitations section acknowledges this and attributes it to "the same model size and NFE configurations as CIFAR-10," but this is not fully satisfying: MTO is motivated precisely by achieving better quality at low NFE, and it clearly does not generalize to slightly larger datasets without additional tuning. The paper does not explain whether the hypothesis space Γ_h is misspecified for higher-dimensional data, whether the pre-training procedure needs scaling, or whether the adversarial objective becomes unstable. This leaves the scalability claim as an assertion rather than a demonstrated result.

### Minor

- **Missing learnable unidimensional baseline.** The ablation in Table 5 includes "EDM-γ + Adv.φ (no multi.)" but not a learnable scalar α_φ(t) trained with the same adversarial framework. Without this control, it is impossible to isolate whether the gains from φ optimization come from *multidimensionality* versus simply *adaptive scheduling*. This is a low-cost experiment that would substantially clarify the paper's core claim.

- **Inconsistent objective for θ (Eq. 14).** During adversarial training, θ is updated with a simulation-free loss (Eq. 14) using an independent noise sample z ~ ρ_T rather than the same x_T used in the simulation rollout. The paper justifies this on grounds of training cost, but this means θ is never trained end-to-end on the full inference chain — only φ is. The consequence is a potential systematic bias: θ is adapted to γ_φ computed with noise z, while at inference it sees γ_φ computed with x_T. No analysis is provided for whether this creates a mismatch or how severe it is in practice. At minimum, this design choice deserves a brief empirical sanity check or discussion.

- **No sensitivity analysis for Γ_h hyperparameters.** The hypothesis space design (sinusoidal basis size M, warping exponent q, LPF kernel width, and scale s) is entirely heuristic with no ablation. If performance is sensitive to these choices, the method's robustness and transferability to new settings are unclear.

- **Total compute accounting in Table 6 is incomplete.** The kimg comparison excludes the pre-training phase for H_θ, which is presumably substantial. The efficiency claim relative to GDD-I should account for total training cost, not just the adversarial fine-tuning stage.

### Tiny

- **Algorithm 1, line 4 labels the pre-training loss as L_θ^MTO** but this quantity was defined in Eq. 10 as L_θ^pre. Minor inconsistency.
- **"5 (+)" NFE notation** is used throughout without quantifying the "+". Since U_φ is described as smaller than H_θ, this is not a large effect, but the actual overhead should be stated (e.g., as a fraction of one H_θ evaluation) for transparent benchmarking.
- **Abstract contains "Stochastic Interpolat"** — a truncation of "Stochastic Interpolant."

---

## Nice-to-Haves

- **Diversity metrics (Precision/Recall).** Since the adversarial objective can improve FID via mode sharpening without preserving coverage, reporting Precision and Recall would verify that MTO maintains diversity, not just fidelity.

- **Geometric or theoretical intuition for per-dimension scheduling.** Even without a formal proof, a discussion of what data structure justifies breaking dimension symmetry — e.g., frequency spectra, semantic channels, or manifold curvature varying across spatial dimensions — would strengthen the conceptual grounding.

- **Coefficient heatmaps on images.** Visualizing γ_φ(t) spatially (e.g., as images showing which pixels receive larger vs. smaller noise schedules) would make the method's mechanism more interpretable and is low effort given U_φ already produces spatial outputs.

- **Inference wall-clock time.** A wall-clock comparison against CTM and GDD at inference time would complement the kimg analysis and clarify whether the U_φ forward pass introduces perceptible latency for practical use.

- **Stability analysis of Figure 7.** The FID curves for different axis combinations show noisy training dynamics. Running multiple seeds or averaging over short windows would make the multidimensionality benefit more statistically credible.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"SOTA claim is misleading because 5 NFE vs. 1 NFE"** (Harsh Critic): Factually incorrect as applied to CIFAR-10 conditional. EDM-MTO's FID of 1.37 beats GDD-1's 1.44 at 1 NFE outright. The SOTA claim is justified. Removed.

- **"Contribution 3 is incremental over CTM/GDD"** (Harsh Critic): While the adversarial training component shares high-level structure with prior work, the joint optimization of both model parameters and trajectory parameters (φ) within a simulation-based loop is distinguishable from fixed-teacher distillation. The degree of novelty is debatable but not clearly dismissible. Removed as stated.

- **Lack of statistical significance/variance estimates for image FID** (Harsh Critic): Single-run FID evaluation is standard practice for large-scale image generation benchmarks. Requiring multiple seeds for competitive GAN training is not standard in this field. Removed as a mandatory criterion; retained only as a nice-to-have.

- **"Coverage of adaptive/per-frequency noise schedules is inadequate"** (Harsh Critic, related works): Per instructions, missing related works are not flagged.

- **Requesting confidence intervals for 2D experiments**: Already present in Table 1 (standard deviations reported). Criticism does not apply.

---

## Novel Insights

The most genuinely novel observation synthesized across reviews is the **conditional generation efficiency effect revealed in Figure 9**: the optimized γ_φ in the conditional setting clusters tightly by class label and shows lower variance than in the unconditional setting. This suggests that label conditioning reduces the effective diversity of optimal trajectories — i.e., for a given class, many starting noises x_T share a near-identical optimal schedule. This sparse structure may explain why MTO outperforms more sharply in the conditional setting and may have implications for how diffusion model conditioning interacts with trajectory geometry. This observation is not fully developed in the paper but warrants attention.

A second insight worth noting: the 2D experiments demonstrate that even starting from an OT-paired model (which enforces straight trajectories during training), MTO can *reverse* the optimized trajectory away from straightness and achieve lower W₂. This suggests that trajectory optimality under finite NFE and discrete solvers is genuinely different from geometric optimality under continuous transport — a conceptually important separation that the paper establishes empirically but does not formalize.

---

## Suggestions

1. **Add a learnable scalar baseline** (single-function α_φ(t) trained adversarially) to Table 5. This is the most important missing ablation for isolating the contribution of multidimensionality.

2. **Directly confront the Table 5 finding** in the main text: quantify what the multidimensional coefficient adds over Adv.θ alone, and provide a hypothesis for why. Does the gain correlate with the degree of trajectory non-linearity (Figure 8)?

3. **Report total training compute** (pre-training + adversarial phase) in Table 6 or a supplementary table, for an honest efficiency comparison.

4. **For FFHQ/AFHQv2**, investigate whether the performance gap is due to (a) model size, (b) NFE, or (c) the hypothesis space. A small targeted ablation (e.g., increasing NFE to 10 on FFHQ or widening Γ_h) would help diagnose the root cause and make the limitations section more actionable.

5. **Quantify the "+" in "5 (+)" NFE** as a concrete fraction of H_θ compute (e.g., "U_φ is 1/k the size of H_θ and adds approximately X ms per sample on hardware Y").

6. **Visualize γ_φ spatially** on a few sample images to confirm the coefficient is learning semantically meaningful structure (e.g., edges vs. smooth regions, or foreground vs. background), not just label-conditioned global offsets.

---

**Evaluation summary:** MTO is a moderately novel contribution with a compelling conceptual argument and a strong CIFAR-10 result. The novelty is real but uneven — the multidimensional coefficient is a fresh idea, though the adversarial training machinery is largely inherited from prior work and explains most of the empirical gain. Technical soundness is adequate but leaves open questions about the split θ/φ objective and hypothesis space robustness. Empirical support is convincing on CIFAR-10 and in 2D but fails to generalize cleanly to FFHQ/AFHQv2, limiting the significance. As it stands, the paper makes a meaningful but not decisive advance; strengthening the ablations (especially the scalar baseline) and the scalability analysis would substantially improve confidence in the core claims.

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 5.0, 6.0]
Average score: 5.5
Binary outcome: Reject
