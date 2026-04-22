Now I have a solid set of calibration anchors. Let me write the final review.

## Summary

The paper introduces the first framework for attributing *global* properties of diffusion models (e.g., Inception Score, demographic diversity, aesthetic quality) to *data contributors* (groups of training data), rather than to individual data points. It formulates this as a contributor-level Shapley value problem and makes computation feasible through "sparsified fine-tuning"—pruning the trained model and fine-tuning on contributor subsets—which provides 5.3–18.6× speedups over full retraining. Experiments across three datasets show the method dramatically outperforms 15 baselines on the linear datamodeling score (LDS).

## Strengths

- **Novel and well-motivated problem formulation.** The shift from local-to-global attribution and datum-to-contributor attribution addresses genuine gaps. Definition 1 (Section 2.2) formalizes contributor attribution, and Section 1 clearly distinguishes this from prior methods like D-TRAK and Journey-TRAK. The contributor-level Shapley formulation avoids the known aggregation error from summing datum-level scores (Koh et al., 2019).

- **Substantial performance gains over all baselines.** Table 1 shows LDS of 61.48%, 26.34%, and 61.44% across three datasets, dramatically outperforming all 15 baselines. Many TRAK-based methods yield negative correlations (e.g., Journey-TRAK at −42.92% on CIFAR-20), confirming that naively aggregating local attributions fails for global properties.

- **Sparsified fine-tuning provides real efficiency-to-accuracy gains.** Figure 2 is a strong result: under equal computational budgets, sparsified-FT consistently achieves higher LDS than both full fine-tuning and retraining from scratch across all three datasets, confirming the efficiency gains translate to better attribution, not just speed.

- **Counterfactual evaluation supports attribution quality.** Figure 3 shows removing top contributors identified by the proposed method causes the largest negative relative change on CIFAR-20 (−23.23%) and CelebA-HQ (−7.83%), and retaining top contributors yields the largest positive change (16.98% and 20.0% respectively), outperforming all comparative baselines on two of three datasets.

- **Diverse experimental settings.** Three experiments span different architectures (unconditional DDPM, LDM, Stable Diffusion+LoRA), dataset scales (20–258 contributors), and global properties (IS, entropy, aesthetic score).

## Weaknesses

### Fatal
None.

### Major

- **The core approximation (Eq. 6) lacks direct empirical validation in the main text.** The entire method rests on the claim that $\mathcal{F}(\tilde{\theta}^*_{S_j,k}) \approx \mathcal{F}(\theta^*_{S_j})$—that sparsified fine-tuning faithfully approximates full retraining. The main text provides no direct comparison for the same subsets. The theoretical results (Propositions 1–2) offer only asymptotic guarantees under convexity assumptions (acknowledged as unrealistic for diffusion training), and the bound $2\sqrt{n}C$ is loose (e.g., ~$32C$ for ArtBench with 258 contributors). The paper itself states (line 135): "We leave theoretical results incorporating finite-step bounds and Shapley value estimation for future work." While empirical validation is promised in Appendix D, this is the crux of the method's correctness—it demands main-text evidence. Without it, we cannot fully assess whether high LDS comes from good Shapley estimation or from artifacts of the approximation procedure. A single scatter plot of $\mathcal{F}(\tilde{\theta}^*_{S,k})$ vs. $\mathcal{F}(\theta^*_S)$ for sampled subsets would significantly address this.

- **The CelebA-HQ LDS of 26.34% is weak and undiagnosed.** An LDS of 26.34% means the additive datamodel explains roughly 7% of the variance ($\rho^2 \approx 0.07$). The paper acknowledges this is "relatively low" but offers no diagnosis. This is the only experiment with a realistic-scale LDM (274M parameters, 50 contributors), making it the setting closest to real-world deployment. Understanding the failure—whether due to the sparsified fine-tuning approximation, the entropy metric's noise, insufficient Shapley samples, or the pruning ratio—is essential for knowing when the method works and when it doesn't.

### Minor

- **Missing ablations on critical hyperparameters.** The number of fine-tuning steps $k$ (200–1000), pruning ratios (44%–74% sparsity), and KernelSHAP samples $M$ are all crucial choices affecting output quality. While Figure 2 shows sparsified-FT beats alternatives under equal compute, this conflates two effects (more subsets evaluated vs. per-subset approximation quality). Ablations on these hyperparameters would clarify which design choices matter most.

- **The ArtBench counterfactual result is ambiguous.** For ArtBench, removing top contributors identified by the proposed method yields a −1.86% change, which is actually *less* impact than D-TRAK (0.58%). The paper does not discuss this anomaly. Since the ArtBench model uses only LoRA parameters (5.1M), it is the simplest setting, and the counterfactual result there is arguably the least informative for showing the method works in challenging configurations.

- **The $\theta_\emptyset$ (untrained model) baseline in Eq. 5 deserves discussion.** KernelSHAP uses $\mathcal{F}(\theta_\emptyset)$ both as a regression intercept and in the efficiency constraint. For diffusion models, an untrained model generates incoherent noise, so $\mathcal{F}(\theta_\emptyset)$ may be poorly defined (e.g., IS on noise). The practical handling of this term is not discussed, and it could affect Shapley value estimates.

### Trivial
- The pruning ratio varies widely across experiments (44%–74%) without explicit justification.

## Nice-to-Haves

- Direct scatter plot of $\mathcal{F}(\tilde{\theta}^*_{S,k})$ vs. $\mathcal{F}(\theta^*_S)$ for sampled subsets to validate the core approximation.
- Diagnosis of the CelebA-HQ failure: does the approximation degrade with larger models, or is the entropy metric inherently noisier?
- Convergence analysis of estimated Shapley values with increasing KernelSHAP samples $M$.
- Discussion of how $\theta_\emptyset$ is handled in practice for each metric.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Baselines are unfairly compared because they were designed for local attribution."** The paper explicitly states (Section 4.3) that local attribution scores are averaged across generated images and aggregated to contributor-level via principled summation. This is a reasonable post-hoc adaptation; the baseline comparison is informative precisely because it shows local methods fail for global attribution. Per the rules, criticizing an asymmetry that favors the baseline (not the proposed method) is not a valid weakness—here, the baselines are given their standard, standardly-applied form.

- **"Theoretical results are unrealistic due to convexity assumptions."** The paper itself acknowledges the limitations (line 135: "we leave theoretical results incorporating finite-step bounds and Shapley value estimation for future work") and states these are standard assumptions for analysis in this line of work (citing Golatkar et al. 2020; Georgiev et al. 2024). The propositions are offered as supporting intuition, not as the sole justification. This criticism has been absorbed into the Major weakness about missing empirical validation.

- **"LDS requires 900 full retrainings—computationally expensive."** This is about the evaluation protocol and is standard practice in data valuation work. The paper reports confidence intervals across 3 random initializations, suggesting actual computation was done. Questioning compute costs of evaluation is a generic concern.

- **"Counterfactual evaluation is selective."** The paper explicitly states the computational reason for only comparing against the best baseline per category. This is a standard trade-off, not a methodological flaw.

- **Missing related works.** Per rules, removed—no external sources to confirm existence.

- **Formatting/style nitpicks.** Removed per hard rules.

## Novel Insights

The paper's most insightful observation is methodological: existing attribution methods for diffusion models (TRAK, D-TRAK, Journey-TRAK) are fundamentally misaligned with the task of global property attribution. Their design around local per-image influence functions means that even principled aggregation (summing datum-level scores) yields negative correlations with global properties. This suggests a deeper structural mismatch—local gradient statistics may carry little signal about distributional properties that emerge only from many training data points interacting. The contributor-level Shapley approach, by directly measuring marginal contributions to the global property of interest, circumvents this mismatch entirely. However, the degree to which this insight is validated depends critically on the approximation quality of sparsified fine-tuning, which remains the paper's central open question.

## Suggestions

- Add a direct validation of Eq. 6 to the main text: for a sample of subsets $S$, compare $\mathcal{F}(\tilde{\theta}^*_{S,k})$ vs. $\mathcal{F}(\theta^*_S)$ where $\theta^*_S$ is fully retrained. This single experiment would do more to validate the framework than the theoretical propositions.
- Diagnose the CelebA-HQ failure: test whether varying pruning ratio or fine-tuning steps improves LDS on CelebA-HQ, or whether the entropy metric itself is too noisy for the additive datamodel to capture well.
- Discuss the practical handling of $\theta_\emptyset$ in Eq. 5—whether clipping, metric-specific baselines, or other adaptations are used.

## Evaluation Dimensions

- **Originality**: High. The problem formulation (global property → contributor attribution for diffusion models) is genuinely novel. The sparsified fine-tuning approach is pragmatic and well-motivated.
- **Importance**: High. Fair data crediting for diffusion models is societally important and technically underserved.
- **Claims support**: Moderate. The method clearly outperforms baselines on LDS, but the core approximation underpinning those Shapley values is not directly validated, and one of three experiments shows weak performance.
- **Experimental soundness**: Moderate-to-good. Three diverse settings, 15 baselines, and two evaluation paradigms (LDS + counterfactual). Missing ablations and direct approximation validation limit confidence.
- **Clarity**: Good. Well-structured, clear figures (especially Figure 1 and Figure 2), and transparent about limitations.
- **Community value**: High. Opens a new problem space with a workable first framework, and the failure of local attribution baselines for global properties is an important negative result.

## Score and Decision

Calibration anchors:
- **High band (>7)**: Diffusion Attribution Score (7.5, Spotlight) — similar topic (data attribution for diffusion), but that paper had strong SOTA on LDS with clearer theoretical grounding. In-Run Data Shapley (7.5, Oral) — also Shapley-based data valuation but with stronger scalability story and oral-level clarity. Scalable Influence Functions for Diffusion (8.0, Oral) — stronger theoretical grounding and more complete validation.
- **Medium band (4–6)**: D-TRAK paper (6.0, poster) — directly comparable topic (diffusion attribution), empirically strong but shallow methodological insight; this paper is stronger than D-TRAK in problem novelty and principled formulation but weaker in direct validation of its core approximation. Journey-TRAK framework (5.75, reject) — similar diffusion attribution with formal notions but primitive method; this paper is clearly stronger.
- **Low band (<3)**: Action Shapley (3.0, reject) — uses Shapley for training data selection but with no baselines, unclear definitions, and no rigorous evaluation; this paper is far stronger.

This paper sits between the medium and high bands. It has genuinely novel problem formulation and strong results, but the two major weaknesses (unvalidated core approximation and undiagnosed CelebA-HQ failure) are significant. Compared to the D-TRAK paper (6.0, poster), this paper is stronger in novelty and principle but weaker in validation completeness. Compared to the high-band papers, it lacks the direct empirical validation of its approximation and has a weaker theoretical contribution. I place it just above the borderline.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>