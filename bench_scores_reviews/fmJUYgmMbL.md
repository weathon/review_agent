## Summary

LRDS (Learned Reference-based Diffusion Sampler) addresses a fundamental practical bottleneck in variational diffusion-based samplers: their performance on multi-modal targets degrades sharply with miscalibration of the reference distribution hyperparameter σ, yet calibrating σ requires ground-truth samples. LRDS sidesteps this by constructing a learned, multi-modal reference distribution from local MCMC chains initialized at known mode locations, then using this reference within the RDS variational framework. Two variants are proposed — GMM-LRDS (lightweight, analytically tractable scores) and EBM-LRDS (more expressive, neural score estimation via a multi-level EBM trained with annealed MCMC) — and both are shown to substantially outperform existing methods at recovering mode weights on high-dimensional Gaussian mixtures and the φ⁴ field theory model.

---

## Strengths

- **Precise and compelling diagnosis of σ-sensitivity.** Figure 1 (left/middle) shows a razor-sharp optimum in σ for LV-PIS and LV-DDS on a 16-D bi-modal target; the optimal σ coincides with the analytically-derived Gaussian approximation variance (Appendix I.2), making the problem concrete and quantitative, not merely intuitive.

- **The robustness asymmetry observation is genuinely novel.** Figure 1 (right) shows that a GMM reference with miscalibrated mode weights is remarkably robust — performance is nearly flat across a wide range of reference weights — whereas a Gaussian reference has a single narrow optimum. This asymmetry is the conceptual core of the paper and is clearly demonstrated.

- **The two-variant design cleanly maps to different problem regimes.** Figure 3 constitutes a crisp, visually unambiguous ablation: on the Rings distribution, GMM-LRDS cannot capture the ring geometry regardless of the number of components (J=16 shown), while EBM-LRDS exactly recovers it. This provides principled guidance on when to prefer which variant.

- **φ⁴ experiment is a strong scientific validation.** On a physically meaningful, non-synthetic target (d=32, two well-separated modes), all competing methods suffer mode collapse while GMM-LRDS correctly tracks the analytical Laplace approximation of the relative mode weight across varying local-field parameter h (Figure 4). The consistency with the Laplace approximation is a meaningful quantitative check.

- **Table 1 elegantly unifies the framework.** By showing PIS and DDS as special cases of RDS with fixed Gaussian reference, the paper provides a conceptually clean perspective on the broader method family, and Table 1 makes the generalisation explicit and graspable.

- **LRDS does not require target score evaluations.** The paper notes (Section 5) that LRDS only needs pointwise evaluations of the unnormalized density γ, not its gradient. This is a practically meaningful advantage for targets where the score is expensive or unavailable, and it distinguishes LRDS from most competitors.

- **The multi-level EBM training formulation solves the negative sampling problem cleverly.** By parameterising the entire noised path (p_t^φ)_{t∈[0,T]} jointly and using annealed MCMC across the path (Section 3.3), the authors simultaneously obtain negative samples for EBM training and leverage annealing for multi-modal coverage — a genuine "two birds, one stone" design that draws on established ideas but applies them non-trivially in this context.

---

## Weaknesses

- **Mode location assumption: minimal practical guidance.** The assumption that mode locations are known a priori is stated clearly in the introduction but receives almost no practical treatment. In real Bayesian inference tasks, obtaining mode locations may itself require MAP optimisation with multiple random restarts, which can be costly and unreliable on rugged landscapes. The paper gives no guidance — not even a brief discussion — on how to obtain modes in practice, how many can be handled, or what happens when mode locations are systematically biased (not merely randomly perturbed as tested in Appendix I.5.1). This gap limits the reader's ability to judge whether LRDS is deployable on their actual problem.

- **Overstatement in Table 2 analysis.** The paper claims "GMM-LRDS outperforms competing methods in all the considered dimensions," but at d=16, PDDS achieves 0.8% ± 0.6% versus GMM-LRDS's 1.7% ± 0.6% — PDDS is numerically superior. PDDS then catastrophically fails at d=32 (66.7%), so LRDS's advantage in higher dimensions is genuine and substantial, but the blanket claim of dominance at d=16 is factually inaccurate and should be corrected.

- **Mode weight estimation is the sole primary metric.** All quantitative main-text evaluation rests on a single scalar — absolute mode weight estimation error. While this is the central problem the paper addresses, it does not characterise intra-mode sample quality (e.g., covariance fidelity, energy distribution within a mode, or sliced Wasserstein distance). Variational metrics from Appendix G/I.5.1 are not discussed in the main text. For a method claiming to produce good samples from multi-modal distributions, this leaves the reader unable to assess whether the samples within each mode are well-distributed.

- **No computational cost comparison.** EBM-LRDS requires (i) running annealed MCMC for negative sampling, (ii) training the multi-level EBM E^φ, and (iii) running the full RDS variational optimisation — a pipeline with potentially 3-5× the cost of baselines. The paper acknowledges this in the Discussion but provides zero wall-clock times or evaluation-budget comparisons. Without this, the empirical advantage cannot be fairly weighed against the computational investment.

- **EBM training stability not characterised.** EBM training is notoriously susceptible to divergence and mode covering artifacts during negative sampling. Section 3.3 provides no convergence curves, stability diagnostics, or ablations on the EBM training hyperparameters (annealing schedule, step size, negative MCMC chain length). There is no evidence that the estimated reference scores s_t^ref = -∇_x E^φ(t,x) are accurate enough across all timesteps to reliably guide the diffusion process.

- **Failure modes of competitors in the φ⁴ experiment are not explained.** The paper simply states that all competitors exhibit mode collapse on the φ⁴ model without explaining whether this is due to the energy barrier structure, the dimensionality (d=32), the specific conditioning number of the covariance, or some other cause. Understanding *why* the competitors fail would substantially strengthen the paper's narrative about when LRDS is needed and when simpler alternatives suffice.

---

## Nice-to-Haves

- **Practical mode acquisition discussion.** A brief experiment or appendix showing a pipeline where mode locations are obtained via L-BFGS with multiple random restarts, followed by LRDS, would significantly strengthen the practical case. Even a discussion of failure conditions (e.g., number of modes beyond which the approach becomes infeasible) would help practitioners.

- **Ablation: competitor methods with GMM reference.** It is unclear whether the performance gains come from the LRDS variational framework or simply from the use of a multi-modal GMM reference that competitors could also adopt. A comparison of LV-PIS/LV-DDS using a GMM reference (instead of their default isotropic Gaussian) would isolate the contribution of the learned reference versus the RDS training framework.

- **Non-diffusion baseline: local MCMC + normalising constant estimation.** When mode locations are known and local MCMC chains are available, one could estimate mode weights via annealed importance sampling or thermodynamic integration within each mode. Including such a baseline would give readers a clearer sense of what the diffusion-based framework adds on top of the simplest possible exploitation of mode location knowledge.

- **Sensitivity to missing or strongly mislocated modes.** The appendix ablation perturbs mode locations lightly, but a more severe test — missing one mode entirely, or having mode locations off by a large multiple of the mode's standard deviation — would establish the robustness boundaries more concretely.

- **Extension to higher dimensions (d ≥ 128).** Table 2 stops at d=64. Diffusion-based methods are often motivated by high-dimensional scalability; demonstrating whether GMM-LRDS's advantage persists at d=128 or d=256 would give a clearer picture of the method's regime of applicability.

- **Real-world multi-modal posterior in the main text.** The Bayesian logistic regression results are in the appendix and the target is noted to be "not explicitly multi-modal." A real-world target with a known multi-modal posterior (e.g., Bayesian neural network with symmetries, or a mixture-model posterior) in the main paper would substantially broaden the demonstrated applicability.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **"Comparison is unfair because baselines receive poor σ calibration" (Harsh Critic):** The paper explicitly sets σ for PIS/DDS/DIS using "a Gaussian isotropic approximation of π̂^ref" (Section 5), which represents the best practically-available σ estimate without ground-truth samples. This is precisely the realistic evaluation setting the paper targets. Demanding oracle σ tuning for baselines would test an unrealistic scenario that contradicts the paper's stated problem setting. The comparison is fair by design.

- **"iDEM at d=64 is competitive" (Harsh Critic):** iDEM achieves 11.7% ± 0.4% at d=64 versus GMM-LRDS's 4.1% ± 0.6% — this is roughly a 3× gap. At d=32, iDEM collapses to 66.7%. This is not competitive.

- **"Listing hyperparameter sensitivity as a contribution is inappropriate" (Harsh Critic):** The paper provides a quantitative, controlled demonstration of this sensitivity with explicit connection to the analytically-derivable optimal σ (Appendix I.2). This constitutes a contribution beyond mere intuition.

- **Missing related works suggestions (all reviewers):** Per evaluation policy, no missing related works are noted, as their existence cannot be independently verified.

- **Pure style/formatting critique on Equation (7) redundancy (Positive Reviewer):** The apparent redundancy in the g − ½g term in (7) may be a text-extraction artifact from the PDF; without access to the original LaTeX, this cannot be reliably evaluated and is a formatting concern rather than a substantive flaw.

---

## Novel Insights

The most genuinely novel observation in this paper — beyond the method itself — is the *asymmetry in robustness between Gaussian and GMM references*, demonstrated cleanly in Figure 1: a Gaussian reference has a single narrow optimal σ requiring ground-truth calibration, while a GMM reference with the correct modal structure is nearly flat with respect to mode weight miscalibration. This asymmetry is not obvious a priori (one might expect the GMM's extra parameters to introduce additional sensitivity), and it provides a principled theoretical intuition for why investing in learning a multi-modal reference yields disproportionately large gains in practical robustness. The secondary insight — that the multi-level EBM parameterisation simultaneously enables tractable EBM score estimation and multi-modal negative sampling via annealing — is a useful connection between the EBM literature and diffusion-based sampling that may have broader applicability.

---

## Suggestions

- **Correct the overstatement in Table 2**: State explicitly that PDDS narrowly outperforms GMM-LRDS at d=16 (0.8% vs 1.7%) but fails catastrophically at d=32+, and that LRDS provides the most *consistent* performance across dimensions.

- **Add a practical mode-acquisition paragraph** in Section 3 or Section 6 discussing how mode locations can be obtained (e.g., gradient ascent from multiple random initialisations, clustering of short MCMC chains), and cite any existing mode-finding literature relevant to the paper's target application domains.

- **Report wall-clock training time** for at least one experiment (e.g., the d=64 Gaussian mixture), broken down by reference training and diffusion training phases, to allow readers to assess the cost-accuracy tradeoff of GMM-LRDS vs EBM-LRDS vs baselines.

- **Include convergence/stability diagnostics for EBM training**: At minimum, plot the ML objective over training iterations for EBM-LRDS and show that the reference score estimates (s_t^ref vs a finite-difference check on small-dimensional examples) are accurate.

- **Explain competitor failures on φ⁴**: Add one paragraph or figure in the appendix decomposing why competitors collapse (e.g., show mode weight trajectories during training, or energy landscape plots showing the barrier), since this provides actionable insight into when LRDS is most needed.

- **Move or expand the Bayesian logistic regression results**: Either move them to the main text with a genuinely multi-modal posterior task, or clearly characterise in the text why this unimodal-in-practice posterior is still a meaningful test case for the method.