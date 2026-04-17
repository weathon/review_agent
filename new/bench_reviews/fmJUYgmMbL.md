Now I have a good sense of the calibration landscape. Let me finalize my review.

## Summary

The paper introduces the Learned Reference-based Diffusion Sampler (LRDS), a variational diffusion-based sampling method for multi-modal distributions that replaces the typically hand-tuned Gaussian reference distribution in prior work (PIS, DDS) with a learned reference—either a Gaussian Mixture Model (GMM-LRDS) or an Energy-Based Model (EBM-LRDS)—constructed from local MCMC samples initialized at known mode locations. The method proceeds in two stages: (i) learn a multi-modal reference process from local MCMC samples, and (ii) run the Reference-based Diffusion Sampler (RDS) variational optimization using the LV loss. Experiments on Gaussian mixtures (up to d=64), φ⁴ field models, and 2D geometric distributions demonstrate that LRDS more accurately recovers mode weights than competing methods.

## Strengths

- **Identifies a genuine and underappreciated problem:** The paper clearly demonstrates (Figure 1) that existing variational diffusion samplers (LV-PIS, LV-DDS) are highly sensitive to the reference σ hyperparameter, with mode weight error varying dramatically depending on this value—which itself typically requires ground truth samples to tune correctly. This is a valuable observation for the community.

- **Principled and general RDS framework:** The RDS formulation (Section 2.2) with arbitrary reference distributions generalizes prior work cleanly. The discrete-time loss (Eq. 7) with tractable coefficients for any noising scheme subsumes LV-PIS and LV-DDS as special cases (Table 1), providing a concrete, implementable objective that was previously only available for Gaussian references.

- **Strong empirical performance on mode weight recovery:** On high-dimensional bi-modal Gaussian mixtures (Table 2), GMM-LRDS achieves 1.7–4.1% error across d=16–64, while most competitors experience mode collapse or large errors (33%+). On the φ⁴ model (Figure 4), GMM-LRDS is the only method avoiding mode collapse.

- **Clean conceptual separation:** The two-step design (learn reference from local data, then learn guidance on top) is well-motivated, and the two reference instantiations (GMM vs. EBM) illustrate the tractability–expressiveness trade-off clearly with the Rings example (Figure 3).

- **Fairness effort in experiments:** The authors ensure baselines receive some prior knowledge (Gaussian approximation of π̂ref for base distributions, σ tuning from reference statistics, standardization for iDEM/PDDS, buffer pre-filling), making comparisons more meaningful than a purely naive baseline comparison.

## Weaknesses

### Major:

- **The assumption of known mode locations fundamentally changes the problem being solved and limits practical scope.** The paper states: "we assume that we have access to the location of the modes as prior information on π" (Sec. 1). In many practical settings where multi-modal sampling is hardest—Bayesian posteriors, Boltzmann distributions—finding the modes is itself the primary challenge. The paper cites Noé et al. (2019); Pompe et al. (2020); Grenioux et al. (2023) to argue that sampling remains hard even with known modes, which is fair, but the claim that LRDS "outperforms previous diffusion-based samplers on challenging multi-modal settings" is only valid under this strong assumption. The paper does not analyze how the method degrades when mode locations are imprecise or incomplete, nor does it discuss how one would obtain mode locations in realistic scenarios. This significantly limits the practical applicability of the method as a general-purpose sampler.

- **Asymmetric use of prior information between LRDS and baselines.** While the authors commendably give baselines some prior knowledge (Sec. 5), the information provided to baselines is weaker: a Gaussian approximation of π̂ref, σ tuned from reference statistics, or standardization. LRDS, in contrast, gets exact mode locations and uses them to (a) run per-mode local MCMC chains, and (b) learn an expressive multi-modal reference distribution. Competing methods are not allowed to initialize separate per-mode chains or learn multi-modal base/reference distributions from the same information. The paper does not disentangle how much improvement comes from the LRDS methodology itself versus from the privileged structural information. A natural ablation—giving competing methods like CMCD or PIS/DDS the same GMM reference distribution as their base/reference—would clarify this, but is absent.

- **Evaluation is heavily concentrated on mode weight recovery, with limited distributional assessment.** For the main high-dimensional experiments (Table 2, Figure 4), the primary metric is mode weight estimation error. While relevant, this is a low-dimensional summary that can be correct even when within-mode geometry or tail behavior is wrong. Additional probability metrics are relegated to an appendix (Appendix I.5.1) and only reported for a subset of experiments. For the φ⁴ model in d=32, where geometry and conditioning are nontrivial, a single scalar ratio is insufficient to establish that the algorithm truly approximates π well.

### Minor:

- **Computational cost is not quantified.** The paper acknowledges that LRDS "comes at the computational cost of the necessary pre-training of the reference process model" (Sec. 6) but provides no wall-clock times, FLOP counts, or training budgets for any method. EBM-LRDS in particular involves pre-training a multi-level EBM via annealed MCMC (itself a non-trivial sampling problem for multi-modal distributions), followed by the RDS variational optimization. Without cost comparisons, it is hard to assess whether the improved accuracy justifies the additional computational overhead.

- **EBM-LRDS training involves its own multi-modal sampling sub-problem.** Training the multi-level EBM requires negative sampling from p^φ, which the paper addresses using annealed MCMC (Sec. 3.3, Algorithm 11 in Appendix F). This means the method partially relocates the multi-modal sampling challenge into the reference training stage. The paper does not analyze how the quality of the annealed MCMC sampler affects EBM-LRDS performance, nor how costly this step is relative to the final RDS optimization.

- **EBM-LRDS's intractable Z^ref and its effect on the RDS objective.** For EBM-LRDS, the reference normalizing constant Z^ref is intractable (Sec. 3.3, Table 1). The term ϱ = log(γ^ref/γ) in the RDS objective (Eq. 7) therefore cannot be computed exactly. The paper does not clearly discuss how this intractability is handled in practice or what approximation errors it introduces.

- **Limited scalability evidence beyond d=64.** The highest dimension tested is d=64 for Gaussian mixtures and d=32 for φ⁴. GMM reference fitting scales quadratically with d (covariance matrices), and EBM training adds further cost. There is no evidence of performance in hundreds of dimensions where practical Bayesian posteriors typically live.

### Trivial:

- The caption of Figure 5 notes that "GMM LRDS and EBM LRDS show more complex structures, possibly due to mode collapse or numerical issues" which somewhat contradicts the otherwise positive narrative, but this is a minor presentation inconsistency.

## Nice-to-Haves

- **Ablation on imperfect mode information:** Testing GMM-LRDS when mode locations are perturbed or when some modes are missed would clarify the method's robustness to the quality of prior knowledge—a critical practical concern.

- **Same-prior comparison with baselines:** Running competing methods (e.g., CMCD) with the same learned GMM as their base/reference distribution would help isolate how much of the improvement comes from the methodology versus the privileged information.

- **Higher-dimensional experiments** (d≥100) on genuinely multi-modal targets would better demonstrate practical scalability.

- **Systematic reporting of probability metrics** (MMD, Wasserstein) across all experiments, not just mode weight error.

## Novel Insights

The paper's key insight—that the reference distribution in variational diffusion samplers should be learned rather than hand-specified, and specifically that mode location information can be leveraged to build a multi-modal reference that dramatically stabilizes mode-weight estimation—is valuable and well-supported by Figure 1 and Figure 2. However, this insight exists in tension with the practical reality that mode locations are often unknown, and the paper does not successfully address this tension. The separation between "learning a reference that mimics the target's multi-modality" and "learning a guidance term that corrects for the remaining discrepancy" is a clean design principle for future sampler development.

## Suggestions

- **Retitle and reframe the contribution more honestly** as "sampling from multi-modal distributions with known mode locations," clearly flagging this as a distinct, restricted problem setting. This would allow the technical contributions to be assessed on their own merits without overclaiming.

- **Add an ablation where baselines receive the same structural information** (e.g., initialize PIS/DDS with the learned GMM as reference, or give CMCD the same per-mode chain initialization) to separate the contribution of the LRDS algorithmic design from the contribution of the prior information.

- **Report wall-clock times** and, at minimum, the number of density/score evaluations per method.

## Score and Decision

**Calibration comparisons:**

- **Improved sampling via learned diffusions** (Richter et al., ICLR 2024) — Accept (poster), scores 6,6,6,8 (avg ~6.5): This is the direct predecessor work (same RDS framework + LV loss). It had a cleaner problem setting (no known-mode assumption) but more limited empirical gains. LRDS builds directly on this foundation with the main addition being the learned reference.

- **DGFS** (Diffusion Generative Flow Samplers) — Accept (poster), scores 8,8,8,6 (avg ~7.5): A sampling paper in the same family with strong theory and novel objectives, but with acknowledged experimental limitations. LRDS has arguably more compelling empirical improvements but under a much stronger assumption.

- **CMCD** (Controlled Monte Carlo Diffusions) — Accept (poster), scores 6,8,6,8,8 (avg ~7.2): A directly competing sampler from the same community. Accepted despite being described as "relatively direct" by one reviewer, with strong empirical results without the known-mode assumption.

- **Annealing Flow** — Withdrawn/Reject, scores 1,5,3,3,6 (avg ~3.6): A weak multi-modal sampling paper with poor writing, theoretical issues, and missing comparisons. LRDS is clearly much stronger than this.

- **a-GPS** — Reject, scores 1,3,3,3 (avg ~2.5): A mode-seeking sampler with severe issues. LRDS is far superior.

The LRDS paper makes a genuine contribution by identifying the reference sensitivity problem and providing a principled solution that works well in the known-modes setting. However, the strong known-mode assumption—which the paper itself acknowledges is central but does not adequately contextualize—significantly weakens the general claims. The paper is essentially solving a different, easier problem than the standard sampling problem, and the comparison with baselines that solve the harder (general) problem is asymmetric. This is not a fatal flaw if the problem setting is clearly scoped, but the paper's current framing overclaims generality. Combined with the incremental nature of the core contribution (learn a GMM/EBM reference instead of a Gaussian in an existing framework) and the limited evaluation metrics, this places the paper below the acceptance threshold for a top venue, but not far below—it is a solid contribution for a more specialized audience.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>