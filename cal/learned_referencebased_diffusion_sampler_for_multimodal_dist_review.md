=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary

This paper introduces the Learned Reference-based Diffusion Sampler (LRDS), a variational diffusion-based method for sampling from multi-modal distributions. The core observation is that existing variational diffusion samplers (PIS, DDS) are highly sensitive to their reference distribution hyperparameter σ, which can only be tuned optimally with ground truth samples. LRDS exploits prior knowledge of mode locations to run short local MCMC chains, from which it learns a multi-modal reference distribution — either a Gaussian Mixture Model (GMM-LRDS) or an Energy-Based Model (EBM-LRDS) — that is then used within the Reference-based Diffusion Sampler (RDS) variational framework. Experiments across Gaussian mixtures (up to d=64), the φ⁴ field system, Rings, and Checkerboard distributions show consistent improvement over a broad set of baselines in mode weight estimation.

---

## Strengths

- **Identifying and empirically diagnosing a concrete failure mode**: Figure 1 provides a compelling and falsifiable demonstration that LV-PIS and LV-DDS performance degrades sharply when σ deviates from the target-dependent optimum — and crucially shows that a well-chosen multi-modal reference makes results robust across a wide range of `w_ref`. This is a focused, reproducible problem statement that grounds the rest of the paper.

- **Two complementary variants with principled trade-offs**: The paper does not just propose a single method; it explicitly characterizes when GMM-LRDS suffices (Gaussian-like modes) and when EBM-LRDS is needed (complex topologies). Figure 3 illustrates this concretely for the Rings distribution, and the paper discusses the computational trade-off honestly. Most papers in this space present a single approach; the comparative analysis of the two variants is a genuine methodological contribution.

- **Density-only oracle — not score oracle**: All competing variational diffusion methods (LV-PIS, LV-DDS, LV-DIS, LV-CMCD, iDEM, PDDS) use the *score* of the target distribution in their training or sampling procedures. LRDS only requires evaluations of the unnormalized density γ(x). The paper makes this explicit in the experimental setup (Section 5): "LRDS only requires evaluations of the target density, which makes it an interesting alternative in settings where the score of π is expensive to compute." This is a practically significant advantage not shared by most baselines.

- **Clean unification of prior work**: Table 1 transparently shows that PIS and DDS are special cases of the RDS framework (with fixed isotropic Gaussian references), situating the contribution precisely in the existing literature. Proposition 1 and the discrete-time objective (Eq. 7) together provide a self-contained and general implementation recipe.

- **Robust performance under all tested settings**: GMM-LRDS consistently dominates across d = 16, 32, 64 Gaussian mixtures (Table 2), the φ⁴ system (Figure 4), and the Checkerboard distribution (Figure 6). The margin is large — most baselines collapse to uniform or single-mode estimates in d ≥ 32, whereas GMM-LRDS maintains errors below 5%.

---

## Weaknesses

### Fatal
None.

### Major

- **Mode location assumption is underexamined**: The paper assumes access to mode locations as prior knowledge and treats this as a given problem setting. This assumption shapes the method's applicability substantially. The paper does not discuss how sensitive LRDS is when mode locations are only approximately known (e.g., obtained from coarse gradient ascent), beyond ablations described as "lightly perturbing the reference distribution." What constitutes "light"? If modes are off by a nontrivial fraction of the inter-mode distance, can the variational optimization still correct for the bias? Without quantitative characterization of this degradation, practitioners cannot know when the method is safe to use. The appendix ablation should be moved to the main text and expanded with a range of perturbation magnitudes.

- **No wall-clock time or computational cost comparison**: The Discussion explicitly acknowledges that EBM-LRDS is computationally intensive and involves a pre-training stage. However, no empirical timing data is provided for either variant. For readers evaluating whether to adopt LRDS, it is essential to know whether, say, a 5× training overhead is incurred for a 3× accuracy improvement. A simple table of training time per method (reference training + diffusion training) and number of function evaluations is essential to assess practical utility.

- **Within-mode quality absent from main text**: Mode weight estimation is necessary but not sufficient. A sampler could assign correct marginal weights while generating poor within-mode samples. Appendix I.5.1 contains probability metrics, but the main text reports only mode weight error. Adding at least one within-mode quality measure (e.g., per-mode Wasserstein or KL) to the main results would substantially strengthen the empirical case.

### Minor

- **GMM component count J is underspecified**: The paper states "we observe that setting J equal or larger to the number of target modes can lead to better performance," and uses J=64 for a 3-mode Rings distribution. The gap between 3 and 64 is unexplained, and no ablation is provided. Without guidance on how to choose J — and evidence that performance is robust to misspecification — J effectively becomes a new hyperparameter replacing σ. The paper should report sensitivity of GMM-LRDS to J.

- **Hyperparameter sensitivity shift not fully characterized**: The central claim is that LRDS bypasses the obstacle of hyperparameter tuning (Abstract). Yet LRDS introduces its own hyperparameters: J (number of GMM components), MALA step size and chain length, and EBM architecture/training details. The paper demonstrates that LRDS is less sensitive than LV-PIS/LV-DDS to σ, but does not show that the total tuning burden is reduced. At minimum, a discussion comparing the sensitivity profiles of LRDS vs. baselines would strengthen this claim.

- **Experimental scale limited to d=64**: Given that the paper identifies protein Boltzmann distributions as a future application (which typically have hundreds to thousands of degrees of freedom), the experiments leave a gap. While d=64 is sufficient to demonstrate the method, it leaves open whether the performance gains persist at larger dimensions. This should be acknowledged as a current limitation.

### Tiny

- **Potential ambiguity in Eq. (7)**: In the discrete-time objective, the first sum contains the expression $g_{T-t_k}^\theta(Y_k)^\top\{g_{T-t_k}^\theta(Y_k) - \frac{1}{2}g_{T-t_k}^\theta(Y_k)\}$, which as typeset algebraically reduces to $\frac{1}{2}\|g^\theta\|^2$ — collapsing the distinction between the live $\theta$ and the detached $\hat\theta$ that appears in the continuous-time loss (6) and in the recursion (8). This may be a PDF extraction artifact, but the paper should explicitly verify this equation is correctly rendered to avoid reproducibility issues.

- **Bayesian logistic regression in appendix is unmotivated**: These experiments are described as "not explicitly multi-modal" and placed entirely in the appendix without explanation of what they are meant to demonstrate. If they support the claim that LRDS degrades gracefully in near-unimodal settings, that should be stated; if not, their inclusion is puzzling.

---

## Nice-to-Haves

- **Ablation on reference perturbation magnitude**: Expand the current "light perturbation" ablation to cover a range of mode location errors, producing a degradation curve that tells practitioners how accurate mode estimates need to be for LRDS to outperform baselines.

- **High-dimensional scaling experiment (d ≥ 100)**: Even a single higher-dimensional experiment (e.g., d=128 or d=256 on a Gaussian mixture) would help bound the method's scalability.

- **Convergence dynamics visualization**: Plotting mode weight estimation error vs. training iteration for LRDS vs. baselines would reveal whether the gain comes from faster convergence or a better final solution.

- **Theoretical variance bound**: Providing even an informal argument (or a formal bound in the appendix) relating $\text{KL}(\pi^\text{ref} \| \pi)$ to the variance of the LV loss would strengthen the claim that a closer reference inherently improves optimization stability.

- **Automatic mode discovery pre-step**: Proposing and evaluating a standard optimization warm-start to approximate mode locations before LRDS would increase practical applicability. This is not a core contribution, but would make the paper more self-contained.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Figure 5 contradiction"** (Harsh Critic): The critic flags an apparent contradiction between the paper text and a figure description stating "GMM LRDS and EBM LRDS show more complex structures, possibly due to mode collapse or numerical issues." This description is an AI-generated alt-text embedded in the PDF extraction, not the paper's own caption. The paper's actual caption reads "Samples obtained for Rings distribution," with text body stating that LRDS recovers mode structure. This is a review artifact and not a real issue with the paper.

- **Missing related works** (Harsh Critic): Per review policy, claims about missing references are excluded, as external sources cannot be confirmed.

- **Nested Sampling baseline** (Spark Finder): Per review policy, claims about missing baselines are excluded when external references cannot be confirmed.

- **EBM circularity** (Spark Finder): The concern that "annealed MCMC used to train the EBM reference suffers from the same mixing issues LRDS aims to solve" is explicitly addressed in Section 3.3. The paper uses the fact that the noising path $(hat{X}_t^\varphi)_{t \in [0,T]}$ defines a path of *increasingly simpler* distributions, making annealed MCMC effective for negative sampling at every level. This is precisely the point of the multi-level EBM design and its connection to prior work on annealed EBM training.

- **Asymptotic unbiasedness** (Harsh Critic): Demanding a formal asymptotic correctness proof is not standard for an empirical variational inference paper of this type and would be a non-standard rigor requirement. Moved.

- **Requesting Bayesian Neural Network posteriors** (Spark Finder): Outside the paper's stated scope and experimental contribution; the paper targets distributions where mode locations are available. This is scope creep rather than a weakness.

---

## Novel Insights

The most genuinely novel observation — which all three reviews touch but none fully articulate — is that the *reference distribution in variational diffusion samplers serves a dual role*: it determines both the tractable base distribution for the noising process and the implicit mode-weighting pressure on the variational optimization. When the reference is a misspecified isotropic Gaussian, the variational objective is structurally biased toward whichever mode is most accessible from the Gaussian center, independently of the true mode weights. By making the reference explicitly multi-modal and learned from local MCMC chains, LRDS breaks this coupling — the reference no longer penalizes paths that visit low-probability inter-modal regions, freeing the variational optimization to allocate probability mass according to the actual energy landscape. The right panel of Figure 1 (robustness to `w_ref`) is a particularly clean empirical demonstration of this: even when the reference weights are substantially wrong, RDS still converges to correct mode weights, because the reference geometry (multi-modality) is correct even when the reference weights are not. This suggests that *mode geometry* matters more than *mode weights* in the reference, a principle that could guide future work on reference design.

---

## Suggestions

1. **Quantify the mode location tolerance**: Run a systematic ablation where mode locations are perturbed by increasing magnitudes (e.g., 0%, 10%, 25%, 50% of the inter-mode distance) and report the resulting mode weight estimation error. Place this in the main paper, not the appendix.

2. **Add a runtime table**: Report total training wall-clock time and number of target density evaluations per method alongside Table 2. This does not require additional experiments.

3. **Move probability metrics to main text**: At minimum, add one non-mode-weight metric (e.g., sliced Wasserstein or per-mode KL) to Table 2 to demonstrate within-mode sample quality.

4. **Clarify Eq. (7)**: Explicitly verify and annotate the detached-gradient distinction in the discrete-time objective. If one instance of $g^\theta$ should be $g^{\hat\theta}$, correct it.

5. **Provide practical guidance for J selection**: Add a brief ablation varying J from below to above the true mode count for a representative experiment, and translate the result into an actionable recommendation.

6. **Discuss mode location acquisition cost**: Add a paragraph (even in the Discussion) estimating the computational cost of obtaining mode locations via gradient ascent with multiple restarts, relative to the diffusion training cost. This grounds the overall pipeline cost.

---

**Evaluation summary**: LRDS is a solid contribution with a clear problem statement, principled methodology, and convincing empirical results within its scope. The novelty is genuine, and the technical execution is sound. The primary gaps are in experimental completeness (no runtime data, no within-mode metrics in main text, limited scale) and in characterizing the robustness of the mode location assumption. These are significant but addressable shortcomings; they do not undermine the core contribution. The paper is strong enough to be accepted at ICLR with revisions, but the empirical section needs to be made more complete before it fully supports all claims.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 3.0, 6.0]
Average score: 6.2
Binary outcome: Accept
