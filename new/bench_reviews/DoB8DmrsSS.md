## Summary

The paper proposes SHIFT, a test-stage adversarial attack on vision-based RL agents that uses a conditional diffusion model with joint history-conditioned classifier-free guidance, policy classifier guidance, and autoencoder-based realism guidance to generate semantically perturbed states outside the traditional $l_p$-norm threat model. The core empirical finding is that these unrestricted, semantics-aware perturbations break multiple state-of-the-art defenses—including diffusion-based denoisers—across four Atari environments, highlighting a critical vulnerability in current robust RL methods.

## Strengths

- **Novel conceptual framework for unrestricted attacks in RL.** Definitions 1–5 (Section 3.1) introduce a useful vocabulary—valid, realistic, semantics-changing, and history-aligned states—that moves beyond the standard $l_p$-ball threat model and gives the field a way to reason about semantic adversarial states.
- **Strong empirical demonstration of defense failure.** Table 1 shows that SHIFT drives cumulative rewards to near-minimum values against five diverse defenses (regularization-based and diffusion-based) across four Atari games. This establishes that defenses designed for $l_p$-bounded attacks are vulnerable to unrestricted semantic perturbations, which is an important finding.
- **Practical feasibility.** Table 2 validates that the EDM architecture reduces per-state sampling time from ~5 seconds to ~0.2 seconds while maintaining comparable attack performance, making the attack feasible for real-time deployment.
- **Enabling theoretical result.** Theorem 1 (Section 3.2.2) proves that classifier-free history guidance and policy classifier guidance can be combined without interference in the reverse process, which is a clean technical result that justifies the proposed architecture.

## Weaknesses

### Fatal
None.

### Major

- **Missing ablations and unrestricted baselines undermine attribution of results.** The paper attributes its success to the combination of three guidance mechanisms (history, policy, autoencoder), yet Table 1 and Figure 3 report results only for the full system. There are no ablations removing individual guidance terms (e.g., unconditional diffusion, diffusion without policy guidance, diffusion without autoencoder guidance) and no quantitative comparison against simpler unrestricted generators (e.g., an unconditional diffusion model with policy guidance only, or a nearest-neighbor training-set frame attack). Without these, it is impossible to tell whether SHIFT’s specific design is necessary or whether any unrestricted realistic-state generator would break $l_p$-targeted defenses.
- **Stealthiness claim is overreached and partially circular.** The introduction states the attack avoids detection “by both humans and AI,” yet the paper contains no human evaluation and no independent automated detector (e.g., a separately trained binary classifier or distinct anomaly detector). The autoencoder is used both as an internal optimization target during generation (Section 3.2.3) and as an external stealthiness metric (Figure 3a), which creates a circular evaluation: low reconstruction error demonstrates only that the attack fools its own auxiliary network, not an independent observer. While the Wasserstein distance metric is independent, the central “stealthy” claim is empirically unsubstantiated.

### Minor

- **True-history approximation deviates from formal framework without empirical analysis.** Section 3.2.1 acknowledges that the implementation conditions on the true history $\tau_{t-1}$ instead of the victim’s observed projected history $H_{t-1}$ (Definition 4) because projection is “computationally expensive.” The paper does not quantify how much the victim’s observed history diverges from the true history under sequential semantic perturbations, or whether this approximation affects dynamic stealthiness over long episodes.
- **Some results exhibit high variance without significance testing.** For example, DP-DQN under attack in Pong reports $0.5 \pm 11.4$ and Diffusion History in RoadRunner reports $1480 \pm 788$ (Table 1). The absence of statistical significance tests makes it difficult to assess the reliability of these comparisons.
- **Theorem 1 is a straightforward application of existing diffusion guidance theory.** While useful for implementation, the result follows directly from the conditional independence of the two guidance terms and does not constitute a deep theoretical contribution.

### Trivial

- Myopic target action selection is already acknowledged as a limitation in Section 5.

## Nice-to-Haves

- Failure-case visualizations showing sequences of perturbed frames over multiple time steps (not just single-frame examples) to verify temporal consistency.
- Analysis of what states diffusion-based defenses reconstruct when given SHIFT perturbed inputs, which would strengthen the explanation for why these defenses fail.

## Removed Points

These points are flagged to be removed; treat them with caution.

- The criticism that the paper “dismisses [Korkmaz 2023] as targeting non-essential semantics without quantitative evidence in the main text” is invalid: the paper explicitly directs readers to Appendix E for evaluation results. Appendix sections are stripped by the parser but exist in the original submission.
- Concerns about missing proofs in the appendix or missing references are invalid for the same reason.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Add ablation experiments removing each guidance term individually and report reward, manipulation rate, and reconstruction error to isolate the contribution of each component.
- Train an independent anomaly detector (e.g., a separate autoencoder or binary classifier) on clean states and report its detection rate on SHIFT perturbed states versus baselines to substantiate the stealthiness claim.
- Quantify the divergence between true history and victim-observed projected history over the course of an episode to validate the approximation in Section 3.2.1.

## Score and Decision

**Calibration papers used:**
- `/home/wg25r/review_agent/human_reviews/F5dhGCdyYh.md` (avg 7.33, Accept spotlight): A directly comparable paper on stealthy RL attacks. It is stronger than the current submission because it includes human evaluation, independent automated detection, and rigorous information-theoretic constraints. The current paper scores well below this anchor.
- `/home/wg25r/review_agent/human_reviews/wZWTHU7AsQ.md` (avg 5.33, Accept poster): An RL robustness paper with concerns about novelty but extensive experiments. The current paper has a more novel framing but larger empirical gaps (missing ablations, overclaim).
- `/home/wg25r/review_agent/human_reviews/a05PWdPKo0.md` (avg 4.50, Reject): An adversarial attack paper rejected for missing ablations and overclaiming. The current paper is above this anchor because its core empirical result—breaking SOTA diffusion-based defenses—is more significant and compelling.
- `/home/wg25r/review_agent/human_reviews/scFfMOOGD8.md` (avg 4.25, Reject): A diffusion-model security paper rejected partly for claiming stealthiness without adequate independent evaluation. The current paper is above this anchor because it does include comparative metrics and broad defense evaluation, but the parallel weakness on stealthiness evaluation pulls it down.

The current paper sits between the rejected 4.5 anchor and the accepted 5.5 anchor. Its core finding—that unrestricted semantic attacks break defenses designed for $l_p$ perturbations—is genuinely important and well-demonstrated in Table 1. However, the missing ablations and the unsupported stealthiness claim are significant enough that the paper cannot score above the mid-5 range. I assign a score of **5.5**, reflecting a borderline paper whose main contribution is compelling but whose evaluation needs substantial strengthening.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>