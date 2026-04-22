Now I have a solid set of calibration anchors. Let me compile my review.

**Calibration paper comparison:**

High-scoring anchors (>7):
- SCALLION/SCAFCOM (8.0): Strong theory matching EF21 + novel algorithms for biased compressors in FL, comprehensive convergence guarantees
- LoCoDL (7.5): Doubly-accelerated convergence guarantees, strong theory + practice alignment
- Sparsity+quantization study (7.5): Mathematical proofs and extensive empirical results on LLMs

Medium anchors (4-6):
- MoTEF (6.6): Novel compression + momentum + EF framework, first to match distributed SGD rate under arbitrary heterogeneity; accepted poster
- ZO hard-thresholding (5.33): Novel variance-reduced ZO algorithm with honest theoretical limitations; accepted poster
- LASER (5.83): Linear compression with SGD-like theory but consistent practical gains; rejected (limited novelty)
- PAdaMFed (7.6): Accepted oral

Low-scoring anchors (<3):
- Compressed decentralized learning (1.67): Weak theory, trivial experiments, restrictive assumptions
- Prodigy (2.5): Claims theoretical improvement but actually worse than existing baselines
- Exact linear-rate GD (2.5): Uses knowledge of x* in step size, fundamentally flawed

Now let me write the final review.

## Summary

The paper introduces Markovian compression operators for distributed optimization — compressors whose stochasticity depends on previous iterations via a Markov chain, so that coordinates recently transmitted are less likely to be selected again. Two concrete instantiations are proposed: BanLast, which hard-bans recently-sent coordinates, and KAWASAKI, which smoothly penalizes them. Convergence guarantees are provided for MQSGD (non-convex/PL) and AMQSGD (strongly convex), and experiments on logistic regression and ResNet-18/CIFAR-10 show practical improvements over Random Sparsification.

## Strengths

- **Novel and well-motivated idea**: Markovian compression — making coordinate selection anti-correlated across iterations — is genuinely new for this setting. The intuition ("we just sent coordinate j, so don't send it again") is natural, and the paper establishes that no prior work combines Markovian stochasticity with compressed communication (Section 1.2).

- **Ergodicity analysis with explicit rates (Theorem 1)**: The paper proves both BanLast and KAWASAKI are ergodic Markov chains with uniform stationary distribution and provides explicit convergence rate bounds ρ, which are non-trivial contributions needed for the main results.

- **Transparent acknowledgment of limitations (Section 2.4)**: The paper is commendably honest about the theory-practice gap (d²/m² vs d/m), the logical contradiction that larger K worsens theoretical bounds but helps in practice, and the suboptimal L/μ dependence. This intellectual honesty is unusual and valuable.

- **Momentum yields provable acceleration**: Corollary 2 achieves (L/μ)^{2/3} dependence for AMQSGD versus L/μ for MQSGD, confirming genuine theoretical acceleration from the momentum scheme.

- **Practical improvements are consistent**: Table 1 shows KAWASAKI achieving 89.05% test accuracy vs. 87.9% for Rand-5%, with visibly lower train loss (0.0305 vs 0.0743) and gradient norm (0.745 vs 1.403). Figure 1 shows consistent improvements across four settings on MNIST logistic regression.

## Weaknesses

### Fatal
None.

### Major

- **Theory-practice disconnect — the theoretical bounds predict the method should be worse, not better**: The convergence bounds in Theorems 2 and 3 contain d²/m² (versus d/m for unbiased compressors) and an unfavorable τ dependence that grows with K. As Section 2.4 explicitly states, "while using a large K will theoretically give poorer convergence, in practice algorithms with non-zero values of K perform better." This means the theory cannot explain the paper's own empirical success. The paper attributes this to the "impossibility of using the expectation trick" and cites other Markovian stochasticity papers facing the same limitation, but ultimately there is no convergence bound under which Markovian compression provably improves over i.i.d. compression — the bound for K=0 (Random) is strictly better. This does not invalidate the paper, but the theoretical contribution falls short of providing the insight that would explain the practical gains, which is the central promise of theory. (Sections 2.2–2.4; Corollaries 1, 2)

- **Experimental methodology: best-run selection and per-method hyperparameter tuning**: Figure captions explicitly state "best runs are selected" (Figure 1) and "Best runs for each method are displayed" (Figure 2). Table 1 reports mean ± std over 5 runs, but hyperparameters were "fine-tuned" separately per method (Section 3.1, 3.2). This setup risks conflating methodological advantages with favorable hyperparameter/seed selection — especially for KAWASAKI which has multiple additional hyperparameters (K, b, π_Δ). The gap between BanLast and KAWASAKI in Table 1 (train loss 0.0734 vs 0.0305) is large enough to warrant scrutiny about whether it reflects genuinely different compression behavior or simply better-tuned hyperparameters. Without a shared hyperparameter search budget or ablations controlling for hyperparameters across methods, the empirical case for superiority is not fully convincing.

- **Missing comparison with error-feedback methods for biased compressors**: The Markovian compressors are only asymptotically unbiased — they are biased at any finite time. The standard approach for biased compressors is error feedback (EF21 and related methods), which can achieve linear convergence without the d²/m² penalty and without data-similarity assumptions. The paper compares against Random Sparsification (unbiased, no error feedback) and PermK/Natural, but not against any error-feedback baseline. Since the paper's main selling point is practical improvement, comparison against the best existing approach for biased compression would be the most informative. The paper acknowledges this in Section 2.4 ("future research" for variance reduction) but the omission weakens the empirical contribution.

### Minor

- **BanLast applicability restriction**: BanLast requires d ≥ (K+1)m, while the ergodicity guarantee requires d > (K+1)m and the explicit ρ requires d > (2K+1)m. The gap between the definition condition and the convergence rate condition (where (K+1)m < d ≤ (2K+1)m is valid for BanLast but has no explicit rate bound) means the theory has limited coverage of parameter regimes, especially for aggressive sparsification (large m). The paper partially addresses this by introducing KAWASAKI, which works for arbitrary d ≥ m, but the theoretical guarantees for BanLast explicitly exclude some feasible parameter settings.

- **Four additional hyperparameters in AMQSGD**: Algorithm 2 introduces θ, η, β, and p on top of the step size γ. The paper provides no principled guidance for setting these; they appear to be fine-tuned in experiments. This makes practical adoption difficult and raises questions about whether the improved KAWASAKI results stem from better hyperparameter search rather than the compressor itself.

### Trivial
None.

## Nice-to-Haves

- Tighter analysis that captures the benefit of anti-correlated sampling (e.g., showing a bound under which K > 0 provably improves over K = 0), which would close the central theory-practice gap
- Fair comparison protocol: shared hyperparameter budget, multiple seeds without best-run selection, and sensitivity analysis for K and b
- Quantification of the transient bias (how quickly the Markov chain mixes relative to the optimization trajectory)

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"The theory predicts the opposite of the claimed practical benefit, undermining the paper's central thesis"** (Harsh Critic #1): The harsh critic frames this as a *fatal* structural flaw. While the theory-practice gap is real and important (and I keep it as a Major weakness), the paper is transparent about it and does not claim theoretical superiority over unbiased methods. The paper's theoretical claim is that AMQSGD converges faster than MQSGD, which is true (Corollaries 1 vs 2 show acceleration). The more serious claim about "the theory contradicts the experiments" is honestly discussed in Section 2.4 — the paper isn't hiding this. Downgraded from Fatal to Major.

- **"Evidential: selective reporting and per-method hyperparameter tuning"** (Harsh Critic #2): The concern about best-run selection is legitimate and kept as a Major weakness. However, the claim that this "compromises" the experimental methodology is somewhat overstated — Table 1 does report means over 5 runs, and the convergence curves are standard in the optimization literature. The concern is real but not fatal. Downgraded slightly in severity.

- **"No comparison with error-feedback methods"** (Harsh Critic #3): This is kept as a Major weakness. However, it is not quite fair to say the comparison is *unfair* — the paper is deploying Markovian compressors within a QSGD framework without error feedback, and comparing against Random Sparsification within the same framework. EF21-style methods use a fundamentally different algorithmic framework (with error correction), so including them would require adapting them to the Markovian setting (which the paper hasn't done). The comparison should be made, but the asymmetry is not inherently against the baseline.

- **"Abstract claims theoretical superiority"** (Harsh Critic): The abstract says "we show theoretically that the accelerated method converges faster than the basic version" — this is about AMQSGD vs MQSGD, which is true. Not a misrepresentation.

- **"Section 1.1 claims strongly convex and non-convex as notable contribution"** (Harsh Critic): These are standard settings, but the paper's contribution is the Markovian convergence analysis in these settings, not the settings themselves. Minor framing issue at worst.

- **"Four momentum parameters with no guidance"** (Harsh Critic): Kept as Minor. This is standard for accelerated methods (the paper follows Vaswani et al. 2019 and Beznosikov et al. 2023b).

- **"KAWASAKI vs BanLast gap needs explanation"** (Harsh Critic): Rephrased as a question about hyperparameter tuning rather than a structural weakness, since the paper notes that "for more complex optimization task, smoother history accumulation (as in KAWASAKI) is required."

- **"Logistic regression comparison is just about coordinate selection"** (Harsh Critic): This is a feature, not a bug — the comparison isolates the effect of the compression scheme when all methods use the same sparsification ratio.

- **Missing related works**: Not included per instructions (I don't have external sources to confirm existence).

- **Reproducibility concerns about hyperparameters**: Removed per rules (nitpick about undisclosed hyperparameters).

## Novel Insights

The core tension in this paper is instructive for the field: Markovian compression is intuitively sensible and empirically effective, but current theoretical tools (the "stepping back" technique, uniform bounds on compressor noise necessitated by the loss of the expectation trick) fundamentally cannot capture the anti-correlation benefit. This is not unique to this paper — it reflects a broader limitation in how Markovian stochasticity is analyzed in optimization. The paper's honest acknowledgment of this gap (Section 2.4) is itself a contribution, as it clearly delineates what the current theory can and cannot explain. The d²/m² vs d/m gap and the τ dependence are features of the proof technique, and the paper correctly identifies them as such, but the fact that no existing analysis framework can explain the empirical improvement suggests that fundamentally new technical tools may be needed to close this gap.

## Suggestions

- Address the theory-practice gap by pursuing analysis that exploits the *beneficial* correlation structure of the Markov chain rather than only bounding the bias it introduces. Even partial results (e.g., for a restricted class of objectives or in a local convergence regime) would significantly strengthen the paper.
- Report experimental results with a fair hyperparameter protocol: same search budget across methods, multiple seeds with full curves (not just best runs), and ablations varying K and b while holding other hyperparameters fixed.
- Add comparison with an error-feedback baseline (e.g., EF21 with Random Sparsification) in the same QSGD framework, even if Markovian compressors haven't yet been adapted for error feedback.

## Score and Decision

Comparing against calibration anchors:

- **High-scoring (>7)**: SCALLION (8.0), LoCoDL (7.5), sparsity+quantization (7.5) — these have tight theory that aligns with practice and strong baselines.
- **Medium (4-6)**: MoTEF (6.6, accepted poster) has novel framework + first matching distributed SGD rate; ZO hard-thresholding (5.33, accepted poster) has honest limitations + practical improvement; LASER (5.83, rejected) has limited novelty but practical gains.
- **Low (<3)**: Compressed decentralized learning (1.67), Prodigy (2.5) — fundamentally flawed or misrepresenting results.

This paper introduces a genuinely novel idea (Markovian compression) with honest theoretical limitations and encouraging but methodologically imperfect experiments. It is above the low-scoring anchors (the idea is sound and novel, the theory is valid though loose, experiments show real improvements). It is below the medium-high anchors like MoTEF (6.6) because (a) the theory doesn't explain the empirical gains, and (b) the experimental methodology has the best-run selection and per-method hyperparameter tuning issue. It is comparable to papers in the 4.5–5.5 range: novel and honest, but with significant gaps that prevent strong confidence in the results.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>