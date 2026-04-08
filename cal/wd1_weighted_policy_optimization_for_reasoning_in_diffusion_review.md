=== CALIBRATION EXAMPLE 37 ===

# Harsh Critic Review
Now I have read the full paper. Let me write a thorough review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is apt. The abstract accurately frames the core motivation (intractability of dLLM likelihoods leading to large variance/bias in GRPO-style ratio estimation) and the proposed remedy (ratio-free weighted log-likelihood). However, the claim "up to a +59% improvement in accuracy" is misleading without context: this refers specifically to 4×4 Sudoku (a relatively niche planning task), not to mainstream math benchmarks. On MATH500 and GSM8K, *wd1* offers negligible gains over *d1*. This selective framing of results in the abstract is problematic.

---

### Introduction & Motivation

The motivation for avoiding policy-ratio computation in diffusion LLMs is well-articulated and genuinely important. Figure 1 makes the empirical point cleanly: ELBO-based ratios have very high variance, and *d1*'s single-step approximation introduces systematic bias. The contributions are stated clearly, and the paper delivers on all four bullet points to varying degrees.

One concern: contribution (3) buries the fact that the large gains (+58.8% on Sudoku, +16% on Countdown) are on planning tasks, while GSM8K/MATH gains are marginal. The introduction implies broadly improved "reasoning capabilities," which is technically true but overemphasized.

---

### Method (Sections 3.1–3.3)

**Derivation of WLL (Sec. 3.1):** The derivation is clean. Starting from reverse-KL regularized policy optimization (Eq. 4), the closed-form solution (Eq. 5) is standard (Lagrangian of KL-constrained RL), and minimizing KL(*π*\*‖*π*_θ) leads to the weighted log-likelihood WLL (Proposition 2). The proof in Appendix A.2 is correct.

**Critical gap: the jump from WLL to wd1 (Sec. 3.2).** The wd1 objective (Eq. 8–9) is not derived from the reverse-KL objective; it is an engineering modification of WLL that adds an explicit negative-sample penalty term *w*^−. The authors acknowledge WLL's two failure modes (vanishing weights for low-advantage samples; undesirable likelihood increases when all rewards are equal) and motivate *w*^−, but this term is not principled from the optimization standpoint. The resulting objective is a heuristic extension — it balances positive reinforcement and negative unlearning, but there is no single clean loss function whose minimization yields Eq. 8. The theoretical grounding claimed for *wd1* properly applies only to its WLL component.

**wd1++ (Sec. 3.3):** The key underspecified point is how rewards are assigned to *intermediate* completions (x₀|ₗ from decoding step l < L). Since math/coding rewards require a complete, parseable answer, intermediate completions may not receive a meaningful reward. The paper never explicitly states whether intermediate completions inherit the final completion's reward. If they do, the group advantage is shared, and this is effectively a data-augmentation trick rather than a fundamentally different optimization. This needs to be stated clearly.

**Computational claims:** The paper claims wd1 reduces NFEs by eliminating the old-policy likelihood evaluation. However, Appendix B.3 notes that sampling from the geometric mixture *π*^{old}_{ref} introduces overhead when β > 0, and that an alternative is to approximate via the reference model log-probability, reintroducing one likelihood evaluation. In practice, β = 0 is used (Sec. 5, Implementation), which means the reference policy is completely removed. This is a significant hyperparameter choice that is only briefly noted, and it changes the theoretical framing (no KL regularization toward a reference).

---

### Theoretical Insights (Section 4)

The theoretical contributions are the strongest part of the paper.

**Lemma 1 / Theorem 1:** These results connect WLL training with energy-guided discrete diffusion (via AW-D-CSM). The proof is technically sound: it traces the energy guidance from clean data (x₀) to intermediate masked states (xₜ), uses Bayes' rule over the masked diffusion process, and then shows that minimizing the AW-D-CSM objective learns the guided concrete score. The derivation is a natural extension of Lu et al. (2023) to the discrete masked setting.

**Remark 1:** The equivalence L_WLL ⇔ L_AW-DCE relies on the chain L_D-CSM ⇔ L_CSM ⇔ L_DSE ⇔ L_DCE, which the paper states holds "up to multiplying a constant." Readers unfamiliar with these results may miss that this equivalence is only up to a positive scalar and holds under specific assumptions about the absorbing kernel. This should be stated more precisely.

**Remark 2:** The interpretation of the w^− term as "data unlearning" (NegGrad) is conceptually interesting. The derivation in Appendix D.1 is correct: via importance sampling, the w^− term can be re-expressed as minimizing the ELBO on samples drawn from a Boltzmann distribution that up-weights low-advantage regions. This is a nice post-hoc interpretation.

**Theorem 2:** The monotonic improvement guarantee for the reverse-KL formulation is established by a straightforward chain of inequalities. The key step (Eq. 29) bounds the reverse-KL penalty by λ·D_KL^{max}, relying on the standard D_TV^{max}² ≤ D_KL^{max} result. This is fine, but the bound is quite loose in practice (as typical for trust-region bounds).

---

### Experiments & Results (Section 5)

**Main results (Table 1 and Table 3):**

1. **Sudoku and Countdown are not representative of "reasoning."** The most dramatic gains (+58.8% Sudoku, up to +25% Countdown) are on narrow, constrained puzzle tasks. On the primary academic benchmarks (GSM8K, MATH500), *wd1*-LoRA vs. *d1* yields: GSM8K 256: 80.8 vs 78.2; GSM8K 512: 82.3 vs 82.0; MATH500 256: **34.4 vs 34.4** (identical); MATH500 512: 39.0 vs 38.0. These improvements are marginal at best and within noise at typical test set sizes.

2. **Comparison is against the authors' own reproduction, not the original reported numbers.** The paper uses "reproduced *d1*" as the main baseline. Table 7 shows that the reproduced *d1* underperforms the reported numbers on several metrics (e.g., MATH500-256: 34.4 reproduced vs 38.6 reported; Countdown-512: 35.2 reproduced vs 42.2 reported). This systematic gap raises questions about whether the reproduction faithfully captures *d1*'s capability, potentially inflating wd1's relative improvements.

3. **wd1++ uses different training data.** The abstract and introduction present wd1++ as a direct comparison to d1 and MDPO, but wd1++ trains on the MDPO dataset (subsampled from OpenR1). Comparing wd1++ (trained on OpenR1 subset) to d1 (trained on GSM8K/MATH train splits) is not controlled. Training data quality and size can dominate method differences. Table 3's caption acknowledges this only obliquely.

4. **No statistical significance testing.** There are no error bars, confidence intervals, or repeated runs. This is especially concerning for the marginal math improvements.

5. **Hardware and resource disparity.** wd1++ trains on 8×A800 (80GB) while d1 trains on 4×A100 (80GB) — twice the total GPU memory. Although the paper argues fewer total rollouts (1,280 vs 30,000), different hardware makes direct comparisons of training cost difficult.

6. **"State-of-the-art" claim:** The abstract claims wd1++ achieves "state-of-the-art" on MATH500 among dLLM methods. This is plausible within the very narrow category of LLaDA-based RL methods. It should not be interpreted as state-of-the-art among all math reasoning systems.

**Ablations (Table 4 and Figure 2):**

The ablation that removes w^− (wd1-P / WLL) is important and the result is striking: WLL alone (with only positive weights) performs drastically worse on Sudoku (6.69%) and math (MATH500-256: 29.4%). This confirms that the negative-sample term is essential and not merely complementary.

The ψ sensitivity study is appreciated but covers a limited range (0.1 to 10.0) with inconsistent comparison base.

The combined weight λ ablation in Table 9 tests only three values (0.4, 0.5, 0.6) and the margin between them is large. λ = 0.5 (equal weighting) is clearly best, but the reason for the sharp cliff (0.4 → 11.77%) suggests high sensitivity that warrants more granular investigation.

---

### Writing & Clarity

The paper is generally well-written, though some structural issues impede comprehension:

- Algorithm 1 (line 4) says samples oi ~ π_old(·|q) but the text (Section 3.2) says samples should come from π^{old}_{ref} — this contradicts the theoretical derivation and the appendix note about β = 0.
- The inline layout of Figure 1 within Section 2.3 is confusing textually (caption appears twice).
- The "reproduced vs reported" split in Table 7 is important but buried in Appendix C.

---

### Limitations & Broader Impact

The limitations section (Appendix D) is honest about degenerate reward cases and text-only applicability. However, several important limitations are unaddressed:

1. **The fair-comparison concern** (different training data for wd1++ vs baselines) is not acknowledged in the main paper.
2. **What does the method do to model diversity / mode coverage?** The negative-sample unlearning interpretation suggests wd1 actively reduces likelihood of low-reward sequences. There is no analysis of whether this leads to mode collapse or degraded out-of-distribution behavior.
3. **Single model (LLaDA-8B-Instruct) only.** No evaluation on Dream-7B or other dLLM architectures reduces confidence in generalizability.
4. **No comparison to AR models fine-tuned with GRPO/PPO** on the same benchmarks. The claim of efficiency is relative to diffusion GRPO, but not placed in the broader context of AR RL methods.

---

### Overall Assessment

*wd1* addresses a real and important problem — ratio-based RL objectives cause compounding approximation errors in diffusion LLMs — and proposes a principled alternative in the WLL objective, backed by clean theoretical connections to energy-guided diffusion and data unlearning. The theoretical contributions (Lemma 1, Theorem 1) are technically sound and provide a satisfying interpretation.

However, the paper overstates its empirical contribution. The headline results (+59% on Sudoku) come from a narrow planning task and a comparison against an apparently under-tuned reproduction of the baseline. On mainstream math benchmarks (GSM8K, MATH500), wd1 offers negligible improvement over reproduced *d1*. The stronger wd1++ numbers use different training data, making the comparison impure. Furthermore, the method's core theoretical claim — that the full wd1 objective is derived from reverse-KL policy optimization — is only partially true; the negative-sample w^− term is a well-motivated but ad hoc addition to WLL that lacks principled derivation from the same objective. The absence of error bars across all experiments is a significant methodological weakness for a venue like ICLR, where empirical rigor is expected.

**Verdict:** The paper makes a genuine conceptual contribution (ratio-free RL for dLLMs with theory), but the experimental support for the claimed improvements on math reasoning is too weak and potentially confounded by non-apples-to-apples comparisons to meet ICLR's bar. Major revisions are needed to: (1) clarify or fix the comparison conditions for wd1++; (2) add significance tests and multiple runs; (3) test on at least one additional dLLM beyond LLaDA; (4) address the principled vs. ad hoc status of the w^− term.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the computational and statistical challenges of applying reinforcement learning (RL) to diffusion-based LLMs, where estimating policy ratios via intractable likelihood approximations introduces high variance and significant overhead. The authors propose `wd1`, a ratio-free policy optimization method that reformulates the RL objective as a weighted log-likelihood, requiring only a single likelihood approximation while explicitly reinforcing high-advantage completions and unlearning low-advantage ones. An extended variant, `wd1++`, leverages intermediate denoising steps to achieve state-of-the-art math reasoning performance with substantially reduced training compute and rollout budgets.

### Strengths
1. **Clear Theoretical Grounding:** The method is rigorously derived from reverse-KL regularized policy optimization (Equations 4-7) and formally linked to energy-guided discrete diffusion training and negative sample unlearning (Theorems 1-2, Appendix A). This provides a solid mathematical justification for the weighted log-likelihood formulation.
2. **Strong Empirical Gains with Notable Efficiency:** `wd1` consistently outperforms the primary baseline `d1` across four diverse reasoning tasks, achieving 76.4% on Sudoku (+58.8%) and 44.2% on MATH500 (Tables 1 & 3). Crucially, it eliminates the need for supervised fine-tuning and reduces FLOPs per step (Table 2), demonstrating 10× greater sample efficiency in rollout usage for `wd1++`.
3. **Thorough Ablation Studies:** The authors systematically validate core design choices. Ablations confirm the critical role of the negative weight $w^-$ in stabilizing training (Table 4, Section 5.2), show that SFT provides diminishing returns for `wd1`, and analyze the sensitivity of the weighting temperature $\psi$ (Figure 4).

### Weaknesses
1. **Reliance on a Biased Likelihood Estimator:** `wd1` inherits the single-step ($t=1$) likelihood approximation from `d1` for computational efficiency. While acknowledged in the limitations (Section D), the paper does not empirically evaluate how sensitive the ratio-free objective is to approximation bias when paired with a more accurate (but costly) ELBO estimator, leaving it unclear whether the performance gains partially depend on this specific bias.
2. **Lack of Statistical Robustness Reporting:** All main results (Tables 1-3) and training dynamics (Figure 2) report single-run or best-checkpoint metrics without standard deviations or multi-seed variance. Given the well-documented instability of online RL and the large claimed margins (e.g., +58.8%), the absence of error bars makes it difficult to assess the statistical significance and reproducibility of the improvements.
3. **Ambiguous Sampling Overhead Analysis:** Section 3.2 and Appendix B.3 note that sampling from the geometric mixture $\pi_{\text{old}^{\text{ref}}}$ requires multiple forward passes. The paper claims overall lower compute but does not provide a transparent wall-clock breakdown of rollout sampling time versus gradient update time for `wd1` vs. `d1`, making the stated efficiency gains slightly difficult to fully verify in practice.
4. **Narrow Focus on Stepwise Sampling Strategy:** `wd1++` samples intermediate clean completions uniformly over denoising steps $l \in \text{Unif}\{1,\dots,L\}$ (Section 3.3). The paper does not discuss or evaluate alternative importance-weighted or confidence-based step selection strategies, which could better align the objective with the actual quality distribution across the diffusion trajectory.

### Novelty & Significance
**Novelty:** High. The work introduces a fundamentally different optimization paradigm for dLLMs by completely bypassing policy ratio estimation, a known bottleneck in diffusion RL. The theoretical equivalence to energy-guided diffusion sampling and the stepwise utilization of intermediate completions represent meaningful methodological advances.
**Clarity:** Good. The paper is well-structured, with clear mathematical derivations and intuitive explanations of the positive/negative weight mechanism. Minor notation density in Sections 3-4 is mitigated by comprehensive appendices and algorithm pseudocode.
**Reproducibility:** High. A public GitHub repository is provided, and the appendix contains detailed hyperparameters, dataset specifications, reward functions, and theoretical proofs. The only reproducibility gap is the lack of reported random seed variance, which ICLR reviewers typically expect.
**Significance:** High. As diffusion LLMs gain traction for their inference efficiency, scalable and stable RL post-training methods are urgently needed. `wd1` addresses a core computational and statistical limitation while delivering strong empirical results on standard benchmarks, making it highly relevant to the ICLR community and the broader ML field.

### Suggestions for Improvement
1. **Report Multi-Seed Variance:** Run main experiments across at least 3 random seeds and report mean ± standard deviation in Tables 1-3. This will establish statistical significance and align with ICLR's standards for robust empirical reporting.
2. **Evaluate with an Unbiased Likelihood Estimator:** Include a small-scale experiment replacing the $t=1$ approximation with an ELBO-based estimator (e.g., averaging over multiple $t$). This would empirically demonstrate that `wd1`'s ratio-free formulation is robust to different likelihood estimators and not uniquely tied to `d1`'s bias.
3. **Clarify Compute Breakdown:** Provide a detailed wall-clock breakdown of training time (e.g., time spent on rollout generation vs. forward pass caching vs. gradient updates) for `wd1` vs. `d1`. This will solidify the efficiency claims and clarify the overhead introduced by geometric mixture sampling.
4. **Analyze Stepwise Sampling in `wd1++`:** Briefly discuss or ablate the uniform step sampling strategy. Experiment with weighting steps by denoising confidence or remaining entropy to show whether targeted step selection could further improve sample efficiency or stability.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a `d1` baseline without SFT because without it, performance gains may be attributed to the absence of SFT rather than the algorithm itself.
2. Include multi-seed results with error bars for main benchmarks because Figure 4 shows seed sensitivity, and single-run results undermine reliability claims.
3. Report total FLOPs or wall-clock time to reach *equal* accuracy because "fewer rollouts" ignores per-step cost differences and misrepresents efficiency.
4. Ablate likelihood approximation methods (ELBO vs. `d1` approx) because the method inherits approximation bias that could confound the "ratio-free" benefit.
5. Compare against RL-tuned autoregressive models (e.g., GRPO on Llama) because efficiency claims require context against the dominant AR paradigm.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze how biased likelihood approximation violates Theorem 2 because the monotonic improvement guarantee assumes exact likelihoods.
2. Track policy entropy over training because it distinguishes whether $w^-$ prevents collapse or merely acts as implicit KL regularization.
3. Analyze advantage distribution across denoising steps because uniform sampling in `wd1++` assumes equal utility which may not hold.
4. Evaluate on out-of-distribution math sets because high scores on MATH500 may indicate verifier overfitting rather than reasoning gains.
5. Monitor weight magnitude distribution because exponential weighting risks gradient instability not captured by average metrics.

### Visualizations & Case Studies
1. Visualize token probability shifts for negative samples because it directly verifies the claimed "unlearning" mechanism.
2. Show side-by-side reasoning traces for success/failure cases because accuracy metrics hide reasoning quality differences.
3. Plot weight histograms over training steps because it exposes potential sample impoverishment from exponential weighting.
4. Correlate completion length with accuracy per bucket because shorter generations (Fig 5) might indicate shortcut learning.

### Obvious Next Steps
1. Derive a score-matching objective without likelihood approximation because the current bias undermines the theoretical soundness.
2. Run consistent full fine-tuning comparisons because LoRA results may not reflect full capacity performance.
3. Validate on non-LLaDA diffusion architectures because reliance on one model family limits generality claims.
4. Test against adversarial verifiers because RL methods are prone to reward hacking which accuracy metrics miss.

# Final Consolidated Review
## Summary
The paper proposes wd1, a ratio-free policy optimization method for diffusion-based large language models (dLLMs). The key insight is that existing GRPO-style methods require computing policy ratios (πθ/πold), which amplifies approximation errors in dLLMs' intractable likelihoods. wd1 reformulates the objective as a weighted log-likelihood requiring only a single likelihood approximation, while introducing a negative-sample unlearning term. An extended variant, wd1++, leverages intermediate denoising steps for further efficiency gains.

## Strengths
- **Principled theoretical foundation:** The derivation of the weighted log-likelihood (WLL) objective from reverse-KL regularized policy optimization is clean (Proposition 2), and the connection to energy-guided discrete diffusion (Theorem 1) provides a satisfying interpretation of the method as learning an advantage-guided score function. The proof that minimizing AW-D-CSM recovers the guided concrete score is technically sound.
- **Addresses a genuine computational bottleneck:** The paper correctly identifies that policy-ratio computation in diffusion RL requires three likelihood approximations (current, old, reference policies), each incurring variance and computational cost. Eliminating this ratio is a meaningful contribution to dLLM fine-tuning.
- **Strong empirical results on planning tasks:** On Sudoku, wd1 achieves 76.4% accuracy compared to d1's 17.6% (with SFT) and 16.1% (without SFT)—a substantial gain. Countdown shows similar improvements (51.2% vs 35.2%). The efficiency improvements (no SFT required, 10× fewer rollouts for wd1++) are real and meaningful.
- **Thorough ablation of core design choices:** The ablation showing that w^− (negative-sample weighting) is essential—not just complementary—is important. Table 4 demonstrates that WLL alone (wd1-P) performs drastically worse (6.69% Sudoku vs 76.4%), validating the design.

## Weaknesses
- **The w^− term lacks principled derivation:** While the WLL objective (Section 3.1) follows cleanly from reverse-KL optimization, the negative-sample term w^− (Section 3.2) is introduced heuristically to address WLL's failure modes. The paper motivates it through "data unlearning" (Remark 2), but this is a post-hoc interpretation rather than a derivation from the same optimization principle. The resulting objective is not the minimizer of any clearly stated loss—only the WLL component enjoys that guarantee.
- **Marginal improvements on mainstream benchmarks:** On GSM8K and MATH500, wd1's gains over d1 are modest (e.g., MATH500-256: 34.4% for both; GSM8K-512: 82.3% vs 82.0%). The dramatic gains are concentrated on Sudoku and Countdown, which are planning/constraint tasks rather than the math reasoning the abstract emphasizes. The paper should more clearly frame where gains occur.
- **wd1++ comparison uses different training data:** wd1++ trains on OpenR1 subsampled data (He et al., 2025), while d1 and other baselines train on GSM8K/MATH train splits. This confounds the methodological comparison—performance differences could stem from data quality/quantity rather than algorithm. The paper acknowledges this only briefly.
- **No statistical significance testing:** All results report single-run values without error bars or multiple seeds. Figure 4 in appendix hints at seed sensitivity (abrupt reward drops early in training), but the main tables lack variance estimates. For ICLR, this is a methodological gap.
- **Single model architecture:** All experiments use LLaDA-8B-Instruct. No evaluation on Dream-7B or other dLLM architectures limits confidence in generalizability.

## Nice-to-Haves
- **Add multi-seed variance reporting:** Run main experiments across 3+ seeds and report mean ± std to establish statistical significance.
- **Evaluate wd1 with an unbiased likelihood estimator:** The method inherits d1's biased t=1 approximation. A small experiment comparing this to ELBO-based estimation would clarify whether gains depend on this specific bias.
- **Clarify the theoretical status of w^−:** Either derive it from a principled objective or more explicitly acknowledge it as an algorithmic heuristic with empirical justification.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"Algorithm 1 contradicts theory about sampling from π^{old}_{ref}"** — In practice, β = 0 is used (Section 5, Implementation), which makes π^{old}_{ref} = π_old, resolving the apparent contradiction.
- **"Compare against AR models fine-tuned with GRPO/PPO"** — The paper's scope is dLLM optimization methods; comparing to AR RL methods would require a different experimental setup and is outside the stated contribution.
- **"Hardware disparity invalidates efficiency claims"** — The efficiency claims focus on algorithmic improvements (rollout count, training steps, NFEs) that are independent of raw hardware speed. Table 2 reports FLOPs per step and NFEs, which are hardware-agnostic.
- **"State-of-the-art claim is narrow"** — The claim is correctly scoped to "diffusion-based LLM reasoning methods" on LLaDA; it does not claim general SOTA on math benchmarks.

## Novel Insights
Beyond the paper's contributions, a key insight emerges from the interplay between theory and practice: the w^− term, while theoretically ad hoc, may be serving as implicit entropy regularization. The critic's observation that the method might "collapse diversity" is partially addressed by the ablation showing w^− is essential, but the paper does not track entropy or diversity metrics. This suggests an unexplored connection between negative-sample unlearning and explicit KL-regularization that could inform future work on dLLM stability.

## Suggestions
- **Report training curves for main benchmarks:** Figure 3 shows reward dynamics for Sudoku/Countdown, but not GSM8K/MATH. Including these would help readers assess convergence and stability.
- **Add a controlled wd1++ experiment on d1's training data:** To isolate the algorithmic contribution, run wd1++ on the same GSM8K/MATH splits as d1 and compare apples-to-apples.
- **Track and report entropy or diversity metrics:** To address concerns about mode collapse from the negative-sample term, report token entropy or response diversity during training.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
