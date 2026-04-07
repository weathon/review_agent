=== CALIBRATION EXAMPLE 79 ===

# Harsh Critic Review
Now I have read the full paper. Let me compose my detailed, section-by-section review.

---

## Section-by-Section Critical Review

---

### Title & Abstract

The title *wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models* is accurate and appropriately scoped. The central claim—eliminating policy ratios while maintaining or improving performance—is clearly stated.

**Concern on "+59% improvement":** The abstract reports "up to a +59% improvement in accuracy." Inspecting Table 1 and 7, *wd1* achieves 76.4% vs. *d1* (reproduced)'s 17.6% on Sudoku-256, a **+58.8 percentage-point** improvement, but a **~334% relative** improvement. Phrasing it as "+59%" conflates percentage-point gains with relative gains, which is misleading. The "+59%" figure also applies only to Sudoku; on GSM8K and MATH500, *wd1* (LoRA) is essentially tied with *d1*. This should be qualified more carefully.

**Reproducibility issue framing:** The abstract benchmarks *wd1* against "diffusion-based GRPO (*d1*)" without noting that the comparison is against the authors' own reproduction of *d1*, which underperforms *d1*'s reported numbers in some settings (e.g., Sudoku 512: 16.2 reproduced vs. 9.5 reported—wait, actually *d1* reported 9.5 but reproduced 16.2; Countdown 256: 25.8 reproduced vs. 32.0 reported). In other words, the reproduced *d1* is not uniformly weaker; the gap with *wd1* may look different when compared against *d1*'s actual published numbers.

---

### Introduction & Motivation (Section 1)

The motivation is well-constructed. The argument that applying approximate likelihoods to compute exponential ratios exp(ϕ^π_θ − ϕ^π_old) amplifies approximation errors is both theoretically and intuitively compelling. Figure 1 (high-variance ELBO-based ratios, biased *d1*-based ratios) provides good empirical motivation.

**Concern: Missing ratio-free baselines in framing.** The introduction frames *wd1* as the natural solution to the policy-ratio problem. However, on-policy approaches (REINFORCE, RLOO) are also ratio-free and have established implementations for LLMs. The introduction should explain why off-policy weighted regression is preferable to simply using on-policy methods. This is partially addressed in Section 6, but the comparison baseline should appear in the experiments, not just in related work.

**Concern: Conflation of the two limitations.** The paper conflates two distinct problems with diffusion GRPO: (1) high variance of ratio estimates, and (2) computational overhead of three separate likelihood evaluations. These are related but separable. It would be cleaner to address them independently, since one could, for example, reduce overhead without changing the variance problem.

---

### Preliminaries (Section 2)

The exposition of masked diffusion, DCE, GRPO, and diffusion-GRPO is clear and complete. The appendix A.1 formally quantifies the exponential error amplification with a clean bound (Equation 15), which is a useful theoretical contribution.

**Minor concern:** Equation (3) (diffusion-GRPO objective) applies clipping with ε, but Figure 1 shows ratios far outside [1−ε, 1+ε] with ε=0.5. It is unclear whether clipping partially addresses the variance problem discussed. The paper should clarify what the clipping actually achieves in practice before arguing it is insufficient.

---

### Method: *wd1* (Sections 3.1–3.2)

**Core derivation (Section 3.1):** The derivation from the reverse-KL regularized objective (Equation 4) to the WLL loss (Equation 6–7) is correct and follows the standard AWR derivation (Peng et al., 2019). The key step is recognizing that the closed-form optimal policy π* allows minimizing D_KL(π*||π_θ) instead of performing the constrained optimization directly. This is a well-known trick (AWR, DPO, etc.) now applied to dLLMs. The contribution is solid, but not surprising.

**Critical concern: The w^− term is not derived, it is engineered.** The theoretical development in Section 3.1 motivates only the w^+ term (WLL, Equation 6). The w^− term in *wd1* (Equation 8–9) is introduced as an *ad hoc* fix for two identified failure modes. While Remark 2 later connects w^− to negative-sample unlearning, this interpretation is post-hoc. The actual *wd1* loss (Equation 8) is not a principled minimizer of any stated objective—it is a heuristic combination of AWR (w^+) and a negative reinforcement term (w^−). The claim of "theoretical soundness" in the abstract overstates matters: the theory establishes that WLL (w^+ only) is equivalent to energy-guided diffusion; it does not establish that the *wd1* objective (w^+ − w^−) is optimal or derivable from first principles.

**Concern: Geometric mixture approximation is unjustified.** Algorithm 1 (line 5) states that samples from π_old^ref are obtained in practice. Appendix B.3 explains that log π_old^ref(x_0^k|x_t, q) is approximated as λ log π_old(x_0^k|x_t, q) + β log π_ref(x_0^k|x_t, q), i.e., mixing in logit space. For softmax-normalized distributions, this is not equivalent to the geometric mixture of distributions in probability space (which the theory uses). The approximation is convenient but unverified, and the authors do not analyze or bound its error.

**Concern: β=0 in all experiments.** Table 6 shows that wd1 sets β=0, which eliminates the reference policy entirely. This means the method in practice is simply AWR (with a negative-sample penalty term) applied to dLLMs with no KL regularization toward a reference model. The elaborate theory involving the geometric mixture π_old^ref and Lagrange parameters (λ, β) is not exercised in the experiments. The practical method is simpler than the theoretical framework suggests, and this discrepancy should be clearly stated upfront, not buried in implementation details.

---

### *wd1*++ Extension (Section 3.3)

The idea of leveraging intermediate denoising completions is creative and shows an awareness of an under-utilized source of training signal in dLLMs.

**Critical concern: Reward computation for intermediate completions is undefined.** The *wd1*++ objective (Equation 10) trains on all intermediate completions {x_0|l} for l=1,...,L. However, how is the reward R(q, x_0|l) defined for intermediate (possibly incomplete) completions that are not the final generation? If R is a verifier that checks correctness, it should be applied only to the final completion x_0|L = o_i. If the reward is shared across all intermediate completions from the same rollout (i.e., all intermediate completions inherit the reward of the final completion), this is an implicit modeling assumption that should be stated explicitly and justified. The paper describes using the "expanded group of completions to estimate both the advantage function and the corresponding weights" (Section 3.3) but does not address this fundamental issue.

**Concern: Conflating exploration benefit with method superiority.** *wd1*++ uses a batch size of 64 (vs. 4 for *d1*; Table 8), 8×A800 GPUs, and different training data (OpenR1/He et al. 2025). The 10× fewer rollout claim (1280 vs. 30000) is based on total rollouts, but *wd1*++ uses 64 rollouts per step while *d1* uses 4—so *wd1*++ samples 16× more per step. The method reaches a better checkpoint in 20 steps vs. 7500 steps for *d1*, but the comparison is confounded by significantly different compute resources and data.

---

### Theoretical Insights: Energy-Guided Diffusion (Section 4)

This section is theoretically the strongest part of the paper. Lemma 1, Theorem 1, and Remarks 1–2 provide a coherent interpretation of WLL as AW-D-CSM (energy-guided discrete diffusion training), and the w^− term as NegGrad-style unlearning. The proofs in Appendix A.3 are clearly written and appear correct.

**Concern: Theorem 1 establishes AW-D-CSM is equivalent to WLL (without w^−), but the actual method uses w^+ − w^−.** The theoretical equivalence (Remark 1: L_WLL ⟺ L_AW-DCE) applies to the positive-weight-only formulation. The augmentation with w^− (unlearning) does not have an equally clean theoretical footing—the unlearning connection (Remark 2, Appendix D.1) is an analogy to NegGrad (Golatkar et al., 2020), not a formal optimality result. The paper should be clearer that the theoretical guarantee applies to WLL, while *wd1* is a heuristic extension of WLL.

**Concern: The monotonic improvement guarantee (Theorem 2) applies to the idealized, unappproximated method.** The practical *wd1* uses biased likelihood approximation (per the d1 approximation, t=1 only) and β=0. Theorem 2 does not cover these approximations or the practical off-policy setting where multiple gradient updates are taken per batch. This gap between theory and practice is not acknowledged in the main text.

---

### Experiments & Results (Section 5)

**Main results (Table 1):**
- The dramatic improvements on Sudoku (+58.8pp) and Countdown (+25pp at len=256) are impressive and consistent.
- On GSM8K and MATH500 (LoRA), *wd1* matches but does not exceed *d1* (reproduced). Specifically, MATH500-256: *wd1*=34.4% vs *d1*=34.4%, exactly tied. This is a significant gap between the strong headline claims and actual math performance.
- Importantly, the baseline *d1* (reproduced) underperforms *d1* (reported) on Countdown (25.8 vs. 32.0 at 256 tokens). If the reported *d1* numbers are used for Countdown-256, *wd1*'s advantage shrinks from +25.4pp to +19.2pp. For math tasks, *d1* reported vs. reproduced are also discrepant (GSM8K-512: 82.1 reported vs 82.0 reproduced—close; MATH500-512: 40.2 reported vs. 38.0 reproduced). The paper should include a row with *d1*'s reported numbers in Table 1, not just in Table 7, to give the reader an unambiguous comparison.

**Fairness of baseline comparisons (Table 3):**
- The comparison of *wd1* (full) at 82.7/43.6 against MDPO (full) at 83.4/43.4 is *not* the same as comparing *wd1* (LoRA) against *d1* (LoRA). The paper should clarify that *wd1* and MDPO are on different footing—different training data, different GPUs, different total compute.
- SDPO (81.2) appears without a MATH500 result, suggesting cherry-picking a favorable comparison point.

**Missing baselines:**
1. **REINFORCE / RLOO for dLLMs**: These are ratio-free by design and are the most natural comparisons for a "ratio-free" method. The paper cites SPG (Wang et al., 2025a) and d2 (Wang et al., 2025c) as ratio-free competitors but does not include them in experiments, citing that they are "concurrent work." Given that the paper's main contribution is being ratio-free, at minimum REINFORCE-based methods should be compared.
2. **WLL alone (*wd1*-P)**: This is included in Table 4, and the results are damning (WLL: 6.69% on Sudoku vs. *wd1*: 76.4%). This confirms that w^− is critical. However, the ablation at 256 tokens doesn't show whether WLL catches up over more steps.
3. **AWR** (Peng et al., 2019) applied naïvely to dLLMs is not compared. The practical *wd1* is essentially AWR + negative reinforcement for dLLMs.

**Statistical significance:** All results are single-run. The paper shows one alternative seed for MATH500 in Figure 4 (right), which produces different early training dynamics. Given the volatile training dynamics visible in Figure 3 (e.g., MATH500 showing a reward drop then recovery), single-run results on all benchmarks are insufficient to draw strong conclusions, particularly for smaller performance differences (e.g., 34.4% vs. 34.4%).

**Training cost comparison (Table 2):**
- The comparison shows wd1 saves ~2 hours of SFT and ~22 seconds per RL step. The per-step speedup is modest (103.5 → 81.2 sec, ~21%). The real savings come from eliminating SFT—but this is as much an experimental choice as a methodological one: the authors chose not to use SFT, and show in ablations that SFT doesn't help *wd1*. Whether this holds generally (e.g., with better SFT data) is not explored.

---

### Ablation Study (Section 5.2)

The ablations are largely convincing:
- The critical role of w^− (Table 4, *wd1*-P collapses without it) is important to establish and is clearly shown.
- The sensitivity to ψ (Figure 4) is appropriately explored.
- The combined weight sensitivity (Table 9) is limited to Sudoku and only three values (0.4, 0.5, 0.6); a wider sweep would be more convincing.

**Concern on the "equal weight" justification (Table 9, Section C.2):** The paper claims the equal-weight design (cw=0.5) is the most robust, and gives a theoretical argument for why extreme values fail. Table 9 shows 25.63% (0.5) vs 11.77% (0.4) vs 14.11% (0.6), which is indeed consistent with the claim. However, the explanation for *why* 0.4 is worse than 0.6 is that "positive weights promote negative samples when all rewards are low," yet cw=0.4 gives *more* negative weight, which should *help* in the all-negative case. The authors' own argument would predict cw=0.4 to be better than cw=0.6, but empirically it is worse. This inconsistency is not addressed.

---

### Related Work (Section 6)

The related work section is thorough and situates the paper well. The distinction between ratio-free on-policy (REINFORCE-style) and ratio-free off-policy (AWR-style, wd1) is clarified correctly.

**Concern:** The paper notes that SPG (Wang et al., 2025a) and d2 (Wang et al., 2025c) are concurrent ratio-free methods but provides no empirical comparison with them. Given that all three papers address essentially the same problem, some comparison—even on a single benchmark—is needed to assess relative merit at ICLR.

---

### Limitations & Broader Impact (Appendix D)

The limitations section is placed in the appendix, reducing its visibility. Key limitations acknowledged include: (1) failure when all rewards are identical, (2) restriction to text modality, (3) bias from the d1-based approximation.

**Missing limitations:**
- The paper does not acknowledge that the impressive Sudoku/Countdown gains may partly stem from *wd1*'s more effective use of negative samples on these highly structured tasks (where most outputs are wrong at the start of training, making negative sample learning particularly impactful), rather than from the ratio-free design per se.
- The mismatch between theory (monotonic improvement with exact likelihoods, β>0 reference regularization) and practice (biased approximation, β=0) is not listed as a limitation.
- The approximation of the geometric mixture in logit space (unverified) is not listed.

---

### Overall Assessment

*wd1* addresses a genuine and well-motivated problem: policy ratio computation in diffusion-based LLMs is unstable due to likelihood approximation errors, and eliminating the ratio is a principled response. The theoretical contribution—connecting reverse-KL policy optimization to energy-guided discrete diffusion training—is elegant and correct, though the practical method (with the ad-hoc w^− term and β=0) diverges meaningfully from the theoretical framework. The empirical results are mixed: the gains on Sudoku and Countdown are striking (+59pp), but the core math benchmarks (GSM8K, MATH500) show no improvement with *wd1* (LoRA) over a reproduced *d1*. The *wd1*++ results on MATH500 (44.2%, SOTA) are the most compelling contribution but are obtained with substantially more compute, different training data, and full fine-tuning, making them difficult to disentangle from confounds. The absence of ratio-free on-policy baselines (REINFORCE/RLOO, SPG) is a notable gap for a paper whose primary claim is the benefit of being ratio-free. For ICLR, I view the paper as borderline: it makes a real contribution to the nascent area of RL for diffusion LLMs, but several experimental and theoretical gaps need to be addressed before the claims can be considered well-supported. The practical insights (the negative sample penalty, denoising-stepwise training) are likely to be useful to the community regardless.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **wd1**, a reinforcement learning method for diffusion-based large language models (dLLMs) that reformulates policy optimization as a weighted log-likelihood objective, effectively avoiding the computationally expensive and variance-prone estimation of policy ratios required by methods like diffusion-based GRPO. The authors extend this to **wd1++**, which leverages intermediate denoising steps, and claim state-of-the-art performance on reasoning benchmarks (MATH500, GSM8K, Sudoku) with significantly reduced computational overhead and no requirement for pre-trained supervised fine-tuning (SFT).

### Strengths
1.  **Addresses a Critical Bottleneck:** The paper correctly identifies and addresses the intractability of likelihood ratios in dLLM policy optimization. By deriving a ratio-free objective (weighted log-likelihood), it avoids the exponential amplification of approximation errors inherent in estimating policy ratios like `pi / pi_old`, which is a valid and significant theoretical contribution.
2.  **Improved Training Efficiency:** Empirically, `wd1` demonstrates substantial efficiency gains, specifically by eliminating the SFT stage required by the baseline `d1` and reducing the number of FLOPs and forward passes per step (Table 2). The claim of `10x` fewer rollouts in `wd1++` to achieve similar performance on Math is impressive if reproducible.
3.  **Strong Theoretical Grounding:** The derivation connecting the weighted objective to energy-guided discrete diffusion and negative sample unlearning (Theorem 1 and Remark 2) provides a solid theoretical framework that bridges RL with diffusion modeling concepts, beyond empirical engineering.
4.  **Robust Performance on Structured Tasks:** The method shows exceptional improvements on structured reasoning tasks like Sudoku (76.4% vs 17.6% for `d1`) and Countdown, suggesting it effectively learns to refine sequences without the instability of ratio-based importance sampling.

### Weaknesses
1.  **Dependency on Likelihood Approximation:** While `wd1` removes the *ratio* approximation, it still relies on the `d1` likelihood approximation (sampling at `t=1`) for the weighted log-likelihood itself (Section 3.2). The paper notes this in limitations, but this introduces a foundational bias in the log-likelihood `log pi_theta` that the theory assumes is accurate, which could still propagate error despite removing the ratio.
2.  **Dataset Misalignment in Comparisons:** The `wd1++` results on MATH500 utilize the OpenR1 dataset (Section 5.1), which is significantly larger and potentially higher quality than the standard GSM8K/MATH train splits used by baselines like `d1`. This makes the direct comparison of "SOTA performance" with "fewer rollouts" slightly confounding, as `wd1++` has access to more data.
3.  **Marginal Gains on Math vs. Sudoku:** While `wd1++` achieves 44.2% on MATH500 (beating `d1`'s 38.0%), the gap is less dramatic compared to the massive gains on Sudoku/Countdown. The improvement over `MDPO` (43.4%) on MATH is narrow (+0.8%) despite significant architectural differences, warranting a more detailed ablation on whether the gain comes from the objective or the OpenR1 data.
4.  **Sensitivity to Identical Rewards:** The paper admits in Limitations that the method fails when all completions in a group receive identical rewards. Given that RL for reasoning often involves binary verifiers, this collapse to uniform advantage is a practical risk that could hinder convergence in early training or hard tasks.

### Novelty & Significance
The paper demonstrates high novelty by adapting Advantage-Weighted Regression (AWR) concepts specifically to the discrete diffusion setting, where likelihood estimation is unique and costly. Interpreting the RL update as "energy-guided diffusion" combined with "unlearning" is a fresh perspective that connects two complex fields. The significance is high for the dLLM community, as it offers a path to align these models without the prohibitive overhead of reference policy likelihood estimation or SFT stages required by current SOTA methods. For ICLR, this represents a methodological advance that solves a specific intractability problem with strong empirical backing.

### Suggestions for Improvement
1.  **Clarify Dataset Disparities:** Please explicitly quantify the data overlap and volume between the dataset used for `d1` (standard GSM8K/MATH splits) and `wd1++` (OpenR1). Including a "standard split" baseline for `wd1++` would isolate the improvement due to the objective from the improvement due to data scale.
2.  **Analyze Likelihood Approximation Bias:** Since `wd1` still uses the `d1` likelihood approximation for `log_pi_theta`, a controlled experiment comparing `wd1` against a version using a higher-fidelity but costlier likelihood estimator would better isolate the benefit of the ratio-free objective versus the underlying likelihood model.
3.  **Ablate on Reward Uniformity:** Conduct a synthetic experiment or discuss mitigation strategies explicitly for the "identical rewards" limitation (e.g., adding uniform noise to rewards or curriculum learning) to show robustness in scenarios where the advantage signal vanishes.
4.  **Expand Baseline Comparison:** Include a comparison with `diffu-GRPO` (non-d1) to ensure the advantage of `wd1` comes from the weighted objective and not from the specific SFT pre-processing `d1` does. The table shows `d1` beats `diffu-GRPO`, but understanding if `wd1` beats `diffu-GRPO` without SFT is crucial for the "no SFT" claim.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Likelihood Approximator Sensitivity:** Compare `wd1` performance using an unbiased ELBO estimator versus the biased `d1` approximator. If `wd1` only works with the biased estimator, the claim that it "mitigates approximation error" is invalid.
2.  **End-to-End Efficiency Measurement:** Quantify the wall-clock time including sampling from the geometric mixture $\pi_{old\_ref}$ described in Appendix B.3. Table 2 excludes sampling cost, which Appendix B.3 admits incurs overhead, potentially negating the reported FLOP savings.
3.  **Stepwise Baseline Comparison:** Evaluate `d1` with identical intermediate step usage to match `wd1++`. Attributing `wd1++` gains to the objective rather than the stepwise data usage confounds the core contribution.
4.  **Task Discrepancy Investigation:** Investigate why base `wd1` yields +59% on Sudoku but only +2% on MATH500. If the method fails on sparse-reward reasoning tasks without stepwise extensions, the general reasoning claim is overstated.
5.  **Ratio-Free Baseline Comparison:** Compare `wd1` against standard REINFORCE or RLOO adapted for diffusion. Claiming benefits from being "ratio-free" requires showing superiority over other ratio-free methods, not just ratio-based ones like `d1`.

### Deeper Analysis Needed (top 3-5 only)
1.  **Empirical Gradient Variance:** Measure gradient variance during training for `wd1` vs `d1`. The core motivation is variance reduction; without empirical metrics, this remains theoretical speculation.
2.  **KL Divergence Trajectory:** Track $D_{KL}(\pi_\theta || \pi_{old})$ throughout training to verify stability. Without clipping or explicit constraints, you must prove the policy does not collapse or diverge beyond the trust region.
3.  **Mixture Sampling Feasibility:** Analyze the practical implementation cost of sampling from $\pi_{old\_ref}$. If this requires multiple forward passes per token, the "single approximation" efficiency claim is misleading.
4.  **Negative Weight Contribution:** Isolate the impact of $w^{[-]}$ on convergence speed versus final performance. The ablation shows performance drop, but it is unclear if this accelerates learning or merely prevents collapse.
5.  **Forward Process Assumption:** Verify the "identical forward process" assumption holds during RL fine-tuning. As $\pi_\theta$ updates, the forward process may drift, invalidating the energy-guided theoretical interpretation.

### Visualizations & Case Studies
1.  **KL Divergence Plot:** Plot KL divergence per training step for `wd1` and `d1`. This reveals whether the ratio-free objective maintains stability without explicit clipping mechanisms.
2.  **Training Cost Breakdown:** Show a bar chart breakdown of Sampling, Forward Pass, and Backward Pass time. This exposes whether the claimed FLOP reduction translates to actual wall-clock speedup.
3.  **Token Probability Heatmaps:** Visualize token probability changes for correct vs. incorrect answers over training steps. This confirms whether "negative unlearning" actively suppresses wrong tokens as claimed.
4.  **Failure Case Examples:** Show specific MATH problems where `wd1` underperforms `d1` or the baseline. Understanding where the method fails is critical for assessing robustness.
5.  **Gradient Norm Distribution:** Plot gradient norm distributions across training steps. High variance in gradients would contradict the claim of improved training stability.

### Obvious Next Steps
1.  **Unbiased Estimator Implementation:** Implement `wd1` with an unbiased ELBO likelihood estimator to isolate the objective's benefit from the approximation artifact.
2.  **Mixture Sampling Ablation:** Compare performance when sampling solely from $\pi_{old}$ versus the theoretical mixture $\pi_{old\_ref}$. This quantifies the necessity of the geometric mixture for stability.
3.  **Standardize Stepwise Extensions:** Develop a stepwise version of the `d1` baseline to ensure `wd1++` gains are not simply due to using intermediate denoising steps available to any method.
4.  **Scale to Larger Models:** Evaluate on larger models (e.g., 70B) to ensure results are not specific to the 8B scale. ICLR expects evidence of scaling behavior for LLM methods.
5.  **Clarify Sampling Code:** Explicitly document how $\pi_{old\_ref}$ sampling is implemented in the released code. Reproducibility depends on knowing if this is approximated or exact.

# Final Consolidated Review
## Summary

The paper proposes wd1, a reinforcement learning method for diffusion-based large language models (dLLMs) that reformulates policy optimization as a weighted log-likelihood objective, eliminating the need to compute policy ratios for importance sampling. Since dLLM likelihoods are intractable and require approximation, ratio computation (π_θ/π_old) amplifies approximation errors exponentially. wd1 derives from reverse-KL policy optimization, requiring only a single likelihood approximation per training step. The authors further extend this to wd1++, which leverages intermediate completions from the denoising process. Experiments on LLaDA-8B show strong improvements on Sudoku (+59pp over d1) and Countdown (+25pp), with competitive performance on GSM8K and MATH500.

## Strengths

- **Addresses a fundamental problem in dLLM RL.** The paper correctly identifies that computing policy ratios in diffusion models requires approximating multiple intractable log-likelihoods, and the ratio exp(ϕ_θ - ϕ_old) exponentially amplifies approximation errors. Figure 1 demonstrates this concretely: ELBO-based ratios have high variance while d1-based ratios exhibit systematic bias. Eliminating ratios is a principled approach to this problem.

- **Elegant theoretical connection.** Theorem 1 establishes that the weighted log-likelihood objective (WLL) is equivalent to training an energy-guided discrete diffusion model, where the energy function is the negative advantage. This bridges RL policy optimization with diffusion model theory, providing intuition for why the method works.

- **Strong empirical results on structured reasoning tasks.** wd1 achieves 76.4% on Sudoku vs. 17.6% for reproduced d1 (Table 1), and 51.2% on Countdown vs. 35.2% for d1. These are substantial improvements on tasks where most initial completions are wrong, making negative sample learning particularly valuable.

- **Computational efficiency gains.** Table 2 shows wd1 eliminates SFT (saving ~2 hours) and reduces per-step FLOPs by ~10% by avoiding likelihood evaluations for π_old and π_ref. The method requires μ likelihood evaluations per step vs. (μ+2) for d1.

- **Negative sample penalty is critical.** The ablation in Table 4 (wd1-P/w^+ only collapses to 6.69% on Sudoku) clearly demonstrates that the w^- term is essential for effective learning. This is an important empirical finding.

## Weaknesses

- **The w^- term is heuristic, not theoretically derived.** The theory in Section 4 establishes equivalence between WLL (w^+ only) and energy-guided diffusion (Theorem 1, Remark 1). The negative weight term w^- in the actual wd1 objective (Equation 8-9) is introduced in Section 3.2 as a post-hoc fix for two identified failure modes. While Remark 2 later provides an analogy to data unlearning (NegGrad), this is interpretation rather than derivation. The claim of "theoretical soundness" in the abstract overstates matters—the theory guarantees apply to WLL, not to the full wd1 objective.

- **Theory-practice gap: β=0 eliminates reference regularization.** Table 6 shows wd1 uses β=0.00, meaning π_ref regularization is entirely removed. The theoretical framework (Equation 4-5) assumes both λ and β are active, and the geometric mixture sampling from π_old^ref is central to the theory. In practice, wd1 is simply advantage-weighted regression with a negative penalty, executed on-policy (single gradient step per batch as shown in Table 8). This discrepancy between the elaborate theoretical framework and the actual implemented method should be stated clearly in the main text.

- **wd1++ reward for intermediate completions is unspecified.** Equation 10 trains on intermediate completions x_0|l for l∈{1,...,L}, but the paper does not explain how reward R(q, x_0|l) is defined for incomplete sequences. If intermediate completions inherit the final completion's reward, this is an implicit modeling assumption requiring justification. If the reward function naturally applies to partial sequences (e.g., Sudoku can check partial correctness), this should be explicitly stated.

- **wd1++ comparison confounded by multiple factors.** Table 3 shows wd1++ achieves 44.2% on MATH500, but this requires full fine-tuning, 8×A800 GPUs, OpenR1 training data, and batch size 64 vs. d1's batch size 48 (Table 8). The "10× fewer rollouts" claim (1280 vs. 30000) obscures that wd1++ uses 64 rollouts/step vs. d1's 4 rollouts/step—wd1++ samples 16× more per step, reaching its best checkpoint in 20 steps vs. 7500. The efficiency claim requires disentangling from these confounds.

- **Math reasoning improvements are modest relative to structured tasks.** On GSM8K-512 and MATH500-512 (LoRA setting), wd1 achieves 82.3% and 39.0% vs. d1's 82.0% and 38.0%—essentially tied. The substantial gains appear on Sudoku and Countdown where negative sample learning has high leverage (most early completions are wrong). This suggests the method's benefits may be task-dependent.

- **Likelihood approximation remains biased.** wd1 uses the d1 approximation (sampling at t=1 only), which Section 2.3 acknowledges is biased. While removing the ratio eliminates error amplification, the single approximation still introduces bias. The paper does not compare wd1 using an unbiased ELBO estimator to isolate whether gains come from the ratio-free property or from the particular approximation.

- **No comparison to other ratio-free methods.** The paper frames wd1 as "ratio-free" but does not compare to on-policy methods like REINFORCE or RLOO, which are inherently ratio-free. The Related Work section mentions SPG and d2 as concurrent ratio-free work but provides no empirical comparison. For a claim centered on the benefits of being ratio-free, demonstrating superiority over other ratio-free approaches would strengthen the contribution.

## Nice-to-Haves

- **Gradient variance measurements.** The core motivation is variance reduction from eliminating ratios, but no empirical variance metrics are provided. Reporting gradient norm variance during training would substantiate this claim.

- **KL divergence tracking.** Since wd1 removes ratio clipping and uses β=0 (no reference regularization), tracking D_KL(π_θ || π_old) during training would verify that the policy does not collapse or diverge excessively.

- **Likelihood approximator ablation.** Compare wd1 performance using the biased d1 approximator vs. an unbiased (but costlier) ELBO estimator to isolate the objective's benefit from approximation artifacts.

- **Stepwise baseline for fair comparison.** Evaluate d1 with the same intermediate completion usage as wd1++ to determine whether wd1++ gains come from the objective or simply from using more training signal per denoising trajectory.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **"+59% improvement" phrasing.** The critic objects to describing the Sudoku improvement as "+59%," but this phrasing (58.8 percentage points, ~59% absolute improvement) is acceptable and clearly refers to the Sudoku result specifically.

- **Statistical significance / single-run results.** Single-run evaluation is standard practice for large-scale LLM training; while multiple seeds would strengthen results, this is not a critical omission.

- **Missing concurrent work comparison (SPG, d2).** These works are truly concurrent, and the paper appropriately discusses them in Related Work. Demanding empirical comparison to methods published contemporaneously is unreasonable.

- **Reported vs. reproduced d1 baseline discrepancy.** The paper transparently presents both reproduced and reported numbers (Tables 1 and 7). The reproduced numbers sometimes exceed reported ones (Sudoku-512: 16.2 reproduced vs. 9.5 reported), so the comparison is not uniformly unfavorable to d1.

## Novel Insights

The connection between reverse-KL policy optimization and energy-guided discrete diffusion training (Theorem 1) is a genuine conceptual insight: optimizing the weighted log-likelihood objective effectively trains a diffusion model where the energy function is the negative advantage. This explains why high-advantage completions are amplified (pushed into higher-probability regions) and provides theoretical grounding for the approach beyond empirical engineering. However, the practical method diverges from this framework by adding the heuristic w^- term and setting β=0.

## Suggestions

- **Clarify wd1++ reward computation.** Explicitly state whether intermediate completions inherit the final completion's reward, or whether the reward function (e.g., format checker, partial credit) naturally applies to partial sequences.

- **Acknowledge theory-practice gaps in main text.** State upfront in Section 3 that β=0 is used in experiments, and that w^- is a heuristic extension of the theoretically-grounded WLL objective.

- **Isolate wd1++ contribution.** Run d1 with intermediate completion usage to fairly attribute performance gains to the objective vs. the training signal.

- **Analyze task-dependent effectiveness.** Discuss why structured tasks (Sudoku, Countdown) show larger gains than math reasoning—hypothesize about the role of negative sample density in early training.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
