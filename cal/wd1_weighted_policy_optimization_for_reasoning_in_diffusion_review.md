=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary

The paper proposes **wd1**, a weighted log-likelihood policy optimization method for reinforcement learning fine-tuning of diffusion-based large language models (dLLMs). By deriving a ratio-free objective from reverse-KL regularized policy optimization, wd1 eliminates the need for likelihood ratio estimation (which suffers from exponential error amplification in dLLMs), requiring only a single likelihood approximation per step. A negative-weight term \(w_-\) actively penalizes low-advantage completions. The extension **wd1++** leverages intermediate denoising completions for stepwise optimization. Theoretical analysis connects the objective to energy-guided discrete diffusion training combined with negative sample unlearning.

## Strengths

- **Ratio-free objective specifically motivated by dLLM intractability.** The weighted log-likelihood formulation (Eq. 7–8) eliminates the need for computing policy ratios \(\pi_\theta/\pi_{\text{old}}\) and \(\pi_\theta/\pi_{\text{ref}}\), which in dLLMs require expensive, error-prone likelihood approximations whose errors amplify exponentially in the ratio (Appendix A.1). This is a targeted and well-motivated design choice—not a generic AWR port—since for AR models where likelihoods are tractable, ratio-free methods offer no particular advantage, as the authors note in Section 6.

- **Principled theoretical interpretation.** The proof that the WLL objective is equivalent to Advantage-Weighted Denoising Concrete Score Matching (Theorem 1, Remark 1) bridges RL policy optimization and score-based generative modeling for discrete diffusion. The interpretation of the negative-weight term as data unlearning via ELBO minimization on a Boltzmann distribution over low-advantage samples (Remark 2, Appendix D.1) provides genuine theoretical insight beyond standard AWR analysis.

- **Elimination of SFT with empirical validation.** Unlike d1, wd1 does not require a supervised fine-tuning stage. Table 4 shows wd1-SFT actually performs *worse* than wd1 on Sudoku and Countdown, confirming that the method's advantage is intrinsic rather than SFT-dependent. This simplifies the training pipeline and reduces cost (Table 2: 0 hrs SFT vs. 2.01 hrs for d1).

- **Comprehensive ablations on the dual-weight mechanism.** Section 5.2 and Appendix C.2 systematically validate the necessity of the \(w_-\) term (Table 4: removing it drops Sudoku from 76.4% to 6.69%), the equal weighting of positive/negative branches (Table 9: \(\lambda = 0.5\) outperforms 0.4 and 0.6), and the \(\psi\) parameter sensitivity (Figure 4).

## Weaknesses

### Major:

- **Data confound in wd1++ experiments undermines the SOTA claim.** Table 3 (left) compares wd1++ against d1, MDPO, and other baselines on GSM8K/MATH500. However, wd1++ is trained on data subsampled from the **OpenR1 dataset** (distilled reasoning data from stronger models), while d1 and the reproduced baselines are trained on **original GSM8K/MATH train splits**. OpenR1 data is generally higher quality. The 44.2% MATH500 and 84.5% GSM8K results for wd1++ may therefore reflect a data advantage rather than a purely algorithmic one. No control experiment training d1 or MDPO on the same OpenR1 data is provided to isolate the algorithmic contribution. This significantly weakens the SOTA claim.

- **wd1++ results in the abstract are not clearly presented in the main comparison table.** The abstract prominently claims wd1++ achieves 44.2% on MATH500 and 84.5% on GSM8K, but Table 3 (left) does not contain a clearly labeled wd1++ row with these exact numbers. The closest entries are wd1 (full) at 82.7/43.6. This presentation gap makes it difficult to verify and contextualize the headline claims.

- **No standard deviations or multiple-seed results reported.** All results in Tables 1 and 3 are single numbers. This is particularly concerning for the Sudoku claim (+59% over d1), where base model performance is only 6.7% and large relative gains may have high variance. Figure 4 (right) shows that changing the random seed for MATH training alters the reward dynamics (removing an early drop), suggesting seed sensitivity. Without error bars, the statistical significance of improvements—especially the modest ones on GSM8K (80.8 vs. 80.7, 82.3 vs. 82.0)—cannot be assessed.

### Minor:

- **Likelihood approximation bias not empirically quantified.** wd1 still relies on the d1-style likelihood approximation (sampling at \(t=1\)), which is acknowledged as biased. While Appendix A.1 shows the error propagation is linear rather than exponential, the *magnitude* of this bias and its practical impact on training dynamics are not measured. A small experiment comparing the \(t=1\) estimator against ELBO with larger sample sizes on a held-out set would clarify this trade-off.

- **Sudoku benchmark uses 4×4 puzzles, limiting reasoning claims.** The primary demonstration of wd1's advantage (76.4% vs. 17.6% for d1) comes from 4×4 Sudoku, which has a small search space and may reflect formatting/decoding artifacts rather than deep reasoning. The base model's 6.7% on 4×4 Sudoku is surprisingly low for an 8B model, suggesting the task may test constraint satisfaction under specific decoding conditions rather than generalizable reasoning. The math benchmark improvements are much more modest.

- **Hardware inconsistency for cost comparisons involving wd1++.** Table 2 reports wd1 vs. d1 costs on 4×A100 (fair comparison), but Table 3 (right) and Appendix B.6 indicate wd1++ was trained on 8×A800 while d1 used 4×A100. The "10× fewer rollouts" claim is valid as a sample-efficiency metric, but wall-clock time and total FLOP comparisons for wd1++ vs. d1 are not provided on equal hardware.

### Trivial:

- The claim of "ratio-free" in the abstract could be slightly more precise—wd1 avoids *ratio estimation* but still requires a single likelihood approximation. However, the paper body is sufficiently clear about this distinction.

## Nice-to-Haves

- Include concurrent ratio-free dLLM methods (SPG, d2) as experimental baselines, since they address the same problem from different angles (policy gradient vs. weighted likelihood).
- Empirically measure the variance of loss estimates during wd1 vs. d1 training to directly validate the central variance-reduction claim.
- Analyze whether the negative-weight term causes mode collapse or loss of general language capabilities on non-reasoning tasks.
- Provide results at longer sequence lengths (>512 tokens) to validate the claim that efficiency gains widen with sequence length.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Ratio-free claim is misleading since likelihood is still approximated"** — The paper is explicit that wd1 avoids ratio computation, not likelihood approximation. This is a mischaracterization of the claim.
- **"Missing comparison with AR-based GRPO baselines"** — Scope creep. The paper's stated scope is RL for dLLMs; comparing against AR models would require a fundamentally different experimental setup and addresses a different question.
- **"Missing broader impact / dual-use discussion"** — Not standard or required for ICLR; this is scope creep.
- **"Data contamination or memorization concerns on GSM8K/MATH"** — Entirely speculative without evidence; the paper uses standard train/test splits.
- **"Insufficient baselines compared to value-based methods or other exploration strategies"** — The paper compares against the most relevant dLLM RL baselines (d1, MDPO, SDPO, TCR). Demanding baselines from other communities (e.g., Atari) is not reasonable.
- **"Reproducibility concerns about undisclosed hyperparameters"** — The paper provides detailed hyperparameters in Table 6 and Appendix B.4.
- **"Formatting/style nitpicks about notation heaviness or equation density"** — Soft rule: remove formatting nitpicks. The notation is standard for the field.
- **"Mode collapse from negative weighting"** — Raised without empirical evidence; purely speculative.
- **"Missing related works"** — Hard rule: do not mention missing related works.

## Novel Insights

The theoretical equivalence between the weighted log-likelihood objective and energy-guided discrete diffusion (Remark 1) reveals a deeper structural property: when applying reverse-KL-constrained policy optimization to masked diffusion models, the optimal policy naturally induces energy guidance at *all* diffusion timesteps (Lemma 1), not just at the clean-data level. This means the advantage function acts as an energy that propagates through the entire denoising chain—a connection that, while derivable from existing energy-guided sampling theory (Lu et al., 2023), has not been explicitly recognized in the RL-for-dLLMs literature. This raises an interesting question: if the guidance is present at all timesteps, could intermediate-time advantage estimates (rather than broadcasting the final-completion advantage to all steps, as wd1++ currently does) provide even richer learning signals?

## Suggestions

- Run d1 and MDPO on the same OpenR1 data as wd1++ to isolate the algorithmic contribution from the data advantage. This is the single most important experiment for validating the SOTA claim.
- Report mean and standard deviation across at least 3 seeds for all main results, especially for Sudoku where the relative gains are largest.
- Add a clearly labeled wd1++ row in Table 3 (left) with the 44.2% / 84.5% numbers from the abstract, along with training configuration details matching the rigor of the wd1 experiments.
- Include a small-scale experiment measuring the bias of the \(t=1\) likelihood estimator vs. ELBO with varying sample sizes to empirically characterize the bias-variance trade-off that motivates wd1's design.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
