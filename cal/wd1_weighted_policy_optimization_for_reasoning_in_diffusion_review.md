=== CALIBRATION EXAMPLE 23 ===

# Final Consolidated Review
## Summary
This paper proposes **wd1**, a ratio-free RL objective for diffusion language models that replaces policy-ratio optimization with a **weighted log-likelihood** objective, motivated by reverse-KL-regularized policy optimization. The paper’s main practical claim is that avoiding explicit ratio estimation reduces both approximation error amplification and training cost for dLLM post-training; empirically, `wd1` improves strongly over reproduced `d1` on Sudoku/Countdown and modestly on GSM8K/MATH500, while the stepwise variant `wd1++` reaches stronger math results with fewer training steps/rollouts.

## Strengths
- **Targets a real dLLM-specific bottleneck with an appropriately specialized objective.** The paper directly addresses the intractable-likelihood issue in diffusion RL and makes the key observation that ratio-based objectives can amplify approximation error: Appendix A.1 explicitly contrasts exponential sensitivity in diffusion-GRPO-style ratios with linear error in `wd1`. This is a concrete and relevant technical contribution for dLLM RL rather than a generic rebranding of RLHF.
- **The negative-sample term is a substantive contribution and is empirically validated.** The extension from the weighted log-likelihood objective (Eq. 6/7) to the full `wd1` loss (Eq. 8/9) is not cosmetic: the paper identifies a genuine failure mode of positive-only weighting—namely that all sampled completions can have their likelihood increased, including uniformly poor ones—and proposes the complementary `w^-` term to suppress low-advantage completions. The ablation is striking: `wd1-P (WLL)` is far worse than full `wd1` in Table 4, and Appendix C.2 further supports that balancing positive and negative weighting matters.
- **There is a meaningful theory-to-method bridge, especially for the diffusion setting.** The derivation from reverse-KL regularized policy optimization to a weighted likelihood objective (Sec. 3.1, Proposition 2) is coherent, and the diffusion-specific interpretation in Sec. 4 is more than decorative: Theorem 1 connects the proposed objective to advantage-weighted denoising score/cross-entropy training. Even if some of the “unlearning” framing is interpretive, the energy-guided diffusion connection is a useful conceptual contribution.
- **Empirically, the method appears materially more efficient per RL update than ratio-based baselines.** Table 2 shows a concrete reduction in RL-step runtime and FLOPs, consistent with the algorithmic simplification of removing old/reference likelihood evaluations from the update. The paper also demonstrates that `wd1` can work **without SFT**, unlike the `d1` pipeline it compares against.
- **The paper surfaces an interesting empirical phenomenon specific to diffusion RL:** positive-only weighted regression appears ineffective, while explicit negative suppression is highly beneficial. This is potentially a useful takeaway for future dLLM RL work beyond the specific algorithm here.

## Weaknesses

### Major:
- **The strongest headline gains are concentrated on relatively non-standard planning/constraint benchmarks, while improvements on standard math reasoning benchmarks are modest for `wd1`.** Table 1 shows enormous jumps on Sudoku and Countdown, but on GSM8K/MATH500 the gains of `wd1` over reproduced `d1` are small (e.g., 82.3 vs 82.0 on GSM8K, 39.0 vs 38.0 on MATH500 at 512). This does not invalidate the method, but it does mean the paper’s most dramatic claims are not primarily supported on the most standard reasoning benchmarks. In particular, the abstract’s emphasis on large percentage improvements could give a stronger impression of broad reasoning gains than the math results alone justify.
- **The `wd1++` comparisons are not cleanly controlled against baselines because the training setup changes.** The paper states that `wd1` is trained on the `d1` setup/datasets, while `wd1++` uses a dataset from He et al. sampled from OpenR1 and also uses a different verifier/evaluation pipeline in some full-parameter experiments (“we leverage a more effective system prompt and Math-Verify…” in Appendix B.1). Since Table 3 is where the paper makes its strongest “state-of-the-art math performance” claim, this mismatch makes it hard to isolate how much of the gain comes from the **stepwise objective itself** versus data / verifier / prompt changes. This is an important comparability issue.
- **The method still relies on a biased likelihood approximation, so the paper reduces one approximation problem without fully resolving the broader one.** The paper is transparent about this in several places, including Sec. 3.2 (“Likelihood approximation in `d1` is directly applicable to `wd1`”) and the limitations section: “Our approach relies on the d1-based approximation, which is computationally efficient but introduces bias.” This does not negate the benefit of removing ratios, but it does narrow the scope of the claim: `wd1` avoids ratio-induced error amplification rather than eliminating likelihood-approximation bias altogether. A more direct analysis of how residual absolute approximation bias affects optimization would strengthen the technical story.
- **The “state-of-the-art” and efficiency claims would be more convincing with stronger statistical and end-to-end controls.** The gains over the strongest math baseline in Table 3 are fairly small (e.g., 44.2 vs 43.4 on MATH500), yet no multi-seed variance or confidence intervals are provided. Similarly, the efficiency discussion emphasizes per-step runtime/FLOPs and total rollouts, but a more direct wall-clock-to-performance comparison under tightly matched setups would better substantiate the practical significance.
- **The stepwise extension `wd1++` is intuitive but not theoretically pinned down to the same degree as `wd1`.** In Sec. 3.3, the method expands the optimization set to include intermediate denoising completions and uses them in the same weighted objective. This is a plausible way to exploit unused signal, but the paper does not provide a corresponding derivation showing that assigning terminal reward-derived weights across correlated intermediate states is the principled solution of an underlying RL objective. As a result, `wd1++` currently reads more as a strong heuristic extension than as a theoretically settled one.

### Minor
- **Evaluation is limited to essentially one model family (LLaDA-8B-Instruct).** Given the paper’s claims about diffusion RL generally, broader evidence across another dLLM architecture or masked diffusion model would improve confidence in generality.
- **The paper does not directly test whether the advantage of ratio-free optimization persists under stronger likelihood estimators.** Since the core claim is partly that ratio computation is especially problematic under coarse approximations like `d1`, it would be informative to compare `wd1` and ratio-based baselines under a higher-fidelity ELBO/DCE estimator on a smaller-scale experiment.
- **The reward setups include substantial shaping / formatting components on some tasks (Appendix B.2), which may contribute to the very large gains on structured benchmarks.** This is not inappropriate, but it makes it harder to disentangle improvements in reasoning from improvements in reward-format alignment.
- **The theoretical “negative sample unlearning” interpretation is suggestive, but less central and less rigorously supported than the main weighted-likelihood derivation.** Remark 2 is interesting, but currently reads more as an analogy than as a key theorem-backed contribution.

### Trivial
- **Some claims would benefit from tighter wording.** For example, “state-of-the-art math performance” is technically plausible from Table 3 but should be phrased more cautiously given the setup differences and small margins.
- **A direct visualization of how `w^-` changes the probability of poor completions over training would make a central mechanism more concrete.**

## Nice-to-Haves
- Run a controlled experiment where `wd1`, `d1`, and `wd1++` use the **same training data, verifier, prompting setup, and rollout budget**, especially for GSM8K/MATH500.
- Add a small-scale experiment using a **higher-fidelity ELBO/DCE likelihood estimator** to test whether the benefit of `wd1` is fundamentally due to removing ratios rather than just pairing well with the `d1` approximation.
- Report **multi-seed results** for the key math benchmarks and for the strongest baseline comparisons.
- Include a direct analysis of **likelihood evolution for high- vs low-advantage samples**, which would concretely validate the proposed negative-sample suppression mechanism.
- Evaluate on at least one additional dLLM backbone to test generality.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that setting `β=0` “invalidates” the theory or reduces the method to an ungrounded objective.** This overstates the issue. The paper explicitly says: “Since previous works … demonstrated that the reference policy is empirically unnecessary, we set `β = 0` and `λ = 1` to eliminate `π_ref` in practice.” When `β=0`, the geometric mixture reduces to `π_old`, which is still a valid special case of the derived objective rather than a contradiction of it. The real issue is not invalidity, but that the empirically strongest setup drops part of the broader reverse-KL/reference-policy framing.
- **Claim that the paper’s theory is merely “post-hoc” and therefore not meaningful.** The energy-guided diffusion connection in Sec. 4 is mathematically developed and relevant to the diffusion setting. One can argue it is more interpretive than operational, but dismissing it as empty is not justified by the paper text.
- **Criticism based on doubting the practical existence/release/verification of cited baselines, models, or datasets.** Removed per instruction.
- **Missing comparison to autoregressive models.** This is outside the paper’s stated scope, which is RL fine-tuning for diffusion LLMs.
- **Concern that multiple inner updates with on-policy samples are inherently invalid.** The paper is explicit that `wd1` is “similar to AWR … inherently an off-policy loss” (Sec. 6), so this is not a misunderstanding the authors overlooked.
- **Potential overfitting/cherry-picking concern based only on the paper improving Sudoku by training longer than an earlier technical report version.** The paper transparently states the longer training schedule in a footnote; that alone is not evidence of cherry-picking.

## Novel Insights
The most interesting cross-review insight is that the paper’s strongest scientific contribution may not be the “ratio-free” slogan itself, but the empirical and conceptual case that **diffusion RL benefits disproportionately from explicit negative reinforcement**. The ablations suggest that for dLLMs, simply reweighting positive samples is not enough; suppressing low-advantage completions appears central. This aligns with the paper’s observation that likelihood-maximizing objectives can otherwise reinforce merely plausible but suboptimal denoising trajectories. If this phenomenon holds more broadly, it could shape how future RL objectives for diffusion LMs are designed, independent of the exact reverse-KL derivation.

## Suggestions
- **Tighten the empirical claim hierarchy.** Emphasize that `wd1` shows clear advantages in efficiency and strong gains on Sudoku/Countdown, while math gains for plain `wd1` are modest; present `wd1++` as a promising extension rather than fully attributing all math gains to the core method.
- **Standardize the `wd1++` comparison setting** with MDPO/`d1` as much as possible, or clearly separate “algorithmic” from “pipeline” improvements.
- **Quantify residual likelihood-approximation bias** within `wd1`, ideally by comparing against a stronger estimator on a smaller-scale benchmark.
- **Add multi-seed reporting** for GSM8K/MATH500 and for the key Table 3 comparisons.
- **Strengthen the analysis of `wd1++`** either with a derivation tailored to intermediate denoising states or with empirical evidence that intermediate-state weighting is not merely exploiting correlated samples.
- **Show direct probability suppression of bad completions** over training to support the negative-sample mechanism beyond final accuracy tables.

In total, this is a **novel and technically interesting paper with a real method contribution**, and the core `wd1` idea appears both sound and useful. The main reasons it falls short of being fully compelling at ICLR level are not that the method is unsound, but that the **strongest empirical claims are not yet supported by equally strong controlled evidence**, especially for `wd1++` and for broad reasoning gains on standard math benchmarks.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
