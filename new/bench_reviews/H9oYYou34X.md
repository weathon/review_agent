Now I have enough information to write the consolidated final review. Let me synthesize everything carefully.

---

## Summary

This paper proposes **Markovian compression operators** for distributed optimization, where coordinate selection follows a Markov chain conditioned on previous iterations rather than being i.i.d. Two concrete compressors are introduced—**BanLast(K,m)** (which hard-bans recently used coordinates) and **KAWASAKI(K,b,π_Δ,m)** (which soft-penalizes them via a score). These are embedded in two algorithms: **MQSGD** (Algorithm 1) and its momentum-accelerated variant **AMQSGD** (Algorithm 2), with convergence results for non-convex, PL, and strongly convex settings. The paper argues empirically that history-aware compression improves practical convergence, while openly acknowledging that the theoretical bounds are strictly worse than standard unbiased compression.

---

## Claims and Support

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Markovian compressors are a novel family | **Supported.** Sec. 1.2 establishes no prior work combines compressed communication with Markovian compressor randomness. The framework is genuinely new, though it rests on only two instantiations. |
| 2 | BanLast and KAWASAKI satisfy asymptotic unbiasedness via ergodic Markov chains | **Supported.** Theorem 1 formally proves ergodicity and uniform stationary distribution for both compressors under stated conditions; proofs are in Appendix C. |
| 3 | MQSGD converges in non-convex and PL settings | **Supported as a feasibility result.** Theorem 2 and Corollary 1 correctly establish convergence. The bounds carry d²/m² and τ penalties, which the authors themselves acknowledge in Sec. 2.4. It is a "convergence-despite-Markovian-bias" result, not a rate-improvement result. |
| 4 | AMQSGD achieves faster convergence than MQSGD | **Partially supported.** Section 2.4 correctly identifies that the condition-number dependence improves from L/μ (MQSGD, Corollary 1) to (L/μ)^{2/3} (AMQSGD, Corollary 2). However, the comparison is not clean: Corollary 1 covers the PL setting while Corollary 2 covers strongly convex; mixing-time dependence worsens to τ^{4/3} in AMQSGD. The paper's own statement after Theorem 3—"constant step-size can attain sublinear convergence"—is misleading for a strongly convex result, where the exponential decay term actually gives linear convergence; this is a writing error. |
| 5 | Markovian compressors outperform existing methods empirically | **Weakly supported.** Figures 1–3 explicitly report "best runs are selected" and "all hyperparameters are fine-tuned." Table 1 appropriately reports mean ± std over 5 runs for CIFAR-10 and does show a genuine advantage for KAWASAKI, but the advantage for BanLast is marginal (88.0% vs 87.9%). Main logistic regression results rely exclusively on best-run curves. |
| 6 | History of transmissions accelerates optimization | **Supported as intuition.** Example 1 demonstrates a concrete 3× speedup in a toy single-coordinate gradient scenario. The broader claim is borne out empirically (with caveats) but is not theoretically explained. |

---

## Strengths

- **Genuinely novel compression concept with principled motivation.** The insight that i.i.d. sparsification can repeatedly miss the same coordinates while others go stale, and that a Markov chain over coordinate subsets can fix this, is both natural and previously unexplored in compressed distributed optimization. Section 1.2 confirms no prior work occupies this niche.

- **Non-trivial "stepping-back" proof technique.** The analysis in Equations (4)–(5) of Section 2.2 is a technically interesting contribution: using the mixing-time threshold τ to "step back" and then recover approximate unbiasedness via Fenchel-Young inequalities. This provides a reusable analysis template for any method involving asymptotically unbiased Markovian randomness.

- **Honest and transparent discussion of limitations (Section 2.4).** The authors explicitly expose the d²/m² vs d/m theory gap, the τ penalty, and—most unusually—acknowledge the logical contradiction between theory (prefer small K) and practice (larger K helps). This intellectual honesty is uncommon and strengthens the paper's credibility.

- **KAWASAKI removes BanLast's dimensional restriction** while maintaining ergodicity. Providing two compressors with different flexibility/constraint profiles demonstrates the design space is non-trivial.

---

## Weaknesses

### Fatal
*(None that individually invalidate the entire paper, but the two Major issues together substantially undermine the practical claim that motivates the work.)*

---

### Major

- **Best-run reporting as primary empirical evidence.** The captions for Figures 1 and 2 both state explicitly: "All hyperparameters are fine-tuned, and best runs are selected." This is not a minor presentation choice—it is the core experimental protocol for the logistic regression results and the main CIFAR-10 curves. Since KAWASAKI introduces three additional hyperparameters (K, b, π_Δ) relative to Rand, selecting best runs gives the proposed method a strictly larger tuning advantage than baselines. Table 1 partially rescues the CIFAR-10 comparison with mean ± std over 5 runs, but the logistic regression claim, which spans four datasets and multiple algorithms, rests entirely on best-run curves with no uncertainty quantification. The paper cannot substantiate "practical superiority" under this protocol.

- **Logical theory–practice contradiction left unresolved.** Section 2.4 explicitly states: "it follows from Theorems 2 and 3 that the convergence rate is improved as τ (and, consequently, K) diminishes… while using a large K will theoretically give poorer convergence, in practice algorithms with non-zero values of K perform better." This is a fundamental gap: the paper's theoretical framework predicts the opposite of what the experiments show. The paper offers no analysis of when or why history dependence helps (e.g., gradient sparsity, temporal correlation structure, heterogeneity level). Acknowledging the contradiction is admirable, but without even a partial theoretical resolution or ablation study that characterizes the regime, the paper currently offers theory and experiments that point in opposite directions rather than a coherent account of the phenomenon.

---

### Minor

- **Missing comparison with error-feedback (EF) methods.** EF21 and similar biased-compressor methods with error compensation also exploit "past information" to correct bias, making them the most natural competitors for the proposed Markovian approach. The paper compares only against unbiased compressors (Rand, PermK, Natural) and DIANA. Without an EF comparison, the contribution's practical positioning is incomplete—it is unclear whether the benefit comes from history-dependence per se or is obtainable via standard error-feedback at lower complexity.

- **BanLast applicability restriction.** The requirement d ≥ (K+1)m (confirmed in Definition 5 and Theorem 1) severely limits BanLast in high-compression regimes (large m/d ratio). The paper acknowledges this and introduces KAWASAKI as the fix, but does not characterize how often the restriction binds in the reported experiments.

- **Ambiguous framing of AMQSGD's convergence type.** Theorem 3 for strongly convex objectives contains a decaying exponential exp[−(T−τ)√(p²μγ/3)] F_τ, which is linear (exponential-rate) convergence—yet the paragraph immediately following the theorem states "AMQSGD with constant step-size can attain sublinear convergence." This self-contradictory framing will confuse readers. The step-tuned corollary shows an iteration complexity that scales polynomially in 1/ε due to the σ²-neighborhood floor, which is why the paper may mean "sublinear in overall accuracy," but this needs clarification.

- **Hyperparameter sensitivity not characterized in main text.** The three hyperparameters of KAWASAKI (K, b, π_Δ) are fine-tuned per experiment, and the only practical guidance is the heuristic K*(α) ≈ 0.73·α derived from Example 1's toy setting. The appendix contains additional tuning analysis, but no sensitivity curves or robustness evidence appear in the main text. This makes it difficult to assess whether KAWASAKI's gains are robust or artifact-sensitive.

---

### Trivial

- The abstract states "Intuitively, this should accelerate the convergence" as motivation but "practical results demonstrate the superiority" as a concluded fact—the latter is too strong and should be hedged consistently with the experimental protocol.
- Theorem 3's dependence on Δ_τ (which involves a sum of gradient norms and distances over the first τ steps) makes the bound less interpretable in the main text; some intuition about its magnitude would help.

---

## Nice-to-Haves

- **Controlled ablation over K (history window size)**: A plot of final accuracy vs. K for fixed datasets and compressors would directly visualize the theory-practice tension and help practitioners choose K without full grid search.
- **Characterize mixing time τ for experimental configurations**: Computing τ_mix for the specific BanLast and KAWASAKI settings used in experiments would ground the theoretical bounds and help readers assess when the theory and practice might re-align.
- **Apply Markovian compressors in a variance-reduced method with theory**: The paper mentions DIANA experiments but no theory for variance-reduced methods. This is acknowledged as future work; even partial results (e.g., for homogeneous data) would substantially strengthen the contribution.
- **Comparison on larger-scale tasks** (e.g., ImageNet or language model fine-tuning) to validate that the CIFAR-10 gains are not limited to the small-scale regime.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"KAWASAKI is underspecified / definition is garbled" (Harsh Reviewer, Sec. 2.1).** Definition 6 and the surrounding text (lines 119–127) are parseable in the extracted PDF. The formula for p̃_j^t is present, the activation function examples are listed, and Theorem 1 gives explicit conditions. Any rendering artifacts are a parser issue, not an author error. Removed.

- **"The paper cannot show this is a framework rather than two ad hoc heuristics"** (Harsh Reviewer, Claim 1). This is scope creep. The paper never claims to present a broader design methodology beyond the two compressors; the "family" language refers to the general Assumption 5 template. Criticizing the paper for not also providing a general design theory exceeds its stated scope. Removed.

- **"Unfair baseline comparison where baselines have fewer hyperparameters"** (Harsh Reviewer framing of tuning parity). The legitimate version of this concern—best-run reporting—is already captured under Major weaknesses. The additional framing about "asymmetry favors the author's method" could be taken as an unfair-comparison criticism, which per our rules is only removed if asymmetry favors the baseline. Here it is ambiguous but the core concern is the best-run protocol, which is kept. Removed as a separate point, subsumed.

- **"All experiments are simulated, no true multi-node results"** (Spark reviewer). Single-machine simulation of distributed optimization is standard in the compressed-communication literature and does not constitute a reproducibility flaw. Removed.

- **"No confidence intervals on logistic regression figures"** — this is a legitimate concern but is subsumed within the Major weakness on best-run reporting, and requesting CIs for every figure in a distributed optimization submission goes slightly beyond field norms. Subsumed.

---

## Novel Insights

The paper's most intellectually interesting contribution is not the compressors themselves but the explicit documentation of a theory–practice inversion: the theoretical analysis (via the stepping-back technique) predicts that smaller K (faster mixing) is better, while empirically larger K consistently wins. This mirrors known gaps in Markovian noise literature (Bresler et al., 2020 lower bounds), but here the gap is unusually sharp because the very mechanism producing practical gains—long-range coordinate avoidance—is exactly what degrades the theoretical bounds. This suggests that the uniform noise bound used in all Markovian-stochasticity analyses (acknowledged in Sec. 2.4) may be fundamentally incompatible with capturing the structured benefit of history-dependent sampling, and that a conditional variance analysis exploiting the Markov structure of the compressor could be a genuinely productive theoretical direction.

---

## Suggestions

1. **Replace best-run figures with average-over-seeds figures** (with standard-deviation bands or box plots) for all main claims. If best-run curves are kept for illustration, they must be supplementary only.
2. **Equalize tuning budgets** across all baselines and report the search protocol explicitly (search space, number of trials, or grid dimensions).
3. **Fix the "sublinear convergence" sentence** after Theorem 3: the exponential decay term gives linear convergence; the σ²-neighborhood floor only makes the limiting accuracy suboptimal. This is a writing fix that prevents a mis-read of the paper's strongest result.
4. **Add an EF21 or EF-SGD comparison** to one main experiment to position the contribution relative to biased-compressor methods.
5. **Add a K-sweep ablation** (convergence speed vs. K for fixed b and π_Δ) in the main text to directly visualize the theory-practice tension and provide practitioners with actionable guidance.

---

## Score and Decision

**Originality:** High. The Markovian compressor concept is genuinely new and the stepping-back proof technique is a non-trivial contribution.

**Importance of research question:** Medium-high. Communication efficiency in distributed learning is a pressing problem; history-aware compression is a natural and underexplored direction.

**Claims vs. support:** Moderate. The theoretical convergence claims are properly supported. The practical superiority claim is not, due to the best-run reporting protocol.

**Soundness of experiments:** Below standard. Best-run selection as primary evidence is a methodological flaw that directly undermines the paper's practical contribution.

**Clarity:** Good overall; the discussion section is particularly clear. The framing around Theorem 3's convergence type is a specific clarity failure.

**Value to research community:** Moderate. The idea and the stepping-back template are useful starting points. The current submission does not deliver a convincing practical case, and the theory explicitly says it cannot yet explain why the method works.

**Overall:** The paper has a real idea, honest theory, and a non-trivial proof technique. However, the central practical claim is supported only by best-run experiments, the theory and experiments point in opposite directions with no resolution, and important baselines (EF methods) are absent. In its current form, the empirical contribution does not meet the bar for publication.

**Score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>