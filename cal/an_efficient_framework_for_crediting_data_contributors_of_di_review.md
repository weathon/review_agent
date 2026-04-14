=== CALIBRATION EXAMPLE 16 ===

# Final Consolidated Review
Now I have enough context. Let me synthesize the final review.

---

## Summary

This paper introduces a framework for attributing *global* properties of diffusion models (e.g., image quality, demographic diversity, aesthetic score) to *data contributors* (groups of training data from a single source), a problem distinct from existing work that focuses on local, per-image attribution to individual data points. The core technical contribution is *sparsified fine-tuning*: prune the full-data model and fine-tune the sparse network on each Shapley subset rather than retraining from scratch, achieving 5.3–18.6× speedups while outperforming all baselines on the linear datamodeling score (LDS) across three datasets and model architectures.

---

## Strengths

- **First framework for global contributor attribution via Shapley values in diffusion models.** The paper cleanly identifies and formalizes two gaps in the literature: (1) local vs. global property attribution and (2) datum-level vs. contributor-level attribution. No prior work addresses both simultaneously for diffusion models, making this a genuine first-mover contribution rather than an incremental one.

- **Large empirical gains over all baselines.** On CIFAR-20 and ArtBench, sparsified-FT Shapley achieves 61.48% and 61.44% LDS respectively—roughly 2× better than the best competing method (LOO at 30.66% on CIFAR-20). These are not marginal improvements; the method substantially re-orders the leaderboard and exposes the systematic failure of aggregated TRAK-style methods (which sometimes yield negative LDS, i.e., worse than random) for global attribution.

- **Computational efficiency validated by budget-matched comparison.** Figure 2 is a particularly strong piece of evidence: under the same computational budget, sparsified fine-tuning outperforms both plain fine-tuning and full retraining at every budget level. This directly justifies the approximation strategy as an accuracy–efficiency Pareto improvement, not just a speed trade-off.

- **Counterfactual analysis on CIFAR-20 and CelebA-HQ is convincing.** Removing the top 40% of identified contributors causes a −23.23% drop in Inception Score (vs. −14.95% and −17.30% for best baselines) on CIFAR-20; retaining only the top 60% yields a +16.98% gain (vs. −9.45% and +9.51% for baselines). This shows the scores predict real downstream consequences, not just correlations.

- **Qualitative case studies reinforce the attribution logic.** The analysis in Section 4.5 ("Who are the top contributors?") shows that identified contributors correspond to classes with lower image entropy (CIFAR-20), underrepresented demographic clusters (CelebA-HQ), and high-vividness art styles (ArtBench), providing interpretable evidence that the Shapley scores capture meaningful signals.

---

## Weaknesses

### Fatal
None.

### Major

- **CelebA-HQ LDS of 26.34% is substantially lower and unexplained.** The best competing method on CelebA-HQ (Pixel similarity max) achieves 21.70%, so the gap is not large. The paper notes the discrepancy but offers no mechanistic explanation. Is the entropy metric based on BLIP-VQA clustering less discriminative for subset-level differences? Is the LDM with 274M parameters harder to fine-tune for 500 steps toward meaningful subset-specific equilibria? Is the 50-contributor CelebA setting structurally harder (e.g., celebrities are more overlapping in feature space than classes)? Without a diagnostic, practitioners cannot know whether this is a fundamental limitation of the approach or a fixable hyperparameter issue. This matters because CelebA-HQ is precisely the most policy-relevant setting (demographic fairness attribution).

- **Missing ablation on pruning ratio.** The paper uses fixed compression ratios (44%, 74%, 49%) chosen without justification. Proposition 1 implies a trade-off: higher sparsity → cheaper subset evaluation → more Shapley samples, but also larger $B$ (the pruning approximation error). There is no experiment varying sparsity to characterize this trade-off. This is a critical missing piece for practitioners who must choose a pruning ratio, and it would also directly substantiate the theoretical claim that pruning quality matters.

- **ArtBench counterfactual evaluation is unconvincing.** When removing the top 40% of contributors, the changes are: D-TRAK +0.58%, Pixel sim max −0.05%, Aesthetic avg −1.27%, Sparsified-FT Shapley −1.86%. These differences are within noise and no confidence intervals are reported for counterfactual $\Delta\mathcal{F}$ values. The advantage of the Shapley method over the aesthetic score average is not statistically established for ArtBench, which undercuts the counterfactual evidence for this dataset.

### Minor

- **The fine-tuning-from-full-model approximation conflates directions that are conceptually distinct from scratch retraining.** Starting from a model trained on all data $\mathcal{D}$ and fine-tuning on subset $S_j$ is not the same as training from scratch on $S_j$: the initialization has "seen" data outside $S_j$, which is the core problem addressed by machine unlearning—a field the paper acknowledges but defers to future work. The paper provides empirical validation via Figure 2 (which shows the approximation works in practice), but does not measure *how much* residual influence from excluded contributors remains after fine-tuning. A diagnostic experiment measuring, e.g., the loss on held-out data from excluded contributors before and after fine-tuning would quantify this gap.

- **Theoretical propositions have a large gap with practice.** Propositions 1 and 2 assume a convex, Lipschitz-smooth objective, which is explicitly stated to be instantiated as a quadratic loss—a setting far removed from training deep diffusion networks. Both results are asymptotic ($k \to \infty$), while experiments use 200–1000 finite steps. The paper is transparent about these limitations (Section 3.2), but the propositions provide motivation rather than meaningful guarantees, and the gap should be acknowledged more prominently (e.g., in the abstract or contributions list) rather than relegated to a footnote.

- **Proposition 2's $2\sqrt{n}C$ bound is uninformative for large $n$.** For ArtBench with $n = 258$, the bound scales by a factor of $\approx 32$. Since $C$ is unestimated, the proposition cannot be used to bound actual Shapley error quantitatively. This limits the theoretical contribution to an asymptotic convergence statement rather than a practical approximation guarantee.

- **Number of KernelSHAP subsets $M$ not reported in the main text.** The convergence quality of KernelSHAP depends critically on $M$ (the number of sampled subsets), especially for ArtBench ($n = 258$). This value and any sensitivity analysis are absent from the main paper.

- **Scalability beyond 258 contributors not demonstrated.** The largest setting (ArtBench) has 258 contributors. Real-world attribution scenarios (e.g., internet-scraped datasets) may involve thousands of contributors. KernelSHAP's variance grows with $n$, and the paper provides no guidance on how LDS degrades as $n$ increases. Even a small synthetic experiment with $n = 500\text{–}1000$ would substantially strengthen the scalability claims.

### Tiny

- The paper does not report the computational cost of generating the 100 held-out retrained models used to compute LDS ground truth. For CelebA-HQ with 274M parameters, this is substantial and should be stated transparently for reproducibility.

- The ArtBench fine-tuning uses LoRA (5.1M parameters), and pruning is applied to these LoRA weights. The generalizability of the pruning-based efficiency argument to full-parameter training regimes (relevant for CIFAR-20 and CelebA-HQ settings) is implicitly assumed but not discussed.

---

## Nice-to-Haves

- **Compute exact Shapley values on a toy scale** (e.g., $n = 5$–10 contributors) to validate that the method converges toward the true Shapley value rather than just correlating with LDS proxies.
- **Ablation on fine-tuning steps $k$**: The paper mentions the asymptotic result but does not show a convergence plot of LDS vs. $k$. Showing stabilization of Shapley estimates with $k$ would increase confidence in the chosen step counts.
- **Comparison to Truncated Monte Carlo (TMC) Shapley**: TMC-Shapley is the standard efficient baseline for Shapley estimation. Including it would clarify whether the sparsified fine-tuning strategy is necessary beyond simply capping the number of model evaluations.
- **Discussion of pruning bias**: Does magnitude pruning disproportionately affect weights associated with rare classes or minority contributors? This is especially relevant for the CelebA-HQ demographic diversity use case, where systematic pruning bias could skew attribution.
- **Investigate unstructured pruning hardware efficiency**: Magnitude-based unstructured pruning may not yield wall-clock speedups during GPU inference without sparse kernel support. The reported speedups are real (since fine-tuning forward/backward passes over sparse weights are faster), but the paper should clarify whether the efficiency gains are measured in actual runtime or theoretical FLOPs.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"TracIn should be included as a baseline"** (Harsh Critic): The paper explicitly and legitimately excludes TracIn because intermediate checkpoints may not be available in practice (Footnote 7). This is a reasonable design choice for a post-hoc framework, not a gap.

- **"Full fine-tuning without sparsification should be in Table 1"** (Harsh Critic): Figure 2 already includes this exact comparison at matched computational budgets, which is more informative than a single-point entry in Table 1.

- **"Negative LDS baselines warrant explanation beyond reporting"** (Harsh Critic): The paper explains (Section 4.5) that aggregated datum-level TRAK scores fail to capture contributor-level global properties and can be anti-correlated. The mechanistic explanation is conceptually clear and the observation itself strengthens the paper's motivation.

- **"Three seeds is thin for confidence intervals"** (Harsh Critic): Three seeds for LDS confidence intervals is standard practice for expensive deep learning experiments requiring full retraining. This is not a meaningful critique at the scale of these experiments.

- **"The efficiency claims may be overstated due to unstructured pruning"** (Harsh Critic): The speedups (5.3×, 10.4×, 18.6×) are empirically measured runtimes (Table 2), not theoretical FLOP counts. The claim is based on measured wall-clock time, so the concern that sparse weights don't automatically speed up GPU inference is already accounted for in practice.

---

## Novel Insights

The most genuinely novel insight in this work is methodological rather than theoretical: the observation that *pruning acts as a structural regularizer that makes subset fine-tuning more data-efficient*. Intuitively, a sparser model has fewer degrees of freedom and thus adapts more quickly to a new data distribution with fewer gradient steps—this is why sparsified fine-tuning outperforms plain fine-tuning under the same budget (Figure 2), even though both start from the same full-data checkpoint. The paper doesn't fully articulate this mechanism, but it has implications beyond diffusion model attribution: any setting where Shapley value estimation requires repeated model re-specialization to subsets could benefit from the prune-then-fine-tune paradigm. The companion finding—that aggregated datum-level attribution methods (TRAK, D-TRAK, influence functions) systematically fail at global contributor attribution, sometimes yielding *negative* LDS—is also a useful empirical observation for the field, suggesting that methods designed for local attribution should not be naively repurposed for global settings.

---

## Suggestions

1. **Diagnose the CelebA-HQ gap**: Conduct ablations isolating whether the LDS drop comes from the entropy metric (add an alternative diversity measure), the BLIP-VQA clustering (try different numbers of clusters), or the LDM fine-tuning regime (increase steps or vary pruning ratio). Present findings in the main paper.

2. **Add a pruning ratio ablation for at least one dataset**: Plot LDS vs. pruning ratio (from 0% to 80+%) for CIFAR-20. This single experiment would answer the most pressing practitioner question and substantially strengthen the core efficiency-accuracy trade-off narrative.

3. **Report Shapley estimate stability**: Add a figure showing Shapley value rank correlation (across contributors) as a function of $M$ (number of subsets) and $k$ (fine-tuning steps) for one dataset. This directly addresses whether the estimates have converged.

4. **Quantify the residual influence of excluded contributors**: For a small-scale case (e.g., CIFAR-20), measure the loss on data from excluded contributors before and after fine-tuning. This would validate that the fine-tuning-from-full-model approximation does not systematically inflate attribution scores for contributors whose data "persists" in the pruned model.

5. **Report absolute GPU-hour costs**: In addition to speedup ratios, provide absolute runtimes for the full attribution pipeline (including LDS ground truth computation). This allows practitioners to assess feasibility without needing to reverse-engineer from relative numbers.

---

**Overall assessment:** This is a solid and timely contribution with a genuinely novel problem formulation and strong empirical results on two of three datasets. The core efficiency claim is well-supported. The primary concerns are (1) the unexplained and relatively weak CelebA-HQ result, (2) the unconvincing ArtBench counterfactual evaluation, and (3) missing ablations on the central pruning ratio hyperparameter. These are significant gaps but do not undermine the overall contribution. The theoretical propositions are motivational scaffolding rather than rigorous guarantees—the paper is mostly honest about this, but could be more explicit upfront. **Novelty: high; technical soundness: moderate-to-good (empirically strong, theoretically limited); empirical support: good for CIFAR-20/ArtBench, weak for CelebA-HQ; significance: high for the generative AI fairness/attribution community; clarity: good.**

# Actual Human Scores
Individual reviewer scores: [5.0, 8.0, 5.0]
Average score: 6.0
Binary outcome: Accept
