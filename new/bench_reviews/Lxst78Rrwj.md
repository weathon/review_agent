## Summary

GLIDE introduces a causal discovery framework that exploits the invariance of the conditional distribution $P(\text{effect}\mid\text{cause})$ under changes to the prior $P(\text{cause})$. It constructs multiple augmented datasets with different cause priors via a downsampling scheme and selects parent candidates whose conditional distribution exhibits minimal variance across augmentations. By reducing the parent search to maximal cliques in augmented bidirectional graphs and exploiting sparsity, it achieves $O(d^2)$ complexity. Empirical results on synthetic (up to 500 nodes) and real-world datasets (Munin, 1041 variables) show superior accuracy and scalability compared to state-of-the-art baselines.

## Strengths
- **Novel causal test**: The distributional invariance principle offers a new, non-parametric lens for causality, avoiding traditional constraint-based multiple testing or score-based heuristics (Section 4.1, Theorem 1).
- **Strong empirical performance**: GLIDE achieves lower SHD and spurious rates than baselines on large benchmarks; on Munin, SHD 883.2 vs. 1235 (GIES) and spurious rate 1.8% vs. 42.4% (Table 2).
- **Scalability**: Up to ~96× speedup over NOTEARS on 100‑node graphs, enabling processing of graphs with hundreds to thousands of nodes where many baselines fail or time out (Section 5.1, Table 2).
- **Versatility**: Validated on continuous (linear-Gaussian, non-linear non-Gaussian) and categorical data without model-specific parametric assumptions (Table 1).
- **Comprehensive evaluation**: Multiple synthetic settings (varying nodes, edges) and seven real-world datasets, using standard metrics (SHD, spurious rate, runtime) with confidence intervals over 10 runs.

## Weaknesses

### Fatal
None.

### Major
1. **No finite‑sample guarantees for the invariance test**. The parent selection criterion (Eq. 3) relies on the sample variance of estimated conditionals, but the paper provides only population‑level guarantees (Theorem 1). There is no analysis of type‑I/II errors, sample complexity, or confidence bounds; the threshold “$\simeq 0$” is heuristic. This undermines the claimed principled test.
2. **Causal sufficiency assumed but not examined**. The method uses Markov blanket recovery (Edera et al., 2014), which requires no latent confounders. The paper mentions this assumption once but neither discusses failure modes nor evaluates robustness when confounders exist. Comparisons with FCI (which handles latents) are only under no‑latent settings, limiting generalizability claims.
3. **Quadratic complexity claim is conditional on unverified sparsity**. The $O(d^2)$ bound holds only if the degeneracy $p$ of the augmented graph $G'(X)$ is a small constant. While experiments show $p\le 13$, no theoretical guarantee ensures $p$ remains bounded for all causal graphs; dense graphs could yield exponential maximal cliques, breaking the bound. The claim is presented as a general property rather than an empirically observed one.
4. **Missing key hyperparameter $m$**. The number of augmentations $m$ directly impacts both computational complexity and test reliability, yet the paper never reports its value in any experiment. This omission hinders reproducibility and proper interpretation of runtime comparisons.

### Minor
5. **Inconsistent speedup reporting**. The abstract states “up to 25×” speedup, the body reports “96.54× and 15.66×” over NOTEARS/MLP‑NOTEARS, and figures suggest at most ~20×. This discrepancy suggests possible numerical errors, mislabeled axes, or scaling issues.
6. **Inadequate justification of basis substitution**. The method replaces source variables with a basis set but does not clearly explain why invariance under basis‑prior changes implies invariance under source‑prior changes; the proof is deferred to the appendix without summary.
7. **Empirical downsampling approximations not analyzed**. The optimal downsampling rates (Theorems 4–5) assume perfect $P(\mathbf{B})$; the paper uses empirical estimates and binning for continuous data but does not analyze how estimation error affects the resulting $P_i(\mathbf{X})$ or the invariance test.
8. **Ablation for $\gamma_0$ missing despite citation**. The text claims an ablation on $\gamma_0$ in Section 5, but no such study appears in the main paper; the impact of this threshold on information loss vs. prior diversification is therefore unknown.
9. **Implementation details for conditional estimation omitted**. For continuous data the paper mentions binning but does not specify bin width, number of bins, or kernel methods; these choices affect estimation accuracy and reproducibility.

### Trivial
None substantive.

## Nice‑to‑Haves
- Theoretical finite‑sample bounds (e.g., via concentration inequalities) for the variance test.
- Robustness evaluation under latent confounders (e.g., FCI‑style data) with comparison to FCI/GFCI.
- Analysis of degeneracy $p$ across graph families or proof that sparsity ensures $p=O(1)$.
- Report $m$, binning strategy, original dataset sizes $n$, and include the $\gamma_0$ ablation.
- Align speedup numbers across abstract, body, and figures.
- Broader discussion of causal sufficiency and its realistic implications.

## Removed Points
These points are flagged to be removed, treat them with caution:
- *“Robustness to other generative mechanisms (e.g., additive noise models with non‑Gaussian errors) is not tested.”* — **Incorrect**: The paper explicitly includes non‑linear non‑Gaussian (nL‑nG) experiments, covering such models (Section 5.1).  
- *“No worst‑case analysis is given for graphs where degeneracy is not constant.”* — **Factually wrong**: Theorem 7 states the bound $O(dp3^{p/3})$; the concern is already captured under Major #3.  
- Any nitpicks about missing related work, formatting, or typos not present in the original submission.

## Novel Insights
Beyond the paper’s own contributions, the insight that one can *engineer dataset shifts* via selective downsampling to test causal hypotheses is broadly useful. The resampling trick to change $P(\text{cause})$ while preserving $P(\text{effect}\mid\text{cause})$ could be applied to other invariance‑based learning problems beyond causality.

## Suggestions
1. **In main text**: Add a concise argument why invariance under basis priors implies invariance under source priors (currently only in appendix), and explicitly discuss causal sufficiency—its role, consequences of violation, and why latent‑variable extensions are left for future work.
2. **Experiments**: Include an ablation on $m$ (number of augmentations) showing trade‑off between accuracy and runtime; run experiments with hidden confounders (e.g., using FCI data generation) and report performance relative to FCI/GFCI.
3. **Reproducibility**: Specify $m$, binning details, and original sample sizes $n$ for all datasets; ensure abstract, body, and figures report consistent speedup numbers.

## Calibration Anchors
- **High** (avg 8.0): `/home/wg25r/review_agent/human_reviews/xByvdb3DCm.md` — Sound theory, important problem, clear writing; minor presentation issues only.
- **Medium** (avg 5.5): `/home/wg25r/review_agent/human_reviews/iaP7yHRq1l.md` — Extensive empirical benchmark under misspecification, accepted despite lacking real data and some SHD inconsistencies; shows empirical value can outweigh theoretical incompleteness.
- **Low** (avg 2.5): `/home/wg25r/review_agent/human_reviews/zgM66fu0wv.md` — Fundamental misunderstandings of causal assumptions, evaluation circular, withdrawn.

GLIDE compares favorably to the medium anchor: it proposes a new algorithmic method rather than a benchmark, and its real‑world results (Munin) are compelling. However, its theoretical gaps are more pronounced than the medium paper’s. Relative to the high anchor, GLIDE lacks comparable theoretical rigor and clarity but still delivers strong empirical gains. The score reflects solid empirical contribution tempered by unresolved theoretical and reporting issues.

## Score and Decision
MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept (Poster)</orange>