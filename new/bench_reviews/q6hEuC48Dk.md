## Summary

This paper introduces Radial Basis Operator Networks (RBON), a closed-form, single-hidden-layer operator architecture built entirely from radial basis functions, along with normalized (NRBON) and frequency-domain (F-RBON) variants. The method solves for network weights analytically via Moore-Penrose inverse, bypassing gradient descent. The paper reports dramatically lower errors than LNO, FNO, and DeepONet on PDE benchmarks (Wave, Burgers, Beam equations) while using orders of magnitude fewer parameters, and includes a real-world CO₂–temperature forecasting experiment with data-driven operators.

## Strengths

- **Novel RBF-based operator architecture with closed-form solution.** The RBON adapts the Chen & Chen (1995b) operator approximation theorem into a concrete, single-hidden-layer RBF branch–trunk architecture where weights are solved analytically via the Moore-Penrose inverse (Section 2.2). This eliminates gradient-based training pathologies (learning rate tuning, local minima, convergence instability), which is genuinely valuable for small-data scientific regimes. Code is provided (Section 3, Julia implementation).

- **Impressive parameter efficiency with strong empirical results.** Despite capping trunk and branch networks at 15 nodes each (≤225 multiplier parameters), RBON variants achieve errors orders of magnitude below LNO on multiple PDE benchmarks—e.g., 4.1E-8 ID error on the Beam equation versus LNO's 1.0E-2 (Table 1). This parameter efficiency is a real contribution, especially given DeepONet required "over 10,000 products between trunk and branch outputs" for comparable tasks (Section 3.1.4).

- **Strong OOD evaluation across function classes.** The Burgers OOD test trains on sine initial conditions and tests on polynomial initial conditions (Section 3.1.2). This is a substantively harder and more informative OOD test than the typical scaling/shifting transformations used in new operator network papers. NRBON achieves 1.0E-1 error on this cross-class test (Table 1).

- **NRBON normalization empirically addresses RBF instability.** The normalization scheme (Eq. 4, Section 2.2) constrains outputs and prevents the extreme peak/valley predictions that plague unnormalized RBF networks with wider capacity (Section 3.2). Table 1 shows NRBON provides consistent variance reduction and competitive OOD stability compared to base RBON.

## Weaknesses

### Fatal
// None identified. The method is structurally sound and results are empirically compelling.

### Major

- **Baseline comparison quality undermines headline performance claims.** Table 1 reports LNO relative errors that are orders of magnitude worse than what LNO achieves in its own paper and in the broader literature (e.g., LNO at 5.6E-1 on Wave, 1.0E-2 on Beam, vs. the ~1E-3 range reported by Cao et al., 2024 on similar problems). The paper provides no baseline architecture specifications, learning rates, optimizer details, regularization schemes, or training schedules for LNO, FNO, or DeepONet (Section 3.1). The claim of "outperforming LNO by several orders of magnitude" cannot be trusted without evidence that baselines were reasonably tuned. This does not mean RBON's results are wrong—the method could genuinely be effective—but the comparison asymmetry makes the headline contribution unverifiable.

- **Overclaim regarding "first operator network in both time and frequency domains."** The paper states RBON is "the first network to successfully learn an operator entirely in both the time domain and frequency domain" (Section 1.2). However, FNO's core mechanism (Fourier neural operator layers) inherently operates in the frequency domain via spectral representations. F-RBON applies a Fourier transform to inputs before feeding them into the same RBF architecture—this is a preprocessing step, not a fundamentally new architectural capability. The contribution is the RBF formulation, not the frequency-domain aspect.

### Minor

- **NRBON universal approximation extension is unproven.** Corollary 2.1.1 claims that extending the universal approximation theorem to NRBON "follows immediately" from redefining ξ̃ᵢᵏ (Eq. 3, Section 2.1). However, normalization transforms the hypothesis space from a linear span of RBFs to a nonlinear rational/softmax-like form (Eq. 4 divides by the sum of all basis products). The Chen & Chen theorem relies on linear independence and closure properties that do not automatically transfer to normalized sums. Without a proof sketch or citation to rational approximation theory, this theoretical claim is unjustified.

- **CO₂–temperature experiment is not genuine operator learning.** The paper frames the CO₂–temperature task (Section 3.2) as "scientific application where the underlying operator is unknown." However, the input functions are parameterized as uₙ(t) with t ∈ {1, …, 12} (months) (Section 3.2, Eq. 5). This maps between 12-dimensional vectors, not infinite-dimensional function spaces. The reported <10% error reflects seasonal interpolation on highly autocorrelated climate data, not proof of learning a complex physical operator. The paper itself acknowledges "this dataset does not naturally lend itself to a Fourier transform" (Section 3.2, final paragraph). This section conflates statistical curve-fitting with operator learning.

- **K-means-dependent RBF centers introduce non-trivial instability.** The paper acknowledges that "the majority of the variation in train/test error is mostly due to the varying results from the location parameters determined by the K-means clustering" and that this "can lead to errors differing by several orders of magnitude between runs" (Section 2.2, Section 4). For scientific computing applications where reproducibility matters, this is a significant practical concern. The suggested workaround of "run K-means multiple times and select the configuration" (Section 4) is ad hoc and not systematically validated.

### Trivial

- None beyond those noted above.

## Nice-to-Haves

- Provide a systematic ablation varying M and N (RBF counts) to quantify the accuracy–cost tradeoff, showing how error scales with the number of basis functions.
- Include spatio-temporal error maps (not just scalar L² errors) for the PDE benchmarks to demonstrate spatial coherence and identify localized blow-up modes.
- Provide baseline training details (architectures, hyperparameters, training budgets) so readers can assess comparison fairness and reproduce the baselines.
- Replace or supplement the CO₂ experiment with a recognized high-dimensional operator benchmark (e.g., Darcy flow with spatially-varying permeability fields) to better demonstrate the method on genuine function-space mappings.
- Formalize or remove the NRBON universal approximation corollary—either provide a proof sketch citing appropriate rational function approximation theory, or reframe normalization as an empirical heuristic.

## Removed Points

*These points are flagged to be removed or downgraded per the review rules. Treat them with caution.*

- **Criticism: "Weight averaging structurally breaks operator coherence" (Section 2.2, Eq. 4).** The harsh critic claims that averaging L per-query-point weight vectors "destroys spatial continuity" and "effectively reduces the method to L disconnected curve-fitting operations." Upon verification against the paper (Section 2.2): the averaged weight vector ξ is a SINGLE shared set of weights used for prediction at ANY query point y via the trunk network t(y). The RBF trunk handles spatial dependence, and each ξ_ℓ is solved from a system incorporating ALL J training inputs. This is a valid collocation-style approach, not disconnected curve fitting. The averaging step is heuristic but does not invalidate the operator structure. The criticism substantially misreads the method and is removed.

- **Criticism: "CO₂ experiment parameterization is incorrect; the paper is learning 12D vector mappings, not infinite-dimensional spaces."** While the reduced scope (12-dimensional monthly vectors) is a valid limitation, the paper acknowledges this framing implicitly when noting that "it is preferable to parameterize the functions across the years" (Section 3.2). The experiment is better characterized as a demonstration on real observational data than as a rigorous operator learning benchmark. The criticism is downgraded from a fundamental/methodological flaw to a minor scope issue.

## Novel Insights

The paper makes a genuine contribution to the operator learning toolbox by exploring the previously underutilized RBF basis within operator architectures. The branch–trunk RBF structure, combined with closed-form weight solving, represents a conceptually distinct approach from the iterative, gradient-based paradigms dominant in neural operator research (FNO, DeepONet, LNO). The most compelling finding is that single-layer RBF networks, when properly regularized via the NRBON normalization, can achieve remarkable accuracy on standard PDE operator learning benchmarks with minimal parameters—suggesting that operator learning may not require deep, iterative architectures for certain classes of problems. However, the paper's framing suffers from overclaiming on its novelty (frequency-domain capability) and mischaracterization of its real-world experiment (CO₂ forecasting), which dilutes an otherwise interesting contribution.

## Suggestions

1. **Re-run baselines with documented training protocols.** Train LNO, FNO, and DeepONet on the same datasets with fully specified architectures, learning rates, optimizers, regularization, and training budgets. Report results alongside RBON. If RBON still outperforms well-tuned baselines, the contribution is much stronger.
2. **Reframe the frequency-domain claim.** Change from "first operator network in both time and frequency domains" to acknowledging that FNO already operates in the frequency domain, and position F-RBON as an RBF alternative that can ingest spectral representations (a preprocessing choice, not a fundamental architectural innovation).
3. **Clarify or prove the NRBON corollary.** Either provide a short proof argument referencing rational function approximation theory, or demote Corollary 2.1.1 to an empirical observation without claiming it "follows immediately" from Chen & Chen.
4. **Report K-means stability statistics.** Run the K-means initialization across ≥10 seeds for each benchmark and report min/mean/max errors. If variance is bounded within acceptable scientific computing tolerances after center refinement, this concern is addressed.
5. **Replace the CO₂ title-framing.** Move the CO₂ experiment to a "time-series application" or "data-driven forecasting" framing rather than positioning it as demonstration of learning an unknown physical operator. Add a proper operator-learning benchmark with continuous function inputs (e.g., Darcy flow with varying spatial fields) to replace or supplement it.

## Score and Decision

**Calibration anchors consulted:**
- **KNO** (UjQthmslFV.md): Scores 5,8,5,1 — kernel-based operator learning with similar baseline comparison issues (unfair FNO baselines, missing training details). Withdrawn. Demonstrates that operator learning papers with strong empirical results but unverifiable baseline comparisons tend to score in the 4–5 range.
- **MgNO** (8OxR034uEr.md): Scores 6,6,6,8 — concise parameterized neural operator accepted as poster. Strong theoretical grounding and clean baselines push this to 6–8. RBON lacks comparable theoretical rigor but has arguably more dramatic empirical results.
- **FEONet** (wwJJUamHVp.md): Scores 3,3,3,3 — rejected for weak baselines, "not even operator learning" concerns, and overclaiming. RBON is fundamentally stronger (genuine operator formulation, strong PDE results) and should score noticeably higher.
- **CViT** (cRnCcuLvyr.md): Scores 6,6,8,8,6 — simple but effective operator architecture accepted as poster. Shows that simple architectures with good results can clear the accept bar.
- **Various borderline 4–5 papers** (Di3VLZHZdj.md, QIsnwejVYE.md, rYhDcQudVI.md): Scored 3–5 with concerns about baseline fairness, limited experimental scope, or overclaiming — closely matching RBON's profile.

RBON sits between MgNO/CViT (accepted, 6–8) and FEONet/KNO boundary cases (3–5). The RBF operator formulation is genuinely novel (beyond KNO's kernel approach), results are impressive on standard PDE benchmarks, and the closed-form solution is a real practical contribution. However, the unverifiable baseline comparison, overclaimed novelty, and unproven theoretical extension pull the score down from a clear accept. Positioned at 5.0 as a borderline paper whose core ideas merit attention but whose evaluation needs strengthening before a confident acceptance recommendation.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>