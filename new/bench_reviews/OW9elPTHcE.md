Now let me read the human review files mentioned by the Human Finder reviewer for calibration.Now let me check the remaining calibration papers quickly:Now I have all the information I need. Let me write the consolidated final review.

---

## Summary

GEFA (Gradient-Estimation-based Explanation For All) introduces a proxy-variable framework for black-box feature attribution. By parameterizing feature presence with Bernoulli probabilities in a continuous proxy space, the method enables path integration and gradient estimation for both discrete and continuous inputs. The paper proves that GEFA is an unbiased Shapley value estimator (Theorem 2), establishes its equivalence with Integrated Gradients under edge paths (Theorem 4), and introduces a control variate for variance reduction. Experiments on text (BERT/Amazon reviews) and image (InceptionV3/ImageNet) classifiers show competitive or superior performance to black-box baselines and, in one evaluation setting, to white-box methods.

---

## Strengths

- **Genuine theoretical contributions**: The proxy-space framework is conceptually clean and well-motivated. Theorems 1–4 are non-trivial: Theorem 2 (GEFA = unbiased Shapley estimator) connects score-function gradient estimation to cooperative game theory; Theorem 4 (equivalence with IG under edge paths) provides valuable conceptual clarity and extends prior observations by Sundararajan & Najmi (2020). These results collectively are interesting and, if the appendix proofs are correct, theoretically sound.

- **Addresses a real gap**: Prior gradient-estimation work (GEEX) was limited to continuous inputs. The proxy variable approach offers a principled and general solution for discrete features (e.g., text tokens), avoiding the need to access internal embedding layers — which would violate the black-box assumption. This is a legitimate and meaningful extension.

- **Control variate is practical and effective**: The control variate is simple to implement and the paper shows clear nAOPC gains over GEFA without it (Tables 1–2). The mechanism is theoretically justified via Theorem 3, and the paper provides an honest discussion of when Assumption 1 may fail (Sec. 5.3 on negation/irony in text).

- **Dual evaluation semantics in Table 1**: The paper reports both embedding-reset and token-removal evaluations, transparently showing where IG leads (embedding reset: 0.6622 vs. GĒFA's 0.6482) and where GĒFA leads (token removal: 0.7366 vs. IG's 0.6677). This avoids cherry-picking.

---

## Weaknesses

### Fatal
*(None that fully undermine the core theoretical contribution, but the evaluation scope is insufficient to support the broadest claims.)*

### Major

- **"For All" claim is empirically unsubstantiated**: The paper's title, subtitle, and abstract assert general applicability to "arbitrary black-box models, regardless of input type." Yet experiments cover only text and image classification. Tabular data with mixed continuous/categorical features is the single most common use case for Shapley value methods (e.g., XGBoost on census/clinical data) and is entirely absent. Without this, "For All" remains aspirational, and the breadth of the practical claim is not validated. This is the most consequential empirical gap.

- **No empirical validation of the key theoretical claim (Theorem 2)**: Theorem 2 states that GEFA produces exactly Shapley Values in expectation, but no experiment verifies this. On any small model (≤15 features) where exact Shapley values are computable via brute force, one could show that GEFA's estimates converge to ground truth with increasing query budget. Without this, the central theoretical claim is proven mathematically but never empirically confirmed, leaving open questions about convergence rate, finite-sample behavior, and whether implementation choices (e.g., numerical handling near γ → 0 or γ → 1 in Eq. (8)) introduce practical deviations.

- **The headline "surpasses white-box" claim uses incommensurable evaluation semantics**: Under *token removal*, GĒFA outperforms IG (0.7366 vs. 0.6677). However, token removal aligns naturally with GĒFA's own perturbation semantics (feature = absent token), while IG's zero-embedding absence representation is evaluated unfavorably under this deletion scheme. Under *embedding reset* — IG's native semantics — IG leads (0.6622 vs. GĒFA's 0.6482). The paper is transparent about this (Sec. 5.2), but the abstract claim "even surpasses white-box approaches" does not adequately flag that the surpass occurs only in the evaluation protocol most favorable to GĒFA. This is not a methodological flaw per se, but the framing is misleading.

- **Evaluation relies on a single metric**: All quantitative comparisons use deletion-based nAOPC exclusively. The paper explicitly acknowledges out-of-distribution concerns (Sec. 5.1 footnote) and claims alignment with retraining-based evaluation (Appendix B), but that retraining analysis remains in the appendix with no summarized evidence in the main paper. For a paper claiming broad superiority, a single deletion metric — especially one that may favor methods whose perturbation process matches the evaluation perturbation — is insufficient. Insertion-based metrics or in-distribution retraining results should appear in the main paper.

### Minor

- **No query-budget efficiency analysis**: The paper fixes budgets at 500 (text) and 5000 (images) without justification or sensitivity analysis. Since query efficiency is the primary cost dimension for black-box methods, a plot of nAOPC vs. query budget for GEFA and competitors is essential for assessing practical trade-offs. At 5000 queries over 299×299 ≈ 90K features, convergence behavior is unclear.

- **Numerical stability near γ endpoints not discussed**: Algorithm 1 samples γ ~ U[0,1] and uses factors ε/γ and ε̄/(1−γ) (Eq. 8), which are numerically unstable as γ → 0 or γ → 1. The paper does not describe how these edge cases are handled, which matters both for reproducibility and for the variance properties of the estimator — the paper's main focus.

- **No variance measurement for the control variate**: The effectiveness of GĒFA's control variate is demonstrated only via nAOPC improvements. The paper does not report the actual variance reduction factor (Var[GEFA] vs. Var[GĒFA]) or how this varies with query budget and Assumption 1's satisfaction level. This gap weakens the variance-reduction claim, which is a key contribution.

### Trivial

- **Minor notation inconsistency in Eq. (1)**: The completeness equation sums $\sum_{i=0}^p \xi_i$, which gives p+1 terms for a p-dimensional attribution vector. Likely a zero-indexing artifact; the meaning is clear from context but should be corrected.

---

## Nice-to-Haves

- **Empirical Shapley convergence experiment**: On a small synthetic model with computable ground-truth Shapley values, show GEFA estimates converging to ground truth as query budget increases. This would directly validate Theorem 2 and reveal convergence rates vs. competitors.
- **Tabular data evaluation** (e.g., Adult, Credit datasets with XGBoost/random forests) to concretely support the "For All" claim.
- **Insertion-based evaluation** alongside deletion, to mitigate known biases of deletion-only metrics.
- **Baseline sensitivity analysis**: The choice of baseline (blurred image, empty token) can significantly affect path-method attributions; even a brief ablation would strengthen trust in the results.
- **Query budget curves**: Performance vs. number of queries for all methods, revealing convergence and practical efficiency.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Comparison with additional Shapley estimators (Spark)**: The Spark reviewer calls for comparing against specific methods (e.g., Antithetical sampling, Unchained from Mitchell et al.). Per the meta-review guidelines, we do not mention missing related works or comparison methods we cannot independently confirm.

- **Feature correlation concern (Human Finder)**: The claim that "marginal contribution approach explicitly handles correlations" is partially correct in cooperative game theory, but Shapley values themselves assume feature independence by construction. This criticism mischaracterizes both GEFA and the baselines.

- **"GEEX excluded from text undermines broad superiority" (Harsh Critic)**: The paper explicitly explains GEEX's incompatibility with discrete inputs in Sec. 1 (the gap GEFA aims to fill). Criticizing the absence of an inapplicable comparison is a strawman.

- **Baseline sensitivity as a "notable gap" (Human Finder)**: The paper follows established recommendations from Sturmfels et al. (2020) and uses standard baselines. Sensitivity analysis would be a nice-to-have but not a critical flaw given that both GEFA and its competitors use the same baselines under the same evaluation.

- **"Large artifact impractical to include" reproducibility concerns**: Requests for full training logs, detailed hyperparameter sweeps beyond what is reported, or wall-clock timing comparisons are removed as implementation nitpicks.

---

## Novel Insights

GEFA's most genuinely novel contribution is the theoretical bridge between score-function gradient estimation in Bernoulli proxy space and unbiased Shapley value computation — a connection that was not previously established. The proof that GEFA and IG coincide under edge paths (and that straight-line path GEFA equals the average over all p! edge-path IG estimates) provides a clean conceptual unification that extends and formalizes partial observations in prior work. The insight that each query contributes to *all* feature attributions simultaneously (unlike marginal-contribution estimators that require paired samples) is a practical consequence of this formulation worth highlighting. The control variate leveraging feature count is a simple but well-motivated instance of this efficiency.

---

## Suggestions

1. **Add a small-scale Shapley validation experiment**: Choose a model with ≤15 binary features (e.g., a shallow tree or small MLP on synthetic data), compute exact Shapley values, and show GEFA estimates converging to them. This is inexpensive and would directly support the paper's central theoretical claim.
2. **Add tabular data experiments** to substantiate "For All" — even one dataset (e.g., Adult with a random forest) would significantly strengthen the generality claim.
3. **Move the retraining-based evaluation summary to the main paper**, even as a table or 1–2 sentence summary. The OOD concern with deletion metrics is well known; not addressing it in the main paper weakens credibility.
4. **Report actual variance reduction ratios** (Var[GEFA] / Var[GĒFA]) alongside nAOPC improvements to validate the variance-reduction motivation directly.
5. **Clarify γ endpoint handling** in Algorithm 1 — a sentence on numerical clipping or rejection sampling near 0 and 1 is needed for reproducibility.
6. **Rephrase the abstract**: "even surpasses white-box approaches" should specify the evaluation setting (token removal) to avoid the impression of unqualified superiority.

---

## Score and Decision

**Calibration:**

- **Fq25rH3ytL** (*Is Forward Gradient an Effective Tool for Explaining Black-box Models?*) — Rejected, scores 3,3,3,5. Very similar premise (gradient estimation for black-box explanation), but lacks the Shapley connection, proxy-space formulation, and rigorous theoretical apparatus. GEFA is clearly superior in theoretical depth.
- **CNZmaInj9n** (*Exploring Unified Perspective for Fast Shapley Value Estimation*) — Rejected, scores 3,6,6. Proposes a unified view of Shapley estimators with limited novelty and experimental issues. GEFA is more novel and theoretically cleaner.
- **1GUTzm2a4v** (*Greedy PIG: Adaptive Integrated Gradients*) — Rejected, scores 6,3,5,3. Proposes an adaptive IG method with strong results on one metric but weak in others. Comparable profile to GEFA in terms of theoretical-vs-empirical balance.
- **rvj1mn8q8D** (*TextGenSHAP*) — Rejected, scores 6,6,5,6. A Shapley method for LLMs with stronger empirical coverage but similar evaluation gaps. Similar quality tier.
- **gzYgsZgwXa** (*Path Choice Matters for Clear Attributions*) — Accepted, scores 8,8,6. A path method with strong theory and compelling experiments on ImageNet. GEFA has comparable theoretical depth but notably weaker empirical coverage ("For All" with only two modalities tested, no Shapley ground-truth validation, single metric).

**Assessment:** GEFA sits above the clearly-rejected papers (Fq25rH3ytL, CNZmaInj9n) due to its genuine theoretical novelty (Theorems 2 and 4) and the real gap it fills (discrete inputs under black-box constraint). However, it falls short of the gzYgsZgwXa tier because its headline claims ("For All," "surpasses white-box") are not adequately substantiated empirically: no tabular data, no ground-truth Shapley validation, single evaluation metric, and the white-box "surpass" relies on a semantics mismatch. The paper lands in the marginal-reject zone alongside Greedy PIG and TextGenSHAP.

**Originality**: Moderate-to-high — the proxy-variable / Bernoulli gradient estimation framework and its Shapley proof are genuinely novel.
**Importance**: Moderate — fills a real gap for discrete input explanation.
**Claim support**: Weak — central empirical claims are not fully supported; key theorem not empirically validated.
**Experimental soundness**: Below threshold — one metric, two modalities, no budget curves, no Shapley ground truth.
**Clarity**: Good — the exposition is clean and the paper is well-organized.
**Value to community**: Moderate — the theoretical insights are interesting and could influence future work.

**Score: 5.0** — Borderline reject. The paper has genuine contributions but requires substantially stronger empirical evidence before acceptance.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>