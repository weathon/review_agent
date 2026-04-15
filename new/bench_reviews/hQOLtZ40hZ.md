Now let me check the calibration papers referenced by the human finder:Now I have sufficient information to write the final consolidated review. Let me synthesize carefully.---

## Summary

This paper develops a dynamic generalization of the R-learner for estimating the difference of Q-functions (Q-contrasts), τ_t^π(s) = Q_t^π(s,1) − Q_t^π(s,0), in finite-horizon offline RL. By framing Q-contrast estimation as a residualized squared-loss minimization, the method achieves Neyman orthogonality, which allows nuisance estimators (Q-function and behavior policy) to converge at slower n^{−1/4} rates while still guaranteeing convergence of the contrast estimate. The paper additionally proves that convergent policy evaluation implies convergent policy optimization under a margin condition (Theorem 3), and motivates the approach by showing that Q-contrasts can be structurally simpler (sparser) than individual Q-functions under different MDP graphical models.

---

## Claims and Support

| Claim | Assessment |
|---|---|
| **C1: Dynamic R-learner derivation for binary actions** | **Well-supported.** Eqs. (2)–(5) provide a clean identification argument from Bellman equations + sequential unconfoundedness. Algorithm 1 is a concrete cross-fitted instantiation. |
| **C2: Orthogonal estimation improves convergence (product-rate robustness)** | **Partially supported.** Theorem 1 states the product-error structure; Theorem 2 is meant to instantiate MSE rates, but as rendered its theorem statement uses π_t^π where τ_t^π or ν_t^π is clearly intended—likely a PDF parser artifact (τ → π), but it makes the central theorem unverifiable from the text. The noisy-nuisance experiment in Figure 2 provides empirical support, though it doesn't fully isolate orthogonality from other design choices. |
| **C3: Convergent policy evaluation ⟹ convergent policy optimization (Theorem 3)** | **Theoretically stated, empirically absent.** Theorem 3 states the result for the finite-horizon backward optimization case and Lemma 2 links contrast error to policy value via margin. However, Eq. (8) contains a trivially vacuous exponent: the expression O(n^{−ρ̄})^{(2+α)/(2+α)} simplifies to O(n^{−ρ̄}) since (2+α)/(2+α) = 1, suggesting a notation/rendering error. More critically, **no experiment reports policy value or regret**—every empirical result measures τ-estimation MSE. |
| **C4: Method adapts to sparsity/structural assumptions in τ** | **Partially supported.** Figure 2 shows that thresholded-LASSO applied to τ estimation adapts to reward-filtered and exo-endo structures better than FQE-Ridge in synthetic settings. But gains are confounded with the LASSO regularizer, not clearly attributable to orthogonalization alone. |
| **C5: Method avoids unstable propensity weighting** | **Well-supported mechanistically.** The residualized loss uses (A_t − π^b(S_t)) rather than inverse propensity weights. This is a correct characterization of the objective. |
| **C6: Extension to multi-action and infinite-horizon settings** | **Unsupported beyond assertion.** Eq. (6) as rendered is garbled (the inner-product structure is broken by the parser). Algorithm 3 for infinite-horizon exists but has no theorem and no experiment. |

---

## Strengths

- **Conceptually clean identification argument (binary finite-horizon).** The derivation from Bellman equations + sequential unconfoundedness to the residualized squared-loss (Eqs. 2–5) is clear, elegant, and provides a well-motivated generalisation of the R-learner to the sequential setting.
- **Novel policy optimization convergence result.** Theorem 3's core argument—that policy-dependent nuisance error is higher-order under the margin condition via an inductive argument, so one does not need to re-estimate nuisances at every policy iteration—is the paper's most original theoretical contribution and addresses a real gap in orthogonal learning for RL.
- **Well-motivated structural examples.** Figure 1 and the surrounding discussion concretely illustrate two distinct MDP graphical models (reward-relevant/irrelevant factored dynamics and exogenous-endogenous) that both imply sparsity in τ but not in Q. This is more specific and compelling than generic "contrast may be simpler" rhetoric.
- **Experimental robustness illustration.** The semi-synthetic noisy-nuisance experiments (τ-TL-η̂_ε in Figure 2) consistently show that the orthogonalized method degrades more gracefully than FQE under n^{−1/4} noise injection, providing empirical corroboration of the rate-robustness claim in at least this controlled setting.
- **Structure adaptation across different DGPs.** The "Misaligned exo-endo" results show that targeting τ with thresholded LASSO is robust to which specific graphical structure generated the data, unlike a method specifically designed for one model. This is a practically useful property.

---

## Weaknesses

### Fatal
*None that completely invalidate the paper's core contribution (the binary finite-horizon identification and orthogonality argument). However, the combination of the two major issues below brings the paper below acceptance threshold.*

### Major

- **Policy optimization is presented as a central contribution but receives zero empirical evaluation.** The abstract, introduction, Algorithm 2, and Theorem 3 all feature policy optimization as a key claim. Yet every experiment reports τ-estimation MSE only; there is no measurement of optimized policy value, regret, or improvement over behavioral policy value. Since the key novelty of Theorem 3 is precisely the gap from estimation to optimization (handling policy-dependent nuisances), the absence of any policy value experiment leaves the most important practical claim entirely unvalidated. This is not "more experiments would be nice"; it means the paper's central stated contribution has no empirical support.

- **Theorem 2 statement is uninterpretable as rendered; Eq. (8) exponent collapses trivially.** Theorem 2 (MSE rates for policy evaluation) writes `E[||π_t^π − π_t^{π^π,o}||_2^2]` where τ or ν is clearly intended—almost certainly a PDF parser substitution (τ → π). As printed, the theorem measures a policy error, not the τ-estimation error the section is about. Similarly, Eq. (8) in Theorem 3 has exponent (2+α)/(2+α) = 1 for any α, making the policy value bound trivially O(n^{−ρ̄}) with no margin-rate improvement. The sup-norm version in Lemma 2 correctly shows n^{−b_*(1+α)}, so the conceptual intention is present but the stated theorems as they appear contain rendering errors in their central rate expressions. For a paper where the theoretical rates are the headline contribution, this is a serious presentational/correctness issue that must be resolved.

- **Table 1 shows the base method underperforms FQE across all sample sizes without acknowledgment.** In the 1D validation experiment, FQE achieves consistently lower MSE than OrthDiff-Q at every sample size tested (n = 50 to 5000), often by a factor of ~2. The paper does not directly address this result, which directly undermines any broad claim that orthogonal contrast estimation is a generally superior estimator. The claimed advantage is narrower—only applicable in the structured/noisy-nuisance regime—but the paper does not clearly delineate this scope in the introduction or abstract.

### Minor

- **Experimental design confounds orthogonalization with sparsity regularization.** The empirical wins consistently come from "thresholded LASSO applied to τ" (τ-TL), which combines two distinct ideas: (a) targeting τ instead of Q, and (b) using support-recovery regularization. There is no ablation comparing (i) direct Q vs. direct τ with identical regularization, (ii) orthogonalized vs. non-orthogonalized τ estimation with the same regularizer, or (iii) effect of nuisance quality holding target parameterization fixed. Without these, it is unclear whether orthogonalization, the τ-target, or thresholded LASSO is the primary driver of the observed improvements.

- **No experiments on standard offline RL benchmarks.** All environments are bespoke synthetic MDPs with hand-crafted sparsity matching the paper's assumptions, or a CartPole-with-distractors variant. Evaluation on standard benchmarks (e.g., D4RL) is not provided, making it difficult to assess whether the benefits materialize in realistic offline RL settings where τ may not be sparse.

- **Infinite-horizon extension (Algorithm 3) is entirely unsubstantiated.** The infinite-horizon case is presented as an extension with Algorithm 3, but has no theorem (no identification or rate result), no experiment, and critically relies on an existing offline RL method to estimate Q^π^* first—substantially reducing the distinct contribution of this extension. It should be presented as future work.

- **Baseline comparison against the most directly related contrast estimator (Shi et al., 2022) is absent.** The paper identifies Shi et al. (2022) as the closest prior work targeting Q-contrasts in infinite-horizon settings. No experimental comparison is provided, making it impossible to assess whether the finite-horizon R-learner approach has empirical advantages over the pseudo-outcome approach.

### Trivial

- **Multi-action extension (Eq. 6) appears garbled.** The rendered Eq. (6) is missing the inner product structure that the surrounding text describes. Given the parser artifacts elsewhere, this is likely an extraction issue, but the multi-action case still lacks experiments and any formal identification proof beyond the binary case.

- **Three-fold cross-fitting for policy optimization is not validated.** Algorithm 2's alternating estimation between fold 2 and fold 3 is motivated theoretically but never tested empirically. With limited data, the small fold sizes may introduce instability—a concern the paper itself raises for the noisy-nuisance case.

---

## Nice-to-Haves

- **An ablation isolating orthogonalization vs. target choice (τ vs. Q) vs. regularization** would significantly sharpen the paper's claims about where the benefit comes from.
- **A scatter plot of nuisance error vs. τ estimation error** would empirically validate the orthogonality claim more directly than the current semi-synthetic noise injection.
- **A clearer statement of when estimating τ is not beneficial** (e.g., when τ is no sparser than Q) would help practitioners understand the method's scope.
- **Discussion of practical implications of the proxy loss variance term (Lemma 1)**—when it is large and how it affects convergence constants—would strengthen the connection between theory and practice.
- **Unified decision flowchart** for when to use Algorithms 1, 2, or 3 would help practitioners navigate the three variants.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **[Harsh Critic] "Multi-action extension framing 'without loss of generality' is not established."** The paper does not claim WLOG without hedging; it explicitly states "We focus on the case of two actions, though the method generalizes to multiple actions." This is appropriate and not overclaiming. REMOVED.

2. **[Harsh Critic] "The key orthogonality proof is deferred to the appendix, which is absent."** The paper explicitly states (line ~205): "We establish Neyman-orthogonality of the loss function in Appendix A.2, though this is overall fairly similar to prior work." Deferring a result that follows prior work to an appendix is standard practice, not a deficiency. REMOVED.

3. **[Harsh Critic + Spark] "Reproducibility concerns about hyperparameters and implementation details."** Nitpick-level reproducibility complaints removed per rules.

4. **[Spark] "Sensitivity to the n^{−1/4} nuisance rate requirement—no analysis of whether FQE-Ridge actually achieves this."** This is asking for an analysis outside the paper's scope (the paper takes nuisance convergence as an assumption, which is standard in the semiparametric estimation framework). REMOVED as scope creep.

5. **[Neutral] "Comparison to advantage learning literature / dynamic treatment regimes."** The paper already situates itself relative to this literature (line ~39). Demanding deeper comparison to tangential literature is scope creep. REMOVED.

6. **[Harsh Critic] General criticism that notation drift/inconsistency throughout is "too unstable to support the headline statistical claims."** Many instances flagged by the harsh reviewer appear to be PDF parser artifacts (τ → π, inner product splitting in Eq. 6). The core theorems' conceptual structure is sound; the issue is rendering, not the underlying argument. This is softened rather than removed—the rendering issues in Theorem 2 and Eq. (8) are real and kept as Major weaknesses, but the global "everything is wrong" framing is not accurate.

---

## Novel Insights

The framing of the Q-function contrast as an estimand that simultaneously (a) enables orthogonal estimation without propensity weighting instabilities, (b) may be structurally simpler than individual Q-functions under common MDP graphical structures, and (c) admits a closed-form residualized loss that generalizes the R-learner to the sequential setting, is a genuinely useful synthesis. The observation in Theorem 3 that policy-dependent nuisance functions introduce only higher-order error under the margin condition—allowing policy optimization consistency without iterative nuisance re-estimation—is the most technically novel contribution and addresses a real gap in the orthogonal learning literature for RL. Whether this advantage is practically significant beyond synthetic structured settings remains the key open question the paper does not empirically resolve.

---

## Suggestions

1. **Add policy optimization experiments immediately.** Report policy value (or value regret against behavioral policy) using Algorithm 2 across at least the synthetic environments already in the paper. This is the most critical missing piece.
2. **Correct the theorem statements in the main text.** Fix the τ/π notation in Theorem 2 and the trivially vacuous exponent in Eq. (8). If these are parser artifacts, ensure the camera-ready version uses notation that survives PDF extraction (or verify the symbols render correctly).
3. **Add a dedicated ablation table** comparing: (i) FQE with sparsity regularization, (ii) τ-estimation with sparsity regularization but no orthogonalization, and (iii) the full orthogonal τ-estimation with sparsity regularization—using the same synthetic DGPs already in the paper.
4. **Either add a theorem and experiment for Algorithm 3 (infinite-horizon)** or move it to an appendix labeled "future work."
5. **Directly address Table 1 in the text.** Explain why OrthDiff-Q underperforms FQE in the 1D setting and under what structural conditions the method is expected to help.
6. **Include a comparison to Shi et al. (2022)** in at least a synthetic setting to clarify when the finite-horizon R-learner approach outperforms the pseudo-outcome approach.

---

## Score and Decision

**Calibration against human-scored papers:**

- **nIEjY4a2Lf** (Sparse Q-learning, accepted 6/6/6/6): A pure theory paper with no experiments but tight matching upper/lower bounds. The present paper has experiments but weaker theorem statements and a major empirical gap. Positioned slightly below this paper.
- **TC9r8gsaoh** (Nuisance-Robust Causal, rejected 8/5/5): Rejected due to missing baselines and unclear advantages over semiparametric approaches. The present paper has stronger theoretical development but a similarly incomplete experimental evaluation of the central claimed benefit. Scores similarly.
- **mwYkVSddzx** (OR-learners, rejected 8/6/3): Rejected partly for missing formal mathematical analysis. The present paper is stronger theoretically but has the analogous gap of missing optimization experiments for the stated central contribution.
- **jO3QEsm15T** (OT for causal inference, rejected 6/5/5/6): Rejected for complexity concerns and limited baselines. Similar positioning.

**Assessment axes:**
- *Originality*: Good. The dynamic R-learner for Q-contrasts is a natural but non-trivial extension with clear intellectual lineage.
- *Importance of research question*: High. Offline RL with causal structure is an important and growing area.
- *Claims vs. support*: Moderate. Binary-action evaluation is well-supported; policy optimization claim is not empirically supported; theorem notation has rendering errors in central expressions.
- *Soundness of experiments*: Weak. Synthetic only, no standard benchmarks, no policy value evaluation, confounded ablations.
- *Clarity of writing*: Moderate. Core ideas are clear; theorem statements have notation issues.
- *Value to community*: Moderate. The framework and Theorem 3 are valuable; the experiments are insufficient to establish practical impact.

**Final score: 5.0** — Borderline reject. The paper has a solid core contribution and a theoretically interesting policy optimization result (Theorem 3), but the central policy optimization claim is entirely unvalidated empirically, the theorem statements have rendering-level errors in their core rate expressions, and the experimental evidence is limited to synthetic settings with confounded ablations. This places the paper below the nIEjY4a2Lf threshold (which was a pure theory paper with clean matched bounds) and at the level of TC9r8gsaoh/jO3QEsm15T (rejected borderline-reject papers with similar structural gaps between claims and evidence).

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>