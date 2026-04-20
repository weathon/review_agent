Now I have enough information to write the final review. Let me synthesize all the inputs.---

## Summary

LOO-StabCP introduces a leave-one-out algorithmic stability framework for conformal prediction that enables a single model fit to serve all *m* test points, versus the *m* model fits required by the prior art RO-StabCP (Ndiaye 2022). The method comes with a finite-sample coverage guarantee (Theorem 1) and concrete LOO stability bounds for regularized loss minimization, SGD, neural networks, and bagging. A screening application (LOO-cFBH) demonstrates improved test power over a state-of-the-art split-CP-based competitor by exploiting the full training set.

---

## Strengths

- **Genuine O(m) → O(1) computational speedup.** Table 1 establishes that LOO-StabCP requires exactly 1 model fit regardless of the number of test points, compared to *m* fits for RO-StabCP and |𝒴|·m for FullCP. Figure 2 empirically confirms that for m=100, LOO-StabCP is orders of magnitude faster than RO-StabCP.

- **Provably valid coverage (Theorem 1).** The coverage guarantee is proved via set containment: C_{j,α}^{full} ⊆ C_{j,α}^{LOO}, so any set that contains the full conformal set inherits its coverage property. The proof is clean and does not rely on unverifiable assumptions beyond LOO stability.

- **Factor-of-2 tighter bound for SGD (Theorem 3, Eq. 5).** The LOO bound τ^{LOO} = Rnη·γν_iρ_{n+j} is exactly half of τ^{RO} = 2Rnη·γν_iρ_{n+j}. The mechanistic explanation—leaving out a data point removes one gradient update, whereas replacing it can reverse the update direction—is a clean and insightful theoretical result that directly translates into less conservative prediction intervals.

- **Breadth of stability derivations.** Theorems 2–5 cover RLM with ridge penalty (including kernel methods as a special case), SGD in convex and non-convex regimes, and derandomized bagging, giving the framework broad applicability.

- **LOO-cFBH screening application.** The FDP control is empirically validated across 1000 repetitions, and the power improvement over cFBH is attributable to using the full dataset for training—a mechanistically sound reason. The factor-of-2 tighter LOO bounds translate directly into less conservative p-values and higher power than RO-cFBH.

---

## Weaknesses

### Fatal
None.

### Major

- **Misleading FullCP accuracy comparison in SGD experiments.** Section 4 states: *"we ran SGD for R = 15 epochs for all methods, except R = 5 for the very slow FullCP."* By Theorem 1, the set containment C_{j,α}^{full} ⊆ C_{j,α}^{LOO} holds for models trained identically—so when the underlying models are the same, FullCP intervals should always be at least as short as LOO-StabCP's. Yet Figure 1 describes LOO-StabCP as having "the shortest predictive intervals" including over FullCP in the SGD columns. This is only possible because FullCP is deliberately undertrained (R=5 vs R=15), inflating its residuals and widening its bands. The claim that LOO-StabCP is "competitive" with FullCP in prediction accuracy is therefore not demonstrated for SGD—the credit belongs to extra training epochs, not the method. The accurate and legitimate accuracy claim is that LOO-StabCP is *competitive with RO-StabCP and superior to SplitCP*, both of which use R=15 epochs and are unaffected by this issue. The RLM experiments (same solver, same budget) are also unaffected. The paper should either (a) reframe the accuracy comparison to exclude FullCP in the SGD setting, or (b) report both R=5 and R=15 FullCP results with a note that the R=15 comparison is impractical but theoretically grounded.

### Minor

- **No formal FDR control theorem for LOO-cFBH.** The paper proves coverage (Theorem 1) for all methods except the screening application, where FDR validity is asserted only empirically. For the BH procedure to control FDR, the input p-values must satisfy super-uniformity under the null. The conservativeness of LOO p-values (which deflate calibration scores and inflate test scores relative to split-CP) is suggestive of super-uniformity, but this reasoning is never formalized. For a paper that is otherwise theory-first, this gap is notable. At minimum, an informal argument (or a short formal statement citing results from Jin & Candès 2023 for analogous structure) should be included.

- **Screening validated on a single small dataset.** Section 6 uses one 215-observation recruitment dataset, yielding approximately 43 test points per run. The abstract claims "improved test power compared to state-of-the-art," but this is based on one study. A second dataset would substantially strengthen the generality of this application claim.

- **Neural-network heuristic not a valid upper bound.** The paper uses τ^{LOO} ≈ Rη·γ‖X_i‖‖X_{n+j}‖ for neural network experiments (Section 5, borrowing from Hardt et al. 2016 and Ndiaye 2022), which is motivated by analogy to linear models but is not a provable upper bound for non-convex networks. Therefore, Theorem 1's coverage guarantee does not formally apply to Section 5. The paper acknowledges this gap and frames it as an open problem, which is honest, but the coverage result in Figure 3 is presented as "valid coverage" without qualifying that the guarantee is empirical rather than theoretical in this regime.

### Trivial

- In Theorem 5 (bagging), the variable *m* is used simultaneously for the subsample size and the number of test points. This notation conflict should be resolved for clarity.

---

## Nice-to-Haves

- Scaling experiments showing runtime curves for m ∈ {100, 500, 1000, 5000} versus RO-StabCP would confirm the linear speedup and better motivate the regime where LOO-StabCP matters most; m=100 already shows clear advantage but larger m would be more compelling for practitioners.
- A table reporting mean τ^{LOO} magnitudes relative to mean interval lengths in experiments would help readers assess the conservatism of the stability corrections in practice.
- The bagging bound's behavior as the subsample size approaches n (where p → 1 and the bound diverges) deserves a brief discussion.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

1. **Critic: "Underdetermined stability bound computation for Huber loss"** — The critic claims that Lipschitz constants for Huber-loss robust regression are never specified, creating a reproducibility gap. The paper references Appendix G.1 for full implementation details, and Huber loss has analytically known Lipschitz constants under standard conditions. This is an appendix-deferred detail per the hard rules; removed.

2. **Critic: "LOO-StabCP achieves the shortest intervals contradicts theoretical containment"** — This was rolled into the FullCP comparison major weakness above, which is the substantive version of the concern.

3. **Critic: "Scaling claim needs m ∈ {100, 500, 1000, 5000} experiments"** — This is a nice-to-have experiment that would strengthen the story; it is not a flaw that invalidates the contribution. Moved to Nice-to-Haves.

4. **Strength Finder: "Clear algorithmic presentation"** — Generic; does not cite a specific table/equation. Removed per the filter rule.

---

## Novel Insights

The most genuinely novel observation across the reviews is the factor-of-2 asymmetry between LOO and RO stability bounds for SGD (Theorem 3). The mechanistic explanation—that removing a data point causes one fewer gradient update (linear perturbation), while replacing it can fully reverse an update (doubling the worst-case bound)—is a clean insight that applies broadly to gradient-based optimization. It shows that the choice of stability type is not merely computational but directly shapes bound tightness, and that LOO is intrinsically preferable to RO for any iterative first-order method. This observation likely has implications beyond conformal prediction, e.g., for generalization bounds in online learning.

---

## Overall Evaluation

**Originality:** Good. The LOO stability framework is a clear and natural departure from RO-StabCP, the derivations are non-trivial, and the factor-of-2 result for SGD is genuinely new. The screening application is a well-motivated extension.

**Importance of research question:** High. Scaling conformal prediction to large numbers of test requests is a practically critical problem, and the O(m) → O(1) model-fit reduction is the right target.

**Claims well supported:** Mostly yes, with one important exception. The speedup and accuracy-vs-RO-StabCP claims are well supported. The accuracy-vs-FullCP claim in the SGD setting is not, due to the mismatched training budget.

**Soundness of experiments:** Good for the RLM comparison; compromised for the FullCP-SGD comparison. The screening experiment is limited to one dataset.

**Clarity of writing:** Good overall; notation conflicts in bagging and the neural-network heuristic framing should be clarified.

**Value to the research community:** Solid. The method is practical, theory-backed (for convex settings), and the code would be straightforward to implement from Algorithm 1.

---

## Score and Decision

**Calibration anchor comparisons:**
- *vcX0k4rGTt* (avg 6.25, Poster Accept): Closely analogous paper that approximates FullCP for neural networks without retraining. Similar theoretical quality, similar experimental scope. This paper has stronger theory breadth (5 algorithm families vs one) and a cleaner speedup story, but has the FullCP comparison flaw.
- *33XGfHLtZg* (avg 7.0, Spotlight): Extends CP to monotone risk control—comparable theoretical elegance and stronger experimental coverage. This paper is somewhat below that bar due to the FullCP issue.
- *aJ3tiX1Tu4* (avg 6.7, Poster Accept): Wasserstein-regularized CP with rigorous theory but limited experiments. Comparable profile.
- *RcNzwKrjTo* (avg 5.0, Reject): Solid theory but minimal empirical improvement. This paper clearly outperforms that anchor—the computational speedup is real and large.

**Assessment:** This paper sits solidly in the 6.0–6.5 range. The core theoretical contribution (LOO stability framework, coverage theorem, factor-of-2 SGD result) and the computational speedup over RO-StabCP are genuine and well-supported. The FullCP comparison issue is a significant presentation flaw that the authors can address in revision; it does not invalidate the core contribution. The screening application and neural-network experiments add breadth. Overall, a reasonable poster-level contribution.

**Final score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>