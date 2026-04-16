## Summary
This paper proposes Leave-One-Out Stable Conformal Prediction (LOO-StabCP), a variant of stable conformal prediction that replaces the per-test-point refitting in RO-StabCP with a single fit on the training data plus leave-one-out stability corrections. The main contribution is a clean finite-sample marginal coverage guarantee together with explicit stability bounds for several learning settings, especially RLM and SGD, and empirical evidence that the one-fit construction materially improves runtime when many test predictions are needed.

## Strengths
- **Clear and meaningful algorithmic idea.** The key distinction from RO-StabCP is genuine and important: in Section 3.1 the correction no longer depends on a model fit involving \((X_{n+j}, y)\), so Algorithm 1 uses **one model fit** for all \(m\) test points. This is a real improvement in the many-query regime, and the motivation is crisp.
- **Strong core theorem for the main task.** Theorem 1 gives a finite-sample marginal coverage guarantee for each test point, preserving the central distribution-free appeal of conformal prediction while avoiding sample splitting.
- **Substantive theory beyond a wrapper method.** The paper does not stop at a generic recipe: Theorems 2 and 3 derive explicit LOO and RO stability bounds for regularized loss minimization and SGD, and the SGD result is especially interpretable: \(\tau^{\text{LOO}}\) is exactly half of \(\tau^{\text{RO}}\) under the stated conditions.
- **Breadth of algorithmic scope.** The paper extends the framework to kernel methods, nonconvex neural-network training, and bagging, which makes the proposal feel like a general stable-CP framework rather than a one-off construction.
- **Empirics target the right use case.** The \(m=1\) vs. \(m=100\) comparisons in Section 5 directly probe the paper’s central claim that LOO-StabCP is most valuable when many predictions must be served. The runtime advantage over RO-StabCP in that regime is plausible and supported by the reported results.
- **Good overall clarity on the main method.** Sections 2–3 are well organized, and the relationship between FullCP, SplitCP, RO-StabCP, and the proposed method is easy to follow.

## Weaknesses
###: Fatal
- **The screening/application claims are not currently supported at the level the paper asserts.** Section 6 is internally inconsistent about the empirical outcome, and it lacks a theorem establishing that the proposed adjusted p-values in Eq. (7), when passed into BH, control FDR. This matters because the abstract makes a strong application-level claim: “improved test power compared to state-of-the-art method based on split conformal.” The paper’s own extracted Figure 4 description says the opposite (“cFBH ... shows lower FDP and higher power compared to RO-cFBH and LOO-cFBH”), while the paragraph beneath claims “our method is more powerful.” Even allowing for possible caption/text mismatch, the paper as presented does not rigorously justify the screening claim. Since this is featured in the abstract, it is a serious overclaim.

### Major:
- **The neural-network experiments are not theoretically aligned with the stated guarantee.** In Section 3.2.3, the paper is appropriately candid that Theorem 4’s nonconvex bound may be very conservative. But in Section 5 it then evaluates neural networks using a heuristic approximation, “\(\tau_{i,j}^{\text{LOO}} \approx R\eta \cdot \gamma \|X_i\| \|X_{n+j}\|\),” rather than the bound from Theorem 4. Theorem 1’s coverage guarantee requires valid upper bounds \(\tau_{i,j}^{\text{LOO}}\), so the neural-network experiments should be interpreted as heuristic demonstrations, not as validating the formally guaranteed method in that setting. The concluding language (“maintained valid coverage across all scenarios” and “highlight the robustness ... for neural networks”) is stronger than warranted by the theory actually used.
- **The empirical comparison to FullCP is weakened by unequal optimization budgets.** In Section 4, SGD is run for \(R=15\) epochs for all methods except FullCP, which is run for \(R=5\) “for the very slow FullCP.” This is understandable computationally, but it weakens any interval-length comparison with FullCP as an accuracy anchor, since part of the difference may come from undertraining rather than the conformal construction itself.
- **The application extension in Section 6 lacks theoretical support commensurate with its claims.** Theorem 1 only gives marginal prediction-set coverage for Algorithm 1; it does not establish that the p-values in Eq. (7) satisfy the assumptions needed for BH-based FDR control. The experiments report FDP empirically over repetitions, which is useful but not a substitute for a theorem when the paper presents LOO-cFBH as a principled extension.
- **Practical deployment of the stability bounds is underexplained.** The usable bounds in Theorems 2–4 depend on quantities such as \(\rho_i,\nu_i,\varphi_i\), but the paper gives little practical guidance on how these are computed or bounded tightly in realistic problems. In practice, overly loose constants can directly inflate \(\tau_{i,j}^{\text{LOO}}\) and hence interval width, so this is not merely a cosmetic omission.

### Minor
- **The empirical scope is somewhat modest relative to the scalability claim.** The central motivation is efficient handling of many predictions, but the largest reported \(m\) is 100 and the real datasets are all small. The runtime argument is still believable, but larger-scale evidence would make the systems-style benefit more convincing.
- **The paper sometimes overstates interval-length advantages.** The figure text around Figure 1 says LOO-StabCP “consistently achieves the shortest predictive intervals,” whereas the surrounding discussion says it is “competitive” with OracleCP, FullCP, and RO-StabCP. The latter is the more defensible framing based on the presented evidence.
- **The bagging result is idealized.** Theorem 5 analyzes derandomized bagging with \(B \to \infty\), while practical bagging/random forests use finite \(B\). This does not invalidate the result, but it limits how directly it transfers to common implementations.
- **Coverage results for some real-data sections are deferred away from the main text.** Since coverage validity is a central claim, pushing those numbers to the appendix weakens the main empirical presentation.

### Trivial
- None.

## Nice-to-Haves
- A clearer practitioner-oriented discussion of how to estimate or upper-bound \(\rho_i,\nu_i,\varphi_i\) in common models.
- Larger-scale experiments with much larger \(m\) to more directly demonstrate the one-fit advantage.
- Additional empirical sensitivity analysis showing how interval width changes with the conservativeness of the stability bound.
- A more explicit statement that Theorem 1 provides **marginal per-test-point coverage**, not a stronger joint guarantee over all \(m\) predictions.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Missing comparisons to jackknife+/CV+ or other related methods.** This may be a reasonable suggestion for future work, but under the review rules I should not treat missing related baselines/works as a core weakness without external confirmation and necessity. The paper’s main comparison to FullCP, SplitCP, and RO-StabCP is sufficient to assess its stated contribution.
- **Criticism that only marginal coverage is provided rather than conditional coverage.** This is a generic limitation of standard conformal prediction rather than a paper-specific flaw, and the paper does not claim conditional coverage for its main method.
- **Formatting/parser artifacts** such as odd figure labels (“CheckCP”) or minor display issues. These are extraction problems, not paper problems.
- **Complaint that FullCP complexity in Table 1 depends on grid search over \(\mathcal Y\).** The paper explicitly defines this as a practical accounting for the implementation used, so this is not a substantive flaw.
- **General reproducibility nitpicks about omitted implementation minutiae.** The paper gives the main experimental settings and such omissions are not central here.

## Novel Insights
The paper is strongest when viewed narrowly as a **many-query acceleration of stable conformal regression**, not as a broad end-to-end theory for all extensions it touches. The most compelling synthesis is that the work makes a clean conceptual shift: instead of approximating the effect of changing the candidate label at a fixed test point (RO stability), it approximates the effect of not augmenting the training set with the test point at all (LOO stability). That shift is what unlocks the one-fit amortization, and it is both algorithmically meaningful and theoretically analyzable. However, the paper’s peripheral extensions—especially neural networks as actually run, and the screening/FDR application—are materially less mature than the core regression CP contribution. A stronger paper would foreground this asymmetry instead of presenting all pieces as equally established.

## Suggestions
- **Narrow the claims in the abstract and conclusion** unless Section 6 is repaired. In the current form, the strongest fully supported contribution is LOO-StabCP for regression prediction intervals with marginal coverage.
- **Either prove a theorem for LOO-cFBH or substantially soften the screening claims.** At minimum, resolve the current contradiction between the Figure 4 description and the surrounding text, and make explicit whether the evidence is only empirical.
- **Reframe the neural-network results as heuristic evidence** unless valid upper bounds are used in the experiment. If keeping the current experiment, explicitly separate “theoretically guaranteed” from “practically motivated heuristic.”
- **Make the FullCP comparison fairer or qualify it more carefully.** If equal training budgets are infeasible, say clearly that the runtime comparison is primary and that interval-length comparisons to FullCP are only approximate.
- **Add practical guidance for stability constants.** Even an appendix explaining how to compute or conservatively estimate \(\rho_i,\nu_i,\varphi_i\) for the models used would strengthen practical value.
- **Expand experiments in the high-\(m\) regime** where the main contribution should shine most clearly.

## Score and Decision
**Evaluation by axis:**  
- **Originality:** good. The shift from RO to LOO stability for amortized conformal prediction is genuinely novel and conceptually clean.  
- **Importance of the research question:** good. Efficient full-conformal-style prediction without sample splitting is a meaningful problem.  
- **Whether the claims are well supported:** mixed. The core regression claim is reasonably supported; the neural-network and screening claims are overstated relative to the actual theory/evidence.  
- **Soundness of experiments:** moderate. The experiments are helpful and generally aligned with the paper’s goal, but the unequal FullCP training budget and the heuristic neural-network bounds weaken parts of the empirical case.  
- **Clarity of writing:** generally good in the main methodological sections, weaker in the application section because of the internal inconsistency.  
- **Value to the community:** meaningful, especially for researchers interested in stable conformal prediction and amortized uncertainty quantification.

**Calibration against human-reviewed anchors:**  
- I compared this paper to **“Approximating Full Conformal Prediction for Neural Network Regression with Gauss-Newton Influence”** (`vcX0k4rGTt`, decision Accept, scores 6/8/5/6). That paper also tackled acceleration of full conformal, had real methodological novelty, but faced questions about validity/approximation and limited baseline scope. The present paper is **similarly strong on core idea/theory**, and in fact stronger on exact finite-sample coverage for its main method, but **weaker in overclaiming** via the unsupported screening extension and mismatched neural-network theory/practice.
- I also compared against **“Conformal Prediction with Model-Aware Debiasing”** (`wdzCyr1stL`, decision Reject, scores 3/6/3/3), which had some interesting ideas but was undermined by limited and unconvincing evidence. The present submission is **clearly above** that level because its central method is sharper, cleaner, and more convincingly supported.
- As a lower-middle anchor, **“Conformal Training with Reduced Variance”** (`LxkgScfHKf`, decision Reject, scores 5/3/5/5) shows how papers with interesting ideas but serious theory/experiment gaps were treated. This submission is **better than that** because its main theorem and main method are much more solid.
- As a positive anchor on sound-but-not-perfect CP work, **“Robust Conformal Prediction with a Single Binary Certificate”** (`ltrxRX5t0H`, decision Accept, scores 6/6/8/8) is stronger than the current paper because its empirical claims and theoretical framing are better aligned.

Relative to these anchors, I place this paper in the **borderline accept / weak accept quality band on its core contribution**, but the abstract-level overclaiming around Section 6 pulls it down. If judged strictly on the main regression method, I would lean somewhat higher; judged on the paper as submitted, I land at:

**Score: 6.0 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>