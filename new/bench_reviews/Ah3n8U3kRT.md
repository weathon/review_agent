## Summary

The paper proposes a median-clipping framework for zeroth-order (ZO) non-smooth convex optimization and multi-armed bandits (MAB) under symmetric heavy-tailed noise with arbitrarily small moment index $\kappa > 0$.  By introducing a novel symmetric ZO oracle (Assumption 3) and a batched median gradient estimator, the authors derive high-probability convergence rates for ZO-clipped-med-SSTM and ZO-clipped-med-SMD that avoid the $\kappa \to 1$ degeneration of prior clipped-mean methods, and apply the same tool to MAB via Clipped-INF-med-SMD.

## Strengths

- **Conceptually interesting gap.** The paper identifies that median estimators can yield finite-second-moment gradient estimates even when the raw noise has infinite variance, opening a path to non-degenerate ZO rates for any $\kappa > 0$.  The link between median-clipped ZO optimization and heavy-tailed bandits is clean and well-motivated.
- **Novel symmetric oracle assumption.** Assumption 3 decouples the noise distribution from the smoothing direction and explicitly models symmetry and heavy tails, which is a meaningful departure from prior fixed-seed oracle models (Section 3.1.1).
- **Core ZO rate.** Theorem 1 and Table 1 show that for the Lipschitz oracle the method achieves $\tilde{\mathcal{O}}(d^{2}\varepsilon^{-2})$ for any $\kappa > 0$, matching the optimal bounded-variance ZO complexity and avoiding the blow-up of prior clipped-ZO methods as $\kappa \to 1$.
- **Extensions to structured objectives.** Remarks 1–2 show transparent adaptations to smooth and Polyak–Łojasiewicz settings, demonstrating the flexibility of the framework.

## Weaknesses

### Fatal
None.

### Major
- **Overclaimed MAB optimality.** Theorem 3 states $\tilde{\mathcal{O}}(\sqrt{dT})$ regret and calls it “optimal” because it matches the lower bound $\Omega(\sqrt{dT})$ for stochastic MAB with *bounded variance* ($\kappa = 2$).  However, the paper’s setting explicitly allows $\kappa < 2$ (unbounded variance).  Standard heavy-tailed bandit lower bounds for distributions with bounded $\kappa$-th moment scale as $\Omega(T^{1/\kappa})$, which is strictly larger than $\sqrt{T}$ when $\kappa < 2$.  The authors neither prove a lower bound for their symmetric class nor rigorously argue why existing lower bounds are inapplicable under symmetry.  Without a matching lower bound for the actual problem class, the optimality claim is unsubstantiated.  (Section 4, Theorem 3, Contribution list.)
- **Empirical claims contradicted by the paper’s own figures.** Figure 1 shows HTINF achieving lower mean regret ($\sim$0.1) and higher best-arm probability ($\sim$0.9) than Clipped-INF-med-SMD ($\sim$0.2 regret, $\sim$0.6 probability).  Nevertheless, the text asserts that “HTINF and APE do not have convergence in probability, while our … does” and that the method demonstrates “superior performance.”  This directly misrepresents the experimental outcome.  (Section 5.1, Figure 1.)
- **Portfolio experiment does not evaluate the proposed bandit algorithm.** The authors explicitly state that the cryptocurrency portfolio task provides full feedback for all assets and that they modified Algorithm 3 to use it (line 4 is executed for every asset).  This removes the bandit structure entirely, so Figure 2 provides no empirical support for Theorem 3.  (Section 5.2.)

### Minor
- **Accelerated method is empirically dominated by a simple baseline.** In all four plots of Figure 3, the theoretically accelerated ZO-clipped-med-SSTM converges slower than the non-accelerated ZO-clipped-med-SGD.  The paper only highlights that median methods outperform non-median baselines; it does not acknowledge that its main accelerated algorithm is outperformed by a trivial SGD baseline in practice.  The practical value appears to come from median clipping rather than from the accelerated scheme.  (Section 5.3, Figure 3.)
- **Theory-practice mismatch on median size.** For $\kappa = 0.75$, the paper reports the best median size is $m = 2$, whereas Lemma 1 requires $m > 2/\kappa \approx 2.67$ (i.e., $m \ge 3$).  The paper offers no explanation for why the theoretical sufficient condition is violated in the reported best setting, leaving it unclear whether the condition is merely conservative or actually necessary.  (Lemma 1, line 392.)
- **Overstated abstract claim.** The abstract states the methods “dramatically outperform” SOTA for $\kappa \le 1$.  This is contradicted by Figure 1 (HTINF performs better on the synthetic bandit task) and only partially supported by Figure 3 (median vs. non-median).  The claim should be tempered to reflect the mixed empirical results.  (Abstract, Section 5.)
- **Missing clipping norm in Theorem 3.** Algorithm 3 applies $\mathrm{clip}_q$, but Theorem 3 lists parameters $m, \tau, \nu, \lambda$ without specifying the norm $q$, creating a minor gap between the algorithm description and the theorem statement.  (Section 4.)

### Trivial
- **Notational inconsistency in the oracle definition.** The paper defines the two-point oracle using “the same realization $\xi$” but then illustrates the independent oracle with independent $\xi_x, \xi_y$.  A cleaner separation of the two models would improve readability.  (Section 3.1.1.)

## Nice-to-Haves
- Report confidence intervals or standard errors in Figure 2 (portfolio profit) and more rigorous statistical summaries in Figure 1; mean-over-runs plots without measures of uncertainty are hard to interpret.
- Include ablations that sweep the median size $m$ (including values below the theoretical threshold) to clarify the necessity of the $m > 2/\kappa$ condition.
- Add a short remark explaining why a lower bound on the noise density at the median is not required under Assumption 3, since standard median-variance analyses often assume $p(0) > 0$.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Invalidity of Lemma 1 due to missing density lower bound.** The reviewer claims that Lemma 1 cannot hold uniformly because Assumption 3 is only an upper bound on the density, permitting distributions with arbitrarily small density at the median.  This overlooks that Assumption 3, combined with the fact that $p(u|x,y)$ is a probability density, yields a uniform bound on the CDF tails (by integrating the pointwise density envelope).  That tail bound in turn controls the moments of the sample median for $m > 2/\kappa$ without requiring $p(0) > 0$.  The proposed counterexample (a bimodal distribution with mass far from zero) cannot satisfy the total-mass constraint for fixed $\Delta$.  Because the proof is deferred to the appendix and the lemma is not self-evidently false, this criticism is unsubstantiated.
- **Formatting and style nitpicks** (e.g., clipping-level specification in Algorithm 1, nested max expression in Theorem 1, Table 1 obscuring $\Delta$, typos/grammar).  These are parser artifacts or minor presentation details that do not affect scientific merit.
- **Missing appendix proofs and missing related works.**  The parser strips appendices and references from all papers; they exist in the original submission.

## Novel Insights

The paper’s central conceptual insight—that symmetry, combined with a median estimator, can remove the $\kappa > 1$ barrier in zeroth-order optimization—is genuinely novel and could catalyze further work on distributional assumptions beyond moment bounds.  If the theoretical guarantees hold up under full scrutiny, the framework could be extended to other robust estimators (e.g., trimmed means) under shape constraints, or to first-order methods with symmetric heavy-tailed noise.

## Suggestions
- Either prove a lower bound for symmetric $\kappa$-heavy-tailed bandits or explicitly reframe Theorem 3 as an upper-bound contribution rather than “optimal.”
- Re-run the MAB experiment under true bandit feedback (one arm per round) with $d > 2$, and report comparisons honestly, including the observation that HTINF outperforms the proposed method on the synthetic instance shown in Figure 1.
- Acknowledge in the text that ZO-clipped-med-SSTM is outperformed by ZO-clipped-med-SGD in Figure 3, and reposition the practical contribution around the median estimator rather than acceleration.

## Score and Decision

**Calibration.**  
- *High anchor:* `/home/wg25r/review_agent/human_reviews/AfhNyr73Ma.md` (avg 7.00, Accept poster) presents a solid ZO stability framework with clear theory and experiments; our submission falls well below because of its unsubstantiated optimality claims and empirical contradictions.  
- *Medium anchor:* `/home/wg25r/review_agent/human_reviews/A7LTIuhH4k.md` (avg 5.00, Reject) has an interesting idea but a flawed computational-cost claim; our paper shares the pattern of overclaiming and adds empirical misrepresentation, pushing it below the medium band.  
- *Low anchor:* `/home/wg25r/review_agent/human_reviews/7t8aKBeATc.md` (avg 3.50, Reject) is a ZO method that claimed optimality without a lower-bound discussion and had weak experiments—issues very similar to ours.  Our paper is more novel and better written, so it sits slightly above this anchor.  Another low anchor, `/home/wg25r/review_agent/human_reviews/IQDSUpp69I.md` (avg 4.33, Withdrawn), was penalized for redefining the optimality gap to make unfair comparisons; our MAB optimality overclaim is comparable.  Relative to these, the paper under review is positioned around **4.0**.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>