Now let me search for calibration papers to properly score this work.Now I have enough to write the final review. Let me synthesize everything.

---

## Summary

The paper proposes median clipping for zeroth-order (ZO) non-smooth convex optimization and multi-armed bandits (MAB) under symmetric heavy-tailed stochastic noise with any κ > 0. By introducing a novel oracle assumption (Assumption 3) and a batched median gradient estimator (Lemma 1), the authors achieve non-degenerating convergence rates that match the optimal bounded-variance case—specifically Õ(d²ε⁻²) iterations for ZO optimization and Õ(√dT) regret for MAB—whereas prior methods require κ > 1 and their rates degenerate as κ → 1. The contribution fills a genuine gap in the literature since no prior ZO work handles κ ≤ 1.

---

## Strengths

- **Theorem 1/2 and Table 1 establish non-degenerating convergence for any κ > 0**: Prior ZO work (ZO-clipped-SSTM [20]) achieves Õ((√dε⁻¹)^{κ/(κ−1)}) iterations, which diverges as κ → 1. Table 1 shows the proposed ZO-clipped-med-SSTM achieves Õ(max{d^{1/2}M₂ʼ/ε, (dM₂ʼ/ε)²/b}) for all κ > 0—a qualitatively different (non-degenerating) rate. This is a genuine advance.

- **Lemma 1 is the pivotal technical result**: It establishes that the BatchMed estimator (Eq. 9) is unbiased for ∇f̂_τ and has bounded second moment for any κ > 0 (Eqs. 10–11), given m > 2/κ median samples. This is non-trivial because the raw noise can have unbounded first moment (Cauchy), and the proof technique differs fundamentally from first-order median clipping.

- **Assumption 3 enables fine-grained exploitation of noise symmetry**: The assumption is placed on φ(ξ|x,y) rather than on ξ itself, which is the key design choice allowing control of symmetry and tail behavior independently. Section 3.1.1 explains this distinction clearly, and Remark 3 confirms backward compatibility with standard assumptions for κ ∈ (1, 2].

- **Theorem 3 gives optimal Õ(√dT) MAB regret for any κ > 0**: Prior heavy-tail MAB results achieve Õ(d^{(κ−1)/κ}T^{1/κ}) [4, 18, 22, 47], which is suboptimal in both d and T for κ < 2. Theorem 3 matches the Ω(√dT) lower bound—a genuine improvement over all prior heavy-tail MAB algorithms for symmetric noise.

- **Figure 3 provides clean experimental validation for the ZO contribution**: The four subplots spanning κ = 0.75, 1.0, 1.25, 1.5 show that median-based methods dramatically outperform non-median methods for κ ≤ 1 (where baselines break theoretically), while remaining competitive for κ > 1. Oracle calls (not just iterations) are used as the x-axis, providing a fair comparison.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Figure 1 directly contradicts the paper's stated conclusion in §5.1**. The extracted figure caption reads: "Clipped-INF-med-SMD (blue) maintains regret around 0.2; APE (red) around 0.25; HTINF (green) decreases to around 0.1" for expected regret; and "HTINF quickly reaches ~0.9 probability of best arm selection, while APE and Clipped-INF-med-SMD both stabilize around 0.6." Yet the paper concludes: *"HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does, which confirms the efficiency of the proposed method."* On both displayed metrics, HTINF outperforms the proposed method (lower regret, higher best-arm probability). The paper likely intends to refer to the *tail behavior* (the 0.05–0.95 percentile bands for regret) being wide for HTINF despite its good mean, indicating that HTINF achieves good results inconsistently across runs—but this is never explained in the text. As written, the verbal conclusion is contradicted by the quantitative values, and a reader cannot independently verify the "convergence in probability" claim without knowing that the shaded bands for HTINF are much wider than for the proposed method. This must be corrected with an explicit description of the percentile bands and a clear explanation of what "convergence in probability" means in this context.

- **The MAB experiment is insufficient to support the abstract's claim of "dramatically outperform SOTA"**. The main-body MAB experiment uses only d = 2 arms with a single Cauchy-type noise at κ = 1. HTINF is designed for κ > 1, so testing at exactly κ = 1 is borderline. The theoretical advantage materializes at κ < 1. An experiment explicitly at κ = 0.5 or κ = 0.75 with d > 2 is needed to substantiate the dramatic outperformance claim. Additional experiments are deferred to Appendix D.1 and are not visible in the main body.

### Minor

- **The cryptocurrency portfolio experiment (§5.2) does not test the MAB problem**. As the authors acknowledge, the portfolio setting has *full feedback* (all assets observed each step), which is not the bandit setting. The baselines (hold ETH, Efficient Frontier) are not bandit algorithms. This makes the section a demonstration of practical usefulness rather than evidence that the MAB algorithm outperforms competing MAB methods on real data. This should be framed more carefully—currently it is presented as supporting the MAB claim, which it does not.

- **The oracle call overhead as κ → 0 is not discussed in the abstract or the comparison section**. Each iteration of ZO-clipped-med-SSTM requires (2m+1)·b = (4/κ+3)·b oracle calls, and the constant σ² in Lemma 1 grows as (4/κ)^{2/κ} → ∞ as κ → 0. Table 1 does note "b/κ calls" honestly, but the headline abstract comparison ("Õ(d²ε⁻²) iterations") refers to iterations only. For κ → 0, the total oracle cost diverges, and the regret constant c² in Theorem 3 also diverges super-polynomially. Section 6.1 briefly notes the κ → 0 limitation in context of the adaptive scheme, but the paper should be explicit upfront that the "κ-uniform optimality" claim holds for fixed κ with a κ-dependent constant.

- **Figure 3 shows SGD variants uniformly outperforming SSTM variants in all plots**, which is flagged in the figure caption but not discussed in the text. If ZO-clipped-SGD (first-order rate, no acceleration) beats ZO-clipped-med-SSTM (accelerated) on every tested value of κ, the motivation for the SSTM-based algorithm needs discussion. This could reflect the constant in the SSTM bound being larger in practice, or the asymptotic regime not yet having been reached with 2×10⁷ samples.

### Trivial

- §1.1 "Theory I" refers to "Assumption 4 (§3.1)" as the novel oracle, but the main body only contains Assumptions 1–3; the actual novel assumption is labeled **Assumption 3**. This is a mismatch that creates confusion.

---

## Nice-to-Haves

- A plot of the effective constant in Theorem 3 (or σ² from Lemma 1) as a function of κ would help readers understand the practical regime where the method's advantage is strongest versus where oracle cost dominates.
- An explicit statement of *total oracle call complexity* (not just iterations) in both the abstract and Table 1 would make the comparison with bounded-variance methods fully transparent.
- The §5.1 figure should label the percentile band widths for each method explicitly in the text, so the "convergence in probability" claim is self-contained and verifiable.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Assumption 4 is missing from the main text"** (Harsh Critic §1.1): The contribution §1.1 refers to "Assumption 4," but the actual novel assumption in the main body is Assumption 3. This is an internal numbering inconsistency, not a missing component. The assumption is present and fully stated. Removing as a fatal/major concern; kept as a trivial note above.

- **"Optimal rates claim conflates iteration complexity with oracle call complexity"** (Harsh Critic #2 as formulated as a methodological gap): Table 1 explicitly states "b/κ calls" and Theorem 1 states "Each iteration requires (2m+1)·b oracle calls." The paper is transparent about this. The concern about the *abstract* not stating total oracle complexity is retained as a minor weakness, but the original formulation as a "systematic misrepresentation" was too strong.

- **"Clipped-INF-med-SMD achieves ~0.6 probability vs HTINF's ~0.9, therefore the method is worse"** (Harsh Critic #1, strong version): The relevant claim is about variance/tail behavior, not just mean performance. The weakness is retained in a more calibrated form as a major issue, but the version claiming the paper's method is "worse" overall goes beyond what can be confirmed without seeing the actual figure's shaded bands.

- **Strength: "MAB experiments demonstrate convergence in probability where baselines fail"** (Strength Finder): This directly conflicts with the verified Major weakness—Figure 1 shows HTINF outperforming the proposed method on both mean regret and best-arm probability. Removed as a strength.

- **Strength: "Real-world cryptocurrency portfolio application demonstrates practical relevance"**: The portfolio experiment uses full feedback, not the bandit setting. It tests a different problem with non-competing baselines and does not demonstrate the MAB algorithm's superiority. Too generic and potentially misleading; removed.

---

## Novel Insights

The paper's most significant insight is that noise *symmetry* can be exploited in the zeroth-order setting through coordinate-wise median of direction-projected gradient differences—an idea that does not trivially transfer from first-order median clipping because the direction vector e is sampled on the unit sphere. The key is that flipping ξ → −ξ (symmetry) flips g(x, e, ξ) → −g(x, e, ξ) + 2∇f̂_τ(x), which makes the component-wise median of 2m+1 samples an unbiased estimator of ∇f̂_τ(x) with bounded second moment, regardless of how heavy the tails of ξ are. This observation—that symmetry rather than bounded moments is what enables optimal ZO rates—is the core insight that unifies the paper's contributions to both optimization and bandits.

---

## Suggestions

1. **Rewrite the §5.1 discussion**: Explicitly state that Figure 1 is evaluated at κ = 1, where HTINF has no convergence *guarantee*. Show the shaded bands for HTINF explicitly and explain that while HTINF's mean looks good on this single instance, it has wide percentile bands indicating high variance. Alternatively, show a κ < 1 case (e.g., κ = 0.5) where HTINF provably fails and the proposed method converges reliably.

2. **Revise the abstract**: Replace "iterations" with "oracle calls after accounting for per-iteration cost" or add a sentence clarifying that each iteration costs O(b/κ) oracle calls, so total oracle complexity is Õ(d²ε⁻²·b/κ) for the Lipschitz oracle case.

3. **Add a MAB experiment at κ < 1 and d > 2**: The theoretical advantage over HTINF/APE materializes exactly for κ ≤ 1 and larger d. A d = 5 or d = 10, κ = 0.5 experiment showing both the mean and the percentile bands would directly confirm the paper's core MAB claim.

4. **Discuss Figure 3's SGD > SSTM finding**: Add two or three sentences explaining why the accelerated SSTM variant does not show empirical acceleration on the tested problem. This could be due to constants or regime issues, but it needs acknowledgment.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Human Score | Comparison |
|-------|------|----------------|------------|
| ZO stability analysis (AfhNyr73Ma) | `/home/wg25r/review_agent/human_reviews/AfhNyr73Ma.md` | 7.0 (Accept poster) | Similar ZO optimization topic; that paper provides a unifying stability framework. The current paper has a comparably strong theoretical contribution (extending κ coverage) with weaker experimental support. |
| ZO minimum-variance estimators (ywFOSIT9ik) | `/home/wg25r/review_agent/human_reviews/ywFOSIT9ik.md` | 6.8 (Accept spotlight) | Similar ZO optimization topic; that paper has cleaner experiments but its theoretical contribution is arguably comparable. |
| Gradient clipping in federated learning (BdPvGRvoBC) | `/home/wg25r/review_agent/human_reviews/BdPvGRvoBC.md` | 6.0 (Accept poster) | Similar clipping analysis; solid theory, moderate scope. The current paper's theoretical contribution is arguably broader (ZO + MAB + κ ≤ 1). |
| Soft clipping analysis (tsNLIBlG4p) | `/home/wg25r/review_agent/human_reviews/tsNLIBlG4p.md` | 4.0 (Reject) | Similar clipping topic; weaker because it lacked clarity about when soft > hard clipping and had slower-than-standard convergence rates. The current paper has clearer and stronger theoretical contributions. |
| Stochastic matching bandits (iKLSISIPH7) | `/home/wg25r/review_agent/human_reviews/iKLSISIPH7.md` | 4.8 (Reject) | Low-medium MAB anchor; rejected for limited contribution scope. |

**Reasoning**: The paper's theoretical core—Lemma 1 plus Theorems 1–3—is sound and fills a genuine gap. However, the major experimental weakness (Figure 1 contradicting the stated conclusion, the MAB experiment being too limited to support "dramatically outperform SOTA") drags the score below the accepted ZO papers in the 6.8–7.0 range. The paper is above the rejected papers at 4.0–4.8 because its theoretical contribution is real and substantial. Accounting for the major Figure 1 issue and the limited MAB experimental support, I position this at **5.5**—above the clear rejects (4.0), below the clean poster accepts (6.0–7.0), as it has publishable theory but presentation and experimental issues that need to be resolved for acceptance.

**Axes summary:**
- *Originality*: Good — extending median clipping to ZO/MAB for κ ≤ 1 is novel and non-trivial.
- *Importance*: Good — κ ≤ 1 noise (Cauchy etc.) is practically relevant and theoretically unexplored.
- *Claims well supported*: Mixed — ZO claims (Figure 3) are well supported; MAB claims (Figure 1) are not.
- *Soundness of experiments*: Weak for MAB, adequate for ZO.
- *Clarity*: Good for theory; problematic in §5.1.
- *Value to community*: Moderate-to-high for the optimization community working on heavy-tailed noise.

**Decision: Borderline — lean toward weak reject pending resolution of Figure 1 and MAB experimental support.**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>