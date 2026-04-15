Now let me search for comparable papers to calibrate the score.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proposes F²SA-*p*, a family of fully first-order methods for stochastic bilevel optimization that leverages *p*-th order finite-difference approximations of the hypergradient. The key conceptual insight is reinterpreting F²SA as a forward finite-difference estimator of the hypergradient, which naturally motivates higher-order variants. Under *p*-th order smoothness assumptions on the lower-level variable **y**, the method achieves Õ(*p*κ^{9+2/*p*}ε^{-4-2/*p*}) SFO complexity, improving the best-known Õ(ε^{-6}) bound for first-order smooth problems. A matching Ω(ε^{-4}) lower bound is proved via a clean fully separable construction, establishing near-optimality of F²SA-*p* in the highly-smooth regime where *p* = Ω(log ε^{-1}/log log ε^{-1}).

---

## Claims and Support

**Claim 1: Reinterpreting F²SA as a forward finite-difference approximation of the hypergradient.**
**Verdict: Well-supported.** The paper establishes the perturbed lower-level problem *g*_ν(**x**,**y**) = *g* + ν*f*, shows ∂²/∂ν∂**x** *ℓ*_ν(**x**)|_{ν=0} = ∇φ(**x**) (Eq. 8), and then explicitly demonstrates (Eq. 9) that the F²SA penalty gradient is exactly a forward-difference quotient. This conceptual reframing is coherent, directly verifiable in Section 3.1.

**Claim 2: F²SA-*p* achieves Õ(*p*κ^{9+2/*p*}ε^{-4-2/*p*}) SFO complexity.**
**Verdict: Supported, with normal theory-paper caveats.** The logical chain is: (a) Lemma 3.1 gives *p*-th order finite-difference error O(ν^p) under *p*-th order Lipschitz smoothness; (b) Lemma 3.2 shows ∂^{p+1}/∂ν^p∂**x** *ℓ*_ν(**x**) is O(κ^{2p+1}*L̄*)-Lipschitz in ν, derived from the Faà di Bruno formula; (c) combined, the hypergradient approximation bias is O(ν^p), enabling ν = O(ε^{1/p}) instead of O(ε); (d) Theorem 3.1 gives the full complexity. The proofs are in the appendix, which is standard for a theory paper. The claim about y-only smoothness (Assumption 2.5) vs. joint (x,y) smoothness (Huang et al. 2025) is stated clearly in the assumptions.

**Claim 3: F²SA-2 improves SFO from Õ(ε^{-6}) to Õ(ε^{-5}) under second-order smoothness.**
**Verdict: Supported** (is the *p*=2 specialization of Claim 2), and the paper makes the additional argument that F²SA-2 requires only 2 lower-level problems per iteration (same as F²SA) and degrades gracefully to first-order guarantees if second-order smoothness fails (Section 3.3). The paper correctly notes F²SA-2 costs the same per-iteration as F²SA for even *p*=2.

**Claim 4: Ω(ε^{-4}) lower bound extends to stochastic bilevel optimization.**
**Verdict: Well-supported.** The construction uses *f*(**x**,**y**) ≡ *f*_U(**x**) and *g*(**x**,*y*) = µ*y*²/2, making the lower-level variable trivially decoupled. The paper verifies this satisfies all smoothness assumptions, and the single-level lower bound transfers cleanly.

**Claim 5: Acceleration relies only on higher-order smoothness in **y**, not joint (x,y) smoothness.**
**Verdict: Supported at the statement level.** Assumption 2.5 explicitly requires only y-directional higher-order derivatives. Lemma 3.2 is stated under exactly these conditions. The comparison to Huang et al. (2025) who require joint smoothness is explicit in Section 2.2. The full technical justification is in the appendix proof of Lemma 3.2 (using Faà di Bruno), which is appropriate.

**Claim 6: The method is effective in practice.**
**Verdict: Partially supported.** Experiments on "20 Newsgroup" learn-to-regularize demonstrate the method works. However, the x-axis is outer-loop iterations rather than SFO calls, which is a genuine issue (see Weaknesses).

---

## Strengths

- **Novel and insightful reformulation.** Reinterpreting F²SA as a forward finite-difference approximation (Eqs. 8–9) is conceptually elegant and provides a principled, non-ad-hoc route to higher-order extensions. This is a genuinely new perspective on penalty-based bilevel methods.
- **Significant and continuous complexity improvement.** The improvement from Õ(ε^{-6}) to Õ(*p*ε^{-4-2/*p*}) is meaningful for all *p* ≥ 2. Notably for *p*=2, the method solves only 2 lower-level problems per outer iteration (identical cost to F²SA), and degrades gracefully if second-order smoothness fails (Section 3.3: "at least as good as F²SA").
- **Clean lower bound construction.** The fully separable construction in Section 4 avoids technical issues present in prior bilevel lower bounds (Dagréou et al. 2024; Kwon et al. 2024a) and cleanly establishes near-optimality in the highly-smooth regime.
- **Tighter auxiliary result (Lemma 3.2).** The bound O(κ^5*L̄*) for *p*=2 improves on the O(κ^6*L̄*) in Chen et al. (2025b), which is noted as independently useful.
- **Clear problem scope and honest open problems.** The paper is candid about unresolved questions (κ-dependence gap, small-*p* regime), and the comparison to joint-smoothness methods (Huang et al. 2025) in Section 2.2 is precise and accurate.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Experimental evaluation is misaligned with the paper's central claim.** The main contribution is an improved **SFO complexity** bound, but Figure 1 plots performance vs. **outer-loop iterations**. For even *p*, F²SA-*p* runs *p* parallel inner-loop problems per outer iteration, and for odd *p*, it runs *p*+1. Thus comparing F²SA (1 problem/iteration) with F²SA-5 (5 problems/iteration) by iteration count conceals the actual cost and can make the proposed method look better per iteration while consuming more total oracle calls. For a paper whose central claim is about oracle complexity, plotting results in a way that ignores per-iteration cost of the proposed method is a fundamental misalignment. The evaluation should be redone with SFO calls (or wall-clock time) on the x-axis.

- **Large condition number dependency remains unresolved and practically limiting.** The bound κ^{9+2/*p*} is very large; even in the near-optimal regime (*p* → ∞) the complexity remains Õ(κ^9 ε^{-4}). The concurrent lower bounds (Ji 2025: Ω(κ^{5/2}ε^{-4}); Chen & Zhang 2025: Ω(κ^4ε^{-4})) indicate a substantial gap of at least Ω(κ^5) in κ-dependence. The paper acknowledges this gap (Table 1, Open Problems) but provides no analysis of whether the gap is inherent to the finite-difference approach or an artifact of the proof technique. This significantly limits practical utility for ill-conditioned problems.

### Minor

- **Normalized gradient step lacks theoretical justification.** Algorithm 1 uses a normalized gradient update *x*_{t+1} = *x*_t − η_x Φ_t / ‖Φ_t‖, which differs from standard bilevel algorithm practice. Remark 3.1 explicitly states this is for analytical convenience and that the authors "believe" standard gradient steps also work — but provide no proof. This leaves the core algorithm partially unvalidated in its most practically natural form.

- **Limited experimental scope.** The main text experiments consist of a single dataset (20 Newsgroup) and single task (learn-to-regularize). No ablation study on *p* with matched total oracle budgets, no variance across seeds, no experiments on the data hyper-cleaning task (Example 2.1 is introduced as motivation but never tested), and no study of how the method behaves as a function of condition number κ or noise level σ. The breadth of practical claims in the introduction is not matched by the experimental evidence.

- **Parameter settings require knowledge of unknown constants.** Theorem 3.1's parameter choices (ν, η_x, η_y, S, K, T) depend on κ, *L̄*, µ, σ², and ε, all of which are typically unknown in practice. The paper does not discuss sensitivity to misspecification of ν, which is the most consequential parameter (controlling the approximation order).

### Trivial

- No error bars are reported in the experiments.
- The odd-*p* algorithm is deferred entirely to the appendix without even a sketch in the main text.

---

## Nice-to-Haves

- A plot of hypergradient approximation error |Φ_t − ∇φ(**x**_t)| vs. ν for different *p* values would directly verify the O(ν^p) bias claim and illustrate the core theoretical mechanism.
- A combined experiment comparing F²SA vs. F²SA-2 at matched **total SFO budget** (not iterations) would be the single most impactful empirical addition.
- Even a brief discussion of whether the finite-difference framework is compatible with variance-reduction (mentioned as future work in Section 6) would help readers assess the long-term trajectory.
- Analysis for the standard (unnormalized) gradient step, at least as a corollary or informal argument beyond the "we believe."

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**From Harsh Critic:**

- *"The key novelty claim rests on a technical lemma not sufficiently substantiated in the paper body"* (as a major criticism): Lemma 3.2 is explicitly stated in the main text with its full conditions (Assumption 2.5 = y-only smoothness, ν ∈ (0, 1/(2κ)]), and the appendix proof is standard practice for a theory paper. The concern that "hidden x-derivatives" might appear is speculative without reading the appendix; as stated, the scope is clear. This is **weakened** to a minor concern about proof transparency, not a major critique.

- *"Claim 2 partially supported because Lemma 3.2 is only summarized"*: For a theory submission, having detailed proofs in an appendix is the norm. The main text provides the result and its derivation path; demanding full proof in the body is not standard in this field.

- *"Claim 5 partially supported — it is not obvious that no hidden mixed x-smoothness is used"*: This is speculative. Assumption 2.5 is stated as requiring only y-directional smoothness, and the comparison to Huang et al. (2025) explicitly distinguishes the two. Without evidence of a flaw in the appendix proof, this is not a valid weakness.

**From Human Finder:**

- *"Missing related works"*: Removed per hard rules — no external sources available to confirm existence.

- *"Parameter settings require knowledge of problem-dependent constants" being cited as a major weakness*: This is a weakness of essentially all complexity-optimal methods in stochastic optimization and is thus generic. Kept as Minor only.

- *"The practical advantage over HVP-based methods is not fully demonstrated"*: The paper explicitly scopes itself to the fully first-order setting (no stochastic Hessian assumption) and provides theoretical justification for the comparison. The claim about "scaling to 32B LLM training" is a citation to Pan et al. (2024) and applies to the broader F²SA family, not a claim made about the F²SA-p extension specifically.

---

## Novel Insights

The paper's most valuable contribution is the identification that **F²SA's gradient approximation is exactly a forward finite difference applied to ∂*ℓ*_ν/**x** as a function of ν** — a reformulation that the original F²SA authors did not make explicit. This observation is more than cosmetic: it immediately reveals that the *p*=1 character of F²SA's approximation error is not fundamental but is rather an artifact of using the simplest finite-difference scheme. The extension to *p*-th order differences is conceptually natural once this is recognized, but the technical work to establish that the higher-order regularity can be controlled using only y-directional smoothness (Lemma 3.2 via Faà di Bruno) is the nontrivial ingredient. This mechanism is orthogonal to variance reduction and provides a new dimension along which stochastic bilevel methods can be improved: rather than reducing variance of the stochastic gradient, one reduces the deterministic bias of the hypergradient approximation by leveraging higher-order regularity of the lower-level solution map.

---

## Suggestions

1. **Replace iteration-axis plots with SFO-call or wall-clock plots.** This is the highest-priority fix. Show F²SA-*p* for *p* ∈ {2,3,5} vs. F²SA at matched total oracle budgets (F²SA-2 uses 2× inner loops per outer step; F²SA-5 uses 5×). This directly validates the paper's core complexity claim.
2. **Add a hypergradient bias ablation.** Plot |Φ_t − ∇φ(**x**_t)| as a function of ν for *p* = 1, 2, 5 to verify the O(ν^p) scaling predicted by Lemma 3.1+3.2.
3. **Strengthen the κ-dependence discussion.** Even a brief derivation sketch explaining which term in Lemma 3.2 contributes κ^{2p+1} would help readers assess whether the gap with lower bounds is proof-artifact or fundamental.
4. **Add an experiment or formal result for the unnormalized gradient step**, even for *p*=2 only. If the normalized step is truly only for analytical convenience, a numerical demonstration of identical behavior would substantially strengthen the practical relevance of Algorithm 1.

---

## Score and Decision

**Calibration:**

- **CvYBvgEUK9** (Penalty methods for nonconvex-nonconvex bilevel, accepted Spotlight): Scores 6,6,6,5,8. Addresses a more general problem class (nonconvex lower-level) with first-order methods and ε^{-7}/ε^{-5} bounds. The paper under review has stronger theoretical tightness (near-optimal lower bound, y-only smoothness condition) but a more restrictive problem class (strongly-convex lower level).

- **ZjOXuAfS6l** (Lower bounds for adaptive algorithms, accepted Poster): Scores 8,6,6,6,5. Pure theory paper, no experiments. Paper under review adds an experimental section (however weak) on top of comparable theoretical contributions.

- **otU31x3fus** (Stochastic second-order method with lower bounds, accepted Poster): Scores 6,6,1,8. A similar "upper bound + lower bound" structure for a related optimization setting.

**Assessment relative to anchors:** The paper under review has a genuinely novel conceptual contribution (the finite-difference reinterpretation), a principled family of algorithms with improved complexity, and a clean lower bound. This places it at or above the CvYBvgEUK9 spotlight level in theoretical content. However, the experimental section is weaker than that paper (single benchmark, misaligned evaluation axis), and the κ-dependence gap remains large. Balancing these, the paper is solidly above the acceptance threshold but not at the level of a high-confidence accept.

**Originality:** High — the finite-difference reformulation is genuinely novel.
**Importance of research question:** High — closing the gap between Ω(ε^{-4}) and Õ(ε^{-6}) for first-order bilevel methods is an important open problem.
**Claims supported:** Mostly yes, with the main weakness being the evaluation metric in experiments.
**Soundness of experiments:** Moderate — only one benchmark, no SFO-based evaluation.
**Clarity of writing:** Good — the paper is well-organized and the contributions are clearly delineated.
**Value to the community:** High — the framework is reusable and the p=2 result ("central difference bilevel") is immediately applicable.

**Score: 7.0** — Solid accept (poster). The theoretical contributions are strong and the paper addresses a genuine open problem. The experimental weakness (iteration-based evaluation) should be flagged as a required revision, but does not undermine the theoretical claims.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>