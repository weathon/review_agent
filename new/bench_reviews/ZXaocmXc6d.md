Now I have a thorough understanding of the paper and calibration anchors. Let me write the consolidated review.

## Summary

The paper derives exact closed-form solutions for the learning dynamics of two-layer deep linear networks under λ-balanced initializations, where λ captures the relative scale between weight layers. The main technical contribution (Theorem 4.3) extends prior zero-balanced solutions (Fukumizu 1998; Braun et al. 2022) to the general λ-balanced case with unequal input/output dimensions, and characterizes how the transition function γ_α(t;λ) shifts from sigmoidal (rich, λ→0) to exponential (lazy, |λ|→∞) learning dynamics (Theorem 5.1). The paper also identifies a "semi-structured lazy" regime and applies these results to continual, reversal, and transfer learning.

## Strengths

- **Theorem 4.3 is a genuine and non-trivial technical advance.** Extending the exact Riccati-equation solutions from zero-balanced to λ-balanced initializations, and from equal to general input/output dimensions, requires careful algebraic work. The eigendecomposition in Lemma 4.2 and the resulting four-quadrant block separation of QQ^T(t) into Z₁(t), Z₂(t), and A(t) (Eqs. 13–15) is a meaningful extension of prior work. Figure 2 shows precise agreement between analytical solutions (dotted lines) and numerical simulations (solid lines) across all five rows (Loss, W₂W₁, W₁ᵀW₁, W₂W₂ᵀ, NTK) and three λ values.

- **The rich-to-lazy transition characterization via relative scale (Theorem 5.1, Eq. 17) is a clear conceptual contribution.** The paper proves that the transition function γ_α(t;λ) converges pointwise to a sigmoidal form as λ→0 and an exponential form as |λ|→±∞, with explicit limiting formulas. This establishes that relative scale—not just absolute scale—independently and continuously controls the learning regime. Figure 3 provides direct visual confirmation of the sigmoidal (λ=0, panel B) vs. exponential (λ=±2, panels A,C) singular value dynamics.

- **The reversal learning result is clean and consequential.** The paper proves that non-zero λ avoids the saddle-point pathology of reversal learning established in Braun et al. (2022), providing both theoretical proof and numerical illustration (Section 6, Appendix D.2). This is a crisp finding with a clear mechanism and a direct connection to neuroscience.

- **The identification of the "semi-structured lazy" regime is a conceptually novel observation.** Large |λ| creates a qualitatively distinct lazy regime where one layer becomes task-agnostic (identity-like) while the other remains small but task-specific (Theorem C.4, Fig. 4C). This partial preservation of task structure is not captured by prior lazy/rich characterizations and could inform understanding of fine-tuning dynamics.

- **Theorem 5.2's recovery of parameter dynamics from QQ^T is a useful structural result.** Showing that W₁ and W₂ can be recovered up to an orthogonal transformation, with singular values determined by S_λ(t) (Eq. 18), provides insight into how λ splits the representation across layers.

## Weaknesses

### Fatal
None.

### Major

- **The most insightful dynamical characterization—the sigmoidal↔exponential transition—requires the additional assumption of task-aligned initialization, and this condition is not made sufficiently prominent relative to the paper's claims.** Theorem 5.1 explicitly requires "a task-aligned initialization, as defined in Saxe et al. (2013)" (line ~163), meaning the initial weights share singular vectors with the task. The paper's contribution bullet states "We model the full range of learning dynamics from lazy to rich" (Section 1), which readers could reasonably interpret as more general than it is. The general solution (Theorem 4.3) applies without task alignment, but the clean regime characterization does not. The paper is transparent about the assumption in Section 5, but the framing of the contribution overstates its generality. This matters because task-aligned initialization is a very strong condition—essentially assuming the network already "knows" the relevant directions at initialization—and it is unclear how robust the sigmoidal/exponential distinction is to violations of alignment.

- **Applications to continual learning, transfer learning, and fine-tuning are listed as explicit contributions but are thin in the main text.** Section 6 devotes 1–3 paragraphs to each application, with key results and supporting figures deferred to the appendix. The transfer learning claim is the most underspecified: the paper asserts that large positive λ improves hierarchical generalization (Section 6, last paragraph), but the mechanism—why small-but-structured input weights lead to better transfer to new features—is not fully developed in the main text, and the key supporting figure (Fig. D.3) is in the appendix. For work listing these as contributions, the main text should contain enough detail for independent evaluation.

### Minor

- **The λ-balanced assumption (A2) limits practical relevance, and this connection is only established for one special case.** The paper shows LeCun initialization at infinite width approximately satisfies λ-balancedness (Fig. 1C, Appendix A.3), but does not analyze how closely He, Xavier, or orthogonal initializations at finite width satisfy this condition, nor how robust the theoretical predictions are to violations. The Discussion acknowledges this limitation but the gap between the theory's assumptions and standard practice remains significant. A perturbation analysis or empirical measurement of how dynamics degrade as initialization deviates from λ-balanced would substantially strengthen the practical relevance claim, though this is not required for the theoretical contribution to stand.

- **The "semi-structured lazy" regime is identified but not formally characterized in the main text.** The qualitative description (Section 5, around Fig. 4C) states that one layer is identity-like while the other is small and task-specific, but there is no formal definition in the main text of the precise conditions under which this regime emerges. The formal result exists (Theorem C.4 in Appendix) but is not presented in the main body, making the concept more of a qualitative observation than a fully developed finding in its current presentation.

- **The "delayed rich" regime description is intuitive but somewhat imprecise.** The explanation that "no least-squares solution exists within the span of the network at initialization" (Section 5, final paragraph) is stated informally. While Theorem C.6 provides formal quantification and Fig. 5C visualizes the delay, the main text presentation could be more precise about what exactly delays the onset of the rich phase.

### Trivial
None.

## Nice-to-Haves

- A perturbation/sensitivity analysis showing how dynamics deviate from λ-balanced predictions as initialization increasingly violates the condition would directly address the practical relevance question and could be a standalone contribution.
- Showing singular-value-level dynamics (analogous to Fig. 3) for non-task-aligned initializations would clarify whether the sigmoidal↔exponential transition is robust or an artifact of alignment.
- A formal definition of the semi-structured lazy regime in the main text (not just the appendix) with precise conditions on λ and task structure.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"RSM_O pseudoinverse requires comment"** — This is standard notation following Braun et al. (2022) and adds no substantive concern.
- **"Numerical instability of exact solution"** — The paper directly acknowledges this and provides a stabilized form in Appendix B.5; this is adequately addressed.
- **"NTK formula stated without derivation"** — Citing Jacot et al. (2018) is standard practice in the field.
- **"Parameter noise robustness not connected to rich/lazy"** — The paper does connect it explicitly: "a rich solution may enable a more robust representation in such systems" (line ~223). The connection is present, just not deeply elaborated.
- **"Missing related works"** — Cannot verify existence of uncited works.
- **"Interplay of absolute and relative scale deferred to Appendix A.4"** — This is a presentation choice, not a flaw; the main text focuses on relative scale as the novel parameter.
- **Formatting and style nitpicks** — Removed per rules.

## Novel Insights

The paper's most insightful contribution is the decomposition of lazy learning into two qualitatively distinct sub-regimes: the standard "fully lazy" regime (both layers identity-like) and the "semi-structured lazy" regime (one layer identity-like, one small-but-task-specific). This distinction, enabled by the λ-balanced framework, suggests that the binary rich/lazy classification in prior work may be too coarse, and that the interplay between which layer is large and which is small creates functionally different lazy regimes with different transfer properties. The counterintuitive transfer learning result—that the semi-structured lazy regime can outperform the rich regime for hierarchical generalization—challenges the common assumption that feature learning is always beneficial.

## Suggestions

- In the contribution statement (Section 1), explicitly qualify that the full rich-to-lazy characterization (sigmoidal↔exponential) is proven under task-aligned initialization, while the general dynamics (Theorem 4.3) apply without this assumption.
- Move at least one key application figure (ideally the transfer learning result, Fig. D.3) into the main text and expand the mechanistic explanation of why small-but-structured input weights aid hierarchical generalization.
- Add a brief formal definition of the semi-structured lazy regime in the main text, even if the full proof remains in the appendix.

## Score and Decision

### Calibration anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Critical Learning Periods (Aq35gl2c1k) | deep linear net, critical periods, analytical | 7.25 | Stronger neuroscience narrative, broader experiments; current paper has more complete mathematical derivation but narrower applications |
| Implicit Bias of SGD (P1aobHnjjj) | deep linear net, SGD dynamics, rank structure | 7.75 | Comparable depth of theory; that paper has a cleaner single story, current paper has broader but thinner applications |
| Scaling Laws Beyond Kernel (dEypApI1MZ) | solvable model, rich/lazy regimes, scaling | 7.2 | More impactful practical implications (scaling laws); current paper is more mathematically complete but less practically impactful |
| Grokking as Lazy→Rich (vt5mnLVIVo) | lazy→rich transition, grokking mechanism | 6.0 | Similar topic (lazy/rich transition), comparable theoretical depth; current paper has more rigorous derivations but both have limitations from assumptions |
| Adversarial Training Theory (SQLDXQ3IG8) | restrictive assumptions, limited practicality | 5.25 | Below current paper — that paper had more severe assumption gaps vs. contributions |
| Weak Correlations (2NwHLAffZZ) | fundamentally flawed, no experiments | 2.33 | Far below current paper |

The current paper sits above the borderline papers (5–5.5 range) because its core theoretical contribution is sound, well-validated, and represents a genuine extension of important prior work. It sits below the spotlight-level papers (7+) because those either had cleaner narratives, broader practical implications, or more complete experimental support for their claims. The overclaiming in the contribution statement and the thin main-text treatment of applications keep it from being higher.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>