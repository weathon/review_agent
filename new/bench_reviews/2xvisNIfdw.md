## Summary
This paper studies global convergence in bilevel optimization through the penalty reformulation rather than the nested hyper-objective. Its main contributions are: (i) a useful conceptual distinction between the poor landscape of the nested objective and potentially more benign penalized objectives, (ii) two sufficient conditions—joint PL and blockwise PL—under which PBGD globally converges to an \((\epsilon,\epsilon)\)-solution, and (iii) trajectory-based verification of these conditions for two specific linear least-squares bilevel problems: representation learning and data hyper-cleaning.

## Strengths
- **The paper tackles an important and genuinely difficult problem.** Global optimality in bilevel optimization is much less understood than stationarity or local optimality, and the paper is explicit that it targets “certain (not all) machine learning applications,” which is an appropriate scoped goal despite the broad title.
- **The landscape perspective is insightful and well supported in the paper.** Section 2 clearly shows that even when \(f\) and \(g\) are individually PL, the nested objective \(F(u)\) can fail to be PL (Example 1), motivating why the penalty reformulation is the more promising object for global analysis.
- **The joint-PL / blockwise-PL split is conceptually sensible.** The paper does not just posit one abstract condition; it ties joint PL to Jacobi updates and blockwise PL to Gauss-Seidel updates, which matches the bilevel structure and gives a coherent algorithm-analysis story.
- **The theory goes beyond a purely conditional statement by verifying the conditions along algorithmic trajectories in two nontrivial cases.** In both Sections 4 and 5, the paper acknowledges that only local/non-uniform PL-style properties are available and then uses trajectory-dependent arguments to bound them. That is more substantial than merely assuming a global PL inequality out of the box.
- **The representation-learning analysis appears technically meaningful within its intended scope.** The paper derives a joint-PL argument for the penalized objective in an overparameterized linear network setting and proves almost linear convergence to an \((\epsilon,\epsilon)\)-solution.
- **The empirical results are aligned with the theoretical claims they are actually making.** Figures 3–4 show almost linear convergence behavior in the stylized settings of the theory and serve as sanity checks for the proposed analysis.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper’s framing materially overstates the breadth of what is proved.** The title (“Unlocking Global Optimality in Bilevel Optimization”), abstract, and some introduction language suggest a broad advance on global optimality for bilevel optimization, but the actual unconditional results are much narrower: the generic theorem is conditional on joint/blockwise PL of the penalized objective, and the concrete verifications are only for two heavily structured linear least-squares settings. The paper does say “for certain (not all) machine learning applications,” but the headline framing still reads broader than the body supports.
- **The data hyper-cleaning result relies on extremely restrictive structural assumptions that sharply limit relevance.** Section 5 assumes a one-layer linear model and, crucially, Lemma 2 / Theorem 3 require \(X_{\text{trn}}X_{\text{trn}}^\top\) or even \([X_{\text{trn}};X_{\text{val}}][X_{\text{trn}};X_{\text{val}}]^\top\) to be diagonal. This is not a mild convenience: the closed-form simplification in Eq. (12), the claim that \(\mathcal S(u)\) is independent of \(u\), and the blockwise PL proof all depend on it. As a result, the theorem is best viewed as a proof-of-concept for a stylized decoupled setting rather than a broadly meaningful result for practical hyper-cleaning.
- **The verified applications are confined to linear/bilinear least-squares models, which substantially limits impact.** Section 4 studies a two-layer linear network with least-squares loss under overparameterization/wide-network/full-row-rank assumptions; Section 5 studies a one-layer linear hyper-cleaning model with sigmoid weights and strong matrix structure assumptions. For a paper centered on “global optimality” in bilevel optimization, the absence of any verified nonlinear setting leaves a significant gap between conceptual ambition and practical reach.
- **Some key assumptions are strong and do important substantive work rather than serving as harmless technicalities.** In particular, Assumption 2 in representation learning requires existence of an \((\epsilon_1,\epsilon_2)\)-solution that also nearly minimizes the training loss, i.e., a compatibility condition between validation-optimal and training-nearly-optimal solutions. The paper notes that sufficient conditions are deferred to the appendix, but in the main text this assumption is central enough that more discussion of when it is plausible would be needed.
- **The experimental section is too limited to materially strengthen the paper beyond theory-consistency checks.** The experiments are confined to the same stylized settings used in the proofs, with convergence curves and ablations, but they do not test robustness outside the assumptions, do not probe the trajectory-dependent PL story directly, and do not meaningfully establish broader practical relevance. Thus they validate consistency with theory, not reach.

### Minor
- **The main generic theorem is still mostly a conditional template rather than a broadly applicable theory.** Theorem 1 is meaningful, but by the authors’ own discussion in Section 3.3, additivity of PL functions fails in general, and the blockwise case further requires \(\arg\min_v L_\gamma(u,v)\) to be independent of \(u\). So while useful, the theorem should be framed more as a framework awaiting verification than as a generally deployable global-convergence result.
- **The distinction between exact global optimality and \((\epsilon,\epsilon)\)-optimality could be communicated more carefully.** The paper is mostly correct here—Theorem 1 and the later results are explicitly about \((\epsilon,\epsilon)\)-solutions via \(\gamma=\mathcal O(\epsilon^{-0.5})\)—but the prose sometimes slides into “global optimum” language in a way that may overstate what the finite-time results literally guarantee.
- **The convergence-rate presentation hides problem-dependent conditioning in big-\(\mathcal O\) notation.** In Sections 4–5, the actual PL/smoothness surrogates depend on singular values and trajectory-dependent quantities. The asymptotic \(\mathcal O(\log^2(\epsilon^{-1}))\) statement is attractive, but the practical significance is hard to judge without more explicit dependence on data conditioning and initialization.

### Trivial
- **The paper would benefit from a clearer main-text proof roadmap.** The inductive argument maintaining rank/singular-value bounds and converting local PL-type properties into uniform trajectory bounds is central, but the main paper gives only a compressed version of that story.

## Nice-to-Haves
- Add experiments on simple nonlinear models to test whether the penalty-reformulation perspective remains empirically useful beyond the proven linear settings.
- Include a direct visualization or logging of the trajectory-dependent PL/smoothness quantities to support the central proof intuition.
- Discuss whether the diagonal/orthogonality assumptions in hyper-cleaning can be relaxed to approximate versions.
- Expand the main text’s discussion of Assumption 2 with concrete intuitive sufficient conditions rather than deferring most of the interpretation to the appendix.
- Sharpen the framing throughout to emphasize that this is a pilot study establishing proof-of-concept global guarantees in narrowly structured settings.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Requests for more related work / comparison to uncited papers.** Removed by instruction: I cannot verify missing external references.
- **Pure formatting/style complaints and typos.** Not substantive for scientific merit.
- **Claims that the experiments must compare to exact global optimization methods to verify global optima.** Too demanding relative to the paper’s scope; the theory itself defines the optimization target and the experiments are positioned as corroboration rather than standalone proof.
- **Criticism that the paper gives no rigorous protocol to “certify” global optimality experimentally.** The paper’s primary support is theoretical, not empirical certification; this would be an unreasonable standard here.
- **Any implication that the paper is wrong because appendices are omitted in the provided extract.** Not a valid criticism of the paper itself.

## Novel Insights
The strongest synthesis here is that the paper’s real contribution is not “solving global bilevel optimization” in any broad sense, but identifying a credible *route* by which global bilevel guarantees may become tractable: abandon direct analysis of the nested hyper-objective, move to the penalized constrained reformulation, and verify PL-type properties only along the optimization trajectory. That is a meaningful conceptual shift. However, the same synthesis also clarifies the paper’s current ceiling: the route is presently demonstrated only in very structured linear settings, and the hyper-cleaning result in particular depends on assumptions that effectively decouple much of the bilevel interaction. So the paper is best viewed as a promising pilot study with a real conceptual insight, but not yet a broadly usable theory of global bilevel optimization.

## Suggestions
- Retitle and reframe the paper more conservatively, emphasizing that this is a pilot study for two structured linear bilevel problems rather than a general unlocking of global optimality in bilevel optimization.
- In Section 5, foreground the diagonality assumption and explicitly discuss its practical implications; ideally, state up front that the theorem is mainly a proof-of-concept for a highly structured regime.
- Strengthen the main-text explanation of Assumption 2 and of the trajectory-based invariants used in Section 4, since these are essential to the validity and scope of the representation-learning result.
- Make the distinction between exact global optimality and finite-time \((\epsilon,\epsilon)\)-global optimality explicit everywhere, including the abstract and conclusion.
- Add experiments outside the exact theorem assumptions—especially mildly non-ideal or approximately diagonal settings—to show whether the perspective has empirical robustness beyond the proved regime.
- Report more explicit dependence of convergence on conditioning, singular values, and initialization to make the almost-linear rate easier to interpret in practice.

## Score and Decision
**Assessment across axes.**  
- **Originality:** Moderate. The penalty-formulation viewpoint and the joint/blockwise PL split are meaningful, and the trajectory-based verification is a nontrivial analytical contribution.  
- **Importance of research question:** High. Global optimality in bilevel optimization is important.  
- **Support for claims:** Mixed. The narrow theorems are reasonably supported, but the broader framing overclaims relative to those theorems.  
- **Soundness of experiments:** Adequate as theory-consistency checks, but limited in scope and not strong enough to extend the claims.  
- **Clarity of writing:** Generally clear conceptually, though some central assumptions and proof mechanics need more transparent motivation.  
- **Value to the community:** Moderate. Useful as a pilot study and conceptual stepping stone, but limited by narrow applicability.

**Calibration against human-review anchors.**  
I compared this paper to:
- **“On Penalty Methods for Nonconvex Bilevel Optimization and First-Order Stochastic Approximation”** (scores 5–8, accepted): that paper earned acceptance because it paired a broadly useful penalty-based theoretical advance with stronger generality. The current paper is **below** that anchor because its verified results are much narrower and more assumption-heavy.
- **“A Local PL and Descent Lemma of GD for Overparameterized Linear Models”** (scores 3–6, rejected): the current paper is **slightly above** this anchor because it has a clearer bilevel-specific conceptual insight and stronger motivation, but it shares the same weakness of being confined to stylized linear settings with trajectory-dependent constants.
- **“Bilevel Optimization without Lower-Level Strong Convexity from the Hyper-Objective Perspective”** (scores 3–6, rejected): the current paper is in a similar quality band—interesting theory, but limited experiments and restrictive assumptions.
- **“Rethinking Moreau Envelope for Nonconvex Bi-Level Optimization”** (scores 5–8, mixed/lean accept): the current paper is **below** this anchor because that work paired theory with broader empirical validation and a more practically relevant scope.

Overall, this paper is a **technically interesting but scope-limited pilot study** whose central idea is worth attention, yet whose title/framing and one of its two flagship applications overreach the actual breadth of the results. That places it slightly below the acceptance bar for a selective venue.

**Score: 5.0 / 10 — Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>