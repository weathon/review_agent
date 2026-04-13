## Summary
This paper studies offline robust learning in finite-horizon tabular two-player zero-sum Markov games under transition uncertainty. It proposes a model-based algorithm, RTZ-VI-LCB, built from empirical robust value iteration plus a pessimistic Bernstein-style penalty, introduces a clipped concentrability notion tailored to unilateral robust best responses, and proves uncertainty-dependent upper and lower sample-complexity bounds; it also sketches an extension to multi-player general-sum games.

## Strengths
- **The paper identifies and formalizes a nontrivial setting that combines three hard ingredients at once: offline learning, robustness to transition uncertainty, and self-play in Markov games.** The contribution is not just “robust RL” or “offline MARL” separately; the analysis is specifically for robust offline two-player zero-sum Markov games with partial coverage.
- **The sample-complexity dependence on state and action spaces is materially improved over the comparison highlighted in the paper.** Table 1 claims a shift from \(S^2AB\) to \(S(A+B)\), and the main theorem/lower bound indeed both scale in \(S(A+B)\), which is a meaningful structural improvement if correct.
- **The robust unilateral clipped concentrability coefficient is a specific conceptual contribution.** Assumption 1 is designed to avoid requiring full coverage of all state-action pairs by clipping occupancy ratios at \(1/(S(A+B))\), which is more nuanced than a raw worst-case density-ratio assumption and is central to the partial-coverage message of the paper.
- **The paper makes uncertainty-level dependence explicit rather than hiding it in constants.** The appearance of \(f(\sigma^+,\sigma^-,H)\) in the upper bound and the two uncertainty regimes in the lower bound are useful theoretical insights about when robust RTZMG learning is no harder than standard TZMG learning and when robustness introduces additional cost.
- **The lower-bound component is substantive and helps frame the contribution.** Theorem 2 is not decorative: it supports the claim that the \(S\) and \(A+B\) dependence is essentially unavoidable and clarifies the role of the uncertainty radius.

## Weaknesses

### Fatal
- **The paper’s core solution concept is not convincingly established under the asymmetric robustification it defines, and this undermines the meaning of the main guarantee.**  
  In Section 2, the max-player and min-player are evaluated in two different robustified problems: the max-player uses \(\inf_{P\in \mathcal U^{+\sigma^+}(P^0)}\) (Eq. 3, Eq. 9), while the min-player uses \(\sup_{P\in \mathcal U^{-\sigma^-}(P^0)}\) (Eq. 4). Eq. (10) then defines a single “robust NE” by requiring both unilateral optimality conditions simultaneously. That is not the standard saddle-point value of one robust zero-sum Bellman operator; it is an equilibrium notion assembled from two player-specific robust objectives, potentially with different uncertainty sets/radii.  
  The paper states after Eq. (9) that “there is at least one policy referred to as \(\mu^*\)... and \(\nu^*\)... [that] simultaneously achieve” the corresponding best-response values, and later says existence “has been proved for general divergence functions ... by Blanchet et al. (2024).” But in the main paper, there is no explanation of why this asymmetric formulation yields a coherent single game object, nor why the gap in Eq. (11) is the right exploitability notion for the returned pair. Since the main theorem is exactly about learning an \(\varepsilon\)-robust NE, this is not a peripheral omission.

- **Algorithm 2’s returned policy pair is not adequately justified as solving the stated equilibrium problem.**  
  The algorithm computes one Nash pair from \(\hat Q_h^+\) and another from \(\hat Q_h^-\), then outputs the cross-paired combination \((\hat\mu,\hat\nu) = (\{\mu_h^-\}, \{\nu_h^+\})\). This is explicitly in Algorithm 2. The main text does not explain why a max-player policy extracted from the “minus” recursion and a min-player policy extracted from the “plus” recursion should jointly form an approximate equilibrium of any single stage game or dynamic game.  
  This is the most serious algorithmic gap in the submission: even granting the definitions, the central construction of the returned policy is ad hoc from the main-paper presentation. Without a clear argument connecting the two separate recursions to the mixed final output, the main performance guarantee is hard to trust.

### Major:
- **The paper overstates optimality/tightness relative to its own theorems.**  
  The abstract and introduction repeatedly claim the method is “optimal” and “tight,” and Table 1 presents this very prominently. But the paper itself also acknowledges in Section 1.1 that optimality holds “except for the finite-horizon \(H\).” This qualification is important, not minor.  
  More concretely, Theorem 1 gives
  \[
  \tilde O\!\left(\frac{C_r^* H^4 S(A+B)}{\varepsilon^2} f(\sigma^+,\sigma^-,H)\right),
  \]
  while Theorem 2 gives lower bounds of order
  \[
  \Omega\!\left(\frac{C_r^* H^3 S(A+B)}{\varepsilon^2}\min\left\{\frac{1}{\min\{\sigma^+,\sigma^-\}},H\right\}\right).
  \]
  So the result supports strong claims about \(S\) and \(A+B\), but not full tightness. The headline language should be narrower and more careful.

- **There are substantial presentation/notation inconsistencies exactly in the quantities the paper claims as main innovations.**  
  Examples visible in the main text include:
  - The uncertainty factor \(f(\sigma^+,\sigma^-,H)\) is written inconsistently across Table 1, Section 1.1, and Theorem 1.
  - Assumption 1 refers in the text to \(C_r^*\) but the displayed equation uses \(C_\epsilon^*\).
  - Eq. (20) and Algorithm 2 appear inconsistent in the Bernstein penalty form.
  - Eqs. (12) and (14) switch to \(N\) after the dataset was defined with \(K\) episodes.  
  Some notation issues are survivable, but here they affect the uncertainty dependence, the data-quality coefficient, and the penalty term—the core technical content. This substantially hurts technical clarity and confidence.

- **The “partial coverage” framing is somewhat stronger than what the theorem actually guarantees.**  
  The paper is right that it does not assume full uniform coverage of all state-action tuples, and the clipped concentrability condition is a meaningful relaxation. However, Theorem 1 still requires a burn-in/sample lower bound depending on
  \[
  d_m^n = \min_{h,s,a,b}\{d_h^n(s,a,b): d_h^n(s,a,b)>0\},
  \]
  through Eq. (24), i.e., inverse dependence on the smallest positive occupancy mass. If \(d_m^n\) is tiny, the required sample size can be very large. So the result is a partial-coverage guarantee under a nontrivial occupancy lower-bound condition, not a broad statement that sparse offline data is generally enough.

- **Computational tractability is not adequately discussed, despite the paper making strong algorithmic claims.**  
  The method requires solving `ComputNash` at every state and time step, and also solving the robust Bellman backup via the TV dual in Eq. (18). Yet the paper provides no overall computational complexity analysis. Moreover, the “Policy estimation” paragraph says “Solving these robust matrix games is generally PPAD-hard,” which is confusing in context because Algorithm 2 applies `ComputNash` to ordinary zero-sum payoff matrices \(\hat Q_h^\pm(s,\cdot,\cdot)\), for which equilibrium computation is standard. The presentation leaves unclear what is actually hard and what is efficiently solved in the proposed method.

- **The multi-player general-sum extension is too underdeveloped in the main paper for the strength of its claims.**  
  Theorem 3 is only stated, with essentially no substantive derivation or discussion in the main text. Yet the paper describes this as “a breakthrough in breaking the curse of multiagency.” For a general-sum robust Markov game extension, equilibrium definition, computation, and selection are delicate; a theorem statement plus slogan is not enough support for such a strong claim.

### Minor
- **The paper is restricted to TV-distance uncertainty in the actual results, despite the problem formulation listing many divergences.**  
  Section 2 discusses KL, Wasserstein, \(\ell_q\), etc., but the algorithmic tractability and analysis in the main results rely on the TV dual form (Eq. 18). This is acceptable as scope, but the paper should more clearly state that the contribution is TV-specific rather than implying broader divergence generality.
- **Assumption 1 is difficult to parse and its interpretation is under-explained.**  
  It quantifies over policies/kernels tied to robust best responses and is analysis-centric rather than observable from data. That is not inherently invalid, but it needs much better intuition in the main text.
- **The paper lacks empirical validation entirely.**  
  For a theory-focused submission this is not fatal, but even a minimal tabular experiment would have helped validate the interaction between the subsampling step, pessimistic penalty, and uncertainty-level dependence.

### Trivial
- None.

## Nice-to-Haves
- Add a short tabular experiment showing NE-gap versus sample size and versus uncertainty radius \(\sigma\), ideally including comparison to the cited prior baseline.
- State the scope of optimality precisely in the abstract/introduction: optimal in \(S\) and action dependence, but not in \(H\).
- Add a concise computational complexity discussion for the TV dual backup and the per-state matrix-game solves.
- Include a short limitations paragraph on rectangularity, TV-only analysis, tabular scope, and the role of \(d_m^n\).
- Give an intuition paragraph before Assumption 1 explaining which occupancy measures are compared and why clipping is the right device.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Eq. (8) has malformed expectations / \(\nu_h(a)\)”** — removed as likely parser/extraction noise rather than a reliable paper issue.
- **“The primal robust backup is exponentially hard in \(S\)”** — the paper indeed says this, but the exact phrasing is more of an overstatement than a substantive flaw in the contribution.
- **Generic complaint that more ablations are needed for the two-stage subsampling and penalty** — removed as too generic for a theory paper, especially without experiments being standardly required for the core claim.
- **Criticism that the multi-player extension is merely ‘expected’ in model-based tabular settings** — weakened/removed in that form. The valid issue is not that the extension is unsurprising, but that it is under-justified relative to the strength of the claim.
- **Broad scope-creep criticisms asking for robustness against unrelated attack models or broader non-tabular evaluations** — removed as outside the paper’s stated scope.

## Novel Insights
The strongest synthesis across the reviews is that the paper’s most interesting idea is not just the \(S(A+B)\) sample-complexity improvement, but the attempt to define a *unilateral* clipped concentrability notion tailored to robust best-response occupancies under partial coverage. That is a potentially valuable lens for offline robust game learning. However, the paper pairs that promising analytical idea with a much shakier equilibrium formulation: the submission effectively learns from two separately robustified objectives and then cross-combines policies from two distinct recursions. This makes the paper feel split between a genuinely interesting sample-complexity analysis direction and an insufficiently grounded game-theoretic target/algorithm interface.

## Suggestions
- **Clarify the solution concept first.** Explicitly define whether the paper studies one robust zero-sum game with a single saddle-point operator, or a pair of player-specific robust best-response problems. If the latter, prove existence and explain why Eq. (11) is the correct gap notion.
- **Justify the cross-pair output in Algorithm 2.** The main paper needs a direct lemma/theorem showing why outputting \((\mu^-, \nu^+)\) yields an approximate equilibrium under the stated definitions.
- **Tone down and localize the optimality claims.** Replace broad “optimal/tight” language with precise claims: optimal in \(S\) and \(A+B\), with an open gap in \(H\).
- **Clean up the technical presentation.** Ensure consistency of \(f(\sigma)\), \(C_r^*\), the penalty formula, and sample-count notation. Right now, these inconsistencies significantly reduce confidence.
- **Add a computational discussion.** Give the complexity of each robust backup and each Nash solve, and clarify the relation between the claimed PPAD-hardness and what Algorithm 2 actually computes.
- **If space allows, add a minimal synthetic experiment.** Even one small tabular RTZMG would help demonstrate that the method behaves as the theory suggests.