## Summary
This paper studies infinite-horizon average-reward restless Markovian bandits and analyzes a rolling-horizon LP / model predictive control policy (“LP-update”) that replans from the current empirical state at every step, then applies randomized rounding to satisfy the hard per-step budget constraint. The main contribution is a new analysis based on dissipativity that yields asymptotic optimality with an \(O(1/\sqrt N)\) gap under a broad mixing assumption, and an exponentially small gap under additional local stability / non-degeneracy assumptions.

## Strengths
- **The core theoretical contribution is real and technically interesting:** the paper gives an infinite-horizon average-reward analysis of a finite-horizon MPC-style policy for RMABs, which the paper explicitly distinguishes from prior uses of finite-horizon LPs mainly in finite-horizon settings or under stronger assumptions. The main theorem (Theorem 4.1) establishes an \(O(1/\sqrt N)\) gap under Assumption 1, and Theorem 4.2 recovers exponential convergence under stronger stability assumptions.
- **The dissipativity viewpoint is a genuine conceptual contribution in this context:** Section 5 does more than repackage existing arguments. The rotated-cost construction, the storage function from the LP dual variable, and the monotonicity of \(L_\tau\) provide a coherent bridge from finite-horizon planning to infinite-horizon average reward. That is the most novel part of the paper.
- **The algorithmic object being analyzed is simple and practically plausible:** at each time step, solve a linear program on the empirical state and apply randomized rounding. This is substantially cleaner than many RMAB policies that require more specialized structural assumptions.
- **The assumptions for the \(O(1/\sqrt N)\) result are meaningfully broader than classical indexability/UGAP-style assumptions:** the paper does support that Theorem 4.1 does not rely on indexability or UGAP, and Assumption 1 is at least a credible broad mixing-type condition. This strengthens the significance of the general theorem.
- **The empirical section contains some genuinely insightful diagnostics, not just reward curves:** Figures 2 and 3 attempt to explain behavior via rotated cost and state-space trajectories, which is well aligned with the paper’s theoretical perspective and is more informative than a pure aggregate-performance table.

## Weaknesses
###: Fatal
None.

### Major:
- **The empirical evidence does not fully support the paper’s broad practical-positioning claims.**  
  The abstract says the method “performs very well in practice when compared to the state of the art,” and the introduction says it “beat[s] state of the art algorithms in our benchmarks.” But Section 6 compares only against LP-priority and FTVA, and explicitly justifies these baselines because they are “natural and simple to implement,” not because they represent the strongest practical baselines for the regimes considered. This leaves the empirical claim overstated relative to the evidence actually shown. The theory remains valuable, but the practical superiority claim should be toned down or supported more convincingly.

- **The paper does not analyze computational cost enough to justify its practical-efficiency messaging.**  
  The method solves a \(\tau\)-horizon LP at every time step. Yet Section 6 presents no timing or scaling study, despite claims in the abstract/introduction/main contributions that the method is easy to implement and performs well in practice, including “in terms of computational time horizon \(T\).” This omission matters because rolling re-optimization is the main practical tradeoff of MPC relative to simpler heuristics. Reward-only plots are insufficient to substantiate practical competitiveness.

- **There is a nontrivial clarity gap between the analyzed algorithm and the algorithm reportedly used in practice due to the treatment of the terminal term \(\lambda \cdot x(\tau)\).**  
  Section 3.1 defines the finite-horizon problem (8) using the terminal term with \(\lambda\) equal to the dual multiplier of the LP, and Section 5’s dissipativity proof relies centrally on this construction. But the paper then states: “our proofs will hold with minor modification by replacing \(\lambda\) by 0 and in practice we do not use this multiplier for our algorithm.” As written, it is unclear whether the experiments solve (8) with \(\lambda\), solve a different LP with \(\lambda=0\), or use some other variant. Since the proof sketch leans heavily on the storage/terminal term, this mismatch should be explained much more explicitly.

- **Some central assumptions / constants are not operationalized well enough for the paper’s “easy to verify / quantified loss” narrative.**  
  Assumption 1 is presented as an easily verifiable mixing assumption, but in the main text it is defined through the worst-case quantity \(\rho_k\) over all initial states and action sequences (Eq. 10). The paper gives a sufficient condition via ergodicity of \(P^0\), which helps, but does not really explain how one would verify or estimate the assumption in realistic models. Likewise, the bound in Eq. (11) depends on \(k,\rho_k,C_\lambda,C_\Phi\) and \(\tau(\epsilon)\), but the main text gives only limited guidance beyond \(\tau(\epsilon)=O(1/\epsilon)\). This does not invalidate the theorem, but it weakens the practical interpretability of the guarantee.

- **Assumption 4 for the exponential result is under-motivated and insufficiently discussed.**  
  Theorem 4.2 additionally assumes uniqueness of the LP solution of (8) for all \(\mathbf{x}\), described only as “a technical assumption that simplifies the proofs.” That is a strong parametric uniqueness requirement, and the paper gives little guidance on when it holds, how restrictive it is, or whether the result can be obtained with a tie-breaking rule or local uniqueness instead. Since Theorem 4.2 is one of the headline results, this deserves better justification.

### Minor
- **The empirical evaluation is somewhat narrow for a paper making broad practical claims.**  
  The experiments cover a few representative examples and useful parameter sweeps, but they do not test failure modes, difficult regimes, or instances specifically designed to showcase the claimed advantage of avoiding stronger structural assumptions. For example, the paper argues its generality relative to prior structurally dependent methods, but the experiments do not clearly demonstrate a regime where that generality is decisive.

- **The main text compresses some important proof steps too aggressively.**  
  In the sketch around Eqs. (16)–(19), the argument switches between stochastic \(U(t)\) and deterministic/mean-field \(u(t)\), and several key bounds are pushed into imported lemmas. The overall line is plausible, but one more level of explicitness in the main text would improve auditability for such a theory-heavy submission.

- **The paper should be more precise about what is novel: the main novelty is the proof framework and infinite-horizon analysis, not the asymptotic rates themselves.**  
  The paper mostly acknowledges this, but some wording could still better distinguish “new algorithmic analysis and viewpoint” from “new convergence rates.” The strongest contribution is the dissipativity-based analysis of MPC for this setting.

### Trivial
- **One small empirical presentation issue is that some figure/caption references are not fully contextualized in the main text** (e.g., Figure 3’s example naming), which slightly hurts readability.

## Nice-to-Haves
- Add a runtime/scaling study as a function of \(|\mathcal S|\), \(\tau\), and perhaps \(N\), alongside reward.
- Clarify whether experiments use the \(\lambda\)-terminal term or \(\lambda=0\), and if the latter, state the theorem variant explicitly in the main text.
- Give more practical guidance on choosing \(\tau\), especially how the truncation error and finite-\(N\) error should be balanced.
- Include at least one example where stronger structural assumptions behind classical approaches are violated, to better illustrate the motivation for this MPC approach.
- If feasible, add exact-optimum comparisons on more small instances, since the LP upper bound can hide absolute finite-\(N\) gaps.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **“Missing Whittle Index comparison” as a core weakness.**  
  It is reasonable as a suggestion, but it should not be elevated to a main criticism here. The paper does not claim to benchmark every classical RMAB policy, and I cannot verify from the paper alone whether the chosen examples are indexable or whether a WI implementation is straightforward/fair in those cases. So this is better treated as a nice-to-have rather than a substantive flaw.

- **Any criticism doubting release status / existence / verifiability of cited methods or references.**  
  Not applicable and should be ignored.

- **Generic complaints about omitted implementation minutiae or appendix placement alone.**  
  For example, the rounding procedure being in the appendix is not itself a meaningful weakness; the real issue is only whether the main text explains the theorem-relevant role of rounding sufficiently.

- **Overstated novelty attack claiming the method is “not new” because finite-horizon LP ideas already existed.**  
  The paper itself explicitly acknowledges that the LP-update idea existed for finite-horizon RMABs and positions its contribution as the infinite-horizon average-reward analysis via dissipativity. So “algorithm not new” is not a fair standalone criticism.

## Novel Insights
The most compelling synthesis across the reviews is that this paper’s real contribution is not a new asymptotic rate, but a new *control-theoretic explanation* for why a rolling-horizon LP policy works in average-reward RMABs. The rotated-cost/dissipativity lens gives a unifying interpretation of “operate optimally at the fixed point” that helps connect mean-field control, dual LP structure, and MPC truncation error in one framework. That perspective may outlast this specific theorem, especially if the authors can make good on the claim that the proof methodology extends to broader constrained MDP settings.

## Suggestions
- **Sharpen the claims.** Rephrase broad statements like “compared to the state of the art” and “beats state of the art algorithms” unless the empirical section is expanded accordingly.
- **Clarify the exact algorithm used in experiments.** State plainly whether Algorithm 1 solves (8) with the terminal term \(\lambda \cdot x(\tau)\) or the \(\lambda=0\) variant; if the latter, include the corresponding theorem statement or proposition in the main text.
- **Strengthen the discussion of Assumption 1 and Assumption 4.** For Assumption 1, provide a clearer checklist or computable sufficient conditions beyond the ergodic \(P^0\) example; for Assumption 4, discuss typicality/restrictiveness or how it might be relaxed.
- **Add computational evidence.** Report wall-clock times or solver scaling with \(\tau\) and \(|\mathcal S|\), and compare against the baselines used in Section 6.
- **Improve the proof sketch around Eqs. (16)–(19).** Make the role of \(u(t)\) vs. \(U(t)\), the rounding discrepancy, and the conditional-expectation step more explicit.
- **If space permits, add one regime specifically illustrating the advantage of avoiding stronger structural assumptions.** That would connect the theory more directly to the motivating narrative.

Overall, this is a strong theory paper with a meaningful conceptual contribution and credible core theorems. Its main limitations are over-ambitious practical framing and underdeveloped discussion of computational cost and some theorem-facing assumptions, rather than a failure of the central theoretical idea.