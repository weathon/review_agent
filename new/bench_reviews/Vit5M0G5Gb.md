## Summary

This paper presents a theoretical framework for understanding simplicity bias—where neural networks learn solutions of increasing complexity during training—through saddle-to-saddle dynamics. The key structural results (Theorems 1, 3) show that fixed points of narrower networks embed as saddles in wider ones, and that gradient flow preserves invariant manifolds corresponding to effectively narrower networks, across a general class of architectures (FC, convolutional, attention). The dynamical analysis identifies two distinct mechanisms: data-induced timescale separation (leading to low-rank weights in linear architectures) and initialization-induced timescale separation (leading to sparse weights in quadratic/homogeneous architectures). The paper makes specific predictions about the effects of width, data distribution, and initialization on learning dynamics.

## Strengths

- **Genuine architectural generality of the structural results.** Theorems 1, 3, and Corollary 2 apply to any architecture fitting Equation (1), encompassing fully-connected, convolutional, and self-attention layers. The extension of Fukumizu & Amari's embedded fixed-point constructions (Eqs. 6–7 for homogeneous and linear activation functions) is non-trivial and directly relevant, as the authors note that "the saddles visited during learning turn out to fall under Equations (5) to (7) but not Equation (4)."

- **Conceptual advance: two distinct timescale separation mechanisms.** The identification of data-induced timescale separation between directions (Section 5.1) vs. initialization-induced timescale separation between units (Section 5.2) is a clean and novel organizing principle. It correctly predicts qualitatively different behaviors: increasing width speeds up learning in attention-based architectures but not in linear networks (Figure 2A), and equalizing singular values eliminates plateaus in linear networks but only shortens them in self-attention (Figure 2B).

- **Falsifiable predictions.** The theory makes concrete, testable predictions about initialization structure (Figure 2C shows that initializing near invariant manifolds but away from saddles produces saddle-to-saddle dynamics without an initial plateau—a regime "not previously observed"), initialization scale effects (Figure 2D), and data distribution effects (Figure 2B). These go beyond qualitative observation to yield specific structural claims.

- **Clean mathematical development.** Theorem 4 (timescale separation in linear networks) and Proposition 5 (rich-get-richer dynamics in quadratic networks) are precisely stated and proved. The proof strategy of connecting data statistics (singular values) to dynamical timescales is well-executed.

- **Honest discussion of scope boundaries.** The paper explicitly identifies conditions where saddle-to-saddle dynamics fails (tanh networks lacking corresponding invariant manifolds, large random initializations away from invariant manifolds, architectures with full expressivity in a single unit). The Discussion section (Section 7) provides a clear condition checklist, which is more forthright than many theory papers.

## Weaknesses

### Major:

- **Overclaiming of "universal" and "across architecture" scope relative to the actual dynamics proofs.** The Abstract claims the framework explains simplicity bias "for a general class of neural networks, incorporating fully-connected, convolutional, and attention-based architectures," and that "linear networks learn solutions of increasing rank, ReLU networks learn solutions with an increasing number of kinks, convolutional networks learn solutions with an increasing number of convolutional kernels, and self-attention models learn solutions with an increasing number of attention heads." However, the rigorous dynamics analysis (Section 5) covers only two-layer linear networks (Section 5.1) and two-layer quadratic networks (Section 5.2). For ReLU, standard convolutions, and realistic attention architectures, the paper provides only heuristic Taylor-expansion arguments and simulation anecdotes. The structural results (fixed points exist, invariant manifolds exist) are architecture-agnostic, but these are necessary—not sufficient—conditions for saddle-to-saddle dynamics. The paper itself acknowledges in Section 7 that "our analysis of dynamics in Section 5 only applies to two-layer networks," yet the framing throughout the Abstract and Introduction does not reflect this limitation. This gap between claimed scope and rigorously established scope is the paper's most significant issue.

- **The iterative saddle-to-saddle mechanism is heuristic, not proven beyond the first transition.** Theorem 4 and Proposition 5 analyze the early-time dynamics near initialization, showing that the first saddle escape follows an invariant manifold. The subsequent "iterations" of saddle-to-saddle dynamics (e.g., Equation 12 and the discussion following it) rely on heuristic arguments that (a) the linearized dynamics remain valid near each subsequent saddle, (b) nonlinear feedback from already-grown units does not qualitatively alter the dynamics, and (c) the trajectory stays close enough to the invariant manifold throughout the transition. None of these are rigorously established. The paper states "we develop heuristic arguments showing that the gradient flow dynamics can, in some cases, naturally evolve near such saddle-to-saddle paths" (Section 4), which is an honest framing, but the paper repeatedly shifts to stronger language ("the network learns solutions of increasing complexity," "learns solutions with an increasing number of attention heads"). The core dynamical claim—that networks iteratively escape saddles by recruiting one effective unit at a time—is supported by simulations but not by proof.

- **Limited experimental validation.** All demonstrations use small-scale synthetic settings (e.g., small matrices for linear networks, simple attention on toy data). No experiments on standard benchmarks, realistic architectures, or complex data distributions are provided. For a paper claiming to explain simplicity bias "across neural network architectures" (from the title), this is a notable gap. The experiments in Figure 2 are illustrative but do not systematically test the theory's quantitative predictions (e.g., predicted vs. observed plateau durations, predicted vs. observed rank of learned solutions at each stage).

### Minor:

- **The ReLU case is underdeveloped relative to its prominence in the paper's claims.** The paper's title and abstract prominently feature ReLU networks ("ReLU networks learn solutions with an increasing number of kinks"), but the dynamics analysis in Section 5 provides no rigorous treatment of the ReLU case. The "positively homogeneous" property (Eq. 6) provides structural fixed points, but the dynamical analysis of how kinks are sequentially acquired is absent. For ReLU, the Taylor expansion around zero picks up the linear term, which would suggest rank-one growth rather than unit-wise kink acquisition—potentially conflicting with the claimed mechanism.

- **The deep network conjecture is speculative.** Section 7 proposes that "whether ϕ(gin(x);ui) is linear or quadratic in ui predicts learning behaviors, including the type of the timescale separation." This is presented as a conjecture with simulation support (Figure 5) but no rigorous treatment for networks deeper than two layers. Given that most practical architectures are deep, this limits the framework's direct applicability.

### Trivial:

- The notation for self-attention in Eq. (2) is acknowledged by the authors as "not a common notation," which adds some parsing overhead but does not affect correctness.

## Nice-to-Haves

- **Quantitative bounds on the dynamics approximations.** Bounding how long the linearized dynamics (Eqs. 10, 14) remain valid, and how closely trajectories follow invariant manifolds, would strengthen the iterative saddle-to-saddle story. Even order-of-magnitude estimates would help.

- **Experiments on more realistic architectures and data.** Even a small-scale experiment with a real transformer on a synthetic sequence task, or a CNN on a simple classification problem, would substantially support the "across architectures" claim.

- **Sharper conditions for when saddle-to-saddle dynamics occurs vs. smooth dynamics.** The Discussion provides qualitative conditions but no quantitative thresholds (e.g., how small must initialization be, relative to data-dependent quantities, for plateaus to appear).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point #1 (Scope of universal claim vs. what is proved):** While the overclaiming concern is valid and retained above in a moderated form, the harsh critic's framing characterizes this as a "structural" issue that "would require either substantially weakening/qualifying the headline claims or adding deep new theory." The paper *does* acknowledge the two-layer limitation in Section 7, so this is not an undisclosed gap—it's a mismatch between framing and rigor. The structural results genuinely span architectures; what's limited is the dynamics story. The harsh critic's point that "the claimed scope is overstated relative to the actual theorems" is valid, but it's not fatal to the paper—it's a framing issue.

- **Harsh Critic Point #3 (Over-reliance on linearization for global claims):** For the linear network case specifically, the dynamics are not an approximation—linear networks are exactly solvable. The "linearization" in the linear case (Eq. 10) is exact for the early phase and the iterative argument uses the residual structure of singular values, which is also exact. For the quadratic case and beyond, the harsh critic's concern about local validity is valid, but it shouldn't be applied uniformly to both cases.

- **Harsh Critic Point #4 (Conflation of architectural symmetry facts with trajectory claims):** The paper does not merely rely on the existence of paths; it provides explicit timescale separation arguments (Theorem 4, Proposition 5) that explain why trajectories approximately follow these manifolds. The harsh critic's characterization that the paper "slides from 'these manifolds exist' to 'gradient descent proceeds along them'" understates the dynamical content. However, the concern that the iterative structure is not rigorously established beyond the first transition remains valid.

- **Neutral Reviewer's point about "lack of comparison with alternative theories":** This is a nice-to-have but not a core weakness. The paper's contribution is primarily theoretical and structural, not empirical. Within its theoretical scope, it provides new mechanisms that differ from prior work (e.g., the data-induced vs. initialization-induced distinction). Detailed comparison with NTK regime or spectral bias would be tangential.

- **Spark Reviewer's point about "experiments on realistic architectures and data":** Retained in moderated form under "Limited experimental validation." The demand for full-scale experiments on transformers/CIFAR is scope creep for a theoretical paper, but some validation beyond toy settings is reasonable to ask for.

- **Spark Reviewer's point about "analysis of noise and stochasticity":** Theoretical papers on gradient flow dynamics routinely use GF as an approximation; demanding SGD analysis is a scope expansion. This is a nice-to-have, not a core weakness.

## Novel Insights

The identification that the *type* of timescale separation—between directions (data-induced) vs. between units (initialization-induced)—is determined by whether the activation is linear or superlinear in the unit-specific weights is genuinely novel. This yields a striking prediction: adding more units to linear networks doesn't change the dynamics (since timescale separation operates across units), while adding more heads to self-attention *does* speed up learning (since timescale separation operates across units, and more units means tighter gaps between initialization scales). This architectural inductive bias—where width helps some architectures but not others—is an unexpected connection between architectural structure and learning dynamics.

## Suggestions

- **Rescope the framing:** The title and abstract should explicitly qualify the dynamical results as applying to "two-layer networks with linear and quadratic activation functions," while noting the broader architectural applicability of the structural results. This would align claims with evidence without diminishing the contribution.

- **Add at least one quantitative prediction test:** Measure the predicted plateau duration (derived from singular value gaps for linear networks, or from initialization gaps for quadratic networks) against simulated plateau durations, and report the correlation/error. This would go beyond qualitative visual agreement.

- **Extend the dynamics analysis to at least one deeper network (e.g., depth-3 linear network):** The paper mentions depth-3 linear networks in the Discussion (Eq. 17) and Figure 5 shows simulations. Even a partial analytical treatment of this case—showing that the layer-wise linearity property holds and produces rank-increment dynamics—would substantively strengthen the deep network claim.

- **Add a dedicated subsection on the ReLU case:** Given that ReLU is the most commonly used activation, explicitly analyzing the dynamics of kink acquisition under the positively homogeneous framework (using Eq. 6 invariant manifolds) for at least a 1D input setting would address the gap between the abstract claims and the actual dynamics.

## Score and Decision

**Calibration:**
- *ogKE7LcvW6* (Loss Landscape of Shallow ReLU-like Neural Networks): Accepted as poster (scores 8,8,6,6). Rigorous analysis of saddle-to-saddle dynamics but limited to shallow ReLU networks. The current paper has broader architectural scope for structural results but less rigorous dynamics analysis for the broader architectures it claims to cover.
- *P1aobHnjjj* (Implicit bias of SGD in L2-regularized linear DNNs): Accepted as spotlight (scores 8,8,5,10). Very clean theoretical contribution within a specific setting (linear DNNs). The current paper has a similar level of theoretical depth for its specific settings but makes broader claims.
- *eQggPqESBr* (Simplicity Bias and Optimization Threshold): Rejected (scores 5,5,6,6). Two-layer ReLU, limited setting, gap between theory and practice. The current paper is stronger than this—it has a broader structural framework and clearer predictions.
- *27SSnLl85x* (Make Haste Slowly): Accepted as poster (scores 8,5,3,8). Strong theory with restricted synthetic experiments and strong assumptions, but novel framework for ReLU dynamics.

The current paper makes genuine and meaningful contributions: the generalized fixed-point and invariant manifold results, the two-mechanism taxonomy, and specific predictions about width/data/init effects. However, the overclaiming of universality and the heuristic nature of the iterative dynamics story beyond the first saddle transition are significant weaknesses. The paper is stronger than eQggPqESBr but weaker than P1aobHnjjj and ogKE7LcvW6 within their respective scopes. Placing it between these anchors:

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>