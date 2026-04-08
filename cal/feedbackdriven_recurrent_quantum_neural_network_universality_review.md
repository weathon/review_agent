=== CALIBRATION EXAMPLE 17 ===

# Final Consolidated Review
## Summary

This paper establishes the first quantitative approximation error bounds and universality results for Feedback-Driven Recurrent Quantum Neural Networks (RQNNs), a class of quantum reservoir computing models that process temporal data through a state-feedback loop. The main results show that RQNNs with linear readouts can approximate contractive Barron-type state-space systems without the curse of dimensionality (error rate $O(1/\sqrt{n})$ independent of input/state dimension), requiring only $O(\lceil\log_2(\epsilon^{-1})\rceil)$ qubits per circuit for accuracy $\epsilon$, and can universally approximate arbitrary fading memory, causal, time-invariant filters.

## Strengths

- **First quantitative approximation bounds for feedback-driven RQNNs with linear readouts.** Prior QRC universality results (Chen & Nurdin 2019; Chen et al. 2020; Nokkala et al. 2021) relied on polynomial output layers, which are harder to train and less common in practice. Theorem 4.6 provides explicit error bounds with linear readouts, directly aligning theory with experimental practice.

- **Novel technical machinery for the recurrent setting.** Extending feedforward QNN approximation bounds (Gonon & Jacquier 2025) to the recurrent case requires controlling approximation errors of functions *and* their derivatives simultaneously (Proposition 4.4, Corollary 4.5), because the state feeds back into the circuit. The proof of Theorem 4.6 carefully propagates these derivative errors through the contraction mapping, which is a non-trivial and original contribution.

- **Weaker smoothness requirements than classical RNNs.** As noted after Theorem 4.6, the Barron-type integrability condition $\int \|\xi\|^4|\hat{F}_j(\xi)|d\xi < \infty$ required here implies the Sobolev condition $s > N/2 + d/2 + 4$, which is strictly weaker than the $s > N + d + 3$ required for classical RNNs (Gonon et al. 2023, Theorem 3). This is a genuine, provable advantage of the quantum parametrization.

## Weaknesses

### Major:

- **No empirical validation of any kind.** The paper is entirely theoretical with zero numerical experiments. There are no simulations verifying the predicted $O(1/\sqrt{n})$ convergence rate, no demonstrations on standard time-series benchmarks, and no experiments on quantum simulators assessing the impact of shot noise or gate errors. For a venue like ICLR, where empirical grounding is highly valued, the absence of even a single proof-of-concept experiment is a significant gap. This is especially notable because the abstract references "promising empirical performance" of prior feedback-based QRC work, yet provides none of its own.

- **The universality proof in Theorem 4.8 externalizes memory via classical preprocessing, obscuring the role of quantum recurrence.** Lemma 4.7 constructs sparse linear preprocessing matrices $P_j$ that impose a shift-register structure on the state vector, effectively limiting memory to $K-1$ steps and guaranteeing the echo state property by construction. Theorem 4.8 then relies on this preprocessing. This means the *memory capacity* is provided by a classical mechanism (storing past inputs in a tapped delay line), while the quantum circuit serves primarily as a nonlinear map applied at each time step. The paper should explicitly acknowledge that Theorem 4.8's universality comes from the combination of a *classical finite-memory structure* plus a *quantum function approximator*, and discuss whether the quantum feedback loop itself can sustain fading memory without this preprocessing. This distinction is crucial for understanding what the quantum dynamics actually contribute.

- **Total resource scaling is understated.** The abstract and introduction emphasize that the "number of qubits only [grows] logarithmically in the reciprocal of the prescribed approximation accuracy." However, this refers to qubits *per circuit*. Section 3 states that $N$ such circuits are run in parallel, where $N$ is the state-space dimension. The total qubit count therefore scales as $O(N \log(1/\epsilon))$, which is linear in the state dimension. While the *rate* of convergence avoids the curse of dimensionality, the *total resource requirement* still scales linearly with $N$. The paper should state this total scaling explicitly to avoid misinterpretation.

### Minor:

- **Circuit depth is not analyzed.** While qubit count scales logarithmically, the paper does not discuss the circuit depth required for the uniformly controlled gate $U_\theta$. Standard decompositions of multi-controlled unitaries on $\sim\log n$ control qubits require gate depth that scales with $n$ or worse. Since $n \sim O(\epsilon^{-2})$, the circuit depth could be substantial, and this matters directly for NISQ feasibility where coherence times are limited. The paper cites Zindorf & Bose (2024; 2025) for efficient decompositions but does not discuss the depth scaling or its implications.

- **Constants in approximation bounds depend on dimension through Fourier norms.** While the rate $1/\sqrt{n}$ is dimension-free, the constants $C_j^\infty$ involve terms like $\|{\hat{F}_j}\|_1$ and $I_{q,j}$ integrated over $\mathbb{R}^{N+d}$. In high dimensions, satisfying the Barron-type integrability conditions becomes more restrictive, and the constants may grow. The paper does not discuss how these constants scale with dimension, which is relevant for assessing whether the "curse of dimensionality" avoidance is meaningful in practice.

- **Shot noise / Monte Carlo error is relegated to Appendix E without quantitative impact assessment.** Appendix E outlines how sampling error scales as $O(R/\sqrt{S})$ but does not analyze how many shots $S$ are needed to keep the total error (approximation + sampling) below a target $\epsilon$, nor how $S$ interacts with the approximation parameter $n$. This is relevant to the claim of "real-time processing capability," since large $S$ could undermine real-time feasibility.

### Trivial:

- The probabilistic proof method (constructing $\theta$ via random weights) is standard in Barron-type approximation theory and not a weakness per se, but it means the results are existence proofs rather than constructive algorithms.

## Nice-to-Haves

- Numerical experiments on standard reservoir computing benchmarks (e.g., Mackey-Glass, NARMA) to validate that the theoretical bounds are achievable in practice, even on classical simulators.
- Analysis of the trainability landscape, including whether barren plateaus arise for the proposed circuit architecture at the stated parameter counts.
- Derivation of generalization error bounds combining the approximation results with risk bounds (as suggested in the paper's own introduction).
- Explicit discussion of how the preprocessing memory depth $K$ in Theorem 4.8 should scale with problem complexity for practical use.

## Removed Points

- **"The abstract claims 'promising empirical performance' without providing any."** Removed because this is a misreading: the abstract says "Motivated by their promising empirical performance," referring to prior work on feedback-based QRC (Kobayashi et al. 2024), not claiming such results for this paper.

- **"The paper should compare RQNNs to classical reservoir computing baselines empirically."** Removed as a weakness because this is a theoretical approximation theory paper; demanding empirical baseline comparisons is outside its stated scope. The paper already provides theoretical comparison of integrability conditions with classical RNNs.

- **"Strong/unrealistic assumptions for NISQ implementations (noise, decoherence)."** Removed as a weakness because the paper is explicit about its scope—it provides approximation theory, not hardware implementation analysis. Complaining that a theory paper doesn't model hardware noise is scope creep. The paper's contribution is proving what is *possible* in principle; the gap to hardware is acknowledged in the conclusion.

- **"Parameter construction is non-constructive."** Weakened to trivial—the probabilistic method is standard in this literature (Barron 1993; Gonon & Jacquier 2025) and the paper clearly states results are formulated for variational circuits with trainable parameters.

- **"Energy efficiency claims for quantum computing."** Removed—the paper makes no such claims.

- **"The Sobolev condition comparison to classical RNNs is marginal."** Removed because this is factually incorrect: $s > N/2 + d/2 + 4$ vs. $s > N + d + 3$ is a meaningful improvement (roughly halving the dimension-dependent term), and the paper correctly identifies this.

## Novel Insights

The architecture's use of uniformly controlled quantum gates (block-diagonal unitaries) creates an explicit connection between the RQNN's expressivity and the Barron-type function class: each block independently parametrizes a rotation whose angle is an affine function of the input/state, and the resulting representation (Proposition 4.1) is essentially a weighted sum of cosines—the quantum analogue of a random Fourier feature model. This means the "quantum advantage" in approximation theory is fundamentally about the specific parametrization enabling weaker smoothness requirements, not about exponential state-space dimension per se. Additionally, the universality construction in Theorem 4.8 reveals an interesting structural point: the quantum circuit's role in the universal approximation result is purely as a nonlinear map, with all memory arising from a classical shift-register preprocessing. This raises a fundamental question for the field about whether quantum recurrence genuinely provides memory advantages over classical recurrence, or whether the quantum advantage is confined to the static nonlinearity applied at each step.

## Suggestions

- Add at minimum a small-scale numerical experiment (e.g., on a quantum simulator) verifying the $O(1/\sqrt{n})$ convergence rate on a low-dimensional contracting map to ground the theoretical bounds.
- Explicitly state the total resource complexity as $O(N \log(1/\epsilon))$ qubits and discuss the circuit depth scaling in the main text.
- Add a paragraph in Section 4.3 discussing the role of the preprocessing matrices $P_j$ and clarifying that the universality result combines a classical memory mechanism with a quantum function approximator, including analysis of how $K$ (memory depth) should scale with target filter complexity.
- Provide a concrete bound on the number of measurement shots $S$ needed to maintain approximation accuracy, combining Appendix E's framework with Theorem 4.6's error bound.

---

**Evaluation on key axes:**

- **Novelty:** High. First quantitative approximation bounds for feedback-driven RQNNs with linear readouts; novel extension of QNN approximation theory to the recurrent setting requiring derivative control through feedback loops.

- **Technical soundness:** Strong. Proofs are rigorous and build logically; the progression from Proposition 4.4 through Theorem 4.8 is well-structured. The probabilistic construction follows established methodology.

- **Empirical support:** Absent. This is the paper's most significant gap—no experiments validate the theoretical predictions.

- **Significance:** Moderate-to-high for the quantum learning theory community; the results provide important theoretical foundations, but the practical impact depends on future validation and the clarified role of quantum vs. classical components in the universality proof.

- **Clarity:** Good. Mathematical notation is dense but consistent; the logical flow is clear. Some notation (e.g., the block-matrix definition of $U$) could benefit from a more algorithmic description, but this is a minor point given the format constraints.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 10.0]
Average score: 8.0
Binary outcome: Accept
