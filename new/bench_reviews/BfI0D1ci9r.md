Now let me verify the harsh critic's claims against the paper in detail and produce the consolidated review.

## Summary

This paper proposes PINCO, a physics-informed graph neural network that solves the AC-OPF problem in an unsupervised manner. The method combines GNNs with hPINN (hard-constraint physics-informed neural networks) using a penalty method and Augmented Lagrangian. It introduces a node-splitting technique for handling multiple generators per bus. Evaluated on IEEE9, IEEE24, IEEE30, and IEEE118 benchmarks, PINCO demonstrates ~50–100× inference speedup compared to the MIPS solver, marginal cost increases (0.6–1.2%), and claims of zero inequality constraint violations.

## Strengths

- **Significant inference speedup with clear empirical evidence:** Figure 4 and Table on page 9 show PINCO inference times of ~0.002–0.005s versus MIPS at ~0.2–0.3s on the same hardware (CSCS/ETH Euler clusters), demonstrating a consistent ~50–100× speedup across all four IEEE test cases. This is directly relevant to real-time grid operations where AC-OPF is solved every ~5 minutes.

- **Effective node-splitting strategy for multi-generator buses (Section 3.1, Figure 2):** The paper introduces a concrete modeling solution where artificial nodes are created for each additional generator at a bus, connected via artificial lines. This is explicitly validated on IEEE24, which features buses with up to 6 generators — a scenario the closest prior work (Owerko et al. 2022) could not address.

- **Fully unsupervised training eliminates solver bias and dataset generation cost:** PINCO is trained purely via the physics-informed loss (Eq. 5) combining the OPF objective, equality constraints, inequality constraints, and Augmented Lagrangian terms. No pre-computed solutions from traditional solvers are required, avoiding the computational burden and possible bias inherent in supervised GNN approaches like Donon et al. (2020).

- **Generalization to unseen loading conditions (Section 4.2, Table 2):** When trained on 500 demand samples from a ±10% uniform distribution, PINCO generalizes to test demands with cost differences of ~0.8–1.1% compared to MIPS across all systems, demonstrating capability as a universal function approximator.

## Weaknesses

### Fatal

*None identified — the core methodology is sound and the speedup is empirically verified.*

### Major

- **Mischaracterization of the MIPS baseline's equality loss undermines the headline accuracy comparison.** The paper reports MIPS equality losses of 6.5 MW (IEEE24) and 20.0 MW (IEEE118) in Table 1, and interprets this as MIPS deliberately "focusing on minimizing costs, even if that results in higher equality losses" (Section 4.1, page 8). However, interior-point solvers like MIPS enforce nodal power balance (equality constraints) to a strict numerical tolerance (typically <10⁻⁵ p.u.). A 6.5–20 MW total mismatch indicates the paper's custom metric (Eq. 7) is measuring something different from solver convergence — likely summing per-node residuals in a way that inflates the figure, or computing the metric on the solver's output incorrectly. In any case, the interpretation that MIPS trades accuracy for cost is fundamentally at odds with how the solver operates. This makes the paper's headline claim of "physically more accurate solutions" relative to MIPS unreliable.

- **The central claim of "zero inequality constraint violations" is entirely unsubstantiated empirically.** The paper states in Section 4 (page 7): "Our approach consistently achieves solutions with zero inequality constraint violations, rendering the need for an inequality violation-based metric unnecessary." Yet the training method (hPINN with penalty + Augmented Lagrangian, Eq. 5) enforces inequality constraints *softly* — they are satisfied approximately to whatever threshold the penalty coefficients produce. No table reports maximum or mean violations, no p.u. threshold is specified, and no empirical verification is provided anywhere in the paper. For a soft-constraint method, claiming exact zero violation is statistically implausible, and the absence of any reported violation metric makes this central contribution unverifiable.

### Minor

- **IEEE118 formulation lacks explicit reference angle handling.** The paper notes on page 7 that IEEE118 has "no reference node, i.e., slack bus, which allows for arbitrary shifts in phase angle," and therefore drops θ comparisons for this case. While the paper correctly identifies this issue, it does not explain whether the GNN output for θ is meaningful without a fixed reference frame, or how the optimization avoids divergence along the null space. This affects the reproducibility and physical interpretability of the IEEE118 results.

- **Narrow ±10% load distribution limits the generalization stress test (Section 4.2).** Sampling demands uniformly from 90%–110% of base load is operationally mild and does not test the model's ability to handle realistic load variability, N-1 contingencies, or renewable injection profiles. While this is acceptable as a first demonstration, it leaves practical robustness unvalidated.

### Trivial

- *None beyond the minor issues listed above.*

## Nice-to-Haves

- **Convergence curves for adaptive penalty coefficients** (μ_H, μ_G, λ) would help demonstrate stable training dynamics, as PINNs often suffer from gradient pathologies where one loss term dominates.

- **Clarification of how "artificial lines" in the node-splitting approach (Section 3.1) are parameterized** — specifically their impedance values and whether they introduce fictitious losses or numerical instability.

- **Per-unit normalization of the equality loss metric** (Eq. 7) would make cross-system comparisons in Table 1 more physically meaningful.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Critic Claim: "IEEE118 formulation is mathematically ill-posed without slack bus."** — *Removed.* The paper does not omit a slack bus constraint; it correctly reports that the standard IEEE118 test case specification does not include a fixed reference angle and transparently drops θ comparisons for this reason. This is a property of the widely used IEEE118 benchmark dataset (via MATPOWER), not a formulation error by the authors. The paper handles this appropriately by not comparing θ values for IEEE118.

- **Critic Claim: "Novelty of combining GNNs with PINNs is overstated; prior work (Owerko et al. 2022) already explored this."** — *Removed (scope).* The paper does not claim to be the first to combine GNNs with PINNs for AC-OPF — it cites Owerko et al. extensively and positions its contribution specifically on (a) achieving zero inequality constraint violations and (b) handling multiple generators per bus, which Owerko et al. did not address. The framing is accurate.

- **Critic Claim: "Claim that PINN methods do not guarantee global optimality is misleading; non-convex AC-OPF inherently has this property."** — *Removed (strawman).* The paper's claim about non-convexity (page 1, lines 21-22) is a straightforward statement of fact: AC-OPF is NP-hard and non-convex. The paper does not claim PINNs are uniquely disadvantaged by this.

- **Critic Claim: "Equality loss metric reported in raw MW is a flaw."** — *Weakened to Nice-to-Have.* This is a presentation preference, not a methodological error. Raw MW is physically interpretable for power systems practitioners. Per-unit normalization would aid cross-system comparison but is not required for the within-system analysis the paper performs.

- **Critic Demand: "Replace narrow load sampling with realistic distributions."** — *Weakened to Minor.* A ±10% uniform distribution is standard for testing generalization in this literature. While real-world distributions and N-1 contingencies would strengthen the paper, the current scope is a reasonable first demonstration.

## Novel Insights

The core novelty of PINCO lies not in the combination of GNNs with physics-informed losses (a growing paradigm), but in the practical engineering solution of node-splitting for multi-generator buses, which resolves a structural representation gap in prior graph-based OPF solvers. The unsupervised formulation is well-executed and eliminates the need for computationally expensive dataset generation via traditional solvers. The ~100× inference speedup is the strongest empirical signal in the paper, as it directly addresses a real operational bottleneck in power system management. However, the paper's interpretation of the MIPS baseline's high equality loss as a "cost-accuracy tradeoff" is almost certainly incorrect, and the absence of any reported inequality violation threshold undermines the central "zero violations" claim — both of which are substantive weaknesses that detract from an otherwise competent methodological contribution.

## Suggestions

1. **Re-run the MIPS baseline analysis and report the solver's actual convergence tolerance.** Compute the equality loss (Eq. 7) on the solver's output with full precision, and clarify whether the 6.5–20 MW figures result from summing per-node residuals across the system. Provide the solver's internal convergence metrics (e.g., max constraint violation, Karush-Kuhn-Tucker residuals) alongside the paper's custom metric to enable direct comparison.

2. **Report explicit inequality constraint violation statistics.** Add a table showing maximum and mean violations for voltage limits, line thermal limits, and generator P/Q bounds across all test cases, expressed both in per-unit and absolute MW/MVAR. Specify the threshold below which violations are rounded to "zero."

3. **Clarify the reference angle handling for IEEE118.** State explicitly whether the GNN output for θ is normalized or constrained, and explain how the optimization handles the flat direction caused by the missing slack bus reference.

4. **Extend the generalization test to wider demand ranges** (e.g., ±20–30%) and at least one N-1 contingency scenario to demonstrate practical robustness beyond narrow interpolation.

## Score and Decision

**Score: 5.0**

Calibration was performed against several anchor papers:

- **High-scoring anchors:** `mtSSFiqW6y` (Judge Decoding, scores 6,10,8,8 — accepted oral) and `ThhQyIruEs` (neural multigrid solver, scores 6,6,6,6 — accepted poster) both featured comprehensive experimental validation with clear baselines and ablation studies. PINCO's speedup claim is comparably strong, but its baseline characterization and missing constraint data are weaknesses these papers did not have.

- **Medium-scoring anchor:** `W8xukd70cU` (physics-informed offline RL for energy optimization, scores 6,8,8,5 — accepted poster) had similar physics-informed ML methodology applied to energy systems with solid experiments. PINCO's methodology is slightly less novel and its empirical reporting is weaker in key areas (constraint violations).

- **Low-scoring anchors:** `iiK1vNRo6I` (scores 3,3,3,3 — rejected) and `TB5THwq1sq` (scores 3,6,3,3 — rejected) both suffered from insufficient baselines and limited experimental validation. PINCO is stronger than these — it uses an established solver baseline, tests on 4 IEEE systems, and demonstrates a clear speedup advantage.

PINCO falls between the accepted papers with solid experimentation and the rejected papers with inadequate validation. The real speedup and clean unsupervised methodology push it above the 3-scored papers, but the flawed MIPS comparison and missing constraint violation reporting prevent it from reaching 6. A score of 5.0 reflects a borderline paper with genuine contributions that requires significant clarification before it would be competitive for acceptance.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>