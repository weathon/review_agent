=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary
FMS PINN introduces flow matching (via optimal transport coupling) as an adaptive collocation sampler for Physics-Informed Neural Networks targeting PDEs with singularities in the source function. At each resampling stage, a weighted bootstrap of high-residual points is used to train a flow matching network, which then generates new collocation points concentrated near singularities. The method is compared against DAS-PINN (KR-net normalizing flows) on multi-peak Poisson problems and linear elasticity with complex geometric inclusions.

---

## Strengths

- **Successful handling of multi-modal singularities without mode collapse.** Figure 4(a) directly demonstrates that the FM sampler places points near all nine singular peaks after a single resampling stage; the MSE convergence curve (Figure 4(b)) shows FMS PINN reaching ~10⁻³ where DAS PINN stalls near 10⁻¹. This is the strongest empirical signal in the paper.
- **High-dimensional advantage over normalizing flows.** The 5D two-peaks result (Table 1, Figure 5) is compelling: FMS PINN achieves MSE 6.1e-3, while DAS PINN collapses to MSE 2.3, visually confirming mode collapse in the normalizing flow baseline. The motivation—that topology-preserving invertible maps are fundamentally limited for disconnected, multi-modal targets—maps directly onto this result.
- **Theoretically grounded choice of generative model.** The paper correctly identifies that KR-net's invertibility imposes a topological constraint (citing Dupont et al., 2019; Wang et al., 2024b) and chooses flow matching specifically to remove it. This is a principled architectural decision, not an ad hoc swap.

---

## Weaknesses

### Fatal
*(None that individually invalidate the method, but the combination of contradictions below seriously undermines confidence in reported results.)*

### Major

- **Direct internal contradiction between Figure 8/9 captions and Table 2.** Table 2 reports FMS PINN outperforming DAS PINN on *both* u_x and u_y for the 2-circles problem (1.5e-3 vs. 1.7e-2 for u_x; 7.9e-3 vs. 1.2e-2 for u_y). Yet Figure 8's caption explicitly states "The FMS PINN solution (b) is smoother *and less accurate*. The DAS PINN solution (c) is *more accurate and closer to the reference solution.*" Figure 9's caption reinforces this: "The FMS PINN error profiles (a, c) show *higher error values* (more red) compared to the DAS PINN error profiles (b, d)." One of these must be wrong. This is not a nuance issue — the captions and the quantitative table give directly opposite conclusions for the same experiment. Until resolved, the reported MSEs for the 2-circles problem cannot be trusted.

- **Table 1 factual mislabeling.** The header of Table 1 reads "Comparison of linear elasticity equation PINN with Normalizing flow PINN in terms of MSE," but the table's contents are entirely about the Poisson equation (2 peaks 2D, 2 peaks 5D, 9 peaks). The linear elasticity results appear in Table 2. This is a concrete error in a key results table.

- **No computational cost data despite "efficiency" claim.** The abstract states the method "enhance[s] the accuracy and *efficiency* of the solution." The method trains a new flow model for 2000 iterations every 5000 PINN iterations—a non-trivial overhead. No wall-clock times, GPU hours, or even relative cost ratios are reported for either the Poisson or elasticity experiments. Without this, the efficiency claim is an unverified assertion.

- **Narrow experimental comparison.** Only DAS-PINN is used as a baseline. The paper itself discusses RAR (Lu et al., 2021), RAD (Wu et al., 2023), importance sampling (Nabian et al., 2021), and AAS-PINN (Tang et al., 2023b) in the related work — none appear in the experiments. Demonstrating that the flow matching model adds value beyond simple residual-weighted resampling (e.g., RAD) is critical for justifying the architectural complexity; this comparison is absent.

### Minor

- **Structural placement of the method.** The proposed method is introduced inside Section 3 ("Related Work"), with the core algorithm in subsection 3.4. The sub-section therein is mislabeled "3.1 Flow Matching" instead of "3.4.1." This conflation of related work and methodology makes the contribution boundaries unclear.

- **Inconsistent terminology for numerical integration.** Line 127 states ODE (10) is solved via "the Euler-Maruyama discretization scheme," which is the method for *stochastic* differential equations. ODE (10) is deterministic. Algorithm 1, Step 4, correctly says "the Euler method." This inconsistency should be corrected throughout.

- **Algorithm 2 notation conflicts with ODE formulation.** ODE (4)/(10) establishes X₀ ~ p₀ (Gaussian) evolving forward to X₁ ~ p₁ (target). Algorithm 2 initializes `x₁ ← Sample(q)` (the base distribution) and iterates *backward* from t=1 to t=Δt, returning x₀ as the output. The notation implies the Gaussian prior is at t=1 and the target is at t=0, inverting the convention used in the paper's own ODE formulation. No explanation is given for this reversal.

- **Reference solution for linear elasticity not described.** The paper benchmarks against a "reference solution" but never states how it was obtained (FEM, analytical, etc.). The MSE values in Table 2 are only meaningful relative to the accuracy of this reference.

- **No discussion of hyperparameter sensitivity.** M (bootstrap size), K (number of stages), flow model architecture, and Δt appear to be fixed without justification or sensitivity analysis. For an adaptive method, this is important for practitioners.

### Tiny

- **Conclusion mentions "future work will examine this method on larger number of epochs"** — suggesting current experiments may not be fully converged. This deserves explicit acknowledgment in the main text rather than being buried in future work.
- **Code is "available upon request"** rather than an anonymous repository; this creates a verification barrier for reviewers.

---

## Nice-to-Haves

- **Ablation vs. RAD/importance sampling**: A direct comparison with residual-weighted random resampling (RAD) would isolate the contribution of the generative model versus simple reweighting.
- **Error-vs-wall-clock curves**: Plotting MSE against actual compute time (not epochs) would make the efficiency trade-off concrete and is more informative than epoch-based curves when comparing methods with very different per-epoch costs.
- **Sampling distribution verification**: KL divergence or other metrics confirming that FM-generated points match the high-residual distribution better than the bootstrap sub-sample alone would strengthen the mechanistic justification.
- **Testing on standard PINN benchmarks** (e.g., Burgers, Navier-Stokes) to demonstrate generalizability beyond synthetic multi-peak functions.
- **Higher-dimensional experiments (>10D)**: The 5D result is encouraging; demonstrating scalability further would strengthen the mesh-free advantage argument.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **"Missing Figure 12"**: The paper references Figure 12 in Section 4.1 to show DAS PINN's failure on the 9-peaks problem. Given that the appendix and references were stripped from the provided text ("Rest of paper (reference and Appendix) is removed"), Figure 12 almost certainly resides in the appendix. This is not a missing figure — it is an appendix figure, a common practice. Removed.
- **Abstract overclaim about "avoids explicit modeling"**: The harsh reviewer argues this is misleading. However, the claim is technically defensible: the method does not fit an explicit density model (like KR-net), even though it uses residuals as bootstrap weights. The distinction is meaningful in the context of this literature. Removed.
- **"No contributions explicitly enumerated"**: The introduction and abstract do clearly state the contribution (FM-based adaptive sampling to handle singularities/mode collapse). The absence of a numbered list is a stylistic preference, not a substantive deficiency. Removed.
- **No comparison with AAS-PINN / selective inclusion of baselines as unfair**: AAS-PINN and RAD are legitimate missing comparisons (kept in Weaknesses). However, the sub-reviewer's framing that DAS-PINN is unfairly chosen is not supported — DAS-PINN is the closest architectural competitor. The issue is *too few* baselines, not a biased selection favoring the proposed method.

---

## Novel Insights

The most genuinely novel insight across the reviews — not stated in the paper itself — is that the topological constraint argument for why normalizing flows fail is *directly testable* through the 5D experiment: the two peaks are in disconnected regions of the input space, which is precisely the topology that invertible flows provably cannot handle without an expansion of intermediate nodes. The 5D result (DAS MSE = 2.3 vs FMS MSE = 6.1e-3) may constitute the first empirical demonstration of this theoretical limitation in the PINN adaptive sampling context. The paper hints at this but does not frame it as explicitly as it could, missing an opportunity to make a stronger theoretical contribution.

---

## Suggestions

1. **Resolve the Figure 8/9 vs. Table 2 contradiction before any further submission.** Either the figure captions are auto-generated and wrong, or the MSEs in Table 2 correspond to different experimental configurations. Provide the correct figure captions and verify MSE values with code output.
2. **Add a timing table** reporting wall-clock time per method (FMS PINN, DAS PINN, and uniform sampling) for the 9-peaks and linear elasticity experiments. Even approximate GPU-hours would suffice.
3. **Correct Table 1's header** to reflect that it contains Poisson equation results, not linear elasticity.
4. **Fix Algorithm 2 notation** to be consistent with the ODE formulation: either relabel Algorithm 2 to initialize from x₀ and integrate forward, or explicitly state that Algorithm 2 uses the inverse-time convention and explain the equivalence.
5. **Add at least one simpler adaptive baseline** (RAD or residual-weighted random sampling) to experimentally isolate the contribution of the flow matching generator.
6. **Relocate and restructure Section 3.4** as a dedicated Method section to separate methodological contribution from related work review.

---

**Overall assessment:**

**Novelty**: Moderate-to-high — applying flow matching as the sampling mechanism for PINN adaptive collocation is a fresh and principled idea, with a solid motivation from the topology-limitation of normalizing flows.

**Technical soundness**: Weak in the current form — the internal contradiction between Figure captions and Table 2, the mislabeled Table 1, and the algorithmic notation inconsistency undermine confidence in the correctness of reported results.

**Empirical support**: Insufficient — a single baseline (DAS-PINN), no statistical replication, no runtime data, and genuine uncertainty about which results to believe make it difficult to accept the empirical claims.

**Significance**: Moderate — if the results are correct and the method is computationally competitive, it addresses a real bottleneck for PINNs on singular PDEs. The 5D result in particular is potentially impactful.

**Clarity**: Poor — structural confusion (method buried in related work), inconsistent numbering, contradictory captions, and mislabeled tables all significantly impede readability for a submission at ICLR.

The paper requires substantial revision: the contradictions must be resolved, the experimental comparison must be broadened, and the presentation restructured before the contribution can be fairly evaluated.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
