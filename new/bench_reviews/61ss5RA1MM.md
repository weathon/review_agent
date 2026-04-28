Now I have enough calibration. Let me write the final review.

## Summary
This paper proposes OC-Flow, a training-free guided flow matching framework based on optimal control theory that provides convergence guarantees for both Euclidean and SO(3) manifolds. The method unifies existing backprop-through-ODE approaches (D-Flow, FlowGrad) as special cases and demonstrates empirical improvements across text-guided image manipulation, conditional molecule generation, and peptide design tasks.

## Strengths
- **Novel SO(3) extension with convergence proof**: Theorem 5 (line 219-221) establishes convergence guarantees for guided flow matching on the rotation group, addressing a genuine gap in the literature for protein design applications where rotational symmetry is critical. This is one of the first theoretically-grounded guided flow methods on non-Euclidean manifolds.

- **Memory-efficient implementation**: The adjoint method with vector-Jacobian products reduces memory complexity from O(ND²) to O(D²) (line 114-116, Table 1), making inference-time optimization more practical. This is a concrete engineering contribution validated by the reported 216s vs 15-minute runtime comparison.

- **Unified theoretical framework**: Section 3.3 (lines 136-152) mathematically derives D-Flow and FlowGrad as limiting cases of OC-Flow (γ→∞ and single control term respectively), providing coherent perspective on the relationship between existing methods.

- **Empirical improvements on key metrics**: OC-Flow achieves lower LPIPS (0.207 vs 0.302), better CLIP scores (0.302 vs 0.299), lower MAE on 4/6 molecular properties, and better binding energy (-50.410 vs -41.665) compared to baselines (Tables 2, 3, 5).

## Weaknesses

### Fatal
None

### Major
- **Theory-practice gap in theoretical assumptions**: The theoretical guarantees rely on assumptions that do not match the experimental setup. Proposition 1 (line 84-90) explicitly assumes an "Affine Gaussian Probability Path" for the KL divergence bound, but the image experiments use Rectified Flow (line 245), which employs deterministic straight-line conditional paths that do not necessarily satisfy the affine Gaussian marginal assumptions. Theorems 2 and 5 (lines 98, 219) require global Lipschitz continuity of the reward function and prior neural vector field, yet the deep networks used (CLIP, EquiFM, PepFlow) are not verified to satisfy this condition. The paper briefly mentions this "can be relaxed to a local Lipschitz condition" (line 94-95) and cites prior work on Lipschitz continuity in deep learning, but provides no empirical verification that the chosen hyperparameters (particularly γ) satisfy the γ > 2C condition required for convergence. This undermines the core claim of being "theoretically grounded" with "convergence guarantee."

- **Unfair baseline hyperparameter configuration**: Section 5.1 states: "We run StyleCLIP and FlowGrad with their official implementation and **default parameter configurations**" (line 247), while OC-Flow parameters (η=2.5, β=0.995, optimization steps=15) are explicitly tuned. Optimization-based guidance methods are notoriously sensitive to step sizes and iteration counts. Comparing a tuned method against baselines with default settings invalidates the strength of the "superior performance" claims in Tables 2 and 3. Without evidence that baselines were tuned to convergence on these specific tasks, the performance gap may be partially attributable to hyperparameter mismatch rather than methodological superiority. This is a significant methodological flaw in the experimental design.

### Minor
- **Selective reporting of performance trade-offs**: The abstract and conclusion claim "superior performance" without acknowledging cases where OC-Flow underperforms: (1) Table 2 shows lower ID preservation than FlowGrad (0.732 vs 0.737), (2) Table 3 shows higher MAE than D-Flow on Δε (367 vs 355), and (3) Table 5 shows higher RMSD than the unconditional PepFlow prior (2.127 vs 1.645). While OC-Flow wins on most metrics, the paper would benefit from acknowledging these trade-offs and discussing when the method may not be preferable.

- **Ambiguous RMSD interpretation in peptide design**: Table 5 reports RMSD as a metric for "structural accuracy" (line 307), where lower is typically better. However, OC-Flow(trans+rot) achieves higher RMSD (2.127) than PepFlow (1.645) while improving binding energy. For *design* tasks aiming to find new binders rather than reproduce native structures, higher RMSD may be acceptable or even desirable. The paper should clarify whether RMSD should be interpreted as a quality metric or a diversity metric in this context, and discuss the energy-RMSD trade-off more explicitly.

### Trivial
None

## Nice-to-Haves
- Direct measurement of KL divergence between prior and guided distributions in experiments would strengthen the empirical validation of Proposition 1's claim, even if the theoretical assumptions are idealized.

- Visualization of state trajectories x_t for OC-Flow vs. baselines would provide intuitive evidence for how the running cost constrains path deviation, supporting the optimal control formulation.

- Discussion of Lipschitz constant estimation or empirical verification that the chosen γ satisfies the γ > 2C condition would help bridge the theory-practice gap.

- Breakdown of runtime comparison (function evaluations, per-step costs) beyond asymptotic complexity would better substantiate the 216s vs 15-minute claim.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Weakness**: "The claim that D-Flow is a special case of OC-Flow (optimizing x₀ vs. optimizing θ_t) is mathematically loose." **Justification for removal**: The paper provides explicit derivations in lines 148-152 showing D-Flow corresponds to the case with a single control term at t=0, with LBFGS providing dynamic learning rate analogous to varying γ. While the connection could be elaborated, calling it "mathematically loose" overstates the issue.

- **Weakness**: "The runtime claim (216s compared to 15 minutes in D-Flow) in the Conclusion is not substantiated in the experimental section." **Justification for removal**: Table 1 provides complexity analysis showing O(nD²) vs O(ND²), and line 323 references Appendix D for detailed computation efficiency analysis. While constant factors aren't detailed in the main text, this is a minor presentation issue rather than a substantive flaw.

- **Strength**: "Theoretical Bound on Distribution Deviation" from Strength Finder. **Justification for removal**: This strength conflicts with the verified weakness that Proposition 1's assumptions (Affine Gaussian path) don't match the Rectified Flow used in experiments. When a strength and weakness disagree, the weakness wins.

## Novel Insights
The paper's most genuinely novel contribution is extending optimal control-based guidance to SO(3) manifolds with formal convergence guarantees—this addresses a real gap since prior backprop-through-ODE methods were limited to Euclidean space. The insight that the running cost in optimal control formulation bounds KL divergence (Proposition 1) provides useful theoretical grounding, even if the assumptions are idealized. However, the theory-practice gap and unfair baseline comparisons prevent this from being a fully convincing demonstration of the framework's advantages.

## Suggestions
1. **Re-run baselines with hyperparameter tuning**: Conduct comparable hyperparameter sweeps for StyleCLIP, FlowGrad, and D-Flow on the specific tasks. If OC-Flow still outperforms fairly-tuned baselines on most metrics, the performance claims would be much stronger.

2. **Qualify theoretical claims**: Revise claims of "convergence guarantee" to "convergence under idealized assumptions" or "monotonic improvement when assumptions hold." Add discussion of when the affine Gaussian and Lipschitz assumptions are likely violated in practice and what empirical behavior to expect in those cases.

3. **Acknowledge trade-offs**: In the abstract and conclusion, note that OC-Flow improves most metrics while acknowledging specific cases where baselines perform better (ID preservation, Δε MAE, RMSD), and discuss when practitioners might prefer alternative methods.

4. **Clarify RMSD interpretation**: For peptide design, explicitly discuss whether higher RMSD represents a limitation (loss of structural accuracy) or a feature (exploration of novel binders), and analyze the energy-RMSD trade-off curve.

5. **Empirical Lipschitz analysis**: Provide empirical estimates or discussion of whether the neural vector fields used approximately satisfy Lipschitz conditions, and whether the chosen γ values are in the regime where convergence theory would apply.

## Score and Decision

**Calibration anchors consulted:**
- EA80Zib9UI (6.5, Safety-Guided Flow): Stronger theory-experiment alignment, more comprehensive ablation, fairer evaluation. This paper's theory-practice gap is more severe.
- NlnDselrtl (6.0, Riemannian VFM): Similar SO(3)/manifold scope but clearer motivation and more careful empirical claims without overclaiming.
- RDerF20JYT (8.0, La-Proteina): Much stronger empirical validation, clearer limitations discussion, no fairness concerns in baselines.
- k9CzIvzfaA (5.33, Embedding Limitations): Similar pattern of strong theory but experimental setup questioned for task-dataset mismatch; that paper scored ~5.
- tiBoNGGaNC (4.0, TCG): Similar theory-practice gap (claims about CFG instability without rigorous proofs), scored 4.
- I3spHvRHqo (4.0, Test Error Bounds): Limited empirical validation of theoretical claims, scored 4.
- G5zJaSxMGN (4.0, Tabular Pretraining): Flagged for unfair evaluation (skipped tuning for some models), scored 4.

This paper has genuine contributions (SO(3) extension, memory efficiency, unified framework) and meaningful empirical improvements, placing it above the 4.0 papers which had more severe flaws. However, the theory-practice gap and unfair baseline comparison are significant methodological issues that prevent it from reaching the 6.0+ range where papers have stronger experimental rigor. The paper is comparable to k9CzIvzfaA (5.33) in having solid core ideas undermined by experimental design issues. The unfair baseline issue is similar to G5zJaSxMGN (4.0) but this paper's empirical results are stronger overall.

Positioned relative to anchors: above the 4.0 papers due to stronger empirical results and genuine novelty, but below the 6.0 papers due to the theory-practice gap and unfair comparisons.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>