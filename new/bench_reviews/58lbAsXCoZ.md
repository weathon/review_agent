Now I have thoroughly read the paper and calibration anchors. Let me compose the final review.

## Summary

The paper proposes NFFS (Neural Functional Flow on Surface), a framework for simulating incompressible fluid on geometric surfaces using neural implicit representations. The core idea combines the Closest Point Method (CPM) with exterior calculus to structurally enforce divergence-free velocity fields on surfaces (including implicit neural representations), and introduces a covariant-derivative-based advection scheme that avoids explicit pressure projection steps. The method is demonstrated on analytic surfaces, explicit meshes, and implicit neural surfaces, with applications to vorticity generation and Helmholtz decomposition.

## Strengths

- **Structural divergence-free guarantee via Theorem 3.1**: The construction v(x) = j⁎((∇(cp⁎σ)∘j(x)) × n(x)) (Eq. 4) cleanly combines CPM with exterior calculus to produce divergence-free velocity fields by design, eliminating projection-induced error. This is a technically sound and practical parametrization that generalizes across surface representations (Sec. 3.2).

- **Simulation on implicit neural surfaces**: The paper is, to my knowledge, the first to demonstrate fluid simulation directly on surfaces represented as neural SDFs (Fig. 7, Sec. 5.2), without requiring mesh extraction or parameterization. The observation that classic functional methods fail to converge on meshes extracted from INRs (noted in Sec. 5.2, with supporting data in Appendix E.4) highlights a genuine robustness advantage.

- **Breadth of demonstrations**: The framework is validated across analytic surfaces, meshes, and INRs (Sec. 5.1–5.2), and extended to conditioning/generation (Sec. 5.3, Fig. 8a) and Helmholtz decomposition on real atmospheric data (Sec. 5.4, Fig. 8b), showing the parametrization's versatility.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed quantitative headline results**: The abstract states "approximately 15 times higher accuracy than other methods with the same storage cost" and "5 times memory savings compared to the classic method." Examining Table 1, the 15× figure primarily compares against PINN (MSE 1.73e5) and INSR (MSE 8.63e4)—methods that are not specialized surface-flow solvers and require surface parameterization not available in general (as the authors themselves acknowledge in footnote 2). The only surface-specific baseline at matched storage is Small-F.S. (MSE 5.34e3), against which the improvement is ~18× in MSE. Meanwhile, the proposed method requires 16.5h versus Small-F.S.'s 0.8h—a 20× compute penalty that is not disclosed alongside the accuracy claim. The "5× memory savings" is a comparison against the GT reference (2643KB), which provides higher accuracy; this is a comparison at unmatched accuracy, not a fair "savings" framing. These headline claims, as stated in the abstract and introduction, misrepresent the actual tradeoffs. This matters because the abstract is the most-read part of the paper and sets expectations that the body does not deliver on.

- **Energy preservation claims lack quantitative main-text evidence**: The phrases "low energy dissipation" and "energy-preserving" appear prominently in the abstract, intro (twice in contribution 2), conclusion, and throughout. However, the main text contains no quantitative energy-over-time plots, decay rates, or energy spectra—only visual comparisons of vorticity fields (Figs. 4–5). The energy validation is deferred to Appendix E.1 (the "sphere rot" case), which is not in the main text. Furthermore, the advection scheme (Eq. 15) results from a first-order truncation of the exponential map (Eq. 14), and no argument or analysis is provided for why this truncation should preserve energy rather than merely reduce dissipation relative to splitting-based methods. For a paper that centrally claims energy preservation, this evidential gap is significant.

### Minor

- **Compute cost not integrated into the evaluation narrative**: Table 1 reports running times (0.8h for Small-F.S. vs. 16.5h for Ours), but the paper does not discuss this tradeoff or evaluate accuracy at matched compute budgets. A reader interested in practical deployment needs to understand when the memory-vs-compute tradeoff favors the neural approach.

- **Qualitative-only evaluation for mesh and INR surfaces (Sec. 5.2)**: Figs. 6–7 show visually plausible results, but there is no ground truth for these cases, and no quantitative metrics (e.g., integrated divergence, vorticity conservation) are reported to verify physical plausibility. Given that these sections showcase the paper's claimed advantage of geometry flexibility, some quantitative sanity check would strengthen the evaluation.

- **No convergence or error analysis for the advection scheme**: The advection loss (Eq. 15–16) is minimized via Adam at each time step, but the paper provides no discussion of optimization convergence criteria, how optimization residuals accumulate across time steps, or the temporal order of accuracy of the scheme. The authors themselves note in the conclusion (line 301) that "a theoretical analysis of convergence and stability of our method would be valuable," acknowledging this gap.

### Trivial
None.

## Nice-to-Haves

- Energy-over-time plots in the main text (even as a small supplementary figure) would substantiate the energy-preservation narrative far more effectively than qualitative vorticity snapshots.
- Comparison at matched compute budget (e.g., running Functional Fluids at 16.5h resolution) would clarify the accuracy-cost Pareto frontier.
- Higher-order time integration (as the authors themselves suggest in the conclusion) could simultaneously improve accuracy and energy behavior.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"PINN and INSR are inappropriate baselines that inflate perceived advantages"** (from Harsh Critic): The paper itself acknowledges in footnote 2 that these methods cannot be applied to general surfaces. On the sphere (Table 1), parameterization to ℝ² is available, so including them is defensible. The real issue is not that they appear in Table 1, but that the "15× accuracy" claim in the abstract averages across methodologically disparate baselines rather than comparing against the most relevant surface-specific alternative (Small-F.S.).

- **"The 5× memory savings is compared against GT which is not a competitor"** (from Harsh Critic): While the comparison framing is misleading (it's at unmatched accuracy), the point about memory efficiency relative to classical grid-based methods is a legitimate contribution. The framing should have been "5× memory savings compared to the reference-resolution classical solver," with the accuracy tradeoff made explicit. The underlying observation—neural representations achieve meaningful compression—has substance, even if the "savings" phrasing oversells it.

- **"The advection scheme's first-order truncation doesn't inherit symplectic structure"** (from Harsh Critic): While technically true that a first-order truncation is not symplectic, the paper's claim is "low energy dissipation" rather than "exact energy preservation." The midpoint-like structure of Eq. 15 (using both v_i and v_{i+1}) does provide some energy-stability properties, and the framework builds directly on Azencot et al. (2014), which demonstrated energy-preserving behavior with this type of covariant-derivative advection. The concern is better framed as "no quantitative energy analysis provided in main text" rather than a claim that the scheme is fundamentally incapable of energy preservation.

- **"Quantitative evaluation missing for explicit/implicit meshes"** (from Harsh Critic, elevated to Minor): Without ground truth, perfect quantitative evaluation is impossible, but metrics like divergence error and vorticity conservation could be reported. The absence of these is minor rather than fatal since the qualitative results are visually informative and the analytic-surface evaluations provide the quantitative backbone.

## Novel Insights

The paper reveals an important practical niche: neural surface-flow methods can succeed on meshes where classical DEC-based solvers fail to converge (implicit neural surfaces with marching cubes). This is more than a convenience—it suggests that the smoothness and differentiability of neural representations circumvent mesh-quality pathologies that plague traditional discretizations. However, the paper's quantitative claims (15× accuracy, 5× memory savings, energy preservation) significantly overstate what the evidence in the main text supports, and the 20× compute cost is underemphasized.

## Suggestions

- Re-phrase the abstract claims to accurately reflect the accuracy-storage-compute tradeoff: e.g., "At comparable storage, our method achieves 18.5× lower MSE than the classical surface-fluid method on the sphere benchmark, at the cost of 20× longer computation time."
- Include at least one quantitative energy-preservation figure (energy vs. time for the sphere rot or jet case) in the main text to support the central energy-preservation claim.
- Report time cost alongside storage in all table entries so the reader can see the full tradeoff surface.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| PhyMPGN | /home/wg25r/review_agent/human_reviews/fU8H4lzkIm.md | 8.0 | Physics-encoded GNN for PDE systems; strong experiments, clear claims, no overclaiming. NFFS is weaker due to overclaimed abstract and missing energy evidence. |
| uKZdlihDDn | /home/wg25r/review_agent/human_reviews/uKZdlihDDn.md | 7.6 | Graph-based latent diffusion for fluid simulation; thorough evaluation. NFFS has narrower but still genuinely novel scope (INR surfaces). |
| HelmSim | /home/wg25r/review_agent/human_reviews/8HG2QrtXXB.md | 5.0 | Helmholtz-based fluid simulation; similar domain, overclaimed novelty relative to prior work. NFFS has clearer technical novelty than HelmSim but shares the problem of overclaiming vs. evidence. |
| clawNOs | /home/wg25r/review_agent/human_reviews/KEpR8hFzvO.md | 5.0 | Divergence-free neural operators; structurally enforces conservation like NFFS, but has novelty concerns and weak baselines. NFFS's CPM+exterior calculus construction is comparably novel. |
| PINN-on-manifolds | /home/wg25r/review_agent/human_reviews/kIZcruKmBg.md | 3.25 | PINNs on surface manifolds; methodologically closer but with weak experiments and missing comparisons. NFFS is stronger with real surface-simulation results. |
| RetNet | /home/wg25r/review_agent/human_reviews/UU9Icwbhin.md | 4.75 | Overclaimed headline results misleading relative to what experiments support. NFFS shares this pattern of overclaiming, though NFFS's underlying contribution is more substantive. |
| Dual-Modal Patch | /home/wg25r/review_agent/human_reviews/OXIIFZqiiN.md | 1.5 | Fundamentally unsound paper. NFFS is far stronger—its overclaiming is about framing, not methodology. |

NFFS sits between the medium-scoring papers (HelmSim, clawNOs at 5.0) and those with serious overclaiming issues (RetNet at 4.75). It has a real, novel technical contribution (CPM-based divergence-free neural fields on surfaces, including INRs) that the medium papers also offer, but the overclaimed abstract and missing main-text energy evidence pull it below a clean accept. Relative to clawNOs (5.0), which also hard-codes conservation but has novelty concerns and weak baselines, NFFS has clearer novelty (CPM+exterior calculus on neural implicit surfaces) but more aggressive overclaiming. A score of 5 reflects this balance.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>