=== CALIBRATION EXAMPLE 31 ===

# Final Consolidated Review
## Summary

This paper proposes a diffusion model operating on toric varieties — the constrained conformation spaces arising from loop closure conditions in proteins and closed kinematic chains. The method enforces loop closure at every diffusion step by computing the Jacobian-based tangent space of the variety (via SVD), sampling noise in that tangent space, and projecting back to the variety via the R6B6 kinematic solver. The approach is demonstrated on MHC peptide conformation prediction (78 test cases) and nanobody CDR3 loop prediction (38 test cases), achieving 15.8% and 22.5% improvement in median RMSD respectively over AlphaFold 2.

---

## Strengths

- **Mathematically grounded geometric inductive bias.** The paper is the first to formulate loop conformation sampling as diffusion on a real algebraic toric variety, rather than a hypertorus or Euclidean space. The Jacobian-based tangent space (Eq. 2) correctly identifies the (n−6)-dimensional manifold of loop-closure-satisfying deformations at each point and is mathematically sound. This is a distinct and non-trivial contribution to geometric deep learning.

- **Hard constraint satisfaction by construction.** Unlike methods that generate structures and then apply post-hoc relaxation (which can alter backbone geometry significantly), this approach maintains loop closure exactly at every step via R6B6. The difference from methods that approximate constraints or rely on energy minimization is concrete and well-motivated.

- **Comparable best-RMSD performance to AF3 on the harder nanobody task.** On the nanobody CDR3 dataset (Table 2), the method achieves median best-RMSD of 1.12 Å versus AF3's 1.17 Å, with mean 1.35 Å versus 1.22 Å. This demonstrates that the diffusion model's conformational sampling quality is in the same range as AF3 — a noteworthy result given the structural constraints involved — even though its confidence-based selection lags.

- **Computational efficiency.** Generating one conformation with 20 denoising steps takes approximately 1 second, with SVD cost O(6N) that is linear in loop length. This makes the approach practical for iterative refinement workflows in drug design and vaccine design pipelines.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Theoretical gap in score matching formulation.** The paper trains the score model to match ∇_{τ_t} log p(τ_t|τ_0), where p is a Gaussian in the *tangential* coordinates τ. However, after applying R6B6 to Δζ_t, the six pivot torsions are modified non-linearly to enforce closure, producing Δζ'_t ≠ Δζ_t. The effective perturbation in tangential coordinates after R6B6 is therefore not the Gaussian Δτ that was sampled. The training target is the score of the *original* Gaussian Δτ, but the system state is determined by the *corrected* Δζ'_t. For small perturbations the discrepancy is minor, but for large t (high noise), the mismatch can be substantial. The paper presents the loss (Algorithm 1, last line) as if the Gaussian score at Δτ is the correct training target, without acknowledging or analyzing this approximation. No bounds on the discrepancy, empirical calibration, or importance-weighting correction are provided.

- **No ablation isolating the contribution of the toric variety constraint.** The paper cannot attribute its improvement over AF2 specifically to the closure-constrained variety treatment versus (a) the diffusion model's diversity, (b) the AF2 refinement+scoring applied afterward, or (c) simply sampling more structures. There is no comparison against: (i) unconstrained torsional diffusion with post-hoc R6B6 correction, (ii) an ensemble of randomly perturbed AF2 structures refined by AF2 (same number of AF2 calls), or (iii) systematic torsional sampling followed by AF2 refinement. Without these controls, the source of improvement is ambiguous.

- **AF2 refinement confounds attribution.** The pipeline applies AF2 refinement and uses AF2 pLDDT to select among the 20 generated conformations. This means AF2 runs an additional 20 refinement passes per case beyond the initial prediction. It is unclear whether the performance gain comes from the diffusion model's loop sampling or from applying 20 AF2 refinements as opposed to 1. A fair ablation would generate 20 starting points by other means (e.g., random torsional perturbation, Rosetta KIC) and apply the same AF2 refinement+scoring pipeline, to check if the diffusion model's specific diversity contributes.

- **Abstract's singularity-free claim is contradicted by the text.** The abstract states the method explores the variety "without encountering singular or infeasible states," but Section 3.3 and Algorithm 2 explicitly describe a rejection-sampling fallback: if R6B6 cannot close the loop, the step is skipped ("the structure will stay at current state"). The success rate of >95% at inference means up to 1 in 20 steps are rejected. Near singular configurations or at high noise levels, this rate may be substantially lower. The paper should replace the categorical claim with an accurate characterization (e.g., "with rare fallback to rejection sampling") and provide singularity encounter statistics as a function of loop length and diffusion time t.

### Minor

- **Test sets are small (N=38, N=78) and no statistical significance is reported.** A 22.5% improvement in *median* RMSD over 38 samples is numerically meaningful but statistically fragile — shifting 2–3 outlier cases could reverse the comparison. No bootstrapped confidence intervals, p-values (e.g., Wilcoxon signed-rank), or sensitivity analysis is provided. This limits confidence that the improvement will hold on a broader, more diverse dataset.

- **pLDDT-based selection uses AF2's confidence on this model's generated structures without validation.** The authors note "we did not learn a model to assess the confidence of the generated ensembles" and use AF2 pLDDT as a proxy. The paper should demonstrate that AF2 pLDDT correlates with actual RMSD *for this model's distribution* of generated structures, not just for AF2's own outputs. Without this, the confidence-based ranking mechanism lacks grounding.

- **Implicit rejection bias in training.** Algorithm 1 retries perturbations until R6B6 succeeds. This introduces implicit rejection sampling: perturbations leading to near-singular configurations are systematically under-represented in training. The effective training distribution of noise levels is therefore non-uniform in t and biased away from singular regions. The paper does not discuss whether this biases the learned score or affects model behavior near singularities.

### Tiny

- **Notation in Section 3.3 conflates several objects.** The paper uses τ_t, Δτ, Δζ_t, and Δζ'_t without a clear summary table. Readers must reconstruct the chain: Δτ (tangential Gaussian) → Δζ_t (ambient, via null vectors) → Δζ'_t (after R6B6 correction). A single paragraph or diagram clarifying the roles and spaces of these objects would substantially improve readability.

- **The claim "1 second per conformation" excludes AF2 initialization.** The total wall-clock time for the pipeline includes running AF2 to generate the starting structure (which is the dominant cost by orders of magnitude). Characterizing inference as "~1 second" without noting this context is potentially misleading for practitioners.

---

## Nice-to-Haves

- **Comparison against Rosetta KIC or other specialized loop modeling tools.** The paper argues in Section 2 that improvements should be measured relative to AF2 (the de facto starting point for modern loop modeling), which is a defensible framing. However, including KIC or NMA (mentioned in the introduction) as supplementary comparisons would clarify whether the approach offers improvements beyond classical kinematic solvers *before* AF2 refinement.

- **Ensemble diversity metrics.** Reporting pairwise RMSD within the 20 generated trajectories would establish whether the diffusion model is genuinely exploring distinct conformational modes or collapsing near its starting point. If conformations are nearly identical, "best-of-20" performance is essentially a single-sample result.

- **Code and weights release.** The R6B6 implementation (Cao et al., 2023) and the trained score model are central to reproducibility. ICLR reproducibility expectations would be well-served by releasing these.

- **Evaluation on longer or shorter loops beyond the tested range.** Loops with n<6 are excluded by the algebraic framework (no closure degrees of freedom). Loops with n=6 have a discrete closure set. Performance on loops at the boundaries of the tested range (15–20 residues) and shorter loops (8–14 residues, which appear in the MHC data) would better characterize the method's operating regime.

- **Visualization of closure error across diffusion steps.** A plot of loop end-to-end closure error as a function of denoising step (for both valid and rejected steps) would directly verify that the method stays on the variety and would strengthen the paper's core claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Abstract misleadingly claims improvement over state-of-the-art" (Harsh Critic).** The abstract says "improve upon the state of the art *open source* AlphaFold." AlphaFold 3 is a server-only proprietary tool as of the time of this paper, not open source. AF2 is the open-source reference, and the claims of improvement are numerically correct. The framing is accurate, not misleading. This criticism is invalid.

- **"Training/testing distribution mismatch" (Harsh Critic).** The paper explicitly addresses this in Section 4: "during validation and testing we start our prediction from AF2 models… to ensure our method is not biased by the information contained in the native structures." The mismatch is a deliberate design choice and is documented. Not a weakness.

- **"Missing related work" (Harsh Critic).** Per review policy, missing related works are not cited as weaknesses since we cannot verify external references.

- **"Unfair comparison with AF3 at top-confidence level" (Harsh Critic, partially).** The paper itself acknowledges (Table 2 footnote and Section 4.2) that AF2 and AF3 confidence functions differ and thus cannot be directly compared at the top-confidence level. The paper presents AF3 as reference, not as a direct methodological competitor under the same selection protocol. The comparison is appropriately caveated.

- **"Dataset size should be larger" — as a standalone weakness.** The field of structural biology frequently evaluates on sets of this scale given PDB availability constraints. Small datasets are a concern worth noting (kept above under Minor) but demanding substantially larger datasets as a binary requirement would be scope creep given community norms. Retained only as a statistical power concern.

---

## Novel Insights

The synthesis across all three reviews surfaces one genuinely underappreciated issue: the paper blends two distinct contributions — a mathematical framework (diffusion on toric varieties) and a biological modeling pipeline (loop prediction via diffusion + AF2 refinement) — in a way that makes it difficult to attribute observed improvements. The toric variety framework is theoretically elegant and novel; the biological results are promising but confounded by the AF2 refinement post-processing. Disentangling these two would substantially strengthen the paper and clarify the mechanism of improvement. A secondary insight is that the method's best-RMSD parity with AF3 (Table 2) suggests the sampling quality is genuinely competitive, and the gap in top-confidence performance is likely attributable to the absence of a task-specific confidence model — an infrastructure gap that could be addressed without changing the core geometric method.

---

## Suggestions

1. **Acknowledge and analyze the score matching approximation.** In Section 3.3, explicitly state that the score target is computed for the tangential Gaussian Δτ, while the applied perturbation is Δζ'_t (after R6B6). Provide at least an empirical analysis of how much Δζ'_t differs from Δζ_t in tangential coordinates across training samples and diffusion times t.

2. **Add an ablation that isolates the toric variety constraint.** Run the identical pipeline (same architecture, same AF2 refinement, same number of samples) but with: (a) unconstrained torsional diffusion + post-hoc R6B6 projection, and (b) random Gaussian torsional perturbation + post-hoc R6B6 + AF2 refinement. This would clarify whether the variety-aware training itself contributes.

3. **Fix the abstract's singularity claim.** Replace "without encountering singular or infeasible states" with an accurate statement such as "with rare encounters of infeasible states, handled by step rejection (>95% step acceptance rate)."

4. **Report statistical significance for RMSD comparisons.** Apply a Wilcoxon signed-rank test (appropriate for paired, non-parametric data) on RMSD differences for both test sets and report p-values. Given N=38 and N=78, this is straightforward and necessary.

5. **Validate pLDDT as a ranking signal for this model's outputs.** Show a scatter plot or Spearman correlation of AF2 pLDDT vs. loop RMSD for the model-generated ensemble. If the correlation is weak, the selection mechanism is unreliable and should be improved.

6. **Report ensemble diversity (pairwise RMSD within 20 trajectories) as a supplement.** This addresses whether the "best-of-20" gain reflects genuine exploration of conformational space.

---

**Overall assessment:** The paper presents a genuinely novel and intellectually interesting mathematical framework for constrained molecular diffusion. The application to protein loop modeling is well-motivated and the empirical improvements over AF2 are real. However, the paper's technical soundness is undermined by an unacknowledged and unanalyzed approximation in the score matching formulation, and its empirical support is weakened by the absence of ablations that isolate the toric variety contribution from the AF2 refinement post-processing. The evaluation scale (N=38, N=78) with no statistical testing is marginal. In its current form the paper is an interesting but incomplete contribution: the mathematical framework deserves to be showcased with an experimental evaluation that is rigorous enough to support the claims made.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 5.0, 5.0]
Average score: 4.5
Binary outcome: Reject
