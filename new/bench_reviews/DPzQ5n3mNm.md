## Summary

The paper introduces Sensitivity-Constrained Fourier Neural Operators (SC-FNO), which augments standard FNO training with a loss term matching the neural operator's predicted Jacobians (∂u/∂p) to sensitivity data precomputed via differentiable solvers or finite differences. The key finding is that standard FNOs, despite accurate solution predictions, produce wildly inaccurate parameter sensitivities—degrading inversion and robustness—and that explicitly supervising these sensitivities dramatically improves performance across multiple PDEs and neural operator architectures.

## Strengths

- **Identifies an important, under-appreciated problem:** The paper systematically demonstrates that standard FNOs can achieve R²≈0.98 for solution prediction while having R²≈0.21–0.78 for parameter sensitivities (Table 1), with practical consequences for inversion and out-of-distribution robustness. This observation is not widely recognized in the neural operator literature and is well-supported by concrete metrics.

- **Strong empirical improvements for inversion:** Parameter inversion accuracy improves dramatically (e.g., multi-parameter PDE1: R²=0.986 for SC-FNO vs. 0.642 for FNO; multi-parameter PDE2: R²>0.96 for SC-FNO vs. ~0.85 for FNO). These gaps are large enough to be practically significant, not marginal.

- **Broad experimental coverage:** Experiments span ODEs, wave equations, Burgers' equation, Navier–Stokes, Allen–Cahn, and zoned high-dimensional (82-parameter) Burgers' equations. The method is also tested across FNO, WNO, MWNO, and DeepONet (Appendix D.1), supporting the generality claim.

- **Practical versatility of gradient computation:** Supporting both AD-based and FD-based gradient generation (Table 5) makes the approach applicable to non-differentiable legacy solvers, broadening practical reach beyond differentiable programming platforms.

- **Data efficiency and performance ceiling:** SC-FNO with 100 training samples achieves lower L² error than FNO with 500 samples on the 82-parameter problem (Table 4), suggesting genuine data efficiency gains rather than just marginal improvements.

## Weaknesses

### Fatal
None.

### Major

- **Overstated novelty relative to prior derivative-supervised training methods.** The core contribution—adding a Jacobian-matching loss to neural operator training—is a direct application of derivative/sensitivity supervision (Sobolev training, gradient-enhanced neural networks) to the neural operator setting. The paper cites Liu & Batill (2000) and Czarnocki et al. (2017) but does not substantively engage with what distinguishes SC-FNO from these predecessors, nor with the broader literature on derivative-informed PDE surrogates. The insight that FNOs have poor sensitivities is valuable, but the algorithmic contribution (add L_s = ‖∂û/∂p − ∂u/∂p‖²) is a straightforward instantiation of a known idea, not a new mechanism. This does not invalidate the work but limits its conceptual novelty.

- **Baseline comparison is too narrow for the strength of claims.** The paper compares only FNO, FNO-PINN, SC-FNO, and SC-FNO-PINN. Missing are: (a) adjoint-based or ensemble-based inversion methods using the differentiable solver directly—the most natural baseline when a differentiable solver is already available; (b) derivative-enhanced surrogates based on simpler architectures (e.g., MLPs or CNNs with Sobolev loss); (c) direct inverse mapping methods (Vadeboncoeur et al., 2023), which the introduction explicitly mentions as an alternative. The headline claim that "SC-FNO exhibits superior performance in parameter inversion tasks" rests on comparisons with only two baselines within the FNO family, leaving open whether a well-tuned alternative approach could achieve comparable results.

- **Potential circularity in the inversion evaluation and missing noise/partial-observation tests.** All inversion experiments use synthetic, noise-free, full solution-path data from the same solver used to generate training data. Since SC-FNO is explicitly trained to match ∂u/∂p from that solver, its inversion advantage over FNO—which receives no such supervision—is partly guaranteed by construction. The paper does not test inversion under observation noise, partial observations, or model-form error, all of which are dominant challenges in real inverse problems. Without these tests, the practical significance of the inversion improvements is uncertain.

- **Suspicious uniformity of results in Figure 2.** For PDE1, all five parameters report identical R² values for FNO (0.635) and SC-FNO (0.945). For PDE2, all four parameters report identical R² values per model (0.85 for FNO, 0.96 for SC-FNO). These identical values across distinct physical parameters (e, γ, c, u, v) are implausible for independently evaluated metrics and suggest either aggregation that obscures per-parameter variation or a reporting error. This undermines confidence in the precision of the reported results and needs clarification.

### Minor

- **No ablation on the relative weighting of L_s versus L_u.** The loss function L_total = L_u + L_s (and optionally L_Eq) is presented without discussion of how the terms are balanced, whether equal weighting was used, or how sensitive results are to this choice. This is critical for practical adoption. The observation that L_Eq provides minimal additional benefit beyond L_s (Tables 1, 2) is noted but not analyzed—understanding why would strengthen the paper.

- **No error bars or statistical significance reporting.** All metrics are from single training runs without variance estimates, making it difficult to assess robustness across random seeds or training conditions.

- **Under-analyzed cost of Jacobian data generation for high-dimensional parameters.** While training overhead (30–130% per epoch) is reported, the one-time cost of computing and storing ∂u/∂p for all training samples—especially for the 82-parameter case—is not transparently analyzed in the main text (referenced to Table D.12 in the appendix). For FD-based gradients, the cost scales linearly with the number of parameters, which could be significant for truly high-dimensional problems.

- **Inversion experiments lack comparison scenarios using noisy or sparse observations.** The current results only use clean, full solution-path data. Real-world applications typically involve noisy, sparse measurements.

- **"Concept drift" terminology overclaims.** The paper uses "concept drift" to describe testing with parameters beyond training range (Section 3.2), but the experiments only scale parameter ranges along the same axes (multiply upper bound by 1+λ). This is extrapolation, not a genuine distribution shift or drift. The language should be toned down.

### Trivial

- The abstract's claim that SC-FNO "decreases training time" is ambiguous. It likely refers to time-to-accuracy (since SC-FNO needs fewer samples), but per-epoch cost increases 30–130%. Clarifying this in the abstract would prevent misinterpretation.

## Nice-to-Haves

- Theoretical or even informal analysis of why supervising ∂u/∂p improves inversion convergence (e.g., via Lipschitz arguments or perturbation analysis), rather than relying entirely on empirical results.
- Comparison with gradient-free inversion methods (e.g., Bayesian/ensemble approaches) to disentangle whether SC-FNO's inversion advantage comes solely from better gradients or from improved landscape geometry.
- Failure mode analysis: no experiment shows SC-FNO performing worse than FNO in any setting, which raises questions about whether negative cases were explored.
- Analysis of the root cause of FNO's poor sensitivity estimation (e.g., spectral bias, implicit regularization of Fourier layers).

## Removed Points

- **"Not yet released / nonexistence" claims about tools:** Removed per hard rules. The paper cites torchdiffeq and other differentiable programming tools; their availability is not in question.
- **Reproducibility concerns about undisclosed hyperparameters:** The paper provides architectural details in Tables C.7 and C.8. Demanding complete training logs or every hyperparameter is beyond what is standard in this community.
- **Formatting/presentation nitpicks:** Removed per hard rules (e.g., complaints about notation or figure quality).
- **Demand for 3D PDE experiments:** The paper tests up to 2D Navier–Stokes and an 82-parameter problem. Requesting 3D experiments is scope creep; the paper's scope is clearly defined and the current experiments adequately support the claims made.
- **Demand for theoretical proofs/guarantees as a fatal requirement:** While theoretical analysis would strengthen the paper, this is an empirical methods paper in the physical sciences tradition. Lack of formal theory is a nice-to-have, not a fatal flaw.
- **Claim that SC-FNO is "not novel" and therefore unpublishable:** While derivative-supervised training is not new in general, its systematic application to neural operators with the specific finding that FNOs have poor sensitivities (despite good solution accuracy) is a genuine contribution. The novelty concern is real but does not rise to the level of making the paper "not even a paper."

## Novel Insights

The most striking and novel finding is the observation that standard FNOs can achieve R²>0.98 for solution prediction while simultaneously having R²<0.3 for parameter sensitivities (e.g., PDE2 ∂u/∂α: R²=0.206). This "accuracy-robustness gap" is not widely appreciated and has serious implications for anyone using neural operators in gradient-based optimization, sensitivity analysis, or inverse problems. The finding that PINN-based equation loss provides minimal benefit for constraining ∂u/∂p—because PDEs implicitly constrain ∂u/∂x and ∂u/∂t but not ∂u/∂p—is a precise and useful diagnostic.

## Suggestions

- **Clarify the Figure 2 results:** Explain why all parameters report identical R² values, and if they are mean values, provide per-parameter breakdowns.
- **Add ablation on L_s weight:** Even a simple sensitivity study (e.g., L_s weight ∈ {0.1, 0.5, 1.0, 2.0, 5.0}) across 2–3 problems would address the most critical practical question.
- **Tone down "concept drift" claims:** Use "extrapolation" or "out-of-distribution" instead.
- **Add at least one noisy/partial observation inversion experiment** to assess practical robustness.
- **Add one comparison to a non-FNO baseline for inversion** (e.g., EnKF or direct adjoint inversion) to establish practical value beyond the FNO family.
- **Discuss or quantify the one-time cost of Jacobian computation** in the main text, not just in the appendix.

## Score and Decision

**Calibration:** I compared this paper against similar papers from the review finder:
- *BiLO* (PDE inverse problems with neural operators, Reject): This paper is stronger than BiLO—it has a clearer observation, broader experiments, and more dramatic improvements.
- *Sobolev acceleration* (derivative-supervised training, scores 6/6/3/3, Reject): Conceptually similar novelty issue. SC-FNO addresses a more applied setting and has clearer empirical impact, but suffers from the same concern about incremental methodology.
- *Spectral-Refiner* (FNO enhancement, Accept Poster): This paper is not as well-polished or as complete as Spectral-Refiner, but makes a comparable empirical contribution.
- *PIDO* and *Pseudo PINO* (physics-informed operators, Reject/Withdrawn): SC-FNO is empirically stronger than these but has similar baseline/evaluation gaps.

The paper makes a genuinely important empirical observation (FNOs have terrible sensitivities) and demonstrates a practical fix with strong results. However, the methodological contribution is straightforward (add Jacobian-matching loss), baselines are narrow, the inversion evaluation is partly circular, and the suspicious identical R² values in Figure 2 are concerning. The paper is above the clear reject line but has significant limitations that prevent a strong accept. It aligns most closely with borderline accept/weak accept territory relative to the calibration set—stronger than the typical reject papers in this space but not at the level of the Accept Poster exemplars.

MY FINAL SCORE: 5.5
MY FINAL DECISION: <orange>Reject</orange>