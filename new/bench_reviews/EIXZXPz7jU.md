## Summary
This paper proposes FMS PINN, an adaptive collocation strategy for PINNs that combines residual-weighted bootstrap resampling with a flow-matching generative model to place new points in high-residual regions. The main empirical claim is that this sampler improves over a normalizing-flow-based DAS/KR-net baseline on multimodal Poisson benchmarks and on some components of a linear elasticity problem with material inclusions.

## Strengths
- The paper introduces a specific and nontrivial alternative to density-estimation-based adaptive PINN sampling: instead of fitting an explicit residual-proportional density with a normalizing flow, it trains a flow-matching model on residual-weighted samples and uses ODE sampling to generate new collocation points. This is a genuine methodological shift relative to the DAS/KR-net baseline discussed in Secs. 3.2–3.4.
- The 9-peaks Poisson result is a meaningful stress test for multimodal sampling, and the reported gap to DAS is large. In Table 1, FMS achieves \(4.2\times 10^{-4}\) vs \(10^{-1}\) for DAS on the 9-peaks problem, and Fig. 4(a) shows generated points concentrated around all nine modes rather than collapsing to only part of the structure.
- The 5D two-peaks experiment goes beyond the usual 2D PINN toy setup. On this task, Table 1 reports \(6.1\times 10^{-3}\) for FMS versus \(2.3\) for DAS, and Fig. 5 indicates that FMS recovers the two-peak structure while DAS misses it. Even if the benchmark remains synthetic, this does strengthen the case that the sampler can help in higher-dimensional settings.
- The paper does not overstate universal superiority in the tables: Table 2 shows mostly favorable but mixed elasticity results, which at least indicates the method is being tested on a more heterogeneous application rather than only on one cherry-picked Poisson example.

## Weaknesses
###: Fatal

### Major:
- The headline **efficiency** claim is not supported by the evaluation. The abstract and conclusion repeatedly claim improved “accuracy and efficiency,” but Sec. 4 reports only error/MSE curves and final MSE tables. There are no wall-clock times, compute-normalized comparisons, memory costs, or error-vs-compute plots, despite the method adding a second learned model and an ODE-based sampling stage. As written, the paper supports an accuracy claim on selected tasks, not an efficiency claim.
- The experiments do not isolate whether **flow matching itself** is responsible for the gains, as opposed to simpler residual-based resampling. Algorithm 1 combines several ingredients at once: residual evaluation, weighted bootstrap, flow-model training, generated-point augmentation, and progressive growth of the collocation set. The only substantive baseline is DAS/KR-net. There is no ablation against weighted bootstrap alone, RAR/RAD-style residual refinement, or a simpler heuristic high-residual sampler under matched point/compute budgets. This leaves the core contribution only partially established.
- The method description overstates what distribution is being learned. The abstract says the approach performs “generative sampling from the distribution of PDE residuals,” but Algorithm 1 actually computes residuals only on the current finite set \(\mathbb{S}_{k-1}\), then forms a residual-weighted bootstrap subsample \(\mathbb{A}_k\), and trains flow matching on that empirical set. That is materially narrower than learning the residual distribution over the domain. The current formulation is better described as learning a sampler concentrated on a residual-weighted empirical point cloud.
- The linear elasticity section is under-specified in ways that matter for validity. Sec. 4.2 states that the plate contains an inclusion made of a second material with different Young’s modulus and “complex geometry,” but the main text does not clearly specify the piecewise coefficient field, interface conditions, or how the reference solution is generated. For heterogeneous elasticity, these details determine what PDE the PINN is actually minimizing and how meaningful the comparison is. Since the abstract highlights this as a key application, the missing specification weakens the empirical case.

### Minor
- The comparison claim should be phrased more narrowly. The evidence supports that FMS outperforms the specific DAS/KR-net baseline on the evaluated Poisson tasks and on some elasticity components, but not that it broadly outperforms “normalizing flows” as a class. The mechanistic discussion in Sec. 3.2 about topology limitations of normalizing flows is plausible motivation, but it is not experimentally demonstrated here.
- The elasticity evidence is mixed, and the presentation does not always reflect that clearly. Table 2 shows FMS is better on 3 of 4 reported entries, but only slightly worse on diamond \(u_y\). The narrative should be more precise about this mixed outcome rather than implying uniformly stronger performance.
- The 5D experiment is promising but somewhat confounded by the training setup: “at initial step of training we draw 100k points from uniform distribution and 60k points from Gaussian centers.” Because the initial sample is already highly structured around the known peaks, this weakens the claim that the adaptive generator alone is discovering difficult regions.
- Important implementation details for reproducibility and interpretation are missing from Algorithm 1 / Sec. 3.4.1: e.g., the exact residual weights used for bootstrap (signed residual, absolute residual, squared residual), whether generated points outside the domain are rejected or projected, whether old points are pruned, and how boundary sampling interacts with adaptive interior sampling.

### Trivial
- There are several notation/technical inconsistencies that reduce clarity: \(\mathbb{A}_k\) vs \(A_k\) in Algorithm 1; Eq. (15) defines \(C=\frac{E}{(1+2)(1-2\nu)}\), which appears suspicious on its face and should be checked against the appendix derivation; and the flow-matching exposition around Eqs. (6)–(8) is hard to follow.
- The figure/text cross-referencing is occasionally inconsistent (e.g., the 9-peaks discussion refers to “Figure 12” in the extracted main text while discussing the comparison in Sec. 4.1).

## Nice-to-Haves
- Add repeated-seed results or at least some measure of variance, especially for the elasticity results where some gains are modest.
- Include direct visualizations of generated samples over multiple refinement stages to show how coverage evolves.
- A small sensitivity study over the number of resampling stages, flow-training iterations, and added points per stage would improve confidence in robustness.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Doubts about existence/release/availability of cited models, repositories, or baselines** — removed per instruction. If the paper cites DAS/KR-net and a public repository, that should be treated as existing.
- **Formatting/parser-artifact complaints about captions or PDF extraction quality** — removed as style/format nitpicks. However, I kept the substantive issue that the presentation of the elasticity results is mixed, because that is reflected in Table 2 itself.
- **Complaints that the comparison is unfair because the baseline may be in TensorFlow while FMS may be in another framework** — removed. The paper states in Sec. 4.1 and Sec. 6 that it used the DAS repository and “the same number of points, epochs, and resampling stages.” Without stronger evidence from the paper text, framework-level speculation would be unfounded.
- **Demands for comparisons to classical PDE solvers** — removed as scope creep. This is a PINN adaptive sampling paper, so the key question is whether the sampling method improves PINNs, not whether PINNs beat finite elements or other classical solvers.
- **Generic criticism that the experiments are ‘too low-dimensional/simple’** — weakened/removed as a standalone flaw. The paper includes a 5D benchmark and an elasticity application; while the scope is still limited, the real issue is insufficient baseline/ablation support, not merely dimensionality.

## Novel Insights
The most important synthesis is that the paper’s strongest evidence supports a *narrower* contribution than the abstract claims. Empirically, the submission makes a plausible case that a residual-bootstrap-plus-flow-matching pipeline can outperform the specific DAS/KR-net baseline on sharply multimodal source-singularity benchmarks, especially the 9-peaks and 5D two-peaks problems. What it does **not** yet establish is that flow matching is the decisive ingredient, that the method is computationally efficient, or that it learns the residual distribution in the stronger sense suggested by the prose. In other words, the work looks more like a promising but under-isolated adaptive sampling variant than a fully validated new principle for efficient PINN training.

## Suggestions
- Replace the current broad claim with a precise one: FMS learns a sampler from a residual-weighted empirical point set and outperforms the DAS/KR-net baseline on the evaluated Poisson benchmarks and some elasticity components.
- Add a compute-aware evaluation: wall-clock time, total sampler + PINN training cost, and error-vs-compute plots.
- Add ablations that isolate the contribution of flow matching:
  - weighted bootstrap only,
  - residual-based refinement without generative modeling,
  - uniform sampling with the same total point budget,
  - ideally a matched-budget comparison against RAD/RAR-style methods.
- Fully specify the elasticity setup in the main text: piecewise material coefficients, interface conditions, and reference-solution generation.
- Clarify implementation details in Algorithm 1, especially how residual weights are formed and how generated points are handled with respect to the domain/boundary.
- Recheck and tighten notation and equations, especially the elasticity constant definition and the flow-matching derivation.

## Score and Decision
**Novelty:** Moderate. Using flow matching rather than explicit density-modeling normalizing flows for adaptive PINN sampling is a real idea, not a trivial variant.

**Technical soundness:** Mixed. The method is plausible and some results are strong, but the paper overstates what distribution is learned, and the elasticity setup is under-specified.

**Empirical support:** Moderate-to-weak for the full claim. The Poisson evidence is good against DAS/KR-net, but the lack of simpler baselines, ablations, and compute analysis leaves the main claim only partially validated.

**Significance:** Potentially meaningful if the efficiency and causal role of flow matching were established, but the current version does not yet support that level of significance.

**Clarity:** Uneven. The high-level algorithm is understandable, but several technical details and parts of the derivation/application are underspecified or inconsistent.

Given the calibration examples and the current level of evidence, this reads like a promising workshop-style or early-stage ICLR submission rather than a fully convincing conference paper. The idea is interesting and some benchmark results are genuinely strong, but the missing compute analysis and missing isolation of the claimed contribution are substantial issues for acceptance.

MY FINAL SCORE: <pineapple>4.9</pineapple>
MY FINAL DECISION: <orange>Reject</orange>