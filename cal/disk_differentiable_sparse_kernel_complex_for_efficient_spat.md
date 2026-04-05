=== CALIBRATION EXAMPLE 68 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title captures the core technical idea: a differentiable sparse-kernel representation for efficient convolution. It is broadly accurate, though “kernel complex” is not standard terminology and obscures the contribution a bit.
- The abstract states the problem, the proposed optimization-based sparse decomposition, and the spatially varying extension. It also reports a speedup claim and comparative advantage over simulated annealing and low-rank methods.
- However, several claims are stronger than what is substantiated in the paper as presented:
  - “higher fidelity than simulated annealing and lower cost than low-rank decomposition” is plausible from the experiments, but the abstract does not specify on which kernels or under what sample budgets.
  - “up to a 20× speedup” appears in Figure 1, but the experiments section does not clearly establish a headline 20× end-to-end speedup in a controlled setting.
  - The abstract says the method supports arbitrary kernels, but the method in Section 4.3 assumes access to the target dense kernel and a one-dimensional parameterization for SV filtering; this is narrower than “arbitrary” in a broad sense.

### Introduction & Motivation
- The motivation is well aligned with an ICLR audience interested in efficient operators and differentiable approximations. The paper identifies a real gap: sparse kernel approximation has largely relied on heuristic search, and spatially varying dense-kernel generation is expensive.
- The contribution list is clear: differentiable sparse-kernel decomposition, initialization strategies, and filter-space interpolation.
- The introduction does slightly overstate generality. In particular, it presents the method as a general solution for arbitrary kernels, but the experimental and methodological discussion suggests the strongest support is for 2D spatial kernels known at optimization time, with a specific one-dimensional basis interpolation for spatially varying effects.
- The novelty relative to prior sparse approximation work should be framed more carefully. The paper’s main distinction is replacing heuristic optimization with end-to-end gradient-based optimization over offsets and weights. That is useful, but the introduction could better clarify why this is a substantive step beyond existing differentiable parameterizations or learned kernel prediction methods.

### Method / Approach
- The method is reasonably described, but there are important reproducibility and conceptual gaps.
- In Section 4.1, the optimization problem is clear at a high level: learn offsets and weights for a multi-layer sparse kernel by matching the impulse response to a target kernel. This is sensible and differentiable.
- A key missing detail is how discrete convolution is made differentiable with respect to continuous offsets. The paper implicitly relies on sampling/interpolation, but the exact operator is not fully specified. For reproducibility, ICRL-level expectations would require:
  - how offsets are applied to a discrete grid,
  - whether bilinear interpolation is used,
  - boundary handling,
  - how gradients are computed when sampled locations move outside the support.
- The “radial initialization” in Eq. (6) is intuitive, but the paper does not fully justify why evenly spaced points on a circle are appropriate for arbitrary target kernels, especially non-radial or anisotropic ones. The claim that the receptive field grows linearly with layer index is plausible, but the derivation of ∆r is only deferred to the appendix and not visible here.
- The “sparse sampling” initialization in Section 4.2 raises more questions than it answers:
  - Eq. (7) gives a radius from support area \(S\) and sample count \(N_s\), but the rationale is not explained rigorously.
  - Rejection sampling from the support sounds reasonable, yet the paper does not explain how it avoids duplicated samples, how it handles very thin or disconnected supports, or what happens when the support is smaller than the requested sample budget.
- Section 4.3 is the most conceptually ambitious part, but also the least fully specified.
  - The basis interpolation assumes a one-dimensional ordered parameter \(p_k\), but many spatially varying effects are inherently multi-parameter (e.g., blur radius plus anisotropy plus orientation). The paper mentions some 2D effects in experiments, but the formulation itself is only clearly defined for a 1D parameter axis.
  - It is not fully clear whether the interpolation is performed over offsets/weights jointly, or whether the basis filters must share a common support pattern.
  - The convex-combination constraint on basis filters is sensible, but the paper does not discuss whether the basis is guaranteed to remain normalized or to preserve energy/brightness.
- The training procedure in Section 4.4 is elegant: optimize the impulse response directly. This is a strong design choice for kernel approximation. But it also creates a limitation the paper should acknowledge more explicitly: the learned sparse complex is tied to matching a fixed target kernel, not directly to image-domain end-task performance. That distinction matters for ICLR standards, which often value principled end-to-end utility.
- The method does not discuss important failure modes:
  - sensitivity to local minima for highly oscillatory or sign-changing kernels,
  - instability when the target kernel has disconnected support,
  - whether negative weights are allowed and whether that affects interpretability or stability,
  - whether the learned multi-layer sparse representation is unique or highly non-identifiable.

### Experiments & Results
- The experiments address the core claims: Gaussian approximation, arbitrary/non-convex kernel approximation, spatially varying filtering, and ablations on initialization/configuration.
- Baselines are plausible: PST is the most relevant heuristic sparse-optimization comparator, and LowRank is a reasonable structured approximation baseline. That said, the comparison set is incomplete for an ICLR-level efficiency paper:
  - There is no comparison to other differentiable kernel parameterizations or learnable sparse mask methods.
  - For spatially varying filtering, the baselines seem somewhat weakly matched to the task, especially if they were not specifically tuned for the same interpolation setup.
- The experimental protocol has some strengths:
  - Multiple kernel families, including non-convex targets and optical PSFs.
  - Runtime benchmarking on a mobile-class device.
  - Multiple metrics including PSNR, LPIPS, and FLIP-LDR.
- However, several issues materially limit the strength of the evidence:
  - The paper does not clearly report error bars, variance over random seeds, or statistical significance. Given that the method involves non-convex optimization, this is important.
  - Many results are described qualitatively, but the paper does not provide enough quantitative tables in the main text to support all claims. For example, “up to 20× speedup” and “1/100 the iterations of PST” need more careful apples-to-apples reporting of wall-clock time and accuracy trade-offs.
  - The sample budget comparisons are not always fully controlled. For instance, the paper compares 8×6, 12×4, 24×4, etc., but it is not always clear whether total sampling counts, FLOPs, and memory are matched fairly across methods.
  - The spatially varying filtering experiments are compelling visually, but they do not clearly establish how much the interpolation basis contributes relative to simply optimizing per-case filters. An ablation on number of basis filters, interpolation granularity, and basis parameterization would materially strengthen this section.
- The ablations in Section 5.4 are useful, especially the comparison of Random, Increasing Radial, and Sparse Sampling initialization. This is one of the paper’s strongest pieces of evidence.
- Still missing are ablations that would be important for ICLR review standards:
  - effect of number of optimization steps on final quality and runtime,
  - effect of the number of sparse layers versus number of samples per layer at fixed FLOP budget,
  - effect of allowing/forbidding negative weights,
  - effect of basis size \(M\) in filter-space interpolation,
  - ablation on whether interpolation in offset-space versus weight-space matters.
- The results generally support the claim that the method is better than PST and LowRank on the evaluated kernels, but the evidence is not yet strong enough to justify the broader implication that this is a generally superior approach to arbitrary kernel approximation.

### Writing & Clarity
- The paper is mostly understandable at a high level, but some key methodological points remain underexplained.
- Section 4.3 is the clearest place where the exposition becomes conceptually dense: the transition from a parameterized family of basis filters to runtime interpolation needs a more explicit algorithmic description.
- The relationship between the single-kernel approximation setting and the spatially varying setting could be clearer. Right now they feel somewhat like two related but not fully unified methods.
- Figures 2–7 appear informative in intent, but the main text does not always guide the reader clearly through what is being compared and what the key takeaway is. The paper would benefit from more explicit quantitative summarization around these figures.
- That said, I do not see major clarity problems that prevent understanding of the main idea.

### Limitations & Broader Impact
- The paper does acknowledge one important limitation in Section 6: access to the target dense kernel is required during optimization and filtering. That is a real and substantial restriction.
- However, several additional limitations are missing or only implicit:
  - The method is primarily a kernel approximation tool, not a learned image restoration model. Its practical usefulness depends on having a known target kernel.
  - The spatially varying formulation appears constrained by a one-dimensional parameterization, which may limit expressiveness for more complex multi-attribute blur fields.
  - The method may be less attractive when kernels change frequently and online optimization time becomes a bottleneck.
  - The differentiable sparse complex may be sensitive to the chosen support size and layer/sample allocation.
- Broader-impact discussion is minimal. That is not necessarily a major issue for this paper, but for an ICLR submission it would be good to mention potential uses in mobile imaging and rendering, and also note that the method is domain-specific infrastructure rather than a general-purpose learning advance.
- No negative societal impact seems obvious, but the paper should still acknowledge that its main practical value is in accelerating existing filtering pipelines rather than introducing a new learning paradigm.

### Overall Assessment
This is a promising and technically sensible paper that addresses a real efficiency problem in convolution-based filtering. Its strongest idea is reframing sparse kernel approximation as differentiable optimization over offsets and weights, and the initialization ablation suggests this is genuinely useful. The main concern is not the core idea, but the level of methodological specificity and empirical support: the paper leaves important implementation details underspecified, the spatially varying formulation is narrower than the headline implies, and the experiments would need stronger quantitative validation and ablations to meet ICLR’s bar for a broadly compelling efficiency method. I think the contribution is credible and likely useful, but as written it does not yet fully justify the breadth of its claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DISK, a differentiable framework for approximating dense convolution kernels using a sequence of learned sparse kernels, optimized directly against the target kernel via impulse-response matching. It also introduces initialization strategies for arbitrary/non-convex kernels and a filter-space interpolation scheme for spatially varying filtering, claiming better fidelity than simulated-annealing-based sparse optimization and better efficiency than low-rank decompositions.

### Strengths
1. **Clear problem motivation with practical relevance.** The paper targets a real bottleneck in computational photography, graphics, and vision: accelerating large, complex convolution kernels, including non-convex PSFs and spatially varying filters. This is a meaningful application area and aligns well with ICLR’s interest in differentiable methods with broad utility.

2. **A principled optimization formulation.** Instead of heuristic search over sparse sampling patterns, the method casts kernel approximation as differentiable optimization over offsets and weights and trains against an impulse response. This is a clean idea and naturally compatible with gradient-based learning pipelines.

3. **Supports arbitrary and non-convex target kernels.** The paper explicitly evaluates beyond Gaussian kernels on shapes such as rings, stars, hearts, animal silhouettes, and optical PSFs. If the reported results hold, this is more general than many kernel-acceleration methods that only work well for analytic or separable kernels.

4. **Addresses initialization, which is important for sparse non-convex optimization.** The proposed radial initialization and sparse sampling initialization are reasonable, concrete design choices, and the ablation section suggests they materially affect convergence. This is evidence that the authors engaged with the optimization challenges rather than only proposing an abstract formulation.

5. **Attempts to extend beyond single kernels to spatially varying filtering.** The filter-space interpolation idea is a useful engineering extension, and the paper’s framing of a compact basis of sparse filters is potentially valuable for resolution-independent per-pixel filter synthesis.

6. **The empirical story is coherent at a high level.** Across figures, the method is presented as improving the quality–latency tradeoff over PST and low-rank baselines, and the paper reports multiple metrics (PSNR, LPIPS, FLIP-LDR) and runtime on a mobile-class device, which is aligned with the application emphasis.

### Weaknesses
1. **The novelty appears moderate relative to the claim.** The core idea is differentiable optimization of sparse sampling offsets/weights for kernel approximation. This is conceptually appealing, but it is also a fairly direct adaptation of a known sparse approximation problem into gradient-based learning. Compared with prior sparse-kernel search methods, the main gain seems to be optimization convenience and better convergence, rather than a fundamentally new representation.

2. **The paper does not clearly establish why the proposed factorization is superior in a general sense.** The composite sparse-kernel chain is presented as a sequence of sparse layers, but the paper gives limited theoretical insight into expressivity, approximation limits, convergence behavior, or why this representation should be easier to optimize than a single sparse kernel or alternative parameterizations.

3. **Reproducibility is incomplete.** Key details are missing or under-specified: exact layer-wise sample allocations for all experiments, the full loss definition details, the precise basis-interpolation procedure for spatially varying filtering, regularization choices, boundary handling, and how target kernels are normalized/centered. For an ICLR submission, this level of ambiguity is a concern, especially since results depend heavily on optimization details.

4. **The evaluation is relatively narrow and heavily application-specific.** The experiments focus on kernel approximation quality and rendering/photography use cases. There is little evidence of broader applicability to learning-based systems, despite the paper’s claim that the method “integrates cleanly into learning pipelines.” ICLR generally expects some broader methodological insight or learning-system validation beyond one niche domain.

5. **Comparisons may not fully cover the strongest relevant baselines.** The paper compares against low-rank decomposition and PST, but does not convincingly position itself against other plausible approaches such as separable approximations, learned dynamic-filter parameterizations, direct discrete sparse optimization with modern gradient estimators, or hardware-aware kernel design methods. This makes the empirical benchmark set feel somewhat incomplete.

6. **The spatially varying filtering section is somewhat underspecified and possibly too specialized.** The method of interpolating among preoptimized basis filters is promising, but the paper does not deeply analyze how well interpolation behaves across the parameter space, how basis count affects memory/runtime/quality, or what happens when the target variation is high-dimensional or non-monotonic.

7. **The claim of “up to 20× speedup” is not contextualized carefully enough.** ICLR reviewers will expect speedups to be measured against strong, well-defined baselines under identical conditions, with details on kernel size, sample count, hardware, and whether preprocessing is amortized. The paper reports latency, but the conditions under which the claimed speedup holds are not sufficiently precise from the text alone.

8. **Some experimental claims are qualitative and may overstate significance.** Phrases such as “visually indistinguishable” and “higher fidelity” are used frequently, but the paper would benefit from stronger statistical analysis or more systematic quantitative reporting across all kernel families and settings.

### Novelty & Significance
**Novelty: Moderate.** The paper’s main contribution is an optimization reformulation of sparse kernel approximation, plus initialization and interpolation tricks. These are useful and nontrivial, but they do not appear to be a major conceptual leap over existing sparse approximation and dynamic filtering literature.

**Significance: Moderate.** For computational photography, graphics, and mobile imaging, the method could be practically valuable if the runtime gains and quality are validated rigorously. However, for ICLR’s standards, the significance is somewhat constrained by the narrow problem domain and the limited evidence of broader ML impact.

**Clarity: Fair.** The overall narrative is understandable, and the main ideas are easy to follow. That said, the paper would benefit from more precise algorithmic descriptions and cleaner exposition of the spatially varying interpolation mechanism.

**Reproducibility: Fair to weak.** The experiments are described at a high level, but not enough implementation detail is given to confidently reproduce the results.

### Suggestions for Improvement
1. **Add a more rigorous algorithmic specification.** Provide pseudocode for training and inference, including exact parameterization of offsets, constraints on weights, handling of kernel centering, interpolation basis construction, and boundary conditions.

2. **Strengthen the theoretical justification.** Include analysis of representational capacity, why stacked sparse kernels help optimization, and whether the decomposition has any guarantees or limitations compared to single sparse kernels or low-rank models.

3. **Expand and sharpen the baselines.** Compare against more alternatives relevant to kernel approximation and acceleration, including separable approximations, modern sparse-optimization methods, and any learned dynamic-filter baselines that can be adapted to this setting.

4. **Improve reproducibility details.** Report all kernel sizes, sample budgets, layer allocations, initialization hyperparameters, regularization terms, optimizer settings, stopping criteria, and exact runtime benchmarking protocol. Make sure these are sufficient for a third party to replicate the results.

5. **Provide broader empirical validation.** Test whether the method transfers beyond hand-designed kernels to learned kernels or to downstream tasks that use kernel modules, to better support the claim that it integrates into learning pipelines.

6. **Analyze the interpolation scheme more deeply.** Show how performance varies with the number of basis filters, the dimensionality of the control space, and the smoothness/monotonicity of the parameter-to-kernel mapping.

7. **Report stronger quantitative comparisons.** Add tables summarizing mean and variance over multiple targets, not just selected visual examples, and include standardized runtime comparisons with clearly matched sampling budgets and hardware settings.

8. **Clarify the practical limits.** Discuss failure cases, when the method breaks down, and whether the approach scales to very large kernels, highly irregular PSFs, or rapidly varying spatially dependent filters.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons to the strongest relevant baselines beyond PST and LowRank, especially direct fast-Gaussian methods for Gaussian cases and modern learned/dynamic filtering methods for SV cases. At ICLR, the current baseline set is too narrow to support the claim of a broadly superior differentiable decomposition method.

2. Add an end-to-end runtime breakdown for optimization time, kernel synthesis time, and filtering time, with the same hardware and the same output quality targets. Right now the paper claims efficiency gains, but the evidence mostly mixes offline fitting with runtime inference, so it is unclear what is actually faster.

3. Add experiments on more diverse kernel families and sizes, including larger supports and harder non-convex PSFs, with a systematic sweep over sparsity budgets. The current examples are visually appealing but too limited to justify the claim that the method works for arbitrary kernels.

4. Add a direct comparison against analytic or exact generation methods for spatially varying blur when applicable, not just approximate baselines. Without this, the claim that filter-space interpolation achieves near-ground-truth quality is not convincingly isolated from the specific test cases.

5. Add generalization experiments where the same optimized basis is reused across unseen parameter maps or unseen blur ranges. This is necessary because the SV claim depends on the interpolation space actually behaving predictably outside the fitted examples.

### Deeper Analysis Needed (top 3-5 only)
1. Add a convergence analysis showing why gradient-based optimization succeeds where simulated annealing fails, including failure cases and sensitivity to initialization. ICLR reviewers will expect evidence that the optimization is robust, not just that it sometimes converges faster on selected kernels.

2. Quantify the approximation-error vs. sparsity trade-off with formal curves, not just a few chosen settings. The paper’s core claim is about efficient sparse approximation, so the missing Pareto analysis weakens the practical value of the method.

3. Analyze when the radial initialization helps and when it hurts, especially for asymmetric or highly irregular kernels. Without this, the initialization is presented as a universal fix, but the paper does not show its limitations.

4. Provide a stability analysis for filter-space interpolation, including whether interpolated sparse kernels remain physically meaningful and whether interpolation can create artifacts. This is essential because the SV method relies on the assumption that basis filters interpolate smoothly.

5. Report variance over multiple optimization runs and seeds. Since the method is an optimization procedure over a non-convex space, a single-run result is not enough to trust the reported quality and speed claims.

### Visualizations & Case Studies
1. Show side-by-side kernel evolution during optimization for representative convex and non-convex targets. This would reveal whether the method truly finds structured sparse solutions or merely overfits the loss in a brittle way.

2. Visualize failure cases where the method cannot match the target kernel within a small sample budget. ICLR readers need to see the boundary of applicability, not just successful examples.

3. Plot the learned sparse offsets and weights for each layer, and how they compose into the final impulse response. This would expose whether the learned decomposition is interpretable and whether the claimed “kernel complex” is actually doing meaningful structured approximation.

4. Show interpolation trajectories across the SV parameter space, including intermediate synthesized kernels. This would make clear whether the basis filters vary smoothly or whether the interpolation is hiding discontinuities and artifacts.

5. Include qualitative comparisons on real images with high-frequency texture and edges under all benchmarked kernels. That is where sparse approximations usually fail, and it would make the method’s claim of perceptual fidelity much more credible.

### Obvious Next Steps
1. Extend the method to learn the target kernel from data or from image supervision, instead of assuming the dense target kernel is known. The current formulation is limited to a narrow offline setting, which weakens the paper’s practical significance.

2. Develop a principled method to choose layer count, sample count, and basis size automatically under a compute budget. Right now the method still requires manual design choices that undercut the claim of a general solution.

3. Generalize beyond 2D spatial kernels to spatiotemporal or anisotropic 3D kernels. This is the most obvious next step if the authors want the method to matter beyond a specialized image-filtering setting.

4. Demonstrate integration into a trainable vision pipeline, with gradients flowing through the decomposed filtering module in a real task. The paper claims differentiability, but does not show that this provides any measurable benefit in a learning setting.

# Final Consolidated Review
## Summary
This paper proposes DISK, a differentiable scheme for approximating dense convolution kernels using a stack of learned sparse kernels, optimized by matching the impulse response of the composed filter to a target kernel. It also adds two initialization heuristics for difficult non-convex targets and extends the idea to spatially varying filtering via interpolation over a compact basis of sparse filters.

## Strengths
- The core formulation is clean and technically sensible: optimize offsets and weights directly in a differentiable way against the target kernel’s impulse response. This is a principled alternative to the heuristic search used by PST and is the paper’s strongest idea.
- The paper addresses a real and practically relevant bottleneck in computational photography/graphics, and it does not limit itself to easy Gaussian cases; the experiments include non-convex shapes and optical PSFs, which is an appropriate stress test for sparse approximation.

## Weaknesses
- The paper overclaims generality relative to what is actually specified and evaluated. The method is presented as an “arbitrary kernel” solution, but the strongest support is for offline approximation of known 2D kernels, and the spatially varying formulation is only clearly defined through a 1D parameterization/interpolation space. That is materially narrower than the headline suggests.
- Reproducibility and algorithmic detail are insufficient for a method whose behavior depends heavily on optimization. The paper does not clearly specify the differentiable sampling operator, boundary handling, exact interpolation procedure for the spatially varying case, normalization/centering conventions, or variance across runs. For a non-convex optimization method, this is a real weakness.
- The empirical validation is too narrow to fully support the efficiency claims. The baseline set is limited, the comparisons are not always clearly matched on compute budget or wall-clock cost, and the paper does not provide the kind of systematic Pareto analysis, variance reporting, or end-to-end runtime breakdown that would be expected for a strong efficiency paper.
- The spatially varying filtering contribution is promising but under-analyzed. The paper shows attractive examples, but it does not rigorously test the interpolation basis size, sensitivity to unseen parameter values, or failure modes when the blur field is more complex than the chosen 1D control axis.

## Nice-to-Haves
- A clearer pseudocode-style description of training and inference, including the exact continuous-to-discrete sampling operator and boundary conditions.
- A more systematic study of the accuracy–sparsity trade-off and the effect of layer/sample allocation under a fixed compute budget.
- A deeper analysis of the filter-space interpolation scheme, including how many basis filters are needed and how well the basis generalizes to unseen parameter maps.

## Novel Insights
The most interesting insight is that the representation is not just “sparse filtering,” but a stacked sparse complex learned in kernel space, where each layer is allowed to move and reweight its samples through differentiable optimization. That makes the decomposition act more like a learnable geometric approximation of the target impulse response than a fixed factorization, which plausibly explains why it can handle non-convex targets better than low-rank methods. The spatially varying extension is also conceptually neat: rather than synthesizing a dense kernel per pixel, the paper amortizes the cost by interpolating among pre-optimized sparse basis filters, effectively shifting the burden from runtime kernel generation to offline basis fitting.

## Potentially Missed Related Work
- Dynamic Filter Networks — relevant as a learned per-pixel kernel prediction framework.
- Decoupled Dynamic Filter Networks — relevant for structured dynamic filtering alternatives.
- Spatiotemporal Variance-Guided Filtering — relevant for spatially varying filter synthesis in graphics.
- Laplacian kernel splatting — relevant for efficient depth-of-field and motion blur synthesis/reconstruction.

## Suggestions
- Add a precise algorithm box with all implementation details needed to reproduce the optimization, especially the differentiable offset sampling and the interpolation rule used for spatially varying filtering.
- Report a true runtime breakdown: offline optimization time, basis precomputation time, per-image synthesis cost, and per-frame filtering latency, all under matched quality targets.
- Include broader and stronger baselines, especially analytic fast-Gaussian methods for the Gaussian setting and learned/dynamic-filter baselines for spatially varying filtering.
- Add seed variance and failure-case analysis to show whether the optimization is robust or only works on carefully selected examples.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
