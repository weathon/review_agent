=== CALIBRATION EXAMPLE 74 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's contribution. The abstract clearly states the method (LBF-NPE), its advantages (flexibility, convex optimization, automatic marginalization), and summarizes key results. However, the claim of "order-of-magnitude improvements in KL divergence" over MDNs and flows on 2D posteriors is strong; Table 1 shows LBF-NPE outperforms MDN by roughly a factor of 10-40 on bands/ring (0.0048 vs. 0.182, 0.0054 vs. 0.205), which is an order of magnitude, but the improvement over flows is more modest (e.g., 0.0048 vs. 0.016 for bands). The abstract should be precise: the order-of-magnitude claim holds versus MDN but not consistently versus flows.

### Introduction & Motivation
The introduction effectively frames the trade-off between flexibility and tractability in NPE and motivates basis expansions by leveraging NPE's ability to target low-dimensional posterior projections. The contributions are clearly stated. However, the paper heavily emphasizes NPE's automatic marginalization over nuisance parameters as a key advantage, but the experiments do not prominently feature problems with high-dimensional nuisance variables. A dedicated experiment demonstrating marginalization over a large number of nuisance parameters would strengthen the motivation.

### Method / Approach
**Section 3.1-3.2**: The variational family and gradient estimator are well-defined. Algorithm 1 is clear but omits a critical detail: the choice of proposal distribution \(r(z)\) for importance sampling. This choice heavily influences gradient variance and stability; the paper should specify how \(r(z)\) is selected (e.g., is it the base measure \(h(z)\), a uniform distribution, or an adaptive proposal?). Without this, reproducibility is hampered.

**Section 4 (Variants & Properties)**:
- **Convexity (Prop. 1)**: The proposition states marginal convexity when one of \(f\) or \(s\) is fixed. However, in the adaptive case (both learned), the alternating optimization is not guaranteed to converge to a global optimum, even though each subproblem is convex. The paper should discuss convergence properties of the alternating scheme and any empirical observations (e.g., does it consistently reach the same solution?).
- **Stereographic Projection (Sec. 4.4)**: The reparameterization mitigates scaling degeneracy but breaks the linear inner product structure. It is unclear whether the convexity properties from Prop. 1 still hold under this nonlinear transformation. The paper should address this or provide empirical evidence that optimization remains stable.
- **Fixed Basis (Sec. 4.3)**: The discussion of local vs. global bases is useful, but no experimental comparison between fixed B-splines/wavelets and adaptive bases is provided. Such an ablation would help practitioners choose between variants.

**Sampling (Appendix C)**: The paper correctly identifies sampling as a limitation. The proposed inverse transform sampling is only feasible for very low dimensions (1D, 2D). For higher dimensions, MCMC or sequential sampling are suggested but not evaluated. In experiments, posterior plots are generated via grid evaluation, which does not scale. The paper should include a sampling evaluation (e.g., efficiency of Langevin dynamics) for at least one non-trivial case to demonstrate that sampling is practical.

### Experiments & Results
**Overall Design**: Experiments cover synthetic and real-world problems, but all latent spaces are low-dimensional (1D or 2D). This limits the claim that LBF-NPE is efficient for "high-dimensional latent spaces" when only low-dimensional projections are of interest. While the 50D annulus (Appendix E.4) is a step in this direction, the posterior is over all 50 dimensions, and the evaluation is still on 2D marginals. A more compelling experiment would involve a generative model with many nuisance variables and a low-dimensional parameter of interest, showcasing automatic marginalization.

**Baseline Comparisons**:
- Table 1 shows LBF-NPE outperforms baselines on 2D problems, but details on baseline architectures and capacities are sparse. For fair comparison, the paper should report parameter counts or ensure baselines are tuned to comparable expressivity. For instance, the MDN uses 10 Gaussian components (50 parameters), while LBF-NPE uses 20 basis functions parameterized by neural networks (likely many more parameters). This makes it unclear whether gains are due to the parameterization or simply increased capacity.
- The standard deviations in Table 1 are small, but the number of independent runs is not stated. Reporting the number of seeds would help assess statistical significance.

**Ablations**:
- Appendix E.3 studies the effect of the number of basis functions \(K\), showing diminishing returns. This is useful, but other important ablations are missing: comparison between fixed and adaptive bases, sensitivity to the proposal distribution \(r(z)\), and choice of basis type (B-spline vs. wavelet).
- The effect of stereographic normalization is shown qualitatively in Appendix E.1, but quantitative metrics (e.g., convergence speed, final loss) would strengthen the argument.

**Comparison to EigenVI**: The paper claims superiority over EigenVI, but the comparison in Appendix E.5 is only visual. Quantitative KL divergences for EigenVI on the 2D test cases should be provided. Additionally, EigenVI uses a fixed orthogonal basis, while LBF-NPE uses adaptive bases; the comparison should discuss whether the performance gap is due to adaptivity or other factors.

**Computational Cost**: Table 6 in Appendix E.7 reports runtime and memory. LBF-NPE often converges in fewer steps but has higher per-step cost. However, the total training time is competitive. This is a reasonable trade-off, but the table lacks details on hardware and whether the reported times are averaged over runs.

### Writing & Clarity
The paper is generally well-written and logically structured. Some sections could be improved:
- Section 3.2: Clarify the choice of proposal distribution \(r(z)\).
- Section 4.4: Elaborate on how stereographic projection affects optimization (convexity, convergence).
- Appendix D: Provide more architecture details for baselines (e.g., number of coupling layers for flows, hidden units for MDN) to ensure reproducibility.

### Limitations & Broader Impact
Section 7 appropriately notes the sampling limitation and suggests future work. However, the paper does not discuss broader societal impacts, which is a minor omission given the methodological nature of the work. A brief statement acknowledging potential indirect impacts (e.g., biases in inferred posteriors for scientific applications) would be sufficient.

### Overall Assessment
The paper introduces a novel and promising variational family for NPE that combines expressivity with favorable optimization properties. The theoretical analysis of convexity is sound, and empirical results on low-dimensional problems demonstrate improved performance over strong baselines. However, several issues weaken the contribution: (1) the claimed advantage for high-dimensional latent spaces with low-dimensional projections is not convincingly demonstrated; (2) the sampling limitation is acknowledged but not thoroughly addressed; (3) experimental comparisons lack thorough ablations and detailed baseline specifications, making it difficult to attribute gains solely to the parameterization. Addressing these concerns—particularly by adding a high-dimensional marginalization experiment and more rigorous baseline comparisons—would significantly strengthen the paper. As is, the paper presents a solid idea with promising results but falls short of ICLR's high bar for novelty and thorough evaluation.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes LBF-NPE, a new variational family for Neural Posterior Estimation (NPE) that models the log-density as a linear combination of latent basis functions. The basis functions can be fixed (e.g., B-splines) or learned adaptively, leading to an exponential-family variational distribution. The method is designed for low-dimensional posterior projections, exploiting NPE's automatic marginalization over nuisance parameters. Empirically, LBF-NPE outperforms mixture density networks, normalizing flows, and a prior basis-expansion method (EigenVI) on synthetic and real-world tasks, including astronomical redshift estimation.

### Strengths
1. **Novel and well-motivated approach**: The paper identifies a clear gap in the NPE literature—the trade-off between flexibility and optimization tractability—and proposes a principled solution via basis expansions in log-density space. The connection to exponential families and the exploitation of NPE's marginalization capabilities are well-articulated (Sections 1, 3).
2. **Theoretical grounding**: The paper provides a convexity analysis (Proposition 1) and discusses global convergence under certain conditions (Section 4.2, Appendix B), linking to neural tangent kernel theory. This strengthens the methodological contribution beyond purely empirical results.
3. **Comprehensive and rigorous experimentation**: The authors evaluate LBF-NPE on diverse problems, including synthetic 1D/2D posteriors, astronomical object detection, and a large-scale redshift estimation task using the LSST DC2 dataset. Quantitative metrics (KL divergences, NLL) are reported, and comparisons are made against strong baselines (MDNs, normalizing flows, EigenVI). The appendix includes extensive ablations (e.g., basis dimension effects, normalization studies) and sampling demonstrations (Sections 6, Appendices D, E).
4. **Practical innovations**: The introduction of stereographic projection to handle identifiability issues (Section 4.4) and the discussion of local vs. global basis functions (Section 4.3) are practical contributions that improve training stability and interpretability.

### Weaknesses
1. **Dimensionality limitation**: The method is primarily suited for low-dimensional latent spaces (as acknowledged in Sections 7 and C). Sampling relies on inverse transform sampling or MCMC, which becomes impractical in high dimensions. While the focus on low-dimensional projections is justified for many scientific applications, this limits broader applicability. The high-dimensional experiment in Appendix E.4 is a 50-D annulus with a simple structure, not a challenging high-D posterior.
2. **Computational overhead per step**: Table 6 shows that LBF-NPE has higher per-step runtime and memory usage than some baselines (e.g., MDN) due to the integral approximation via importance sampling. Although convergence is faster in steps, the total cost can still be significant, especially for adaptive bases requiring alternating optimization.
3. **Limited exploration of adaptive basis functions**: While adaptive bases are presented, the paper does not deeply analyze what kinds of basis functions are learned or how they relate to the posterior structure. The visualizations (Appendix E.3) are informative but qualitative; a more systematic study of the learned representations would strengthen the claim of "adaptive" flexibility.
4. **Comparison to recent NPE advances**: The paper compares to standard variational families (MDNs, flows) and EigenVI, but does not discuss more recent advances in NPE, such as sequential methods (e.g., SNPE-C) or attention-based architectures. A comparison to state-of-the-art NPE techniques would better position the contribution.

### Novelty & Significance
**Novelty**: The core idea of using basis expansions for variational families in NPE is novel. While basis expansions have been used in VI (e.g., EigenVI), their application to amortized, likelihood-free NPE—with adaptive bases and stereographic projection—is a new contribution. The theoretical convexity results specific to this parameterization are also novel.

**Significance**: The method offers a compelling middle ground between simple and flexible variational families, with improved optimization properties. The empirical gains on scientific problems (e.g., redshift estimation) demonstrate practical impact. The work could influence how variational families are designed for amortized inference, particularly in low-dimensional settings common in scientific applications.

### Suggestions for Improvement
1. **Address high-dimensional limitations**: Explore structured basis expansions (e.g., tensor products, conditional independence assumptions) or sparse approximations to scale to moderately high dimensions. Alternatively, provide a more thorough discussion of when low-dimensional projections are sufficient and how to identify such scenarios in practice.
2. **Deepen analysis of adaptive bases**: Include a quantitative study of how the learned basis functions evolve during training and how they correlate with posterior features (e.g., multimodality, skewness). This could involve measuring basis orthogonality or visualizing their activation patterns for different observation types.
3. **Expand comparisons**: Include comparisons to recent NPE methods (e.g., SNPE-C, transformer-based approaches) and other flexible VI techniques (e.g., score-based VI) to better establish the state-of-the-art. Additionally, compare runtime and memory trade-offs more comprehensively, perhaps reporting wall-clock time to convergence rather than just steps.
4. **Clarify practical guidelines**: Provide clearer recommendations for practitioners on choosing between fixed vs. adaptive bases, selecting the number of basis functions \(K\), and tuning hyperparameters (e.g., proposal distribution for importance sampling). The discussion in Sections 4.3 and 7 is helpful but could be more prescriptive.
5. **Improve presentation of theoretical results**: While Proposition 1 is proved, the global convergence discussion in Appendix B relies heavily on prior work (McNamara et al., 2024a). A more self-contained summary of the assumptions and implications would make the theory more accessible. Additionally, discuss how stereographic projection affects convexity (since it breaks the linear parameterization).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Systematic evaluation on higher-dimensional posteriors (e.g., d=5-10).** The paper focuses on 1D/2D latents and only has one 50D example in the appendix. A core claim is suitability for "low-dimensional projections," but the boundary of this regime is unexplored. Without testing moderate dimensions, it's unclear where the method's computational advantages break down due to the curse of dimensionality in the integral computation.
2.  **Comparison against a broader suite of modern normalizing flows.** Baselines are limited to RealNVP and NSF. To substantiate claims of superior performance, comparisons against more recent, high-performance flows (e.g., FFJORD, CNF, autoregressive flows) are necessary, especially on the presented 2D benchmarks.
3.  **Ablation on the alternating optimization scheme.** The method alternates between updating `f_φ` and `s_ψ`. The impact of the update schedule (e.g., 1000 steps each) and the necessity of alternation versus joint training are not studied. A poorly chosen schedule could lead to suboptimal results, undermining the convergence claims.
4.  **Quantitative evaluation of sampling efficiency and accuracy.** The paper acknowledges sampling is a limitation and suggests inverse transform sampling or MCMC. However, there is no quantitative comparison of sample quality (e.g., ESS, MMD) or speed between these schemes and sampling from flow/MDN baselines. This is critical for practitioners.

### Deeper Analysis Needed (top 3-5 only)
1.  **Empirical analysis of the convexity and optimization landscape.** Proposition 1 states marginal convexity, but there is no empirical verification (e.g., visualizing loss landscapes for LBF-NPE vs. MDN/flow). The claim that convexity leads to better convergence needs direct support beyond the single sinusoidal likelihood trace plot.
2.  **Analysis of what the adaptive basis functions `s_ψ(z)` learn.** The paper shows basis visualizations but does not analyze their interpretability or how they adapt to different problem classes. An analysis linking basis function structure to posterior geometry (e.g., do they specialize to modes?) would strengthen the "adaptive" claim.
3.  **Sensitivity analysis of key hyperparameters: number of basis functions `K` and the scaling weight `w`.** While some `K` sweeps are shown, a systematic study of how performance and training stability scale with `K` across problem types is missing. The role of the fixed scale `w` in Eq. 12 is also not analyzed.
4.  **Investigation of the identifiability and degeneracy issue.** The paper mentions rotational degeneracy persists even after stereographic projection. The practical impact of this on optimization and the final solution is not discussed. Analyzing the variance of solutions from different random seeds could reveal instability.

### Visualizations & Case Studies
1.  **Visualizations of failure cases or limitations.** All shown posteriors are high-quality. To build trust, the authors should visualize cases where LBF-NPE fails or underperforms relative to baselines (e.g., for very high `K`, very rough posteriors, or when the proposal distribution `r(z)` is poorly chosen).
2.  **Case study on a posterior with complex correlation structure beyond the shown 2D examples.** The "spiral" test is good, but a case with strong, nonlinear correlations in higher dimensions (e.g., a Bayesian neural network posterior) would better test the method's ability to capture dependencies.
3.  **Visual trace of the basis functions `s_ψ(z)` during training.** Animations or a series of snapshots showing how the basis functions evolve during alternating optimization would provide unique insight into how the model builds up the posterior representation.

### Obvious Next Steps
1.  **Benchmark against Hamiltonian Monte Carlo (HMC) ground truth on a challenging, moderate-dimensional problem.** For a problem where MCMC is feasible (e.g., d<20), comparing LBF-NPE's posterior approximation fidelity (in KL divergence) to an HMC baseline would provide a strong, missing reference point for accuracy.
2.  **Integrate and evaluate the proposed sampling methods.** The appendix mentions Langevin dynamics and inverse transform sampling. These should be integrated into the main experimental workflow, with a clear comparison of their accuracy and computational cost for generating samples used in downstream tasks (e.g., expectation estimation).
3.  **Explore the use of more sophisticated proposal distributions `r(z)` for the importance sampling gradient estimator.** The paper uses a simple `r(z)`. The gradient estimator's variance and training stability likely depend heavily on this choice. A natural next step is to adapt `r(z)` during training, which should have been explored.

# Final Consolidated Review
## Summary
This paper introduces Latent Basis Function Neural Posterior Estimation (LBF-NPE), a variational family for amortized, likelihood-free inference that models log-densities as linear combinations of basis functions over the latent space. The method offers theoretical convexity properties when either the basis functions or the coefficient network is fixed, and it leverages NPE’s ability to target low‑dimensional posterior projections. Experiments demonstrate improved performance over mixture density networks and normalizing flows on synthetic 2D tasks and a large‑scale astronomical redshift estimation problem.

## Strengths
- **Novel and well‑motivated variational parameterization.** LBF‑NPE fills a gap between simple (e.g., Gaussian) and highly flexible (e.g., normalizing flow) families by using basis expansions in log‑density space, yielding an exponential‑family distribution that is both expressive and optimizable.
- **Theoretical grounding.** Proposition 1 establishes marginal convexity of the NPE objective when either the basis functions or the coefficient network is fixed, linking to prior work on global convergence in wide neural networks and providing insight into the optimization landscape.
- **Comprehensive empirical evaluation.** The method is tested on a range of problems—from synthetic multimodal posteriors to real‑world astronomical object detection and redshift estimation—and consistently outperforms strong baselines (MDNs, RealNVP, neural spline flows) in forward/reverse KL divergence and negative log‑likelihood.

## Weaknesses
- **Insufficient demonstration of automatic marginalization.** A key motivation is NPE’s ability to marginalize over high‑dimensional nuisance variables when only a low‑dimensional projection is of interest. However, the experiments do not prominently feature problems with many nuisance parameters; the 50‑D annulus (Appendix E.4) and the BNN example (Appendix E.8) are steps in this direction but do not fully showcase this claimed advantage.
- **Sampling limitations are acknowledged but not thoroughly addressed.** While inverse transform sampling and Langevin dynamics are proposed, their efficiency and accuracy are only qualitatively illustrated for 2D cases. A quantitative evaluation of sampling quality (e.g., effective sample size, MMD) and computational cost is missing, which is important for practitioners who need posterior samples.
- **Experimental comparisons could be more rigorous.** Quantitative results for EigenVI are omitted (only visual comparison in Appendix E.5), making it hard to assess the claimed superiority. Ablations between fixed and adaptive basis functions are not provided, and baseline architectures (e.g., flow depth, MDN component count) are not matched in parameter count or expressivity, leaving open whether gains stem from the parameterization or increased capacity.
- **Overstatement in the abstract.** The abstract claims “order‑of‑magnitude improvements in KL divergence over both MDNs and normalizing flows” on 2D posteriors, but Table 1 shows that while improvements over MDNs are indeed order‑of‑magnitude, improvements over flows are more modest (e.g., 0.0048 vs. 0.016 for bands—roughly 3×). The wording should be precise.

## Nice-to-Haves
- Deeper analysis of what the adaptive basis functions learn and how they relate to posterior geometry (e.g., specialization to modes, correlation structure).
- Sensitivity study of key hyperparameters: the number of basis functions \(K\), the scaling weight \(w\) in the stereographic projection variant, and the proposal distribution \(r(z)\) used in importance sampling.
- Comparison to a broader set of normalizing flow architectures (e.g., autoregressive flows) and recent NPE advances (e.g., sequential NPE methods) to better situate the contribution.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **“Alternating optimization is not guaranteed to converge globally.”** The paper only claims marginal convexity (Proposition 1) and does not assert global convergence for the alternating scheme; this criticism misreads the theoretical claims.
- **“Stereographic projection breaks convexity.”** The paper introduces stereographic projection to mitigate identifiability, not to preserve convexity; the impact on optimization is empirical and the method remains stable in practice.
- **“The paper does not discuss broader societal impacts.”** Given the methodological focus, a societal impact statement is not required; its absence is not a substantive weakness.
- **“Request for benchmark against Hamiltonian Monte Carlo.”** HMC requires likelihood evaluations and is not directly comparable to likelihood‑free, amortized NPE methods; this is outside the paper’s scope.
- **“Formatting nitpicks (e.g., missing hardware details in Table 6).”** These are minor presentation issues that do not affect the scientific contribution.

## Novel Insights
The paper’s core insight is that modeling the log‑density via a basis expansion yields an exponential‑family variational distribution that can be optimized through convex subproblems while retaining high expressivity. This bridges the gap between simple variational families (with stable optimization) and flexible black‑box families (which are harder to train). Additionally, the connection to angular‑distance optimization through stereographic projection reparameterization offers a fresh perspective on stabilizing amortized inference. The empirical finding that even a small number of adaptive basis functions can capture complex multimodal posteriors (e.g., 20 basis functions suffice for intricate 2D shapes) suggests that low‑rank representations in log‑density space are surprisingly powerful for many inference tasks.

## Suggestions
- Include a dedicated experiment that clearly demonstrates marginalization over a high‑dimensional nuisance space (e.g., a hierarchical model with many nuisance variables) to solidify the motivation.
- Provide quantitative KL divergence numbers for EigenVI on the 2D test cases and an ablation comparing fixed B‑splines/wavelets against adaptive bases on the same task.
- Add a quantitative evaluation of sampling efficiency (e.g., ESS, wall‑clock time to generate a fixed number of samples) for at least one non‑trivial posterior to guide users on practical deployment.
- Clarify the choice of proposal distribution \(r(z)\) in Algorithm 1 and the experimental settings (e.g., uniform over the latent domain) to enhance reproducibility.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
