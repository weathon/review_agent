=== CALIBRATION EXAMPLE 76 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the contribution. The abstract succinctly summarizes the motivation, method, and key claims (superior performance, convex optimization, computational efficiency for low-dimensional projections). The claims are generally supported by the paper, but the abstract lacks any mention of limitations, such as the difficulty of sampling from the variational distribution in higher dimensions.

### Introduction & Motivation
Well-written and clearly motivates the problem: the flexibility-tractability trade-off in NPE variational families. The introduction effectively highlights NPE's advantages (amortization, likelihood-free inference, automatic marginalization) and positions LBF-NPE as a solution that leverages these. Contributions are stated clearly: parameterization via latent basis expansions, fixed/adaptive basis variants, convexity properties, and stereographic projection for identifiability.

### Background
Accurate and concise. Correctly distinguishes ELBO-based VI (which struggles with marginalization) from NPE (which naturally handles it via simulation). The forward KL objective for NPE is properly defined. No major issues.

### Method (LBF-NPE)
**Section 3.1**: The variational family as an exponential family with neural basis functions is clearly defined. The use of a base measure \(h(z)\) is noted, but its practical choice (e.g., uniform over bounded latent space) is not discussed in the main text, which could confuse readers.

**Section 3.2 & Algorithm 1**: The gradient estimator using self-normalized importance sampling (SNIS) is derived correctly. However, key implementation details are omitted:
- The choice of proposal distribution \(r(z)\) is not specified. This is critical for efficient and low-variance gradient estimation. The appendices reveal that in some experiments numerical quadrature is used instead of SNIS, but this should be clarified in the main text.
- The bias of the SNIS estimator is acknowledged but not quantified; the impact of the number of importance samples \(P\) on optimization stability is not discussed.

Overall, the method is novel and well-founded, but the lack of details on the proposal distribution and the practical handling of the log-normalizer gradient could hinder reproducibility.

### Variants & Properties
**Section 4.2 (Convexity)**: Proposition 1 (marginal convexity) is a strong theoretical point. The proof in Appendix B is correct. The connection to NTK theory for infinite-width networks is mentioned, but the practical relevance for finite networks is not explored—this is acceptable as the empirical results support the benefits.

**Section 4.3 (Fixed Basis)**: The discussion of local vs. global bases is helpful. The recommendation of B-splines and wavelets is practical.

**Section 4.4 (Stereographic Projection)**: The reparameterization to address rescaling degeneracy is clever. However, the rotational degeneracy remains. The scaled loss (Equation 12) introduces a hyperparameter \(w\); its selection is not discussed. Appendix D mentions using stereographic projection but does not specify \(w\), leaving an implementation gap.

### Related Work
Appropriately covers exponential families in VI, NPE with mixtures/flows, and EigenVI. The distinction from EigenVI (amortization, adaptive basis) is clear. One minor omission: recent work on neural empirical Bayes or score-based density estimation could be relevant but is not essential.

### Experiments
The experiments are extensive and generally support the claims. However, several important details are missing or require clarification:

**Section 6.1 (Sinusoidal Likelihood)**: Demonstrates consistent convergence of LBF-NPE vs. MDN. However, only negative log-likelihood curves are shown; final KL divergence values would strengthen the quantitative comparison.

**Section 6.2 (2D Case Studies)**: LBF-NPE shows strong performance. Table 1 reports forward/reverse KL and NLL. Notably, for the "spiral" case, LBF-NPE has higher NLL than NSF (0.838 vs. 0.727) despite better forward/reverse KL. This discrepancy is not explained—perhaps due to different tail behavior or estimation error.

**Section 6.3 (Object Detection) & 6.4 (Redshift Estimation)**: Impressive real-world applications. The redshift experiment shows a significant NLL improvement.

**Appendix Experiments**: The comparisons with EigenVI (E.5) and score-matched neural exponential families (E.6) are valuable. The high-dimensional annulus (E.4) shows good marginal density estimation, but full joint sampling is not demonstrated.

**Key Experimental Concerns**:
1. **Proposal Distribution \(r(z)\)**: Not detailed. In Appendix D, for 2D cases, the integral is approximated by trapezoidal quadrature on a grid; for object detection, Monte Carlo with 22,500 uniform samples is used. This inconsistency should be clarified, and the rationale for choosing \(r(z)\) should be discussed.
2. **Alternating Optimization**: The schedule (e.g., 1000 steps for \(f_\phi\), then 1000 for \(s_\psi\)) is heuristic. Sensitivity to this schedule is not ablated.
3. **Stereographic Projection Scaling \(w\)**: Not specified; likely set to 1, but this should be stated.
4. **Computational Cost**: Table E.7 shows LBF-NPE sometimes has higher per-step cost but converges faster. Total training times are competitive, but memory usage is higher. This is acceptable.
5. **Sampling Demonstration**: For the 50-dimensional annulus (E.4), only marginal densities are shown, not samples from the full joint posterior. The paper’s sampling limitations are acknowledged, but the experiments do not demonstrate sampling in dimensions >2. The inverse transform sampling method described becomes impractical in moderate dimensions.

### Discussion & Limitations
The discussion highlights advantages (log-space modeling, unconstrained optimization, likelihood-free compatibility) and acknowledges the main limitation: sampling difficulty. However, the sampling issue is underemphasized: while inverse transform sampling works in low dimensions, it does not scale. The method is essentially limited to very low-dimensional latent spaces (e.g., the 2D experiments). The 50-dimensional example only evaluates marginal densities via Monte Carlo integration, not full sampling. This restricts the applicability to problems where only low-dimensional projections are needed, which is consistent with the paper’s motivation but should be stated more clearly.

Other limitations not fully discussed: sensitivity to the choice of \(K\), the base measure \(h(z)\), the proposal \(r(z)\), and the alternating optimization schedule. The rotational degeneracy after stereographic projection might affect interpretability but not performance.

### Writing & Clarity
The paper is well-written and logically structured. The figures and tables are informative. Some methodological details are buried in appendices, but overall the presentation is clear. A few typos/formatting artifacts exist (e.g., missing spaces, inconsistent equation numbering), but these are minor.

### Overall Assessment
This paper presents a novel and compelling approach to variational families for NPE. The core idea—using neural basis expansions for the log-density—is well-motivated, and the theoretical convexity properties are attractive. Empirically, the method outperforms strong baselines on a diverse set of benchmarks, including real scientific applications. The main weakness is the limited sampling capability, which constrains the method to low-dimensional targets, but this is consistent with the paper’s focus on posterior projections. Several implementation details (proposal distribution, alternating optimization schedule, hyperparameter choices) need clarification to ensure reproducibility. With revisions addressing these points, the paper makes a solid contribution suitable for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Latent Basis Function Neural Posterior Estimation (LBF-NPE), a novel variational family for amortized inference that models the log posterior density as a linear combination of basis functions over the latent space. The method leverages the automatic marginalization properties of Neural Posterior Estimation (NPE) and is designed for low-dimensional posterior projections of interest, common in scientific applications. LBF-NPE offers a flexible yet tractable alternative to existing families like Gaussian mixtures or normalizing flows, with favorable optimization properties.

### Strengths
1. **Novel and well-motivated variational parameterization**: The use of basis expansions (fixed or adaptive) to model the log-density within an exponential family framework is innovative. It effectively bridges the gap between simple, interpretable families and flexible but hard-to-optimize black-box models. The connection to convex optimization when basis functions are fixed is a strong theoretical point.

2. **Strong empirical performance**: The paper demonstrates consistent improvements over Mixture Density Networks (MDNs) and normalizing flows across diverse tasks, including synthetic multimodal problems, astronomical object detection, and cosmological redshift estimation. Quantitative results (e.g., lower KL divergences and negative log-likelihoods) are convincing and supported by thorough experiments.

3. **Exploitation of NPE's strengths**: The method explicitly leverages NPE's ability to automatically marginalize over nuisance parameters and its suitability for low-dimensional posterior projections. The design is tailored to practical inference scenarios where only a few parameters are of scientific interest.

### Weaknesses
1. **Sampling limitations**: The primary weakness is the difficulty in drawing samples from the fitted variational distribution, especially in higher dimensions. While inverse transform sampling or Langevin dynamics are suggested, their scalability and efficiency are not thoroughly evaluated. This could limit the method's utility in applications requiring posterior samples (e.g., for downstream uncertainty propagation).

2. **Heuristic choices for basis functions and dimensionality**: The performance depends on the number of basis functions \(K\) and their type (fixed vs. adaptive). The paper provides some ablation but lacks clear guidelines for practitioners on how to select these hyperparameters optimally. The diminishing returns with increasing \(K\) are noted, but a principled selection method is missing.

3. **Incomplete comparison with related work**: The comparison to EigenVI, another basis-expansion VI method, is only qualitative and brief (Appendix E.5). Quantitative metrics (e.g., KL divergence) on the same benchmarks would strengthen the claim of superiority. Additionally, the computational cost analysis (Appendix E.7) is useful but could be more detailed regarding memory and time scaling with dimension.

### Novelty & Significance
The work is novel in combining basis expansions with the NPE framework, offering a new point in the trade-off between flexibility and optimization stability. Theoretically, it provides convexity guarantees under certain conditions, contributing to the understanding of NPE optimization. Practically, it demonstrates significant performance gains on real-world scientific inference problems, which is highly relevant for the ICLR community interested in simulation-based inference and probabilistic deep learning.

### Suggestions for Improvement
1. **Address sampling challenges more comprehensively**: Include a systematic evaluation of sampling methods (e.g., inverse transform, Langevin dynamics, sequential Monte Carlo) for higher-dimensional targets (beyond 2D). Discuss computational trade-offs and provide recommendations for practitioners.

2. **Provide guidance on hyperparameter selection**: Conduct a more detailed ablation study on the choice of \(K\) and basis type (e.g., B-splines vs. wavelets) across problem classes. Offer heuristic rules or adaptive schemes (e.g., increasing \(K\) until performance plateaus) to aid users.

3. **Strengthen comparisons with EigenVI**: Add quantitative results comparing LBF-NPE and EigenVI on the same 2D benchmarks (e.g., KL divergence, negative log-likelihood) to substantiate the claim of better performance with fewer basis functions. Discuss differences in assumptions (orthogonality constraints) and optimization.

4. **Clarify limitations and scalability**: While the method excels in low-dimensional projections, explicitly discuss its applicability to higher-dimensional latent spaces. Explore structured basis expansions (e.g., tensor products) or conditional independence assumptions to extend scalability, and provide preliminary results if possible.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Controlled ablation on the effect of basis dimension *K* and basis type (fixed vs. adaptive).** The paper claims expressivity improves with *K* and that adaptive bases are beneficial, but provides only scattered examples (e.g., object detection with K=9,20,36,64). A systematic sweep on key benchmarks (like the 2D cases) measuring error vs. K is needed to substantiate claims about efficiency and to guide practitioners.
2. **Direct comparison to the state-of-the-art NPE methods on standard simulation-based inference (SBI) benchmarks.** The paper compares to MDNs and flows, but not to recent advanced NPE methods like SNPE-C/SNL or more expressive flow architectures (e.g., MAFs, NSF with more layers/coupling transforms). The claimed superiority is not convincing without these baselines.
3. **Quantitative evaluation of optimization stability/consistency beyond the simple sinusoidal 1D example.** The core claim of better optimization landscape is supported only by a single toy example (Figure 1). Need to show metrics like variance of final loss or KL across many seeds for the more complex 2D/astronomy problems to prove this is a general advantage.
4. **Analysis of computational cost vs. accuracy trade-off compared to flows.** Table 6 reports runtime but not a direct Pareto curve (e.g., KL vs. wall-clock time or number of parameters). The method's practicality hinges on being more efficient, but this is not rigorously shown.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the failure modes and limitations of the sampling approaches (inverse transform, Langevin).** The paper briefly mentions sampling is a limitation but then shows sampling results in Appendix C.1 that look good. A critical analysis is missing: when do these sampling methods fail (e.g., in higher dimensions, for very complex geometries)? Quantify error of sampled statistics vs. true posterior.
2. **Investigation into what the learned adaptive basis functions actually represent.** The paper shows visualizations but no analysis linking basis functions to posterior structure. For example, do individual basis functions correspond to modes or features? This is key to the claim of interpretability and adaptivity.
3. **Sensitivity analysis of the hyperparameters: scaling factor *w*, proposal distribution *r(z)* for importance sampling, and alternating optimization schedule.** The method's performance likely depends on these choices; without ablation, it's unclear how robust the method is.
4. **Theoretical/empirical analysis of the identifiability issue and the effectiveness of stereographic projection.** The paper mentions rotational degeneracy persists even after projection. How does this affect optimization? Does it lead to multiple equivalent solutions, and does that matter?

### Visualizations & Case Studies
1. **Visualization of optimization trajectories (loss landscape slices) for LBF-NPE vs. MDN/flows on a non-trivial 2D problem.** This would directly support the claim of a more favorable optimization landscape. The current evidence (Figure 1) is too weak.
2. **Case studies showing clear failures of the method, especially when the low-dimensional projection assumption is violated or when *K* is too small.** The paper only shows successes. Showing and diagnosing failures would provide a more honest assessment of limitations.
3. **Visualization of the approximation quality as a function of *x* (the observation) for the amortized network.** The paper shows posteriors for a few selected *x*. A plot of KL divergence vs. *x* (or some summary statistic of *x*) would reveal if the amortization fails for certain regions of observation space.

### Obvious Next Steps
1. **Include a standard SBI benchmark suite (e.g., from `sbi` library) to position the work within the field.** The current benchmarks are mostly custom. Using community standards is essential for fair comparison and credibility.
2. **Provide a quantitative analysis of the marginal convexity claim in practice.** Proposition 1 states marginal convexity, but does that translate to faster/more reliable convergence in realistic neural network training? Plot loss over iterations for fixed *s* vs. joint training to demonstrate this benefit.
3. **Discuss and experiment with more sophisticated methods for sampling from the fitted density in moderate dimensions (e.g., Hamiltonian Monte Carlo).** The sampling discussion is cursory; exploring and comparing sampling methods is a logical next step that should have been initiated.
4. **Compare to variational inference methods that also target low-dimensional projections (e.g., via variational marginalized inference).** The paper emphasizes NPE's automatic marginalization, but does not contrast with other VI methods that can marginalize nuisance parameters.

# Final Consolidated Review
## Summary
This paper introduces Latent Basis Function Neural Posterior Estimation (LBF-NPE), a variational family for amortized inference that models the log posterior density as a linear combination of basis functions over the latent space. The method leverages neural posterior estimation's automatic marginalization and is designed for low-dimensional posterior projections, offering convex optimization guarantees with fixed bases and adaptive flexibility. Experiments show superior performance over mixture density networks and normalizing flows on synthetic and real-world tasks.

## Strengths
- **Novel variational parameterization via latent basis expansions:** The method models log-densities as linear combinations of basis functions, forming a flexible exponential family. This bridges the gap between simple interpretable families and black-box flows, with convex optimization properties when bases are fixed (Proposition 1).
- **Strong empirical performance across diverse benchmarks:** LBF-NPE consistently outperforms mixture density networks and normalizing flows in forward/reverse KL divergence and negative log-likelihood on synthetic 2D problems, astronomical object detection, and cosmological redshift estimation (Tables 1, 2).

## Weaknesses
- **Sampling scalability is limited:** Drawing samples from the fitted variational distribution is non-trivial beyond very low dimensions. While inverse transform or Langevin dynamics are suggested, their efficiency and accuracy in moderate to high dimensions are not demonstrated, restricting applications where posterior samples are required.
- **Performance hinges on heuristic hyperparameter choices:** Key design decisions—number of basis functions \(K\), basis type (fixed vs. adaptive), alternating optimization schedule—lack principled guidance or systematic ablation. This affects reproducibility and optimal deployment.
- **Quantitative comparison to EigenVI is insufficient:** The related basis-expansion method EigenVI is compared only qualitatively (Appendix E.5), without reported KL divergences or likelihoods to substantiate claims of superiority with fewer bases.

## Nice-to-Haves
- Systematic ablation studying the effect of basis dimension \(K\) and type (e.g., B-splines vs. wavelets) on approximation error.
- Analysis of optimization stability (e.g., loss variance across seeds) for complex problems beyond the sinusoidal example.
- Exploration of more scalable sampling techniques (e.g., Hamiltonian Monte Carlo) for moderate-dimensional targets.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add quantitative metrics (e.g., KL divergence) to the EigenVI comparison on the 2D benchmarks to rigorously demonstrate advantages.
- Provide practical guidelines for selecting \(K\), such as using a validation set to monitor performance plateau.
- Evaluate sampling methods on a higher-dimensional target (e.g., the 50D annulus) to quantify sampling error and computational cost.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
