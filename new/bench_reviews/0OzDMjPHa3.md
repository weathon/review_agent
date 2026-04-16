## Summary

The paper presents "Pruning AMR," an algorithm for efficiently visualizing pre-trained implicit neural representations (INRs) by constructing an adaptive mesh that concentrates resolution in high-detail regions. The key idea is to use interpolative decomposition (ID) pruning on the INR's weight matrices restricted to subdomains corresponding to mesh elements; elements that can be significantly pruned (suggesting low local variation) are left coarse, while elements that resist pruning are refined. The algorithm iterates until each element's pruned INR meets both an error threshold and a neuron-proportion threshold, producing variable-resolution visualizations with fewer degrees of freedom than uniform discretization.

## Strengths

- **Well-motivated and novel problem formulation.** The paper identifies a real and growing need: pre-trained INRs from dynamic micro-CT can encode ~3.6 TB of volumetric data in a few MB of weights, but visualization requires discrete sampling. The question of how to efficiently visualize a pre-trained INR without training data is clearly articulated and timely.

- **Clean and well-described algorithm.** Algorithm 1 is precisely specified with all inputs, and Tables 1–2 provide hyperparameter descriptions and heuristic values. The mapping from pruning decisions to refinement decisions is logically sound and easy to follow.

- **Theoretically grounded pruning method.** Leveraging the interpolative decomposition (ID) pruning method from Chee et al. (2022) provides a principled foundation with theoretical guarantees, rather than relying on ad hoc heuristics.

- **Honest reporting of limitations.** The authors openly acknowledge that Pruning AMR shows only marginal improvement on the highly-detailed experimental CT data (Figure 3b, Section 4.3), and explicitly discuss conditions under which the method is less beneficial. This transparency strengthens credibility.

- **4D demonstration.** Figures 5 and 7 show that the mesh adapts across time slices, validating applicability to the time-varying setting central to the motivating application.

## Weaknesses

### Major:

- **Core hypothesis lacks formal justification or direct empirical validation.** The algorithm rests on the hypothesis (stated in Section 3) that "the less detailed a function is on a region of the domain, the smaller an INR needs to be to accurately describe the function in that region," i.e., that prunability on a subdomain correlates with local function detail. This is plausible but never formally analyzed or directly tested. A straightforward validation would be to compute ground-truth local variation (e.g., gradient magnitude) on the analytical benchmark and plot it against pruning proportion. Without this, the mechanism linking weight-matrix rank to functional complexity remains speculative, and it is unclear whether the method's partial success is due to the posited mechanism or merely coincidental.

- **Evaluation is narrow with example-specific hyperparameter tuning.** The paper tests on three examples (one 2D analytical, one simulated CT, one experimental CT), and each requires different hyperparameters ($\varepsilon$ ranges from $10^{-3}$ to $10^{-2}$, $P$ from 0.075 to 0.1). No principled strategy for selecting these parameters is given beyond empirical search. The paper provides no sensitivity analysis showing how robust the method is to these choices on a given example, making it difficult for practitioners to deploy the method on new INRs without extensive tuning. On the most realistic example (experimental CT), the improvements are characterized by the authors themselves as "minimal" and "marginal" (Section 4.3), and only 5 iterations are run due to compute constraints, leaving the key claim of scalability unverified.

- **Baselines are weak and no computational cost analysis is provided.** The only adaptive baseline ("Basic AMR") uses a simple random-sampling error estimator against the bilinear interpolant, which is a fairly naïve approach. More natural competitors—such as gradient-magnitude-based refinement (computing $\nabla$INR is cheap and directly measures local variation)—are absent. Additionally, the paper claims "efficient visualization" and "significant memory savings" but only measures output mesh DOFs, not the computational cost of the pruning process itself. Each iteration requires running ID pruning per element plus INR evaluations, which could be substantially more expensive than simply evaluating the INR densely. For a paper motivated by 4D CT-scale data where computational resources are the bottleneck, this omission is significant.

### Minor:

- **The claim that DOF savings "would only further improve with more iterations"** (Sections 4.2, 4.3) is speculative and unsupported by experiment or scaling argument, particularly since the experimental CT example already shows diminishing returns.

- **Restricted architecture scope.** The method assumes fully-connected INR layers and does not discuss how it would apply to hash-grid-based (Instant-NGP-style) or other modern architectures where weight matrices are not the primary information carrier. This is acknowledged implicitly but not discussed as a limitation.

### Trivial:

- The "ground truth" comparison uniform mesh in Figure 2 is a fine discretization, not an analytical ground truth. This is minor since the comparison is qualitative.

## Nice-to-Haves

- Direct empirical validation of the prunability ↔ detail hypothesis, e.g., a scatter plot of pruning proportion vs. local gradient magnitude on the 2D analytical benchmark where the true function is known.
- Comparison against a gradient-based or residual-based AMR error indicator to isolate the specific benefit of using pruning vs. a simpler adaptive approach.
- Wall-clock timing breakdowns (pruning time per element, total pruning time, mesh refinement time) to assess whether the DOF savings justify the computational overhead.
- Results on additional INR architectures (e.g., SIREN, hash-grid INRs) to demonstrate broader applicability.

## Removed Points

- **Code availability during review**: The reproducibility concern about code not being released during review is removed per the hard rules against nitpicks about reproducibility and unavailability of artifacts.

- **Claim that Basic AMR is disadvantaged by more error_samples**: The harsh critic stated that Basic AMR uses error_samples=256 vs. 32 for Pruning, implying unfair advantage for Pruning. However, using more samples gives Basic a *better* error estimate, which generally helps adaptive methods—not disadvantages them. This criticism is factually incorrect about the direction of the bias.

- **Claim that "no prior work" is overstated**: This is removed per the hard rules against requesting missing related works, since we cannot verify whether other related work exists.

- **Mention of treating uniform mesh as ground truth as problematic**: The paper's comparison to a fine uniform mesh is standard practice in AMR literature. The caption says "Treating Uniform as 'ground truth'" for qualitative comparison purposes, which is appropriate.

- **Concern about what happens when INR(X) is near zero (relative error explosion)**: While potentially valid, this is a standard numerical issue with relative error metrics that is not unique to this method and would be a minor implementation detail.

- **Detailed mesh implementation concerns (anisotropy, hanging nodes in 3D/4D)**: The paper uses MFEM for mesh management, which handles these issues. This is a nitpick about implementation details.

- **Formatting concerns about hyperparameter tables or algorithm presentation**: Removed per formatting nitpick rule.

## Novel Insights

The paper raises an interesting and underexplored question: given a pre-trained INR as a compressed data format, how can one extract adaptive-resolution information *directly from the weights* without access to training data? The idea of using network prunability on subdomains as a proxy for local functional complexity is creative, even if the current validation leaves the strength of this proxy unclear. The key insight—that a network restricted to a small subdomain may be more compressible, and that this compressibility varies with local detail—is intuitive and worth investigating further, but the paper would be substantially strengthened by directly validating this correlation rather than relying on indirect evidence through final mesh quality.

## Suggestions

- **Validate the prunability ↔ detail correlation directly.** On the 2D analytical benchmark, compute the known local variation (since $f(r) = \sin(1/(\alpha + r))$ is available analytically) and plot it against pruning proportion across elements. This single experiment would strongly support or undermine the core hypothesis.
- **Add a gradient-based AMR baseline.** Computing $\|\nabla \text{INR}(x)\|$ via autodiff or finite differences on each element is cheap and directly measures local variation; this is the most natural competitor for pruning-based refinement.
- **Report wall-clock times.** Even rough timing comparisons would clarify whether the pruning overhead is justified. If pruning per element is 10× slower than evaluating the INR on a uniform grid of that element, the practical case weakens considerably.
- **Run more iterations on the experimental CT example** or at least provide a scaling argument for why the gap between methods would widen, since the current results at 5 iterations are marginal.

## Score and Decision

**Calibration papers compared against:**

- **ASMR** (INR inference efficiency, kMp8zCsXNb): Scores 5–8, accepted poster. Had novel method, stronger experimental results across multiple settings, but some baseline concerns.
- **Smooth Real-time INR** (mMjSc5fspq): Scores 3–6, rejected. Had interesting idea but limited demonstrations and weaknesses in validation.
- **DSF/weight matrix decomposition** (DwiwOcK1B7): Scores 5–8, accepted poster. Had novel methodological contribution but hyperparameter sensitivity concerns.
- **Better Neural PDE Solvers** (hj9ZuNimRl): Score 6, accepted poster. Good idea, limited experiments, missing runtime analysis.
- **CoINR** (ZWi6RpT4mJ): Scores 1–5, withdrawn/rejected. Limited experimental validation for INR compression claims.

This paper has a clearly novel and well-motivated problem formulation with a reasonable algorithm, but the validation is narrow, the most realistic example shows only marginal improvement, the core hypothesis is unvalidated, and computational cost is unanalyzed. It is stronger than the rejected CoINR paper because the algorithm is better specified and the problem is more clearly defined, but weaker than accepted papers like ASMR or DSF which had broader experimental support. The marginal results on the realistic example and the weak baselines are significant concerns. The paper sits between the rejected and accepted papers in this calibration set.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>