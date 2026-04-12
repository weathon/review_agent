=== CALIBRATION EXAMPLE 24 ===

# Final Consolidated Review
## Summary
This paper argues that local generalization geometry is fundamentally a two-operator problem: beyond the spectra of train and test Hessians, one must also account for the alignment of their eigenspaces. It develops a general overlap-based fluctuation formula, a free-probability transfer law, an exact asymptotic treatment for anisotropic ridge regression under covariate shift, and scalable estimators for overlap functionals, then validates parts of the framework on small MLPs and uses it diagnostically on a CIFAR-10 ResNet-20.

## Strengths
- **Clear identification of a genuinely underemphasized object: train–test eigenspace overlap.** The paper makes a concrete and useful conceptual move from one-loss curvature analysis to two-loss geometry. The central trace quantity  
  \[
  \mathbb E[\Delta L] = \tfrac12 \mathrm{tr}\,\bar{}[H_{\text{test}} C_{\text{train}}]
  \]
  and its decomposition into spectral scales plus overlap kernel gives a clean language for separating “how much variance is induced” from “where that variance lands” in test-sensitive directions.
- **A strong analytical treatment in ridge regression that makes the overlap perspective operational rather than rhetorical.** In Section 3.2 and Appendix C, the paper does not stop at a formal decomposition; it derives asymptotically exact overlap formulas under arbitrary covariate shift and uses them to interpret multiple descent and isospectral shifts. The controlled rotation experiment in Fig. 1 is particularly effective because it holds spectra fixed and varies only alignment, isolating an effect that spectrum-only views cannot distinguish.
- **Insightful geometric reinterpretation of multiple descent.** The paper’s analysis of two-scale and multi-scale covariances gives a compelling picture: peaks occur not merely when small train eigenvalues appear, but when the induced high-variance directions overlap particular test subspaces. Even if related risk formulas are already known, the overlap map in Figs. 2–3 provides a more directional explanation of which components drive the peaks.
- **Novel numerical machinery for cross-operator overlap estimation.** The Overlap-KPM construction in Appendix F extends standard spectral-density tools to pairwise overlap functionals using Hutchinson estimation plus Chebyshev approximations. This is a specific technical contribution, not a generic “we ran Lanczos” implementation detail.
- **Convincing local validation in the intended small-perturbation regime.** The MLP experiments in Section 3.3 are appropriately set up to test the local theory rather than overclaiming broad realism: train to near convergence, inject controlled input/label noise, and compare measured \(\Delta L\) to the quadratic prediction. The agreement in Fig. 4 supports the internal correctness of the local framework.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper sometimes overstates the novelty and explanatory status of Theorem 1.** The “overlap local fluctuation law” is mathematically clean, but it is not an independently predictive law in the sense of introducing a new estimable quantity from first principles; it is a decomposition of \(\tfrac12\mathrm{tr}\,\bar{}[H_{\text{test}}C_{\text{train}}]\) into spectral measures and their joint overlap structure. Appendix B.2 makes this explicit: the overlap function is defined as the Radon–Nikodym derivative of the joint spectral measure, and Equation (6) follows by rewriting the trace in that basis. That does not make the result uninteresting—the decomposition is useful—but the paper should present it more as a principled reformulation that exposes the missing directional term, and less as if a new predictive mechanism had been discovered independently of the underlying quadratic model.
- **Empirical evidence for modern deep networks is still mostly diagnostic/observational rather than predictive.** The ResNet-20/CIFAR-10 study in Section 3.4 shows that class imbalance changes train–test overlap structure, but it does not demonstrate that overlap metrics predict test degradation better than standard spectrum-only summaries, nor does it quantify how much of the observed change is attributable specifically to misalignment versus concurrent spectral changes. The class-imbalance result is therefore suggestive rather than decisive support for the stronger claim that overlaps “govern” generalization in modern neural networks.
- **The deep-learning validation is narrow relative to the breadth of the paper’s claims.** The only quantitative test of the fluctuation law is on a very small teacher-student MLP, and the larger-scale experiment is a single pretrained ResNet-20 checkpoint. This is enough to show feasibility and some plausibility, but not enough to substantiate broad statements about “modern neural networks” or practical generalization analysis at contemporary scale.
- **The practical reliability of the overlap estimators is not benchmarked rigorously enough.** Appendix F gives synthetic demonstrations and complexity discussion, but there is no direct small-scale accuracy benchmark against exact eigendecomposition for realistic Hessians, nor a careful sensitivity study for kernel width / Chebyshev order / probe count on neural network Hessians. Since the ResNet conclusions depend on these estimates, stronger empirical calibration of the estimator would materially improve confidence.

### Minor
- **The scope of validity is strongly local, and that limitation should be foregrounded more explicitly in the main narrative.** The paper does acknowledge the quadratic regime and even gives a surrogate-free formulation with an effective Hessian in Appendix B.2.1, but many of the headline claims are broader than the validated regime. For nonlinear SGD-trained networks, higher-order terms, optimizer dynamics, and movement across regions may matter substantially beyond the perturbative setting actually tested.
- **The transfer law and asymptotic free-probability machinery are compelling for the ridge model, but their relevance to finite neural-network Hessians is not empirically characterized.** The paper does not need to prove such results for deep nets, but some finite-size sanity checks would help bridge the theory-to-practice gap.
- **The paper hints at actionable consequences (“alignment-aware optimization”) without yet demonstrating them.** This is fine as future work, but currently the work is much stronger as an analysis/diagnostic paper than as a method with established utility for improving models.

### Trivial
- The main text could do a better job distinguishing which claims are exact structural decompositions, which are asymptotic theorems for ridge regression, and which are empirical hypotheses about deep networks. This is mostly a framing issue, not a technical flaw.

## Nice-to-Haves
- Show a predictive sweep under varying class-imbalance ratios and compare overlap-based predictors against spectrum-only baselines such as trace, top eigenvalue, or sharpness proxies.
- Add at least one more modern architecture or training setup to test whether the overlap diagnostics remain informative beyond ResNet-20 and toy MLPs.
- Include a calibration experiment for Overlap-KPM on smaller models where exact overlaps can be computed, to quantify estimator error.
- Track overlap evolution through training to connect the static local picture with actual learning dynamics.
- If feasible, test a simple overlap-aware regularizer or intervention to establish causal utility rather than post-hoc descriptiveness.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Theorem 1 is just a tautology / not a paper.”** This is too strong. While Equation (6) is indeed a decomposition of a trace quantity rather than a standalone predictive principle, the paper does more than rename a trace: it builds a two-loss formalism around that decomposition, derives transfer rules, solves a nontrivial ridge model, and develops practical estimators. So the criticism should be weakened, not treated as fatal.
- **“The paper unfairly ignores that the literature already accounts for directional sensitivity.”** The paper’s introduction may be rhetorically broad, but within the submission it does not claim absolute novelty of all directional notions; it specifically argues that train–test eigenspace overlap is missing from common spectrum-centered loss-geometry analyses. That is a defensible framing.
- **“The two-scale covariance model is too simplified, so the multiple-descent analysis is invalid.”** The paper explicitly uses the two-scale model “for clarity” after deriving general formulas in Appendix C. This is a reasonable illustrative choice, not a misrepresentation.
- **“The transfer law is invalid because freeness fails in finite networks.”** The paper applies Theorem 2 in the ridge-regression asymptotic setting where freeness is the intended tool. It does not claim the theorem is exact for finite neural networks.
- **“Scalability to very large LLM-scale models is not shown.”** True but outside the core demonstrated scope. The paper claims scalable estimators and shows feasibility on ResNet-20 with complexity analysis; not reaching LLM scale is not by itself a substantive weakness for this submission.

## Novel Insights
The paper’s strongest new insight is not merely that overlaps matter, but that they provide the right language to factor local generalization into three distinct pieces: test sensitivity scale, train-induced variance scale, and a routing term that specifies where variance flows. The isospectral rotation experiment is especially revealing because it isolates an effect that spectra provably cannot see. A second useful synthesis is that multiple descent can be viewed as a sequence of overlap reallocations between emergent train eigenspaces and different test subspaces, rather than only as a story about near-zero eigenvalues; this gives a more directional interpretation of why error can decrease even as the minimum train eigenvalue continues to shrink.

## Suggestions
- Reframe Theorem 1 more carefully as a structural decomposition of local two-loss geometry, not as a wholly new predictive law.
- Strengthen the ResNet/CIFAR section by turning it into a prediction task: vary imbalance systematically and test whether overlap-based quantities forecast degradation better than spectrum-only baselines.
- Add direct estimator-validation experiments on smaller networks where exact overlap computations are possible.
- Moderate claims about “modern neural networks” unless additional architectures/settings are added.
- Sharpen the paper’s claim hierarchy: exact local decomposition, exact/asymptotic ridge results, and suggestive deep-net diagnostics should be clearly separated.
- If space allows, include a brief quantitative discussion of when the quadratic approximation breaks down in the MLP experiments (e.g., as noise magnitude increases).

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 2.0, 4.0]
Average score: 5.6
Binary outcome: Accept
