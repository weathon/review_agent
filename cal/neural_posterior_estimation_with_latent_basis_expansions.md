=== CALIBRATION EXAMPLE 71 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper does introduce an amortized posterior estimation method based on basis expansions in latent space.
- The abstract clearly states the high-level problem, the proposed LBF-NPE approach, and the kinds of empirical gains claimed.
- However, the abstract makes several strong claims that are not yet justified from the paper’s main text alone:
  - “optimizes over the class of all exponential families of a fixed dimension K” is too strong as stated. In practice, the family is restricted by the chosen neural/basis parameterization and the finite Monte Carlo approximation used in training.
  - “convex optimization” and “stable convergence to global optima” are presented in a stronger form than the actual method seems to guarantee, especially for the adaptive-basis and stereographic-projection variants.
  - The abstract highlights superiority over normalizing flows and MDNs, but the paper’s evidence is mostly on low-dimensional synthetic tasks and a few application-specific settings, so the breadth of the claim should be tempered.

### Introduction & Motivation
- The motivation is strong and ICLR-relevant: the paper targets a real gap in simulation-based inference, namely the tension between expressive posterior families and tractable optimization, especially when only low-dimensional posterior projections are required.
- The introduction does a good job positioning NPE, nuisance marginalization, and the appeal of a likelihood-free amortized framework.
- That said, the novelty claim needs sharper calibration. The paper states this is “the first method” to use neural exponential families for posterior representation in amortized inference. This may be defensible if narrowly phrased, but the relation to existing work on neural exponential families, likelihood-free posterior/likelihood estimation, and basis-expansion VI should be made more precise.
- The introduction also risks over-selling the convexity angle. The paper’s strongest optimization claims are for the fixed-basis case, yet much of the empirical appeal comes from adaptive-basis variants where convexity no longer applies jointly. That distinction should be highlighted more carefully.

### Method / Approach
- The core formulation in Section 3 is clear: model the log-density as an inner product between an inference network output \(f_\phi(x)\) and latent basis functions \(s_\psi(z)\), with normalization handled by explicit numerical integration.
- A key strength is that the method is simple in principle and matches NPE’s training objective naturally.
- But there are several important issues that need better justification or clarification:
  - **Gradient estimation**: Section 3.2 uses self-normalized importance sampling to estimate gradients of the log normalizer. The estimator is biased, and the paper acknowledges consistency only as the number of proposal samples grows. For practical training, the variance/bias tradeoff and sensitivity to the proposal \(r(z)\) are not well analyzed.
  - **Reproducibility / practical implementation**: the method depends heavily on numerical integration over latent space. This is manageable in 1D and some 2D settings, but the paper’s own high-dimensional discussion is much less concrete. The exact integration strategy changes across experiments (trapezoidal sums, Monte Carlo, inverse transform, sequential sampling), which makes the method less uniform than the main framing suggests.
  - **Convexity claims**: Proposition 1 is about marginal convexity in \(f\) and in \(s\), not joint convexity. The paper sometimes reads as if this gives strong optimization guarantees, but alternating minimization over two nonconvex neural nets is still not globally convex. This should be stated more explicitly.
  - **Proof concerns**: Appendix B’s proof of Proposition 1 is not cleanly written and appears to conflate pointwise convexity with convexity of the function class under the neural parameterization. The statement about “marginal convexity in the arguments” is fine at the functional level, but readers need a clearer separation between convexity in function space and nonconvexity in network weights.
  - **Identifiability / stereographic projection**: the paper correctly identifies scale degeneracy in the inner product formulation, but the stereographic projection discussion leaves open what exact invariances remain, how much it helps beyond normalization, and whether it introduces optimization pathologies of its own. The claimed “unit hypersphere” mapping is plausible, but the paper should more carefully justify why this particular reparameterization is preferable to simpler normalization.
  - **Sampling limitation**: the paper admits sampling is difficult, which is an important limitation, but the main method is still framed as a posterior approximator. For a posterior approximation method, weak sampling support is a real practical drawback and deserves more emphasis.
- On balance, the method is interesting and coherent, but the paper overstates the generality and optimization guarantees relative to what is actually established.

### Experiments & Results
- The experiments do test several of the central claims:
  - multimodal low-dimensional posteriors,
  - advantages over MDNs and flows,
  - the effect of basis-function dimension,
  - a realistic astronomical application,
  - and a higher-dimensional illustrative extension.
- The synthetic experiments are particularly relevant for the claimed optimization advantages. The sinusoidal example in Section 6.1 does support the claim that the fixed-basis convex formulation can reduce bad local minima relative to an MDN.
- However, several aspects limit how strongly the results support the paper’s broader conclusions:
  - **Baselines are reasonable but incomplete**: MDN, RealNVP, NSF, and EigenVI are appropriate comparisons, but the paper does not compare against other relevant SBI/NPE methods such as robust NPE variants or more recent conditional density estimators, beyond the limited appendix references.
  - **Fairness of comparisons**: the training budgets, architectures, and proposal/integration strategies differ substantially across methods and tasks. For example, the basis-expansion method sometimes uses exact or grid-based integration, while other methods are trained in standard likelihood-based ways. More detail is needed to ensure the comparisons are fair and not accidentally favoring the proposed approach.
  - **Ablations**: the most important missing ablations are on the gradient estimator and proposal distribution \(r(z)\), the number of importance samples in Algorithm 1, and the effect of stereographic projection versus simpler normalization. These are central to the method’s practical behavior and could materially change the conclusions.
  - **Uncertainty reporting**: some tables report means ± standard deviations over seeds, which is good, but the main application results are uneven. In particular, Table 2 reports held-out NLL on redshift estimation, but the paper does not clearly describe variability across runs, nor whether the reported test score is stable across random initializations and data splits.
  - **Metrics**: forward/reverse KL on fully tractable 2D synthetic problems is appropriate. But in the object detection and redshift settings, the evaluation is less directly comparable to the core posterior quality claim. The redshift result is held-out NLL of the true redshift under the learned density, but because of the BLISS pipeline and tiling approximations, it is not obvious that this is a clean measure of posterior quality.
  - **Cherry-picking risk**: the paper sometimes emphasizes its strongest cases without equally foregrounding weaker ones. For example, in Table 1, LBF-NPE is not best on Spiral for NLL, where NSF is slightly better. That should be acknowledged more directly rather than presented as uniformly dominating.
- Overall, the experiments are promising and likely persuasive for low-dimensional SBI, but the evidence is less complete than the paper’s strongest claims suggest.

### Writing & Clarity
- The high-level narrative is understandable, and the section organization is sensible.
- The clearest parts are the motivation for basis expansions, the distinction between fixed and adaptive bases, and the practical relevance of low-dimensional projections.
- The main clarity issues are conceptual rather than stylistic:
  - Section 3.2’s derivation of the gradient estimator is hard to follow, and the transition from the exact gradient of the log normalizer to the SNIS approximation would benefit from a cleaner, more explicit derivation.
  - The distinction between “basis functions” as neural-network outputs and classical fixed bases is somewhat blurred in places. Readers would benefit from a sharper conceptual separation between fixed local bases, learned bases, and the coefficient network.
  - The convexity discussion in Section 4 can mislead readers into thinking the whole training problem is convex, when in fact only marginal convexity in one component is established.
  - The sampling discussion in Appendix C is somewhat optimistic relative to the method’s actual use cases; for a method whose output is a density, sampling is not a side issue.
- The figures and tables are generally informative, especially Figures 1–3 and Tables 1–6, though some appendix figures are more diagnostic than necessary for the main argument.

### Limitations & Broader Impact
- The paper does acknowledge one key limitation: sampling from the fitted density is difficult.
- But it misses or under-develops several important limitations:
  - **Scalability beyond low dimensions**: although the paper gestures at high-dimensional extensions, the core practical story relies on low-dimensional posterior targets where numerical integration is feasible. This should be stated much more prominently as a fundamental scope limitation.
  - **Dependence on integration strategy**: the method’s behavior depends on whether one uses trapezoidal quadrature, Monte Carlo, inverse transform sampling, or sequential sampling. This makes the method less turnkey than standard NPE families.
  - **Optimization complexity in adaptive-basis mode**: the alternating optimization and stereographic normalization add nontrivial complexity. The paper presents these as fixes, but they may also introduce new tuning burdens.
  - **Scientific impact / misuse**: the paper’s applications are benign scientific inference problems, so there is no obvious negative societal impact. Still, the authors should note that posterior approximations in safety-critical settings require careful calibration, and density quality alone may not be sufficient.
- The broader-impact discussion is minimal, but that is not itself a fatal issue for this kind of paper. The more important gap is that the limitations are not framed strongly enough relative to the method’s apparent scope.

### Overall Assessment
This is a creative and technically plausible paper with a clear idea: use latent basis expansions to make amortized posterior estimation more stable and expressive in settings where only low-dimensional posterior projections matter. The fixed-basis convexity angle is genuinely interesting, and the synthetic and applied experiments suggest the method can outperform MDNs and normalizing flows in several low-dimensional SBI problems. That said, the paper’s strongest claims are broader than the evidence supports. In particular, the optimization guarantees are only partial, the adaptive-basis variant is still nonconvex and somewhat under-analyzed, the gradient estimator’s bias/variance and dependence on numerical integration are underexplored, and the scalability story remains limited to settings where low-dimensional integration is feasible. For ICLR standards, this is a promising and potentially publishable direction, but it needs stronger methodological and empirical substantiation before the claims can be considered fully convincing.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes LBF-NPE, a neural posterior estimation method whose variational family is defined by an exponential-family log density parameterized through learned or fixed latent basis functions. The central idea is to combine NPE’s amortization and automatic marginalization with a basis-expansion posterior family that is intended to be more optimization-friendly than flows or mixture models, while remaining expressive enough for multimodal low-dimensional posteriors. The paper reports empirical gains on synthetic multimodal targets and two scientific applications, including astronomical object detection and photometric redshift estimation.

### Strengths
1. **Clear and relevant problem setting for ICLR.** The paper targets an important practical limitation of NPE: the tension between expressivity and optimization stability in the variational family. This is a meaningful issue for simulation-based inference and amortized inference more broadly, which are active topics at ICLR.

2. **Interesting methodological angle.** The idea of representing the posterior log density as an inner product between amortized coefficients and latent basis functions is conceptually elegant and naturally connects NPE to exponential families. The paper makes a plausible case that this can retain optimization benefits while increasing flexibility beyond Gaussian families.

3. **Exploits NPE’s marginalization property well.** A notable contribution is the explicit design for low-dimensional posterior projections while marginalizing nuisance variables through joint simulation. This is well aligned with many scientific inference problems where only a few latent quantities are of scientific interest.

4. **Strong empirical story on multimodal toy problems.** The synthetic experiments show the method outperforming MDNs and normalizing flows on challenging 2D posteriors, and the fixed-basis variant appears to converge more reliably than MDN on the sinusoidal example. The reported KL improvements on bands/ring are substantial.

5. **Application-driven evaluation.** The paper goes beyond toy examples and includes object detection and redshift estimation on astronomy-related tasks. This helps support the claim that the method is useful in realistic likelihood-free settings.

6. **Reproducibility signals are present.** The paper states that code is available and provides many implementation details in the appendix, including architectures, training schedules, and dataset sources. That is a positive sign for reproducibility.

### Weaknesses
1. **Novelty is somewhat incremental relative to existing exponential-family and basis-expansion VI work.** The core recipe—learned exponential-family parameterization with basis functions—is closely related to prior neural exponential family / basis expansion literature. The main novelty appears to be adapting this specifically to NPE and exploiting low-dimensional posterior projections, but the conceptual gap from prior work is not very large.

2. **Theoretical claims are stronger than the evidence supports.** The paper emphasizes convexity and global convergence properties, but these results are quite limited in scope. Convexity is marginal—holding when one of the two networks is fixed—and the more interesting adaptive-basis setting is not covered by the same strong guarantees. The empirical reliance on stereographic projection also appears to move the method outside the exact assumptions used in the convexity discussion.

3. **The optimization story is not fully convincing for the adaptive variant.** The adaptive-basis method alternates between optimizing coefficients and basis functions, but the paper does not provide a rigorous convergence analysis for this nonconvex alternating procedure. The identifiability issue is noted, but the proposed reparameterization only partially addresses it.

4. **Comparisons are incomplete for ICLR-level claims.** The main baselines are MDN and normalizing flows, with EigenVI and a score-matching method relegated to appendices. There is limited evidence against stronger or more recent posterior estimation methods, and some comparisons are on different problem formulations or with different sampling procedures, which weakens the universality of the empirical claims.

5. **Evaluation is somewhat narrow on posterior quality.** Most results focus on KL on synthetic data or held-out NLL on astronomy tasks. There is limited assessment of calibration, sample quality, downstream decision-making, uncertainty quality, or robustness to misspecification. Since NPE is often used for approximate Bayesian uncertainty, these additional metrics would be valuable.

6. **Sampling from the variational family is a real practical limitation.** The paper acknowledges that sampling is difficult, especially in higher dimensions, and suggests inverse transform or MCMC-like workarounds. This is important because many users of posterior approximations need efficient sampling, not just density evaluation.

7. **Clarity of the method is uneven.** The high-level idea is understandable, but several derivations and variants are not crisply separated. In particular, it is sometimes hard to tell which parts are core method, which are theoretical motivation, and which are experimental convenience choices. This makes it harder to assess what is truly required versus optional.

### Novelty & Significance
**Novelty: Moderate.** The paper combines known ingredients—exponential families, basis expansions, amortized inference, and NPE—in a way tailored to low-dimensional posterior approximation. The adaptation to simulation-based posterior estimation and the focus on optimization-friendly basis functions are useful, but the method does not feel like a major conceptual departure from prior neural exponential-family approaches.

**Significance: Moderate.** If the empirical advantages hold broadly, this could be a useful alternative to flows/MDNs for low-dimensional posterior projections in scientific SBI applications. That said, the significance is limited by the method’s apparent dependence on low-dimensional targets and by the absence of strong evidence that it consistently dominates more general-purpose posterior estimators across a broader benchmark suite.

**Clarity: Fair.** The paper communicates the main idea and motivation reasonably well, but the exposition is uneven, and the relationship between theory, architecture choices, and practical implementation could be presented more cleanly.

**Reproducibility: Good but not perfect.** The appendix provides many training details and the code is said to be available, which is positive. However, some claims depend on implementation details, and the adaptive optimization plus sampling procedures would benefit from even more precise algorithmic description.

**Overall significance relative to ICLR bar: Borderline to moderate.** ICLR typically rewards methods that offer clear algorithmic novelty, strong empirical evidence across diverse benchmarks, and broad relevance. This paper has a solid applied motivation and promising results, but the theoretical contribution is limited in scope and the novelty over prior basis-expansion / exponential-family methods is not fully compelling enough on its own.

### Suggestions for Improvement
1. **Clarify the exact novelty relative to prior neural exponential-family and basis-expansion methods.** A sharper discussion of what is new beyond “basis expansions + amortized inference” would strengthen the paper’s position.

2. **Provide a more rigorous treatment of the adaptive-basis optimization.** If strong guarantees are not possible, the paper should explicitly frame the adaptive version as a heuristic alternating-minimization method and characterize its failure modes.

3. **Expand the benchmark suite.** Include stronger posterior estimation baselines and a broader range of tasks, especially cases where the posterior dimension is not just 1–2 and where sampling quality matters.

4. **Evaluate calibration and downstream utility.** Add metrics such as credible-interval coverage, posterior predictive calibration, or decision-theoretic performance to complement KL/NLL.

5. **Separate fixed-basis and adaptive-basis variants more clearly.** The paper would benefit from a concise algorithm box for each variant, with clear statements of when each should be used.

6. **Discuss sampling more concretely.** Since density evaluation is not enough for many Bayesian applications, the paper should provide a more direct and practical sampling algorithm, complexity analysis, and empirical comparisons to baseline samplers.

7. **Temper claims about convexity and global convergence.** The current theoretical claims should be stated more carefully, especially given the mismatch introduced by stereographic normalization and the limited scope of the convexity result.

8. **Strengthen ablations.** The paper would benefit from systematic ablations on basis family choice, number of basis functions, importance-sampling sample count, and the effect of stereographic projection across multiple tasks.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to a strong NPE baseline trained under the same objective and compute budget, especially the Gaussian/canonical exponential-family setup from McNamara et al. (2024a). Right now the paper claims better optimization and global-convergence behavior, but it never shows whether the gain comes from the basis expansion or simply from using a different, more favorable parameterization and training recipe.

2. Add ablations that isolate the contribution of each claimed mechanism: fixed vs adaptive bases, with vs without stereographic normalization, and with vs without alternating optimization. Without these, the paper’s main claim that the method is both more stable and more expressive is not attributable to any specific design choice.

3. Benchmark against additional expressive amortized density estimators that are standard ICLR baselines for simulation-based inference, not just MDNs and flows. In particular, include modern conditional diffusion/score-based posterior estimators or stronger flow variants tuned carefully, because the current baseline set is too weak to support “outperforms existing variational families” as a broad claim.

4. Add a fairness-controlled compute comparison reporting wall-clock to target quality, not only final KL/NLL. ICLR reviewers will expect evidence that the method’s apparent optimization advantages are not offset by expensive integral estimation, alternating training, or large Monte Carlo grids.

5. Add a real posterior-prediction downstream task evaluation on the scientific applications, not just density metrics. For the astronomical cases, show whether better NLL translates into better calibration, coverage, or scientifically relevant decisions; otherwise the claimed practical superiority is not convincing.

### Deeper Analysis Needed (top 3-5 only)
1. Add sensitivity analysis for the number of basis functions K and the proposal/sample count used in the SNIS gradient estimator. The paper repeatedly claims that small K is sufficient and training is stable, but there is no analysis showing when the method breaks, whether gradients become high-variance, or how performance scales with approximation budget.

2. Add an analysis of identifiability and optimization under the adaptive basis parameterization. The paper states that rescaling and rotation degeneracies remain after stereographic projection, but it does not quantify whether alternating optimization truly resolves them or merely hides them on the presented tasks.

3. Add calibration and coverage diagnostics for the learned posteriors, especially in the redshift and object-detection settings. Forward KL/NLL alone do not tell whether the learned posterior uncertainty is well calibrated; without reliability or coverage analysis, the scientific utility claims are incomplete.

4. Add a careful study of approximation error versus posterior dimensionality and topology. The paper argues that low-dimensional posteriors make the method feasible, but the claims about “high-dimensional latent spaces” are only weakly supported by one synthetic 50D annulus experiment and do not establish general scalability.

5. Add variance/bias analysis of the gradient estimator used for the log-normalizer. Since the method’s main novelty is an importance-sampling-based estimator inside amortized training, the paper needs to show that the estimator is numerically reliable across the tasks and not just empirically adequate in a narrow regime.

### Visualizations & Case Studies
1. Add training-trajectory visualizations showing coefficient evolution, basis-function evolution, and loss landscape behavior for LBF-NPE versus MDN/flow baselines. This would reveal whether the claimed convexity/stability actually produces smoother optimization or whether the method merely converges in these particular tasks.

2. Add failure-case posterior plots for difficult observations, not only representative successes. ICLR reviewers will want to see where the method collapses, over-smooths modes, or misplaces mass; otherwise the presented visual evidence is too selective.

3. Add side-by-side visualizations of learned basis functions across tasks and K values, with the corresponding posterior density decomposition. This would expose whether the basis network learns meaningful local structure or simply memorizes task-specific partitions.

4. Add uncertainty-calibration plots for the scientific case studies, such as coverage vs nominal level and PIT-style diagnostics for redshift. These are the most direct way to show whether the posterior densities are usable in downstream scientific inference.

### Obvious Next Steps
1. Add a rigorous scaling study on latent dimension, K, and Monte Carlo budget to establish when the method remains practical. The current paper implicitly relies on low-dimensional targets; it should quantify the boundary of that claim.

2. Add an end-to-end comparison on a standard SBI benchmark suite, not just hand-designed toy problems and two astronomy tasks. ICLR expects generality beyond a narrow application stack, and the paper needs evidence that the method transfers across diverse posterior geometries and simulators.

3. Add a principled sampling algorithm for LBF-NPE that is actually usable in the general case. The paper acknowledges sampling difficulty as a main limitation, so a method whose posterior is easy to fit but awkward to sample from is incomplete as a variational inference contribution.

4. Add a clearer theoretical statement connecting the exponential-family basis expansion to approximation guarantees. The current convexity discussion is not enough; the paper needs to specify what posterior classes are approximable with finite K and under what conditions the basis expansion is expressive enough to justify the method.

# Final Consolidated Review
## Summary
This paper proposes LBF-NPE, an amortized neural posterior estimation method that parameterizes the posterior log density as an inner product between an observation-dependent coefficient vector and latent basis functions. The main appeal is that, for fixed bases, the resulting objective is convex in the amortized coefficients; for adaptive bases, the method alternates optimization and uses stereographic projection to mitigate scale degeneracy. The paper shows promising results on low-dimensional synthetic posteriors and a few astronomy applications, but the core story is still much narrower than the claims suggest.

## Strengths
- The core formulation is elegant and well aligned with NPE: it leverages NPE’s ability to marginalize nuisance variables while restricting attention to low-dimensional posterior projections, which is genuinely relevant in scientific SBI.
- The fixed-basis variant has a real optimization advantage in the toy multimodal example, and the paper provides evidence that it converges more reliably than an MDN on the sinusoidal benchmark and achieves strong KL performance on the 2D test problems.

## Weaknesses
- The strongest theoretical claims are overstated relative to what is actually proved. The paper establishes only marginal convexity in one block at a time, not joint convexity, and the adaptive-basis variant with stereographic projection falls outside the clean convexity story. Why it matters: the headline optimization narrative is substantially weaker than the paper implies.
- The method’s practical value is limited by reliance on numerical integration and a biased SNIS gradient estimator whose variance, sensitivity to the proposal distribution, and dependence on the number of importance samples are not analyzed. Why it matters: the training procedure may be much less stable or efficient than the paper’s headline results suggest, especially beyond the small-dimensional settings used in experiments.
- The scalability story is still thin. Most evidence is for 1D or 2D posterior targets, with one synthetic high-dimensional appendix example that does not establish general applicability. Why it matters: the paper presents itself as broadly useful for low-dimensional projections, but the boundary of where the method remains practical is not characterized.

## Nice-to-Haves
- A cleaner separation between the fixed-basis method, the adaptive-basis method, and the stereographic reparameterization would make it easier to see what is core and what is an implementation choice.
- A more direct discussion of sampling from the learned density would help, since density estimation without a practical sampler is incomplete for many Bayesian workflows.

## Novel Insights
The genuinely novel part of the paper is not just “basis expansions for VI,” but the way it turns posterior estimation in NPE into a low-dimensional exponential-family fitting problem with amortized coefficients and latent basis functions. That is a sensible and potentially useful reframing for SBI problems where only a small set of scientifically relevant latents matters. The paper’s most convincing insight is that this structure can improve optimization relative to mixtures/flows in difficult multimodal settings, but the benefit seems strongest in the fixed-basis regime; the adaptive regime is more of a heuristic extension than a fully characterized method.

## Potentially Missed Related Work
- McNamara et al. (2024a) — closely related convex NPE theory for simpler exponential-family parameterizations; useful for clarifying what is and is not new here.
- Pacchiardi & Dutta (2022) — neural exponential families in likelihood-free inference; relevant for positioning the “neural exponential family” aspect of the method.
- Cai et al. (2024) EigenVI — the closest basis-expansion VI comparator, especially for understanding how this work differs from fixed orthogonal expansions.

## Suggestions
- Add a direct comparison against the canonical Gaussian/exponential-family NPE setup from McNamara et al. under matched compute and objective, to isolate the gain from basis expansions.
- Include ablations for fixed vs adaptive bases, with vs without stereographic projection, and sensitivity to the number of basis functions and SNIS samples.
- Report wall-clock time to reach a target KL/NLL, not just final performance, and include calibration/coverage metrics on the scientific applications.
- Make the theoretical claims more precise: state clearly that convexity is marginal and does not apply to the full adaptive network parameterization.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
