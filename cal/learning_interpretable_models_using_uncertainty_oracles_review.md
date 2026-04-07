=== CALIBRATION EXAMPLE 34 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and reflects the core idea. The abstract outlines the problem, approach, and claims. However, it is a bit verbose and could more crisply state the **novelty** relative to the precursor work (Ghose & Ravindran, 2020). The claim of "only one hyperparameter" is slightly misleading; while the iteration budget \(T\) is the primary knob, the user must also set box constraints for the optimization variables (e.g., ranges for \(a, b, \alpha, N_s\)), which the paper later says have "reasonable defaults." This should be clarified.

### Introduction & Motivation
The motivation is strong: small interpretable models are desirable but suffer from an accuracy trade-off, and existing levers may be insufficient. The toy example (Figure 1) is effective. The contributions are listed clearly. One minor point: the phrase "model-agnostic" is used but defined in a footnote as agnostic to model *family*. This is fine, but it should be made clear that the technique is not agnostic to the **loss function** (it's designed for non-differentiable losses via BayesOpt).

### Method / Approach
This is the core of the paper and has both strengths and weaknesses.

**Strengths:**
*   The high-level idea (project to 1D via oracle uncertainty, learn a sampling distribution via DP mixture, optimize with BayesOpt) is innovative and well-motivated.
*   The use of a Dirichlet Process Infinite Beta Mixture Model (IBMM) is appropriate for flexibility on [0,1].
*   The decision to use Bayesian Optimization to handle non-differentiable losses is a key design choice that broadens applicability.

**Concerns & Gaps:**
1.  **Justification of Choices:** The paper provides intuitive reasons for choices (e.g., Beta distribution support, margin uncertainty) but lacks deeper justification or ablation. Why is a *mixture* of Betas necessary? Could a single, flexible distribution (e.g., a Kumaraswamy) suffice? The smoothing transformation in §A.5 is presented as a practical heuristic but its impact on the optimization objective (Eq. 1) is not analyzed. It changes the density the IBMM models, which could introduce bias.
2.  **Algorithmic Clarity:** Algorithm 1 is clear, but the dependency on Algorithm 2 (in Appendix) for sampling from the IBMM is critical. The description of sampling (lines 6-10 of Alg. 2) is confusing. It states that for each component \(c_i\), it calculates \(p_j\) for *every* training instance \((x_j, y_j)\), normalized to sum to 1, and then samples \(n_i\) points from the **entire training set** based on these probabilities. This seems computationally heavy per iteration and implies components are not associated with specific regions of the uncertainty space, which contradicts the typical DP mixture model intuition. This needs clarification.
3.  **Optimization Details:** The statement "Because there is no tight coupling between our formulation and the optimizer, it is possible to use a different BayesOpt library" (end of §3.3) is important but undercuts the novelty of the specific implementation. What are the necessary properties of the optimizer? The paper later (§5, A.11) shows that switching from *hyperopt* (TPE) to *BoTorch* (GP+LogEI) yields massive speedups. This raises the question: how much of the reported performance is due to the core idea of learning a distribution on uncertainty, and how much is an artifact of the specific (and perhaps suboptimal) *hyperopt* configuration used for the main results?
4.  **Theoretical Grounding:** Equation 1, the objective, is sensible. However, there is no discussion of convergence properties or what the learned distribution \(p'\) represents. Does it up-weight instances near the decision boundary? The intuition is given, but a more formal connection to techniques like hard example mining or importance weighting would strengthen the foundation.

### Experiments & Results
The empirical evaluation is extensive, which is a major strength.

**Effectiveness (§4.1):**
*   The improvements in Table 1/4 are often substantial, especially for very small model sizes. This strongly supports the primary claim.
*   The use of F1-macro and multiple trials is good.
*   **Critical Issue: Statistical Testing Protocol.** The procedure in §4.1.2 is problematic. For each configuration, they run 5 trials, perform a t-test on the **validation** scores between the baseline and their method, and only report the test set improvement if the null is rejected (p<0.1). Otherwise, they report \(\delta F1_{test}=0\). This is a form of **post-selection inference** that inflates apparent significance. The test set should be strictly held out for final evaluation, not used to decide whether to report a result. A cleaner approach would be to report the mean \(\delta F1_{test}\) over all 5 trials for all configurations, and then use a paired Wilcoxon test (as done later in A.8) across datasets/sizes to establish overall significance. The current method could hide cases where the method slightly underperforms on the test set even after validation-significant improvements.
*   The observation that improvements diminish with model size is intuitive and well-discussed.

**Benchmarking & Competitiveness (§4.2, A.9-A.10):**
*   The comparison to density trees (A.9) is favorable and demonstrates advancement over the precursor.
*   The "competitiveness" experiments are compelling. Showing that CART + your method can rival specialized algorithms like IMM for cluster explanation is a strong result. The prototype-based classification experiment is similarly effective.
*   The experiments on multivariate size (A.12) and different feature spaces (A.13) nicely demonstrate versatility.

**Missing Analyses:**
1.  **Ablation Study:** What is the contribution of each component? Most critically, how does performance change if you **remove the oracle** and use a simple 1D projection (e.g., PCA first component)? Or if you use a simpler distribution model (e.g., a single Beta)? Or if you do simple uncertainty sampling (e.g., always sample high-uncertainty points) instead of learning a distribution? The claim that "we can’t just pick highly uncertain points" is supported by a citation but not validated in this paper's context.
2.  **Oracle Sensitivity:** How does the choice/accuracy of the oracle affect results? What if the oracle is poor? Is there a correlation between oracle accuracy and final improvement?
3.  **Computational Cost:** While runtime is discussed as a limitation in §5, the main results do not report wall-clock times or computational budgets. For practicality, readers need to know the cost of, e.g., the ~100% improvement for a depth-1 tree.

### Writing & Clarity
The paper is generally well-written. The figures are helpful. Some sections, particularly the method details, are dense and could benefit from a more pedagogical structure. The appendix is massive (40+ pages) and contains essential details (e.g., Algorithm 2), which is good for reproducibility but suggests the main paper could be more self-contained on key mechanics.

### Limitations & Broader Impact
*   **Limitations:** The runtime limitation is honestly discussed, and the preliminary mitigation using BoTorch is promising. However, this mitigation is only briefly mentioned; more details and results would strengthen the case for practicality. Another limitation is the assumption of a **probabilistic, calibrated oracle** to provide uncertainty scores. The paper does not discuss the behavior with poorly calibrated oracles.
*   **Broader Impact:** A "Broader Impact" statement is **missing** and is a required section for ICLR submissions. The paper should include a discussion of potential societal benefits (e.g., enabling more accurate interpretable models in high-stakes domains) and risks (e.g., the computational cost could limit accessibility; the dependence on an oracle could introduce biases).

### Overall Assessment
This paper presents a novel and generally effective technique for improving the accuracy of small, interpretable models by learning a resampling distribution over training data, projected via an oracle's uncertainty. The core idea is sound, and the empirical results across a wide range of datasets, model families, and tasks are impressive and support the claims. However, the acceptance hinges on addressing significant concerns: (1) the **statistical evaluation protocol** is flawed and must be revised, (2) the **methodological description** needs clarification, particularly regarding the sampling process and the role of the optimizer, (3) **critical ablations** are missing to justify design choices, and (4) a **broader impact statement** must be added. If these issues are adequately addressed, this represents a valuable contribution to the interpretable ML literature suitable for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a technique for improving the accuracy of small, interpretable models (e.g., decision trees, linear models) by learning a modified training distribution. The core idea is to project data into a 1D space using prediction uncertainties from a separate "uncertainty oracle" (e.g., a GBM or Random Forest), then use Bayesian Optimization to fit a Dirichlet Process mixture model over this space to sample a new training set. This approach is model-agnostic, handles non-differentiable losses, and requires setting only one hyperparameter (the optimization budget). Extensive experiments on 13 datasets demonstrate significant accuracy improvements for constrained model sizes, versatility across model families, and competitiveness with specialized methods in tasks like cluster explanation and prototype-based classification.

### Strengths
1. **Strong empirical validation**: The paper provides exhaustive experiments on 13 public datasets, using two interpretable model families (LPMs and DTs) and two oracle families (GBMs and RFs). Results show substantial relative improvements (often >100% for very small sizes) in F1-macro score, with statistical significance confirmed via Wilcoxon signed-rank tests (see Table 1, Figure 5, and Appendix A.8).
2. **Versatility and generality**: The method is shown to work with different notions of model size (e.g., depth, non-zero coefficients, multivariate sizes like depth and number of trees in GBMs—§A.12), and with different feature spaces between oracle and target model (e.g., GRU on character sequences for a DT on n-grams—§A.13). This flexibility is a key practical advantage.
3. **Competitive benchmarking**: The technique outperforms the closely related density-tree approach (Ghose & Ravindran, 2020) in most settings (Table 5) and can elevate simple baselines (e.g., CART, RBFN) to be competitive with recent task-specialized methods like IMM and SNC in explainable clustering and prototype classification (§A.10, Figures 7 & 8).
4. **Clear methodology and reproducibility**: The algorithm (Algorithm 1) is well-described, with detailed appendices covering uncertainty metrics, smoothing, parameter defaults, and sampling. Code is provided as supplementary material, and all datasets are publicly available.

### Weaknesses
1. **High computational cost**: The reliance on Bayesian Optimization (using hyperopt) leads to long runtimes—e.g., ~1 hour for a single configuration on a1a with T=3000 (§5). While preliminary results with BoTorch suggest potential speedups (∼2 minutes), these are not fully integrated or evaluated across all experiments, leaving practicality in doubt.
2. **Limited analysis of smoothing impact**: The smoothing transformation (§A.5) is applied uniformly, but its necessity and effect are only briefly analyzed on two datasets (Table 2). A more thorough ablation study across all datasets and model sizes would strengthen the claim that smoothing is generally beneficial without negative side-effects.
3. **Moderate novelty**: The core idea—learning a training distribution to improve small models—builds directly on prior work (Ghose & Ravindran, 2020). The main advances are the use of an uncertainty oracle for projection and Bayesian Optimization for flexibility, but the conceptual leap is incremental. The paper could better delineate its novelty against broader areas like data reweighting, active learning, and distillation.
4. **Incomplete statistical reporting**: While Wilcoxon tests are used, the paper does not report effect sizes or confidence intervals for the improvements, making it harder to assess practical significance. Additionally, the use of a one-sided test at p=0.1 for accepting models (§4.1.2) is relatively lenient and may inflate positive results.
5. **Sparse discussion of limitations**: Beyond runtime, limitations are under-explored. For instance, the method’s performance on very high-dimensional data (e.g., text with thousands of features) or with extremely small sample sizes is not tested. The sensitivity to oracle quality (e.g., poorly calibrated probabilities) is also not addressed.

### Novelty & Significance
**Novelty**: The technique is a non-trivial extension of prior work on learning training distributions. The integration of an uncertainty oracle for 1D projection and the use of Bayesian Optimization to handle non-differentiable losses are novel contributions. However, the overall framework is conceptually similar to existing re-sampling and weighting strategies.

**Significance**: The work addresses a core challenge in interpretable ML—the trade-off between size and accuracy—and offers a general, model-agnostic tool that can boost the performance of simple models. The ability to work with different feature spaces and multivariate size constraints is practically valuable. If computational costs can be reduced, the method could have substantial impact in domains requiring small, accurate, and interpretable models.

**Clarity**: The paper is generally well-written, with clear figures and algorithmic descriptions. Some sections (e.g., the smoothing transformation in Appendix A.5) are technical but accessible.

**Reproducibility**: High. Code is provided, datasets are public, and experimental settings are detailed in the appendix.

### Suggestions for Improvement
1. **Address computational efficiency more concretely**: Integrate the BoTorch experiments fully into the main evaluation (e.g., replace hyperopt or compare both) and report runtimes and accuracy trade-offs across all major configurations. Discuss parallelization or early-stopping strategies to make the method more practical.
2. **Deepen the analysis of smoothing and parameter sensitivity**: Conduct an ablation study to show the impact of smoothing across all datasets and model sizes. Also, explore the sensitivity to the Dirichlet Process parameters (e.g., scale, α) and provide guidance on setting bounds.
3. **Strengthen statistical reporting**: Report effect sizes (e.g., Cohen’s d) or confidence intervals for improvements. Consider using a stricter significance threshold (e.g., p=0.05) for model acceptance, and include a power analysis for the five trials.
4. **Clarify novelty and related work**: Expand the related work section to explicitly contrast the method with data reweighting, curriculum learning, and distillation (despite the authors’ distinction, the use of an oracle is reminiscent of distillation). Highlight what specific limitations of prior work are overcome.
5. **Explore failure modes and limitations**: Test on more challenging data (e.g., high-dimensional sparse features, very small datasets) and discuss when the method might fail (e.g., with a poorly performing oracle). Include a subsection on limitations in the main paper, not just runtime in §5.
6. **Improve presentation of results**: In Table 1, use a consistent color scale or symbols to highlight significant improvements. Consider including a summary table of average improvements across all configurations (model, oracle, size) to give an overall picture of gains.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study removing the uncertainty oracle.** The core claim is that the 1D projection via oracle uncertainty is crucial. Without comparing to a variant that uses the raw features (or a random projection) to learn the distribution, it's unclear if the oracle adds value beyond a generic dimensionality reduction.
2. **Comparison to simpler reweighting/resampling baselines.** The paper lacks comparison to standard techniques like importance weighting, SMOTE, or hard example mining. This omission makes it impossible to judge whether the complex Dirichlet Process and Bayesian optimization are necessary.
3. **Sensitivity analysis of the oracle's accuracy.** The method assumes a high-quality probabilistic oracle. Experiments varying oracle quality (e.g., using a weak model or noisy probabilities) are missing; without them, we cannot trust the method's robustness in practice.
4. **Direct comparison to gradient-based methods for differentiable losses.** For Linear Probability Models (differentiable), the paper should compare to standard reweighting techniques that use gradient information (e.g., learning instance weights via bilevel optimization). This would justify the choice of Bayesian optimization only for non-differentiable cases.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the learned sampling distributions.** The paper does not show what distributions are actually learned (e.g., do they upweight high-uncertainty points?). Visualizing the learned Beta mixtures and the corresponding sampled instances is needed to understand why the method works.
2. **Interpretability audit of the resulting models.** The paper claims to produce interpretable models but does not analyze whether the models remain interpretable after distribution learning. For example, do the decision trees become more complex in structure even if depth is constrained? A qualitative analysis of the resulting rules is missing.
3. **Statistical robustness of improvements across all settings.** While some statistical tests are reported, a comprehensive analysis (with appropriate multiple-testing corrections) across all datasets, model sizes, and oracle combinations is lacking. This is critical for claiming consistent improvements.
4. **Breakdown of computational cost.** The paper mentions high runtimes but does not provide a detailed analysis of time spent on oracle training, Bayesian optimization, and model training. This is essential for assessing practical utility.

### Visualizations & Case Studies
1. **Visualization of sampled instances in feature space.** For a few datasets, use PCA/t-SNE to plot original vs. sampled training points. This would reveal whether the method selectively samples near decision boundaries, as hypothesized.
2. **Case study on a real-world high-stakes domain.** The paper uses standard ML datasets. A case study on a dataset where interpretability is critical (e.g., medical diagnosis) would demonstrate practical impact and show that interpretability is preserved.
3. **Visualization of the optimization trajectory.** Plot validation accuracy vs. Bayesian optimization iterations, alongside the evolution of distribution parameters. This would illustrate whether the optimizer reliably finds better distributions and how quickly.

### Obvious Next Steps
1. **Extend to differentiable models using gradient-based optimization.** The paper notes this as future work, but given that linear models are differentiable, a comparison to gradient-based instance weighting methods should have been included to position the contribution.
2. **Joint optimization of model size and distribution.** The model size is fixed a priori. A natural extension is to make size a variable in the optimization, which would fully automate the size-accuracy trade-off.
3. **Comparison to state-of-the-art interpretability methods.** The paper compares to a few specialized techniques (e.g., IMM, ProtoNN) but omits comparisons to other contemporary interpretable model families (e.g., rule sets, sparse generalized additive models). This limits the claim of versatility.
4. **User study on interpretability.** Since interpretability is human-centric, a user study evaluating the comprehensibility of models trained with vs. without the method would strengthen the claim that interpretability is maintained.

# Final Consolidated Review
## Summary
This paper proposes a technique to improve the accuracy of small, interpretable models by learning a modified training distribution. The core idea is to project data into a one-dimensional space using prediction uncertainties from a separate "uncertainty oracle," model a flexible Dirichlet Process mixture over this space, and use Bayesian Optimization to learn a sampling distribution that maximizes the accuracy of a size-constrained target model (e.g., a shallow decision tree or sparse linear model).

## Strengths
- **Strong and Extensive Empirical Validation:** The method is evaluated across 13 datasets, two interpretable model families (Linear Probability Models and Decision Trees), and two oracle families (Gradient Boosted Models and Random Forests). Results show substantial relative improvements in F1-macro score, often exceeding 100% for very small model sizes, with statistical significance established via Wilcoxon signed-rank tests (Table 4, Figure 5).
- **Demonstrated Versatility and Generality:** The technique is shown to be agnostic to the specific notion of model size (e.g., tree depth, non-zero coefficients, multivariate sizes like tree depth and count for GBMs in §A.12) and can function even when the oracle and target model use entirely different feature spaces (e.g., a GRU on character sequences for a decision tree on n-grams in §A.13). This flexibility is a significant practical advantage.
- **Competitive Performance Against Specialized Methods:** The approach outperforms its direct predecessor (density trees) and, more impressively, can elevate simple baselines (e.g., CART, RBF networks) to be competitive with recent task-specialized algorithms like Iterative Mistake Minimization (for cluster explanation) and Stochastic Neighbor Compression (for prototype-based classification) (§A.9, A.10).

## Weaknesses
- **Flawed Statistical Reporting Protocol:** The evaluation method described in §4.1.2 uses the validation set to perform a t-test and only reports test set improvements if the null hypothesis is rejected (p<0.1). This constitutes post-selection inference and can artificially inflate the perceived significance of the results. The test set should be used for final evaluation only, not for deciding whether to report a result.
- **Missing Critical Ablation Studies:** The paper lacks experiments that justify key design choices. Most notably, there is no ablation comparing the use of the uncertainty oracle to a simpler 1D projection (e.g., PCA) or to direct resampling baselines (e.g., importance weighting, hard example mining). Furthermore, there is no analysis of the method's sensitivity to the quality and calibration of the oracle, which is a core dependency.
- **High Computational Cost and Incomplete Efficiency Analysis:** The reliance on Bayesian Optimization with `hyperopt` leads to long runtimes (e.g., ~1 hour per configuration as noted in §5). While preliminary results with `BoTorch` suggest massive speedups (~2 minutes), this alternative is not integrated into the main evaluation. The computational practicality of the method, a key concern for adoption, remains unclear.

## Nice-to-Haves
- A deeper analysis and visualization of the learned sampling distributions to better understand what regions of the data are being up-weighted.
- A more thorough investigation of the smoothing transformation's impact across all datasets and model sizes, beyond the limited analysis in Appendix A.6.
- A case study applying the method in a real-world, high-stakes domain where interpretability is critical, to qualitatively assess the preservation of model understandability.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Claim of Missing Broader Impact Statement:** The paper includes "Ethics Statement" and "Reproducibility Statement" sections, which fulfill the required broader impact considerations for ICLR.
- **Criticism of "Only One Hyperparameter" Being Misleading:** The paper clarifies in §3.3 that "reasonable default bounds exist for parameters Φ" and that "in practice, T is the only parameter that a user needs to set." This is a reasonable claim given the provided defaults.
- **Confusion About Algorithmic Clarity of Sampling (Algorithm 2):** The sampling procedure is detailed in Appendix A.2 and Algorithm 2. The description, while computationally involved, is clear: for each DP component, probabilities are calculated for all training points based on that component's Beta distribution, and points are sampled accordingly. This is a valid implementation of a mixture model sampler.
- **Request for Theoretical Convergence Analysis:** The paper is primarily an empirical, methodological contribution. Demanding theoretical proofs for convergence is not a standard expectation for this type of work in the field.

## Novel Insights
The most novel insight emerging from the synthesis of the results is the method's ability to effectively bridge different representational spaces. The experiment in §A.13 demonstrates that an oracle trained on one feature space (a GRU on character sequences) can successfully guide the training of an interpretable model on a completely different, human-engineered feature space (n-grams for a decision tree). This suggests the technique can act as a translator, allowing the informational value captured by a powerful, potentially black-box model to be transferred into the structure of a simple, interpretable one without requiring them to share a common input representation. This property significantly broadens the potential applicability of the method beyond standard settings.

## Suggestions
- **Revise the Statistical Evaluation:** Report the mean test set improvement over all trials for every configuration, regardless of validation score significance. Use aggregate statistical tests (like the already-employed Wilcoxon test) across datasets and sizes to establish overall method efficacy, avoiding the conditional reporting scheme.
- **Conduct Key Ablation Studies:** In a revised manuscript or appendix, include experiments that: (1) replace the oracle uncertainty with a simple random or PCA-based projection, (2) compare against standard reweighting/resampling baselines, and (3) test robustness with progressively weaker or miscalibrated oracles.
- **Integrate and Report on Computational Efficiency:** Either replace the main `hyperopt` results with those from the more efficient `BoTorch` setup mentioned in §5 and Appendix A.11, or provide a direct comparison of runtime versus accuracy for both optimizers across several representative configurations. Clearly state the computational cost trade-off.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 0.0]
Average score: 3.3
Binary outcome: Reject
