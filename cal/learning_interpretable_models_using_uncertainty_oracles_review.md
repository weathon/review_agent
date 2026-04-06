=== CALIBRATION EXAMPLE 40 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the core idea: using an uncertainty oracle to learn interpretable models. The abstract succinctly states the problem (accuracy vs. size trade-off) and the proposed solution. The claims are specific: (1) enhances small model accuracies, (2) versatile across model families, (3) convenient (one hyperparameter, works with non-differentiable losses), (4) works across different feature spaces, (5) can augment old techniques. These claims are supported by the experiments described later, though the abstract could benefit from a brief mention of the empirical scope (e.g., number of datasets, model families) to set expectations.

### Introduction & Motivation
The introduction effectively motivates the need for accurate small interpretable models in high-stakes domains. The trade-off between size and accuracy is well-articulated, and the limitations of existing hyperparameter-based levers are clear. The contributions are listed explicitly and align with the abstract. Figure 1 provides a compelling toy example that illustrates the potential gain. One minor note: the term "model-agnostic" is used in the sense of agnostic to model family, which is acceptable, but could be clarified early on to avoid confusion with algorithm-agnostic.

### Methodology
The methodology is described in detail with an overview (Figure 2) and algorithmic steps (Algorithm 1). Key components are justified:
- **Uncertainty scores**: Margin uncertainty is chosen for its ability to handle multi-class settings and account for probabilities of different classes. Alternatives are discussed in the appendix.
- **Density model**: An infinite Beta mixture model via a Dirichlet Process is used for flexibility. The parameterization and sampling process (Algorithm 2) are clear.
- **Optimization**: Bayesian Optimization (hyperopt) is chosen to handle non-differentiable losses. The optimization variables (7 total) and their reasonable default bounds are provided. The authors note that only the iteration budget \(T\) must be set in practice.
- **Smoothing**: A practical smoothing transformation (Algorithm 4) is introduced to improve the optimization landscape, with empirical evidence (Appendix A.6) that it helps for non-smooth uncertainty distributions.

**Critical concerns:**
1. **Dependence on the oracle**: The method assumes the availability of a well-calibrated, accurate probabilistic oracle. The paper does not discuss robustness to poor oracle calibration or accuracy. What happens if the oracle is biased or poorly calibrated? This could affect the uncertainty scores and thus the learned distribution.
2. **Information loss**: Projecting to a 1D uncertainty score may discard important information about instance difficulty. While the justification (proximity to decision boundaries) is intuitive, it is not theoretically guaranteed that uncertainty scores alone are sufficient for optimal re-weighting. Some discussion or ablation on this point would be helpful.
3. **Computational complexity**: The BO process with \(T=3000\) iterations is expensive, as acknowledged in Section 5. The preliminary results with BoTorch suggest speedups, but these are not integrated into the main experiments. The runtime limitation is a significant practical barrier.
4. **Smoothing heuristic**: The smoothing transformation (Algorithm 4) is ad-hoc. While it appears helpful, its impact on the optimization dynamics and potential introduction of bias are not thoroughly analyzed. A more principled approach (e.g., kernel smoothing) might be considered.
5. **Hyperparameter interactions**: Although the authors state that only \(T\) needs to be set, the box constraints for the optimization variables still require user specification (even if defaults are provided). The sensitivity of results to these bounds is not explored.

### Experiments & Results
The experimental validation is extensive, covering 13 datasets, two interpretable models (LPM and DT), two oracles (GBM and RF), and multiple model sizes. The use of F1-macro accounts for class imbalance, and statistical testing (t-test on validation, Wilcoxon signed-rank on test) adds rigor.

**Key strengths:**
- Results (Table 1, extended in Table 4) show substantial relative improvements, especially for small model sizes (often >100%). The trend of diminishing gains with increasing size is intuitive and well-explained.
- The comparison against the density tree approach (Ghose & Ravindran, 2020) shows clear improvements (Table 5), establishing advance over prior work.
- The competitiveness experiments (cluster explanation and prototype-based classification) demonstrate that the method can elevate simple techniques (CART, RBFN) to be competitive with recent specialized algorithms (IMM, SNC). This is a compelling result.
- The experiments with multivariate model sizes (Appendix A.12) and different feature spaces (Appendix A.13) showcase flexibility and are convincing.

**Critical concerns:**
1. **Missing absolute performance**: The results report only relative improvements (\(\delta F1\)). Without baseline F1 scores, it is difficult to assess the practical significance. A large percentage improvement on a very low baseline may still yield an unacceptably low absolute F1. The authors should include absolute F1 scores (at least for key cases) to contextualize the gains.
2. **Comparison to standard regularisation baselines**: The baseline is the model at iteration 0 (which is essentially the standard training without distribution learning). However, it would be important to compare against standard regularisation techniques (e.g., L1 for linear models, cost-complexity pruning for trees) that also control model size. The paper does not establish that the gains are beyond what could be achieved by careful tuning of these existing levers.
3. **Oracle selection and cost**: The method requires training an oracle (e.g., GBM or RF) which itself can be computationally expensive and requires tuning. The overall cost (oracle training + BO) is not compared to simply training a more complex interpretable model with hyperparameter tuning. The net benefit in terms of total compute time is unclear.
4. **Statistical significance reporting**: The Wilcoxon signed-rank test results (Figure 5) are provided, but the description of the binning procedure (using Sturges’ rule) is somewhat arbitrary. A more conventional approach (e.g., per-size analysis) might be more straightforward.
5. **Reproducibility**: The code is provided, and datasets are publicly available, which is good. However, the exact versions of libraries and detailed configuration (e.g., hyperopt settings) would be needed for full reproducibility.

### Writing & Clarity
The paper is generally well-structured and clearly written. The figures are informative, and the algorithm descriptions are detailed. Some sections are dense (e.g., the DP mixture details), but the appendix provides additional explanations. There are minor formatting artifacts from PDF extraction (e.g., broken formatting in Table 4, stray characters like "~~"), but these do not impede understanding. The writing meets ICLR standards.

### Limitations & Broader Impact
- **Limitations**: The runtime issue is explicitly discussed in Section 5, and potential mitigation (faster BO) is noted. However, other limitations are understated:
    - The method is currently limited to classification tasks; extension to regression is not discussed.
    - The dependence on a well-calibrated oracle is a critical assumption that is not examined.
    - The smoothing transformation, while helpful, is heuristic and may not generalize.
    - The BO process can get stuck in local optima, and the convergence properties are not analyzed.
- **Broader Impact**: The ethics statement is standard. The work has positive societal impact by enabling more accurate interpretable models in sensitive domains. No negative impacts are identified, which is reasonable.

## Overall Assessment
The paper presents a novel, general method for improving the accuracy of small interpretable models by learning a training distribution via Bayesian Optimization over uncertainty scores from an oracle. The core idea is creative and well-executed. The empirical validation is extensive, demonstrating significant improvements across diverse datasets, model families, and tasks. The flexibility to handle different feature spaces and multivariate size constraints is particularly compelling.

However, the paper has notable weaknesses: the computational cost is high, the reliance on a good oracle is not critically examined, and comparisons to standard regularisation baselines are missing. Additionally, the reporting of only relative improvements obscures the absolute performance gains.

Given ICLR's emphasis on novelty, technical soundness, and empirical rigor, the contribution is substantial but would be strengthened by addressing these concerns. With revisions (especially adding absolute performance, comparison to regularisation baselines, and deeper analysis of oracle dependence), this paper would be a strong candidate for acceptance.

**Overall recommendation**: Borderline accept / weak accept. The paper presents a valuable idea with solid empirical support, but the concerns above should be addressed in a revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a model-agnostic technique for improving the accuracy of small, interpretable models (e.g., decision trees, linear models) by learning a biased training distribution. The core idea is to project data into a one-dimensional space using an uncertainty oracle (e.g., a GBM or RF), model the distribution of uncertainty scores with a Dirichlet Process mixture of Beta distributions, and optimize the sampling distribution via Bayesian Optimization to maximize validation accuracy. The method claims to enhance accuracy significantly for constrained model sizes, works with non-differentiable losses, and requires only one hyperparameter (optimization budget).

### Strengths
1. **Extensive Empirical Validation**: The paper provides thorough experiments across 13 datasets, two interpretable model families (LPMs and DTs), and two oracle families (GBM and RF), demonstrating consistent improvements in F1-macro scores, especially for small model sizes (e.g., improvements often >100% for very small models). Statistical tests (Wilcoxon signed-rank) support the significance of gains.
2. **Model and Size Agnosticism**: The method is flexible—it can handle different interpretable model types and various notions of “size” (depth, non-zero coefficients, even multivariate sizes like depth and number of trees in GBMs). This is demonstrated in experiments and appendices.
3. **Practical Convenience**: The authors emphasize that only one hyperparameter (the optimization budget \(T\)) needs to be set, making the approach user-friendly. It also works with non-differentiable losses, broadening applicability.
4. **Beyond Standard Baselines**: The technique outperforms the closest prior work (density trees) and shows competitiveness with specialized methods in tasks like cluster explanation and prototype-based classification, elevating older methods (e.g., CART) to match recent algorithms.
5. **Innovative Extensions**: The method is shown to work even when the oracle and interpretable model use different feature spaces (e.g., GRU on character sequences for a DT on n-grams), expanding potential use cases.

### Weaknesses
1. **Limited Theoretical Foundation**: The paper lacks a rigorous theoretical justification for why learning the training distribution via uncertainty scores should improve accuracy for size-constrained models. While intuitive, the connection between uncertainty-based sampling and model performance is not deeply analyzed.
2. **Heavy Reliance on Heuristics and Engineering**: Several components are motivated empirically rather than theoretically: the choice of margin uncertainty, the smoothing transformation (Algorithm 4), and the fixed scaling factor for Beta priors. Their necessity and impact are not thoroughly ablated.
3. **Computational Cost**: Despite claims of practicality, the runtime is high (e.g., ~1 hour per configuration with hyperopt). The suggested mitigation using BoTorch (2 minutes) is preliminary and only tested on two datasets; more evidence is needed to confirm scalability.
4. **Incomplete Analysis of Oracle Influence**: The paper treats the oracle as a mere tool for dimensionality reduction, but its quality and calibration likely affect results. No sensitivity analysis is performed—what happens with a poor oracle? How does oracle accuracy correlate with improvements?
5. **Ambiguous Novelty Claim**: The core idea of learning a training distribution to improve small models builds directly on Ghose & Ravindran (2020). The extension using an uncertainty oracle and Bayesian Optimization is non-trivial but may be perceived as an incremental engineering improvement rather than a conceptual leap.

### Novelty & Significance
**Novelty**: Moderate. The paper builds on prior work that learns training distributions for interpretability. The key novelties are: (1) using an uncertainty oracle for 1D projection, (2) employing Bayesian Optimization to handle non-differentiable losses, and (3) demonstrating applicability across differing feature spaces. However, the foundational idea (distribution learning for accuracy-size trade-offs) is not new.
**Significance**: The work is practically significant for interpretable ML, offering a versatile tool to boost performance of simple models. It could impact applications where model size is critical (e.g., edge devices, human-in-the-loop systems). The empirical results are convincing, and the code is provided for reproducibility.

### Suggestions for Improvement
1. **Strengthen Theoretical Grounding**: Provide a theoretical analysis linking the uncertainty-based sampling distribution to the generalization error of size-constrained models. Even a simplified analysis (e.g., under margin theory) would add substantial value.
2. **Conduct Ablation Studies**: Systematically ablate the impact of key design choices: the uncertainty metric (margin vs. entropy), the smoothing transformation, the Dirichlet Process parameterization, and the choice of oracle. This would clarify which components are essential.
3. **Expand Runtime Evaluation**: Provide a more comprehensive evaluation of runtime improvements with BoTorch (or other optimizers) across all datasets and model types. Discuss trade-offs between optimization budget, noise handling, and final performance.
4. **Analyze Oracle Sensitivity**: Explore how oracle accuracy, calibration, and architecture affect results. Could a simple model (e.g., logistic regression) serve as an oracle? What are the minimal requirements?
5. **Improve Presentation and Clarity**: The main paper is clear, but many details are relegated to the appendix. Integrate critical details (e.g., smoothing, parameter bounds) into the main text to improve readability. Also, clarify the limitations section to explicitly discuss the heuristic nature of some components.
6. **Compare to More Baselines**: While density trees and specialized methods are compared, consider adding comparisons to other methods for improving small models, such as distillation, pruning, or regularization techniques, to better situate the contribution.

**ICLR Suitability**: The paper aligns with ICLR’s focus on empirical rigor and novel methods. However, the lack of theoretical insight and incremental nature may hinder acceptance. The extensive experiments and practical utility are strengths, but the authors should address the weaknesses above to increase chances.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study on the uncertainty oracle**: Compare with simple 1D projections (e.g., random, PCA) or direct uncertainty-based sampling (e.g., sample proportionally to uncertainty). Without this, it's unclear if the complex Dirichlet Process and Bayesian optimization are necessary vs. a simple heuristic, undermining the novelty claim.
2. **Comparison with state-of-the-art methods for learning small interpretable models**: The paper only compares to a specific predecessor (density trees) and task-specific techniques. To establish broad impact, compare against recent general methods for small interpretable models (e.g., Bayesian rule lists, optimal decision trees, L1-regularized logistic regression with advanced solvers).
3. **Experiments with more diverse interpretable model families**: The paper only tests Linear Probability Models and Decision Trees. To substantiate the claim of model-agnosticism, include other families like rule sets, sparse logistic regression, or interpretable neural networks.
4. **Sensitivity analysis of the iteration budget \(T\)**: The paper claims only one hyperparameter (\(T\)) must be set. Show how performance and runtime scale with \(T\); if performance is highly sensitive, the claim of convenience is weakened.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the learned sampling distributions**: Visualize or quantitatively characterize the learned Beta mixture distributions across datasets. Do they consistently emphasize high-uncertainty regions? Without this, it's unclear what the method actually learns and whether it aligns with the intended mechanism.
2. **Statistical significance of improvements across datasets and sizes**: The paper reports average improvements and uses a t-test for model selection, but does not provide confidence intervals or per-dataset statistical tests for the improvements. This is needed to trust that gains are robust and not due to random variation.
3. **Failure mode analysis**: Investigate cases where the method yields negative improvements (∼1.88% of non-null cases). Understanding when and why it fails is critical for assessing its reliability in practice.
4. **Computational cost analysis**: The paper mentions runtime as a limitation but provides no analysis of scaling with dataset size, model complexity, or iterations. A thorough complexity analysis is needed to judge practicality.

### Visualizations & Case Studies
1. **Visualization of decision boundaries on real datasets**: Beyond the toy example, show how the learned sampling changes the decision boundaries of interpretable models (e.g., DTs) on at least one real dataset. This would visually confirm that the method improves models by focusing on uncertain regions.
2. **Case study on a high-impact domain dataset**: The paper motivates with healthcare, law, etc., but uses only standard UCI/LIBSVM datasets. Include a case study on a real-world dataset from such a domain (e.g., medical diagnosis) to demonstrate improved accuracy and interpretability in a meaningful context.

### Obvious Next Steps
1. **Include a strong, simple baseline that uses oracle uncertainty directly** (e.g., sampling instances with probability proportional to uncertainty) in the main experiments. This is necessary to justify the complexity of the proposed optimization framework.
2. **Integrate the faster Bayesian optimization approach (BoTorch) into the main experiments** instead of only a side note in the appendix. This would address the runtime limitation and strengthen the practicality claim.
3. **Evaluate interpretability beyond size**: The paper equates small size with interpretability, but no user study or quantitative interpretability metrics (e.g., simulatability, faithfulness) are provided. At minimum, discuss how the method affects interpretability metrics other than size.

# Final Consolidated Review
## Summary
This paper introduces a technique to improve the accuracy of small, interpretable models (e.g., decision trees, linear models) by learning a biased training distribution. The distribution is modeled as a Dirichlet Process mixture of Beta distributions over the uncertainty scores of a separate probabilistic oracle, and its parameters are optimized via Bayesian Optimization to handle non-differentiable losses. Extensive experiments show significant relative improvements in F1-macro scores, especially for very small model sizes, across diverse datasets, model families, and even when the oracle and interpretable model use different feature spaces.

## Strengths
- **Empirically robust and extensive validation**: The method is evaluated on 13 datasets, two interpretable model families (Linear Probability Models and Decision Trees), and two oracle families (Gradient Boosted Machines and Random Forests), with statistical tests confirming significant improvements—often exceeding 100% relative gain for very small models.
- **Model-agnostic and flexible**: The technique works with different interpretable models and varying notions of model size (e.g., tree depth, non-zero coefficients, multivariate sizes like depth and number of trees in GBMs) and is demonstrated to function even when the oracle and target model operate on different feature representations (e.g., a GRU on character sequences for a decision tree on n-grams).
- **Competitive with specialized methods**: The approach outperforms its direct predecessor (density trees) and, when applied to older methods like CART or RBF networks, elevates their performance to be competitive with recent task-specific algorithms for cluster explanation and prototype-based classification.

## Weaknesses
- **Missing comparisons to simple baselines and standard regularization**: The paper does not compare against straightforward heuristics like sampling instances proportionally to uncertainty without the full optimization framework, nor against standard regularization techniques (e.g., L1 for linear models, pruning for trees) that also control model size. This leaves unclear whether the complexity of the proposed method is necessary for the gains observed.
- **Dependence on oracle quality without sensitivity analysis**: The method relies on a well-calibrated, accurate oracle for uncertainty estimates, but no analysis examines how performance degrades with a poorly calibrated or low-accuracy oracle. This is a practical concern for deployment.
- **Information loss from one-dimensional projection**: While projecting instances to a single uncertainty dimension is intuitive, the paper provides no evidence that this projection retains all information needed for optimal re-weighting. An ablation comparing to using the full feature space (or other projections) would strengthen the claim.
- **High computational cost and preliminary mitigation**: The primary experiments use a Bayesian Optimization setup (hyperopt) requiring up to an hour per configuration. Although the authors suggest faster optimizers (BoTorch) can reduce runtime to ~2 minutes, this is only demonstrated on two datasets and not integrated into the main evaluation, leaving runtime a practical limitation.

## Nice-to-Haves
- Ablation studies on the choice of uncertainty metric, the smoothing transformation, and the Dirichlet Process parameterization.
- Visualization or quantitative characterization of the learned sampling distributions to better understand what the method optimizes for.
- Analysis of failure cases (the ~1.88% of runs with negative improvements) to identify when the method may underperform.
- More comprehensive evaluation of faster Bayesian Optimization methods across all datasets and model types.
- Evaluation of interpretability beyond model size (e.g., simulatability, faithfulness), though the paper rightly focuses on size as a primary interpretability proxy.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Include a baseline that samples training instances proportionally to oracle uncertainty (or other simple schemes) to isolate the benefit of the full optimization framework.
- Compare against standard regularization techniques (e.g., L1 for linear models, cost-complexity pruning for trees) that are commonly used to achieve small model sizes.
- Conduct a sensitivity analysis of the method to oracle quality, e.g., by using oracles of varying accuracy and calibration.
- Integrate the faster Bayesian Optimization approach (e.g., BoTorch) into the main experiments and report runtime comparisons across all configurations.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 0.0]
Average score: 3.3
Binary outcome: Reject
