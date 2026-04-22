# Robust Mixture Models for Algorithmic Fairness Under Latent Heterogeneity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Standard machine learning models optimized for average performance often fail on minority subgroups and lack robustness to distribution shifts. This challenge worsens when subgroups are latent and affected by complex interactions among continuous and discrete features. We introduce ROME (RObust Mixture Ensemble), a framework that learns latent group structure from data while optimizing for worst-group performance. ROME employs two approaches: an Expectation-Maximization algorithm for linear models and a neural Mixture-of-Experts for nonlinear settings. Through simulations and experiments on real-world datasets, we demonstrate that ROME significantly improves algorithmic fairness compared to standard methods while maintaining competitive average performance. Importantly, our method requires no predefined group labels, making it practical when sources of disparities are unknown or evolving.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ROME (Robust Mixture Ensemble), a novel framework that addresses algorithmic fairness by discovering and optimizing for *latent* subgroups, rather than relying on predefined demographic categories. By integrating Distributionally Robust Optimization (DRO), the method explicitly optimizes for worst-group performance, providing a more robust and practical approach to equity in machine learning models.

### Strengths
1. Dual-Model Framework: The paper provides two flexible implementations: ROME-EM for linear models and ROME-MoE for non-linear, neural network settings.
2. Closed-Form Solution: The ROME-EM approach offers a closed-form solution for the optimal robust predictor in linear contexts.
3. Non-Linear Adaptation: The ROME-MoE model effectively adapts the fairness framework to complex, non-linear problems using a Mixture-of-Experts architecture.
4. Comprehensive Validation: The methodology is validated using both synthetic data (confirming its ability to recover true group parameters) and three real-world datasets (demonstrating practical utility).

### Weaknesses
Although the paper is interesting and its contributions are relevant, it has several weak points that should be addressed prior to publication. Specifically:

1. Poor Presentation: The paper's notation is confusing, particularly in Section 2.3. For example, $\mathcal{C}$ is introduced but never used, the exact definition of $Q_X$ is unclear, $\mathbb{P}$ is never introduced, the definition of $\mathcal{H}$ is repeated, and the distinction between $\hat{f}$, $f^*$, and $f^{(k)}$ is misleading. This is compounded by other symbols that are either not introduced or introduced but never used. Another example, outside this section, is the definition of $\mathbf{X}$ and $X_i$: the former is never introduced, and the latter seems to be a vector (features of sample $i$) rather than a matrix. The overall flow and clarity of the paper should be thoroughly revised.
2. Minor Contribution of ROME-MoE: The differences between Vanilla-MoE and ROME-MoE appear to lie just in the loss function's use of robust optimization (accounting for worst-case performance). While this is a contribution, it is a minor one, and its relationship to the linear case is also tenuous. It would be interesting to see an intermediate model where expert weights are learned via some version of the linear model, while still using neural networks for the experts to capture non-linearities.
3. Overstated "Guarantees": The authors mention “theoretical guarantees” and “convergence guarantees” for the linear model in Section 3, but these are not provided. A closed-form solution for one part of the algorithm is not a theoretical guarantee, and convergence is never proven. The authors should soften these claims regarding the linear model.
4. Lack of Clarity (Linear Model): It is unclear why hard prior group assignments are needed, only to then compute a “membership probability” (is this similar to a soft assignment?). It is also unclear if these probabilities are computed using Equation (1), as the notation changes (introducing $\gamma$). Furthermore, the group assignments $z_i$ mentioned in the initialization should be $z_{ij}$ to be consistent with the notation in Section 2.1.
5. Limited SOTA Comparisons: The comparisons are limited to Vanilla-MoE (where differences are minor, as discussed) and standard MLPs. The authors should strengthen the numerical comparisons against other SOTA methods, even if they are not optimized for fairness or do not use group-specific architectures.
6. Weak Experimental Design: The authors state that “ground-truth groups are unknown in real data,” but datasets exist where sensitive attributes (e.g., race) provide clear group divisions. Using such data would offer a valuable benchmark, as the number of groups and assignments would be known (likely removing the need for a gating mechanism) and would better isolate the performance gains from the robust optimization. Furthermore, the linear model's performance is not evaluated on any real-world data. If space is a concern, the figures displaying parameter estimates are unnecessary when MSE is already reported.
7. Missing Fairness Metrics: While the structural fairness approach is valid, the paper would benefit from comparing its predictions against baselines using established fairness metrics from the literature.
8. Unclear Synthetic Setup: For the synthetic experiment, "misspecified" starting conditions are created by setting 50% of observations to incorrect groups. It is unclear if this refers to the initial group assignments for parameter estimation. If so, it's not obvious why this method was chosen over a standard uniform random assignment.
9 Robustness to G: The paper mentions that the number of groups (G) can be fine-tuned using information criteria, but it never analyzes the algorithm's robustness to an incorrect estimation of G.

Minor Issues:

* In the E-step, the term "posterior responsibilities" is used. Is this standard, or should it be "posterior probabilities"?
* In Algorithm 1, if the gating network $g$ already outputs a probability distribution, the `Softmax` call in lines 9 and 11 appears redundant. Also, in line 13, estimates for groups whose coefficient is not significant are not needed, meaning a computational overhead for the algorithm.
* The authors mention generating “covariates” in the synthetic experiment, and it is unclear what this refers to.
* In Table 2’s caption, "MSE" is mentioned, while the table reports $R^2$ values.

### Questions
See weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ROME (Robust Mixture Ensemble) to improve algorithmic fairness when harmful disparities arise in latent subgroups rather than predefined demographics. It has two versions: ROME-EM, a finite-mixture linear model trained via EM with a post-hoc distributionally robust (DRO) closed-form aggregation over group predictors; and ROME-MoE, a neural mixture-of-experts that routes by a gating network (which may use sensitive attributes) while restricting experts to non-sensitive features and training with a worst-group objective. The performance of this method is showcased via simulation and three real-world datasets. The approach requires no predefined group labels but does rely on sensitive attributes during training.

### Strengths
Fairness with latent structure is important and underserved. The paper argues that discretizing continuous S or predefining groups is limiting and aims to build a model to enforce algorithmic fairness in the presence of latent subgroups. The paper is easy to read and well-structured.

### Weaknesses
1. **Theoretical guarantees and fairness notion.** The paper’s theoretical footing, in particular regarding algorithmic fairness, is unclear. It is not specified which formal criterion of algorithmic fairness (e.g., Demographic Parity, Equalized Odds, etc.) is being optimized here. Can the authors provide finite-sample guarantees in a simple linear setting to anchor the claims?

2. **Sensitivity to the number of groups ($G$).** How dependent is performance on knowing or correctly selecting $G$? Some analysis of robustness to misspecification and practical guidance for choosing $G$ would be helpful.

3. **Use of sensitive attributes in gating.** Allowing $S$ in the gating network induces an indirect $S \rightarrow$ prediction pathway via routing, even if experts do not observe $S$. Could this, in practice, violate algorithmic fairness?

4. **Empirical evaluation.**  
   - **Baselines.** The current baselines are weak: training on non-sensitive features $A$ alone is known not to ensure fairness when $S$ correlates with $A$. Please include stronger, established fair-learning baselines for a meaningful comparison.  
   - **Effect sizes.** Reported improvements over a simple MLP baseline appear small. While they may be significant from the standard error point of view, their practical significance is minimal, especially given the training cost.

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose a Robust Mixture Ensemble (ROME) framework to address the problem of group fairness.  
Built upon the EM and Mixture-of-Experts (MoE) algorithmic paradigms, the proposed method can automatically identify latent intersectional subgroups within data and employs DRO to minimize the worst-case performance, thereby enhancing fairness.  
Experimental results on both synthetic and real-world tabular datasets validate the effectiveness of the proposed approach.

### Strengths
- The consideration of intersectional effects arising from combinations of sensitive attributes is both interesting and important.  
- The proposed EM- and MoE-based DRO method is well aligned with the formulated fairness problem and theoretically grounded.  
- The paper is overall well-organized and easy to follow, with a clear problem formulation and experimental structure.

### Weaknesses
- Lack of experimental details.

  For example, the paper does not clearly describe the process of synthetic data generation, nor does it provide sufficient information about the neural network architecture used for the gating network and predictors. 

- Fairness metric and definition are insufficiently discussed.

  The performance comparison mainly focuses on subgroup differences under specific evaluation metrics (e.g., MSE), but the scope or interpretation of fairness is not clearly defined.  

- Presentation could be further refined.

  Certain explanations lack precision or consistency; please refer to the questions below for clarification suggestions.  

- Theoretical analysis needs stronger connections.

  While the theoretical results mainly build upon prior work [1], it would be valuable for the authors to further explain how these existing results specifically motivate or strengthen the current method.

> [1] Distributionally Robust Machine Learning with Multi-Source Data.

### Questions
- What is the definition of the outcome prediction model mentioned in line 077?  
  What is the difference between $\mathbf{S}\_{i,\text{mem}}$ and $\mathbf{S}_{i,\text{out}}$?  
  How are these two sets of sensitive attributes determined in practice?

- For ROME-EM and ROME-MoE, how sensitive is the method to initialization?  
  How is the number of groups G selected?  
  What does “sufficiently large” mean for $n_j$ in the context of convergence?

- The definition of $\mathcal{H}$ appears to be inaccurate.  
  $\mathbf{v}$ should belong to a $(G-1)$-dimensional probability simplex $\Delta^{G-1}$.

- Could the authors further explain why the constraint set adopted in Eq. (7) clarifies or improves the ROME-EM formulation?

- Since subgroup structure is often subtle or latent, how does the method determine which attributes are non-fairness-related (i.e., can be safely used in predictors)?

- Are there any detailed analyses about the intersectional unfairness mitigation or detection?

### Soundness
3

### Presentation
2

### Contribution
2
