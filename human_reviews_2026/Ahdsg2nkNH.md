# Multilevel Control Functional

- Decision: Accept (Poster)
- Scores: 8, 8, 8

## Abstract
Control variates are variance reduction techniques for Monte Carlo estimators. They play a critical role in improving Monte Carlo estimators in scientific and machine learning applications that involve computationally expensive integrals. We introduce \emph{multilevel control functionals} (MLCFs), a novel and widely applicable extension of control variates that combines non-parametric Stein-based control variates with multi-fidelity methods. We show that when the integrand and the density are smooth, and when the dimensionality is not very high, MLCFs enjoy a faster convergence rate. We provide both theoretical analysis and empirical assessments on differential equation examples, including Bayesian inference for ecological models, to demonstrate the effectiveness of our proposed approach. Furthermore, we extend MLCFs for variational inference, and demonstrate improved performance empirically through Bayesian neural network examples.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel variance-reduction method for computing expectations using Markov chains and variational inference. The proposed approach, Multi-Level Control Functionals (MLCF), integrates multi-fidelity modeling with non-parametric control variates based on the kernelized Stein discrepancy. The authors prove that the estimator is unbiased, establish theoretical bounds on its variance, and derive the level-wise optimal sample allocation. The method is empirically validated on inference tasks involving dynamical systems and Bayesian neural networks.

### Strengths
- The paper presents a novel and well-motivated approach to variance reduction, a key challenge in Bayesian inference.  
- The method is theoretically rigorous, providing explicit variance bounds and optimal sample allocation formulas.  
- The empirical results are convincing, demonstrating that MLCF consistently yields lower-variance estimators across the tested scenarios.

### Weaknesses
- The impact of varying the fidelity level $L$ is not explored, making it difficult to assess its practical importance.  
  *(See related questions below.)*

### Minor
- In the BNN example, it is not immediately clear from the main text that variance reduction is applied to  
  the gradient estimator of the ELBO. This is mentioned in the appendix; consider moving or referencing it earlier for ease of reading.  
  (A single clarifying sentence would suffice.)

### Questions
- Please provide the converged training ELBO and test log-likelihood values corresponding to Figures 5 and 6.  
- At what dimensionality does the integrand cease to be considered “moderate”?  
- What can be said about the relationship between the magnitude of $L$ and the extent of variance reduction?  
- How does $L$ affect computational cost?  
- How should practitioners choose or tune $L$ in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Multilevel Control Functionals (MLCFs), a novel variance reduction technique for Monte Carlo estimators. MLCFs extend traditional control variates by combining two powerful ideas: non-parametric Stein-based control variates and multi-fidelity methods . The authors provide theoretical analysis showing that MLCFs achieve faster convergence rates when integrands and densities are smooth and dimensionality is moderate. They validate MLCFs empirically on differential equation (DE) tasks (e.g., Bayesian inference for ecological models) and extend the framework to variational inference (VI), demonstrating improved performance on Bayesian neural network (BNN) benchmarks.

### Strengths
1.The combination of Stein-based control variates with multi-level methods is innovative.
2. The convergence rate analysis is a key strength, even though I did not validate the details of mathematical theory.
3. The empirical evaluations are well-chosen and cover diverse use cases:

### Weaknesses
1. The abstract and theory section reference MLCFs working when “dimensionality is not very high,” but this is underspecified. What is the upper bound of dimensionality for MLCFs to remain competitive? 
2. Maybe more baseline methods (mentioned in related works) could be included in the experiments.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
his paper proposes Multilevel Control Functionals (MLCFs), a novel variance reduction technique for efficiently estimating computationally expensive integrals. The core idea is an elegant combination of two powerful variance reduction strategies: Multilevel Monte Carlo (MLMC) and non-parametric, Stein-based Control Functionals (CFs).

MLMC methods accelerate estimation by using a telescoping sum of a hierarchy of low-fidelity, cheap approximations to the expensive, high-fidelity integrand. The variance is reduced by estimating the small differences between levels.

This paper's key insight is to treat the MLMC difference terms themselves as integrands that can be further variance-reduced. The authors propose applying a non-parametric Stein-based Control Functional to each level of the MLMC estimator. This results in an estimator for the sum of these variance-reduced differences, which has significantly lower variance.

The authors provide a solid theoretical analysis, including a variance bound that demonstrates a faster convergence rate than standard MLMC, particularly for smooth integrands and densities in low-to-moderate dimensions. They also derive the optimal sample allocation across levels to minimize this variance bound under a fixed computational budget.

Furthermore, the paper extends this framework to variational inference (VI) by proposing the Multilevel Control Functional Re-parameterized Gradient (MLCFRG) estimator. This estimator applies the MLCF idea to the multilevel gradient estimator for the ELBO (MLRG), and a practical, efficient recursive update is provided.

The method's effectiveness is demonstrated empirically on a synthetic example, a boundary-value ODE, a Bayesian inference problem (Lotka-Volterra system) using MCMC samples, and a variational inference problem (Bayesian Neural Network). In all cases, the proposed MLCF and MLCFRG methods outperform their respective baselines (MLMC, CF, MLMCRG).

### Strengths
Novelty and Significance: The paper's main contribution—hybridizing MLMC with Stein-based Control Functionals—is both novel and highly intuitive. It directly addresses a practical and important problem: the high computational cost of integration in many scientific and machine learning applications. This combination is powerful, as it leverages the "divide-and-conquer" strength of MLMC and the non-parametric variance-reducing power of CFs.

Theoretical Soundness: The method is supported by a strong theoretical foundation. The paper provides a clear variance bound that explains why and when MLCF should be effective (dependence on smoothness and dimensionality). The derivation of the optimal sample allocation adds to the method's practical utility.

### Weaknesses
Computational Cost of Control Functionals: The primary weakness, acknowledged by the authors, is the cubic computational cost of inverting the kernel Gram matrix to construct the control functional, where the cost scales with the number of design points. This cost is incurred per level for the standard MLCF estimator and per iteration for the MLCFRG estimator. While the authors argue this is negligible if the integrand is sufficiently expensive, this scaling severely limits the number of points that can be used to build the CF, which in turn limits the achievable variance reduction.

Practical Complexity: The method adds a new layer of complexity compared to the relatively simple MLMC. A user must now select an appropriate kernel (e.g., Mateŕn, SE) and its hyperparameters (e.g., length-scale) for each level. The paper suggests maximizing the marginal likelihood (in the Appendix), but this is a non-trivial, costly optimization problem in itself, adding to the overall computational burden.

### Questions
Hyperparameter Sensitivity: How sensitive is MLCF's performance to the choice of kernel and its hyperparameters? The paper uses several different kernels. How much of the performance gain is attributable to careful, and potentially expensive, hyperparameter tuning via marginal likelihood maximization at each level?

Relation to Multi-Fidelity BQ: How does MLCF compare to other methods that combine multilevel/multi-fidelity approaches with kernel-based methods, such as the multilevel Bayesian Quadrature (Li et al., 2023) you cite? A direct empirical comparison, even on the ODE problem, would be very insightful to position MLCF in the literature.

### Soundness
3

### Presentation
3

### Contribution
3
