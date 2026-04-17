# Revisiting Nonstationary Kernel Design for Multi-Output Gaussian Processes

- Decision: Accept (Poster)
- Scores: 10, 8, 4, 6

## Abstract
Multi-output Gaussian processes (MOGPs) provide a Bayesian framework for modeling non-linear functions with multiple outputs, in which nonstationary kernels are essential for capturing input-dependent variations in observations. However, from a spectral (dual) perspective, existing nonstationary kernels inherit the inflexibility and over-parameterization of their spectral densities due to the restrictive spectral–kernel duality. To overcome this, we establish a generalized spectral–kernel duality that enables fully flexible matrix-valued spectral densities — albeit at the cost of quadratic parameter growth in the number of outputs. To achieve linear scaling while retaining sufficient expressiveness, we propose the multi-output low-rank nonstationary (MO-LRN) kernel: by modeling the spectral density through a low-rank matrix whose rows are independently parameterized by bivariate Gaussian mixtures. Experiments on synthetic and real-world datasets demonstrate that MO-LRN consistently outperforms existing MOGP kernels in regression, missing-data interpolation, and imputation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
I applaud the authors for revisiting multi-task GPs through non-stationary kernels. I fully agree with the paper's motivation. The manuscript is well written and a delightful, helpful read. My comments and questions mainly seek clarification.

### Strengths
- The paper proposes a new, very flexible kernel for multi-task GPs, which is an important area of research
- The manuscript is well-written and comprehensible without too much effort.
- Sound theory.
- A reasonable number of test cases.
- A good choice of competitor methods.

### Weaknesses
- Some extra implementation details would be good. I found it hard to reproduce the kernel from Definition 1. For instance, one could describe typical bounds for the hyperparameter.

### Questions
- Are there restrictions or bounds on the m_ij, z_ij, S_ij^(q), and other hyperparameters? 
- What is the reasoning behind the choice of competitor methods?  
- Why is there no comparison to a baseline coregionalization?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a multi-output version of the next-gen SM kernel from Yang (2025).

### Strengths
* The proposed kernel gives an elegant solution to the problem.
* The method has good results.

### Weaknesses
* The appendix contains the hyperparameter values, but only for MO-LRN, not for the other kernels.
  * The appendix also specifies how the kernel was optimized for MO-LRN (500 iterations of Adam), but not for the other kernels. Is the same procedure used for all kernels?
  * It is not clear how the hyperparameters like learning rate were chosen.
* The source code is not released (yet)

### Questions
* Table 2: "$I$ denotes the index in the LMC summation over latent processes, with $I < V$."
   Should this be $I \le V$?
 * Table 2: what does "Expressive kernel" mean?
 * Eq (1) is missing parentheses around the 4 terms in the integral.
 * "When ω_1 = ω_2, this theorem reduces to Bochner’s theorem."
    ω are not parameters of the theorem, so this statement makes no sense. Do you mean to say that the theorem reduces to Bochner's theorem when u(ω_1,ω_2)=δ(ω_1-ω_2)u(ω), so when u is non-zero only when ω_1 = ω_2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript presents a novel formulation of a multi-output non-stationary spectral kernel. A low rank approximation is adopted in order to maintain a manageable number of trainable parameters. The model is benchmarked against synthetic and real world data, and shown to be competitive for point prediction tasks. The approach also demonstrates favourable computational efficiency.

### Strengths
The manuscript is well written and clearly presented

The approach is a novel solution to tackling a challenging problem in the literature.

The experimental results are clearly presented along with a suitably broad selection of benchmarks and a good set of baselines.

### Weaknesses
My primary concern with this work is the absence of quantitative probabilistic performance metrics. The metrics shown, such as MAE and RMSE, test only the quality of the point predictions of the model. But here we are working with probabilistic models, so without metrics such as NLPD to verify the uncertainty calibration, it is difficult to recommend acceptance. This concern would hold for any probabilistic model, but is particularly acute here where we have many degrees of freedom (due to non-stationarity and multi-output), as it becomes more challenging to regulate the full posterior.

It is unclear what we are sacrificing when moving to the low rank setting. 

Scalability remains a significant concern that is only briefly discussed in the Appendix. For multi-output datasets that are long enough to reveal detectable non-stationary features, we quickly hit the upper limits of Figure 4, where the computational cost becomes prohibitive.

### Questions
Under what conditions should we expect this formalism to fail? It might be helpful to construct a somewhat adversarial synthetic example where we know the model cannot reproduce the cross-covariances, and see how it performs. 

As pointed out in 4.1, spectral kernels tend to be highly sensitive to the initialisation strategy, but it was not clear to me what initialisation is used for MO-LRN? And how should practitioners decide on a suitable configuration such as Q for a given dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Multi-Output Low-Rank Nonstationary  kernel for multi-output Gaussian processes. By introducing a generalized spectral–kernel duality and modeling the matrix-valued spectral density via a low-rank Gaussian mixture, MO-LRN achieves linear scalability and superior expressiveness, outperforming existing MOGP kernels on regression and imputation benchmarks.

### Strengths
The paper’s strengths lie in its clear theoretical and practical contributions.

1 It establishes a new spectral–kernel duality that removes conventional restrictions and enables fully flexible matrix-valued spectral densities for multi-output Gaussian processes.

2 This paper introduces the MO-LRN kernel, a parameter-efficient yet expressive nonstationary kernel that reduces parameter growth from quadratic to linear through a low-rank spectral density with independent Gaussian-mixture factors. I really like this idea.

3 Extensive experiments validate  its effectiveness and scalability in modeling complex multi-output, nonstationary processes.

### Weaknesses
I really enjoy reading this paper but I am not an expert in GP. From a general perspective, I cannot find obvious flaw in this paper. I will read other reviewers' comments along with authors' feedback and ajust my score.

### Questions
I do not have quetions.

### Soundness
3

### Presentation
4

### Contribution
3
