# Uncertainty-Aware Diagnostics for Physics-Informed Machine Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Physics-informed machine learning (PIML) integrates prior physical information, often in the form of differential equation constraints, into the process of fitting ML models to physical data. Popular PIML approaches, including neural operators, physics-informed neural networks, and neural ordinary differential equations, are typically fit to objectives that simultaneously include both data and physical constraints. However, the multi-objective nature of this approach creates ambiguity in the measurement of model quality. This is related to a poor understanding of epistemic uncertainty, and it can lead to surprising failure modes, even when existing metrics suggest strong fits. Working within a Gaussian process regression framework, we introduce the Physics-Informed Log Evidence (PILE) score. Bypassing the ambiguities of test losses, the PILE score is a single, uncertainty-aware metric that provides a selection principle for hyperparameters of a physics-informed model. We show that PILE minimization yields excellent choices for a wide variety of model parameters, including kernel bandwidth, least squares regularization weights, and even kernel function selection. We also show that, prior to data acquisition, a special data-free case of the PILE score identifies a-priori kernel choices that are "well adapted" to a given PDE. Beyond the kernel setting, we anticipate that the PILE score can be extended to PIML at large, and we outline approaches to do so.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Physics-Informed Log Evidence (PILE), a novel, uncertainty-aware model selection criterion for Physics-Informed Machine Learning (PIML). The work is situated within the Physics-Informed Kernel Learning (PIKL) framework, which uses Gaussian Processes (GPs) to solve linear PDEs. The key insight is to use the Bayesian marginal likelihood (or Bayes free energy) to navigate the multi-objective trade-off between data fidelity and physics constraints—a common pain point in PIML. The authors demonstrate that minimizing the PILE score effectively selects hyperparameters (e.g., kernel bandwidth, regularization weights) and even the kernel function itself. A particularly innovative contribution is the "data-free" PILE, which connects to Fredholm determinants and allows for a priori kernel selection suited to a given PDE, before any data is observed. Empirical results on Poisson and convection equations show that PILE reliably diagnoses and prevents common PIML failure modes.

### Strengths
This paper makes a clear and original contribution by introducing the Physics-Informed Log Evidence (PILE) score as an uncertainty-aware diagnostic and model selection principle for physics-informed machine learning. The work is significant in that it addresses the long-standing ambiguity in balancing data loss and physics loss in PIML, by providing a single, principled metric rooted in Gaussian process regression. A notable strength is the dual perspective: the authors present both a priori diagnostics (via a Fredholm determinant in the data-free case) and a posteriori diagnostics (via PILE after training).

The quality of the work is high. The theoretical development is rigorous, linking PIML to marginal likelihood in GPs, and the empirical case studies (Poisson, wave, and convection PDEs) clearly show how PILE can guide hyperparameter selection, prevent over/under-smoothing, and diagnose failure modes. The exposition is well-structured and accessible, with illustrative figures that support the main claims. Overall, the paper advances originality by integrating uncertainty quantification into model evaluation, and its significance lies in offering a practical tool for robust diagnostics in PIML.

### Weaknesses
The primary limitation is that the methodology is restricted to kernel-based Gaussian process models. While the authors outline possible extensions to neural network–based PIML methods (e.g., PINNs, neural operators), these are not demonstrated. Empirical validation on nonlinear PDEs or higher-dimensional benchmarks would substantially strengthen the generality claims.

A second limitation is computational scalability. The PILE score requires cubic time in the number of data and quadrature points, which the authors acknowledge. They cite approximate methods for marginal likelihood computation, but no experiments demonstrate that these approximations keep PILE practical at scale. More discussion or demonstrations here would increase the impact.

Finally, while the connection to the Fredholm determinant is mathematically elegant and provides a novel perspective on data-free kernel selection, its practical utility is demonstrated only in a single anisotropic kernel case study. Further evidence that this insight generalizes would help substantiate its significance.

### Questions
How does the computational cost of PILE scale with both the dimension of the PDE domain 𝑑 and the number of data/quadrature points What are the practical limits for its application in higher-dimensional or large-scale problems?

The extension to neural networks via the NTK is mentioned. Could you elaborate on the practical challenges of computing or approximating the PILE score in this setting, given the known limitations of the NTK approximation in finite-width networks?

How sensitive is the PILE score to the choice and quality of the quadrature scheme used to approximate the physics loss? Would different quadrature methods or levels of accuracy substantially change the diagnostic outcome?

In the data-free kernel selection, the Fredholm determinant is used. Are there intuitive interpretations of its value that could help a practitioner understand why one kernel is “better adapted” to a PDE than another?

Is code available to reproduce the experiments and facilitate the adoption of the PILE score by the community?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a metric, named PILE, to evaluation principle for hyperparameters for physics informed models. The metric is for kernel learning only.

### Strengths
1. The paper addresses an important topic for incorporating the knowledge into data-driven learning. 
2. Thorough theoretical treatment of the problem.

### Weaknesses
1. Applicable to kernel learning only. It's still an open problem whether it can be extended to other ML techniques especially neural networks. Although kernel learning is highly capable, there are still a large selection of NN-based ML methods which would greatly benefit from physics knowledge.
2. The organization and exposition of the paper can be further improved. It is good to be mathematically rigorous, the exposition can be improved to give the readers more intuition, instead of piles of math equations.

### Questions
1. I have concerns over the phrase "multi-objective nature of the approach" of balancing both the data loss and physical constraints. Although it is true that the two components may draw the overall loss to different ends of the loss function spectrum, the ultimate goal is to train a model most generalizable. In my view, the multi-objectiveness is only the superficial aspect of the optimization. 
2. It appears that the metric is only applicable to physical systems modeled by PDEs, hence the background information in section 2.1. Would the method applicable to other systems, such as differential-algebraic equations? 
3. Issue with Fig 1, although the general trend of three component is very clear, the magnitude as shown in the y-axis is very small. Would the numerical noise affect the solution?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the problem of measuring the model quality in the case of physics-informed machine learning (PIML). Since PIML companies data and physical constraints (like differential equations) to train models, this leads to multi-objective approach which is hard to evaluate. Current methods can produce models that look good by standard metrics but still fail in surprising ways due to poor understanding of epistemic uncertainty. The authors work within a Gaussian process regression framework and introduce the Physics-Informed Log Evidence (PILE) score. The PILE score provides a single, uncertainty-aware metric that bypasses the ambiguities of traditional test losses. The paper addresses a critical gap in PIML methodology.

### Strengths
- The multi-objective evaluation issue is an important problem in PIML that practitioners struggle with. Having a principled diagnostic is valuable.
- A single metric for hyperparameter selection is very usable (as opposed to juggling multiple competing objectives such as data loss vs. physics loss vs. test error).

### Weaknesses
- The current set-up is limited to Gaussian processes (not major, but it would obviously great to have something model agnostic). 
- This also limits the problems in which PILE would be useful to rather smaller scale problems, as GPs don't scale well. The PILE score itself introduces additional complexity. Hence, it is not fully clear what is practical limitations in terms of the computational cost and up to what degree it would be feasible to use it in practice.

### Questions
- How would PILE score behave when a model completely violates a hard constrain and when it has an ok residual error in a soft constraint? Is is possible to distiguish between hard and soft constraints?  

- How does the computational cost of evaluating the PILE score scale as constraint complexity increases? What are computational limitations? Or in other words, up to what dimensionality is it reasonable to use it (given it's higher computational complexity and cubic complexity of GPs)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes PILE (Physics-Informed Log Evidence), a principled and uncertainty-aware model selection metric for Physics-Informed Machine Learning. Working within a Gaussian Process-based Physics-Informed Kernel Learning framework, the authors show that PILE resolves the core multi-objective tension between data-fit and physics-fit by providing a single score that quantifies model quality, enabling reliable tuning of kernel hyperparameters, regularization weights, and noise levels. They further show that in the absence of data, PILE converges to a Fredholm determinant, which can be used to select kernels a priori that are well-adapted to a given PDE. Through case studies, including a challenging convection PDE known to cause PIML failures, the paper demonstrates that PILE not only identifies when a model or kernel is misspecified, but also guides the choice of kernels and parameters that yield high-quality solutions, thereby offering a robust diagnostic and model-selection tool for PIML.

### Strengths
The paper offers a novel contribution to physics-informed machine learning by introducing PILE, a principled and uncertainty-aware model selection criterion based on Bayesian marginal likelihood. Unlike existing PIML approaches that rely on heuristic loss weighting or manual hyperparameter tuning, this work reframes the problem through Bayesian evidence maximization, providing a single score that jointly captures data-fit, physics-fit, model complexity, and uncertainty calibration. A particularly original aspect is the theoretical result showing that, in the absence of data, the PILE score converges to a Fredholm determinant of a PDE-informed integral operator. This reveals a connection between PIML, Gaussian processes, and spectral operator theory, opening a new line of investigation.

The paper is technically strong, with a rigorous derivation of PILE from the Bayes free energy of the physics-informed GP model. The theory is well supported by clear assumptions and proofs, and the empirical results, though modest in scale, effectively illustrate the practical value of the method. In particular, the convection PDE case study demonstrates that PILE can identify kernel choices that significantly improve solution quality, underscoring the method’s practical utility.

The presentation is generally clear and well organized. The progression from standard GP regression to physics-informed kernel learning and finally to the PILE formulation is logical and accessible for readers with a background in GPs or kernel methods. While some of the operator-theoretic aspects may remain abstract for a broader audience, the core ideas are well communicated and supported by helpful examples.


This work is significant because it directly tackles a key limitation of current PIML methods: the lack of a principled mechanism for model and hyperparameter selection. By enabling both data-informed and data-free kernel selection, the proposed framework has the potential to meaningfully reduce reliance on trial-and-error and improve robustness in scientific ML workflows. The core ideas are sufficiently general that they could influence future developments beyond GP-based methods, including uncertainty-aware neural operators and physics-based deep learning. Overall, the paper makes a meaningful, well-founded, and potentially impactful contribution to the field.

### Weaknesses
While the paper is strong overall, a few areas could be improved to enhance its practical impact. The work is primarily method-driven and introduces a novel and well-motivated framework, and I did not identify major weaknesses in the core methodology or theoretical development. My comments are therefore more about opportunities to further strengthen the empirical validation. The experiments are limited to relatively low-dimensional PDEs with simple settings e.g., the main results focus on a 1D Poisson equation and a 1D convection problem and demonstrating the method on more challenging benchmarks (e.g., 2D/3D heat equation, 2D convection-diffusion equation etc.) would better showcase its broader applicability. In addition, the connection to Fredholm determinants is theoretically elegant but only demonstrated through a single kernel-selection case study. Further analysis of stability and generality across PDE types and kernels would help strengthen confidence in its practical utility.

### Questions
1. In the data-free PILE formulation, the theory requires $m\rightarrow\infty$ for convergence to the Fredholm determinant. In practice, how do the authors recommend choosing $m$ for a given PDE and kernel? Are there heuristics for determining when $m$ is "large enough" for the PILE score to be reliable?

2. Have the authors observed how sensitive the PILE score is to the number and distribution of collocation/quad points? For example, do the kernel rankings stabilize beyond a certain $m$? Any such empirical guidance would be useful for practitioners.

### Soundness
4

### Presentation
4

### Contribution
4
