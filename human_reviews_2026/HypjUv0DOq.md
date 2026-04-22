# Preconditioned Norms: A Unified Framework for Steepest Descent, Quasi-Newton and Adaptive Methods

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Optimization lies at the core of modern deep learning, yet existing methods often face a fundamental trade-off between adapting to problem geometry and leveraging curvature utilization. Steepest descent algorithms adapt to different geometries through norm choices but remain strictly first-order, whereas quasi-Newton and adaptive optimizers incorporate curvature information but are restricted to Frobenius geometry, limiting their applicability across diverse architectures. In this work, we propose a unified framework generalizing steepest descent, quasi-Newton methods, and adaptive methods through the novel notion of preconditioned matrix norms. This abstraction reveals that widely used optimizers such as SGD and Adam, as well as more advanced approaches like Muon and KL-Shampoo, and recent hybrids including SOAP and SPlus, all emerge as special cases of the same principle. Within this framework, we provide the first systematic treatment of affine and scale invariance in the matrix-parameterized setting, establishing necessary and sufficient conditions under generalized norms. Building on this foundation, we introduce two new methods, $\texttt{MuAdam}$ and $\texttt{MuAdam-SANIA}$, which combine the spectral geometry of Muon with Adam-style preconditioning. Our experiments demonstrate that these optimizers are competitive with, and in some cases outperform, existing state-of-the-art methods. Our code is available at https://anonymous.4open.science/r/LIB-2D05

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes preconditioned matrix norms that unify steepest-descent-style LMOs with quasi-Newton and adaptive (diagonal) preconditioning. Within this framework, the LMO in the transformed gradient space yields a general update, capturing classical quasi-Newton (e.g., Shampoo/K-FAC), Adam/AdaGrad-type methods, Muon/Scion spectral-norm LMOs, and recent hybrids like SOAP/SPlus as special cases. The paper derives conditions for affine and scale invariances in the matrix-parameterized setting and introduces MuAdam and MuAdam-SANIA, which combine a spectral LMO with Adam/SANIA-style diagonal scaling. Experiments on a synthetic scale-invariance test, GLUE fine-tuning, LLM LoRA fine-tuning, and character-level language modeling demonstrate comparative performance with AdamW and Muon.

### Strengths
1. The idea of combining problem geometry with curvature utilization by integrating the steepest descent algorithm with preconditioning methods is insightful.
2. The paper provides a systematic invariance analysis on matrix-parameterized methods, and the proposed algorithm retains Adam-style scale invariance.

### Weaknesses
1. Algorithmic design feels mechanically compositional. Based on Defs. 2.1/2.2 and Alg. 1, the method “stacks” preconditioners into the norm and applies an LMO on the transformed gradient; Thm. 2.3 shows the composition identity, but not why this specific stacking is preferable or improves optimization beyond the base components.
2. The gains in the experiments appear occasional, making the benefit of combining the two less convincing.
3. The empirical evaluation uses fine-tuning tasks rather than pre-training, which seems like a less common choice for evaluating optimizers and how conclusions would transfer to model training.
4. Other preconditioned / more quasi-Newton-like methods (e.g., SOAP) are not compared; it remains not well-justified that Adam can be regarded as a representative of quasi-Newton methods.
5. No standard deviations reported in Tables 2 and 3. No learning rate grid-search performance results were reported.

### Questions
1. Can the authors disclose the hardware settings, e.g., GPUs, used in the experiments?
2. How does the proposed method compare with baselines in terms of per-iteration computational complexity / wall-clock time?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a single optimization framework based on "preconditioned matrix norms" to demonstrate that steepest descent and quasi-Newton, and adaptive optimizers follow from a common fundamental principle. The method allows researchers to create new hybrid optimization methods through the development of MuAdam, which unites spectral geometry with adaptive preconditioning techniques.

### Strengths
1. The algorithm provides a principled and modular methodology for designing novel hybrid optimizers by combining different curvature approximations with various descent geometries, as demonstrated by the creation of the MuAdam optimizer.
2. The paper introduces a complete method to analyze affine and scale invariance for models that use matrix parameters. The paper establishes exact testable requirements for these desirable properties, which will guide researchers developing robust optimizers.

### Weaknesses
1. The paper did not evaluate the computational requirements of the proposed MuAdam optimizer. The Newton-Schulz iteration in the algorithm performs matrix multiplications, which cost more than Adam's element-wise operations, yet the paper does not assess how the trade-off between accuracy gain and training duration affects the method.
2. While the framework introduces a vast and promising design space for creating new hybrid optimizers, the paper only explores a single new combination (a spectral LMO with a diagonal preconditioner). The paper did not investigate two promising combinations that would unite spectral geometry with KFAC preconditioners from Shampoo.
3. The experimental data indicate MuAdam functions as a competitive optimizer, yet it fails to outperform AdamW in all situations. The optimizer shows limited performance advantages, which seem to rely on particular task requirements and model structures and training methods, thus reducing its current adoption potential.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Paper that combines adam and muon

### Strengths
The paper does a review of recent methods related to gradient whitening. 

Has some good empirical results on finetuning

### Weaknesses
Unfortunately the paper does not consider the original line of work for gradient whitening. PSGD from Xilin Li published in 2015 covers all the methods in table 1. Some resources can be found here

Resources Preconditioned stochastic gradient descent, arXiv:1512.04202, 2015. (General ideas of PSGD, preconditioner fitting criteria and Kronecker product preconditioners.) 

Preconditioner on matrix Lie group for SGD, arXiv:1809.10232, 2018. (Focus on affine Lie group preconditioners, including feature normalization or whitening (per batch or layer) as special affine preconditioners. Use PSGD for gradient whitening.) 

Black box Lie group preconditioners for SGD, arXiv:2211.04422, 2022. (Mainly about the LRA preconditioner.)

Stochastic Hessian fittings with Lie groups, arXiv:2402.11858, 2024. (Properties of PSGD, also a good summary of PSGD. The Hessian fitting problem is shown to be strongly convex in GL(n,R) under certain mild assumptions. Will keep updating it to align with the code.) Curvature-informed SGD via general purpose Lie-group preconditioners, arXiv:2402.04553, 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)

Also there are a plethora of methods trying to combine adam and muon but none of them actually worked better.

### Questions
Please run your method in the modded NanoGPT speedrun. If the record is broken then the method will be accepted by the community.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper provides a descent framework that generalizes steepest descent, quasi-Newton, and adaptive methods, introduces MuAdam and MuAdam-SANIA, and presents empirical results.

### Strengths
The paper combines existing ideas based on a linear minimization oracle.

### Weaknesses
The main result of the paper is a simple observation when viewed in light of Bernstein and Newhouse (2024b) and Pethick et al. (2025), and neither the theory nor experiments are sufficient evidence to support the paper's claimed benefits.

The empirical results are severely underwhelming. Though the authors list 16 algorithms in Table 1 (including their own), they only compare with AdamW and Muon. In the GLUE benchmarks for full fine-tuning (Table 2), MuAdam does better in only 2 out of 8 datasets. This does not provide a strong basis to support claimed improvements.

The paper places a lot of emphasis on the importance of scale-invariance, for example Section 3 and Section 4.1. However MuAdam-SANIA is absent from "Optimizers compared" in the LLM fine-tuning experiments. This is a glaring oversight for a paper that espouses "one of the important properties of optimization algorithms is affine and scale invariance".

Likewise the theoretical contributions are weak, with basic "theorems" that are merely brief calculations of equivalence (for example Theorem 2.3 and its proof on lines 787-792 and 796-801), and the experiments are not convincing or thorough enough to make up for the extremely spare theory.

### Questions
Why are there no empirical comparisons with another "Hybrid" algorithm, when Table 1 states MuAdam and MuAdam-SANIA are themselves "Hybird" algorithms?

### Soundness
1

### Presentation
2

### Contribution
1
