# Enhancing Stability of Physics-Informed Neural Network Training Through Saddle-Point Reformulation

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Physics-informed neural networks (PINNs) have gained prominence in recent years and are now effectively used in a number of applications. However, their performance remains unstable due to the complex landscape of the loss function. To address this issue, we reformulate PINN training as a nonconvex-strongly concave saddle-point problem. After establishing the theoretical foundation for this approach, we conduct an extensive experimental study, evaluating its effectiveness across various tasks and architectures. Our results demonstrate that the proposed method outperforms the current state-of-the-art techniques.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses a fundamental challenge in training Physics-Informed Neural Networks (PINNs): the instability and poor performance caused by conflicting gradients from the multiple loss terms (e.g., PDE residuals and boundary conditions). The authors propose a novel solution by reformulating PINN training as a nonconvex-strongly concave saddle-point problem (SPP). The authors provide a solid theoretical foundation for their proposed Bregman Gradient Descent Ascent (BGDA) algorithm, proving its convergence to a stationary point even in this non-Euclidean geometry setting. Extensive experiments on the comprehensive PINNacle benchmark demonstrate that AdaptiveBGDA significantly outperforms a wide range of state-of-the-art optimizers and weighting schemes across different PDE problems.

### Strengths
1. The saddle-point reformulation is a principled and theoretically grounded approach to the known problem of loss imbalance in PINNs..

2. The results are extensive, covering wide variety of PDEs (Poisson, Heat, Navier-Stokes, etc.) and challenging features (complex geometry, multiple domains, long time intervals). 

3. The proposed AdaptiveBGDA algorithm is a concrete and practical contribution that can be readily adopted by other researchers and practitioners to improve the stability and performance of their PINN models.

### Weaknesses
1. Although rich experiments are conducted based on PINNacle, the relative error values in Table 3 are relatively bad and did not reflect the state of the art results, and the improvement seems modest. For example, the Poisson and Heat equations are simple and can be easily solved by traditional numerical methods with high accuracy, however, the errors stays around $\mathcal{O}(10^{-2})$, which are too large. For the complex chaotic KS equation, the improvements are from 9.57E-1 to 9.53E-1, however, for the same solution, work [a] in the year 2023 has reported a much better result 1.61E-1.
[a] AN EXPERT’S GUIDE TO TRAINING PHYSICS-INFORMED NEURAL NETWORKS, 2023.

2. The experiments in the main text seem to use a standard PINN architecture.  However, vanilla PINN architectures are hard to present the sota results[b]. While Appendix A shows results on more advanced architectures, a deeper discussion on how the saddle-point optimization interacts with specific architectural choices (e.g., Fourier feature networks, transformers) would be valuable.
[b] Causality-enhanced Discreted Physics-informed Neural Networks for Predicting
Evolutionary Equations,2024

3. While the paper shows a runtime speedup, the saddle-point formulation inherently introduces additional computational overhead per iteration due to the proximal step for updating π

4. The method introduces new hyperparameters (λ for the regularizer, step sizes γ_θ and γ_π). Although the paper uses fixed values across all benchmarks to show robustness, optimal performance in new, highly complex problems might still require tuning.

### Questions
1. See weakness above.

2. Could you give insights on how practitioners should go about selecting the regularization parameter λ and the initial learning rates for a new problem?

3. The experiments on high dimension is simply on d=5 dimensions, which is very weak indeed. Could you report results on higher dimensions like d=100?

4. The theory and experiments consider a small number of loss terms (M). How would the method scale and perform if M were very large, for instance, in problems with a very large number of boundary condition constraints or coupled multi-physics systems?

5. Assumption 1 needs the Lipschitz continuous for the loss function, which is widely used in the optimization literature. However, since the PINN loss is highly non-convex and ill-conditioned, does assumption 1 really hold for the PINN loss? How about the convergence

### Soundness
3

### Presentation
3

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
Although PINNs have attracted significant attention and found wide applications, their performance often suffers from instability due to the complexity of their loss functions. To address this, the research team reformulated PINN training as a nonconvex–strongly concave saddle-point problem. They first established a solid theoretical foundation by studying this class of saddle-point problems under non-Euclidean geometry and proposing a Bregman proximal mapping–based method with rigorous guarantees on optimization dynamics. Extensive empirical validation followed, demonstrating that the proposed method consistently outperforms existing approaches and achieves state-of-the-art results across nearly all benchmark PDEs, effectively unifying previously task-specific dominant methods. Finally, the paper presents a comprehensive theoretical and empirical analysis of the proposed approach and highlights its key contributions: (1) establishing the theoretical foundation, and (2) conducting extensive empirical verification.

### Strengths
1.  Reformulating PINN training as a nonconvex–strongly concave saddle-point problem and introducing the AdaptiveBGDA optimizer based on Bregman divergence is innovative. The method offers a fresh perspective distinct from traditional gradient-based techniques.

2. Theoretical analyses are rigorous, with proven convergence guarantees. Experimental results across multiple PDE benchmarks show consistent improvements over state-of-the-art methods.

3. The paper is well-structured, with clear explanations of the algorithms and mathematical derivations, intuitive figures, and transparent proofs.

4. To some extent, it addresses the instability issues in PINN training and has potential application value for solving PDEs in scientific computing.

### Weaknesses
1. The theoretical analysis (Theorem 1 and Lemma 2) establishes convergence under idealized assumptions, but these are not explicitly connected to observed empirical behaviors. The discussion lacks explanation of how theoretical stability guarantees translate to improved convergence in practice.

→ Improvement: Add a section explicitly mapping theoretical assumptions (e.g., strong concavity, bounded divergence) to empirical observations in experiments.

2. Although the paper includes a link to anonymous code for review, it does not ensure full reproducibility—some implementation details and hyperparameter settings are missing.

→ Improvement: Provide complete training scripts, configuration files, and datasets used for each benchmark.

3. The theoretical role of Bregman divergence is mathematically well defined, but its intuitive contribution to training stability in non-Euclidean spaces is underexplained.

→ Improvement: Include visual or conceptual explanations illustrating how Bregman divergence shapes the optimization trajectory or balances competing losses.

### Questions
1. What is the computational complexity (in both time and space) of AdaptiveBGDA compared to Adam, NTK, or AL-PINN? Please provide either a theoretical complexity analysis or detailed runtime profiling (e.g., per iteration cost, GPU utilization, or convergence rate per epoch). This would clarify whether AdaptiveBGDA’s superior stability comes at a computational trade-off.

2. The paper mentions fixed hyperparameters (γπ = 0.1, γθ = 0.008, α1 = 0.9, α2 = 0.999, β = 0.999). How sensitive is the method’s performance to these choices? Please include a sensitivity analysis or guidelines for hyperparameter selection would strengthen the practical usability of the method.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the instability problem of PINNs by reformulating the training process as a saddle-point problem (SPP). Specifically, the authors introduce trainable loss weights $\pi$ and impose a divergence regularization term to ensure strong concavity with respect to $\pi$. This transformation is claimed to improve regularity and stability during optimization. The resulting formulation is optimized using a BGDA algorithm and its adaptive variant (AdaptiveBGDA). Theoretical guarantees are provided for convergence in nonconvex–strongly concave settings, and experimental results on multiple PDE benchmarks demonstrate faster convergence and improved accuracy compared to baseline PINN training schemes.

### Strengths
1. **clear motivation**: The work addresses a well-known bottleneck in PINN training: gradient imbalance between boundary and residual terms. The motivation for stabilizing the training through adaptive loss weighting is well articulated and relevant.

2. **Novel regularization for strong concavity**: Introducing a divergence-based regularization (via Bregman divergence) ensures strong concavity in the inner maximization problem, which theoretically stabilizes the optimization dynamics. 

3. **simple algorithm**: The BGDA framework is conceptually simple and can be implemented with minor modifications to existing PINN optimizers.

### Weaknesses
1. **Overall poor presentation including unlear definitions**: for example, in Equation (1), quantities such as $M_r$, $M$, and the role of $S$ and $\hat pi$ are not clearly defined. 

2. **Potentially incorrect criticism of prior work**: I think the claim that Liu & Wnag (2021) used $\pi \in \mathbb R^d$ and suffered instability due to the unconstrained weight space may be inaccurate. The dimension of $\pi$ in Liu & Wang (2021) is also the number of losses. 

3. **Justification for reformulation**: Transforming a nonconvex minimization problem into a nonconvex–strongly concave SPP can make optimization harder, as it introduces an inner maximization step that must be approximated in practice. The paper lacks a quantitative discussion on whether the stability gain outweighs the computational burden..

4. **Limited theoretical scope**: The analysis guarantees stationarity of the envelope function $\Phi(\theta)=L(\theta, \pi^*(\theta))$ but does not ensure convergence to a joint stationary point. Therefore, the derived resutls are somewhat weak compared to standard convergence analysis in the literature on minimax optimization . 

5. **Lack of fair computational comparison**: computational costs are not reported. Since SPP training increases per iteration cost, it should be reported. 

6. **Insufficient experimental details and reproducibility**: Standard deviations are missing. Deatils for selection for hyperparameters are not rerported. 

7. **Missing baseline**: Comparison with dual-dimer approaches (Liu & Wang (2021)) is missing.

### Questions
1. How sensitive is the algorithm to the choise of  hyperparameters? 

2. Please include computational overhead of the proposed method.

3. Pleae include experimental comparison with dual dimer method

4. How does the method perform on high-dimensional or more challenging PDEs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to reformulate the PINN loss as a min–max (saddle-point) problem with a Bregman regularizer to stabilize the weights. Using a simple optimizer, Bregman Gradient Descent–Ascent (BGDA), the authors empirically show improved performance on numerous PDE benchmarks. To explain where the gains come from, the paper presents a gradient conflict analyses showing reduced conflict, which supports the reported performance improvements.

### Strengths
1. High novelty. The paper reformulates PINN into a new optimization framework that mitigates gradient-conflict issues.

2. Clear and rigorous. The problem setup and method are explained clearly, and the paper provides rich theory, such as complexity bounds, that demonstrates the strength of the approach.

3. Strong experiments. The evaluation across 22 PDEs is comprehensive and shows the stability of the proposed method.

### Weaknesses
Although this paper have already compared to many baseline optimizers. It would better to include some SOTA method like [1] and [2], especially [2] also experimented on Burgers equation and achieves a much lower error.


[1] Rathore, Pratik, et al. "Challenges in training pinns: A loss landscape perspective." arXiv preprint arXiv:2402.01868 (2024).

[2] Kiyani, Elham, et al. "Which Optimizer Works Best for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks?." arXiv preprint arXiv:2501.16371 (2025).

### Questions
None.

### Soundness
3

### Presentation
4

### Contribution
3
