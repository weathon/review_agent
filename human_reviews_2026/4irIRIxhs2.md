# Quantum Speedups for Sampling and Non-convex Optimization with Stochastic Zeroth Oracles

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 8, 4, 6

## Abstract
We propose quantum algorithms with provable speedups for sampling from probability distributions of the form $\pi \propto e^{-f}$, where $f:\mathbb{R}^d\mapsto \mathbb{R}$ is a potential function. In particular, we consider access only to a stochastic evaluation oracle, allowing simultaneous queries of the potential value at two different points under the same stochastic parameter. By introducing novel quantum algorithms for stochastic gradient estimation in this setting, our algorithms improve the evaluation complexities of classical samplers, such as Hamiltonian Monte Carlo (HMC) and Langevin Monte Carlo (LMC) in terms of dimension, precision, and other problem-dependent parameters. Furthermore, we demonstrate that our quantum sampling algorithms can be used to achieve quantum speedups in optimization, particularly for minimizing nonsmooth and approximately convex functions that commonly appear in empirical risk minimization problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a novel quantum algorithmic framework for accelerating sampling and non-convex optimization in a stochastic zeroth-order setting. The theoretical contribution is significant, as it cleverly combines quantum gradient estimation, quantum mean estimation, and classical sampling theory to provide a provable quantum speedup under specific conditions. However, the paper suffers from fundamental issues regarding the justification of its core model and its general applicability, which currently limit the solidity of its claims and the breadth of its impact.

### Strengths
This work presents a comprehensive theoretical framework that integrates tools from quantum computing (gradient estimation, mean estimation) and classical numerical analysis (MLMC, convergence theory for samplers) to achieve an end-to-end complexity analysis with polynomial speedups.

### Weaknesses
There are some major concerns affecting the score of this paper, as follows:

1. Questionable Fairness and Realism of the Core Oracle Model

The paper's technical approach relies crucially on a strong oracle assumption: the ability to query the stochastic function at two different points using the same random seed (i.e., "reproducible randomness"). This is a fundamentally more powerful model than the standard stochastic zeroth-order oracle used by the classical baselines, which typically allows for independent sampling on each query. Demonstrating a quantum speedup against classical algorithms that operate in a weaker, standard model is arguably unfair. The claimed speedup might be a direct consequence of this stronger assumption rather than a pure algorithmic improvement. 

Further, this strong assumption severely restricts the generality of the proposed algorithms. They are primarily applicable to finite-sum problems, where fixing the random seed corresponds to selecting a specific data index. For many important real-world problems (e.g., optimization based on physical experiments, interactions with non-stationary systems), the algorithm is not directly applicable. The paper should more explicitly acknowledge this limitation rather than presenting its results as a general "stochastic zeroth-order" acceleration.

2. Insufficient Analysis of Quantum Resource Costs

While the focus on query complexity is standard for a theoretical paper, the complete omission of other quantum resource costs may mislead readers about the algorithm's practical feasibility. What is the asymptotic scaling of the number of qubits required to construct the phase oracle (Proposition 2.3) and run the robust estimation framework (Algorithm 1)? Is this scaling polynomial in the dimension and the precision? This information is crucial for assessing practical viability.

3. Lack of Comparison with Relevant Quantum Works

The paper chooses to compare its performance against classical zeroth-order algorithms. However, it lacks a critical comparison with relevant quantum algorithms. For the most natural application scenario—finite-sum optimization—there exist other quantum-accelerated methods. How does the proposed sampling-based framework compare to these approaches? Is it superior in terms of query complexity, generality, or implementation difficulty?

### Questions
1. The introduction and discussion should clearly state that the work relies on a "strengthened oracle model with reproducible randomness."  Meanwhile, the authors should discuss whether this strong assumption is necessary for achieving the speedup. 

2. A fairer comparison would be against a classical algorithm that is also granted the same powerful oracle, or the paper should explicitly frame the speedup as being achieved at the cost of reduced generality.

3. A rough asymptotic analysis of the quantum resource requirements should be provided in the appendix or discussion.

4. A dedicated paragraph in the related work should discuss the anticipated performance of the proposed algorithm against existing quantum optimization/sampling algorithms under the same (finite-sum) setting.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the potential of quantum algorithms to accelerate the optimization of functions with only access to zeroth oracle, in particular nonsmooth and almost convex functions. The authors demonstrate that using quantum mean estimation and jordan's algorithm, it is possible to achieve quadratic speedups over classical methods on various problems. This work derives an algorithm to efficiently and accurately estimate gradients even in the presence of noise and approximation errors. Additionally, this paper shows that the result can be further generalized to non-smooth scenarios via gradient estimation.

### Strengths
The algorithm proposed in the paper is explained very well. The paper has a good presentation where it focuses on not only the technical details but also the intuition behind the algorithm. Besides, showing an elegant algorithm for gradient estimation on nonsmooth functions is interesting.

### Weaknesses
No significant weekness

### Questions
- Is it possible to also prove a lower-bound for the optimization scenario considered in this work?
- Would it be possible to extend the result to some extent to non-convex landscapes?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes quantum algorithms that achieve provable speedups for sampling from Gibbs distributions $\pi \propto e^{-f(x)}$ and for optimizing nonconvex objectives, when only stochastic zeroth-order (function value) access to $f$ is available. The authors introduce new quantum stochastic gradient estimation methods that improve classical query complexities from$\tilde{O}(d^2\sigma^2/\varepsilon^2) $ to $\tilde{O}(d\sigma/\varepsilon)$ or even $ \tilde{O}(d^{1/2}\sigma/\varepsilon)$ under additional smoothness assumptions. These estimators are then used to obtain LMC and HMC with reduced oracle complexity, leading to polynomial quantum speedups in dimension, precision, and noise parameters for both sampling and optimization tasks. The paper further extends these results to nonsmooth and approximately convex optimization, showing that quantum sampling techniques can yield faster convergence in empirical risk minimization–type problems. Theoretical guarantees are established for all algorithms, assuming fault-tolerant quantum computation.

### Strengths
- Originality: Introduces the first quantum algorithms achieving provable polynomial speedups for stochastic zeroth-order sampling and optimization, extending quantum gradient estimation to a realistic noisy-oracle model. Provides rigorous convergence and complexity analyses for quantum variants of LMC and HMC, connecting Jordan’s gradient estimation, quantum mean estimation, and MLMC in a novel way.
- Breadth of applicability: Framework covers both strongly convex and nonconvex settings, and further applies to nonsmooth approximately convex optimization, showing broad theoretical relevance.
- Clarity: The paper is well-structured, making the logical flow of ideas easy to follow.
- Significance: Establishes new theoretical baselines for quantum advantages in sampling and optimization, potentially guiding future algorithm design once fault-tolerant quantum hardware becomes available.

### Weaknesses
- Proof missing details: could the author explain in section B.1, proof of theorem 3.2, there seems to be a mismatch of $\kappa$ and $\sigma$ in the proof and statement. Could the authors clarify a bit on this?
- Unclear treatment of bounded gradients: Several proofs rely on a global bound $\|\nabla f(x)\|\le M$ without establishing or bounding M in terms of problem parameters. Could the authors clarify a bit on this?
- Novelty of their techniques: much of the techniques of quantum speedups seems to come from quantum gradient estimation and mean estimation. Could the authors explain more about their technical novelty?

### Questions
- See the weaknesses part.
- Line 1372: "where the last inequality is due to the fact that tails of $\pi^{\beta}$ is upper bounded by a Gaussian with variance $\Omega(1/\beta)$." could the authors explain more about why this holds, especially relying on what kind of assumptions? It seems not immediately clear to me.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel quantum algorithm for stochastic gradient estimation under various smoothness assumptions, leading to quadratic speedups for smooth potential functions. By leveraging this new stochastic gradient estimation subroutine in zeroth-order sampling tasks, this paper proposes two new quantum algorithms that achieve polynomial speedups over existing classical methods. The application of this approach to non-smooth and approximately convex optimization has also been discussed in the paper.

### Strengths
- A new quantum gradient estimation subroutine is proposed that overcomes the drawbacks of existing gradient estimation methods. In particular, this paper only requires the expectation value of the Lipschitz constant to be bounded. This is a slightly weaker assumption than many previous results. This is achieved by a careful combination of quantum mean estimation and Jordan's gradient estimation algorithm. 
- This quantum gradient estimation subroutine has been applied to both LMC and HMC, and the convergence is analyzed. 
- Applications to noisy, approximately convex optimization problems are discussed. This is a prominent problem class with important applications in ML, such as empirical risk minimization.

### Weaknesses
- The distance metric ($W_2$) used in Theorem 3.2 appears to be weaker than those in Theorem 3.4. Is this because the analysis of the base classical algorithm (HMC) is less explored compared to LMC? Does the quantum algorithm improve the distance metric?
- The approximate convexity assumption (Assumption 4.1) is very weak in high dimension ($d \gg 1$). Is it possible to relax this assumption further and still obtain quantum speedups? Will quantum algorithms be more competitive in the more "noisy" regime?

### Questions
See comments above.

### Soundness
3

### Presentation
3

### Contribution
3
