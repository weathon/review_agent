# Faster Sampling from Gibbs Distributions with Quantum Variance Reduction

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
We present quantum algorithms that provide provable speedups for approximate sampling from probability distributions of the form $\pi \propto e^{-f}$, where $f$ is a potential function that can be written as a finite sum, i.e., $f= \frac{1}{n}\sum_{i=1}^n f_i$. Our approach focuses on stochastic gradient–based methods with only oracle access to individual gradients \{$\nabla f_i$\}$_{i\in [n]}$. The techniques of our quantum algorithms are based on a non-trivial integration of quantum mean estimation techniques and existing variance reduction techniques such SVRG and CV. 

 As these techniques often require occasional full-gradient calculations, the key challenge is that an unbalanced weighting between variance reduction and quantum mean estimation results in a regime where the quantum advantage is lost due to frequent full-gradient computation. We overcome this difficulty by carefully optimizing the target variance level. Our algorithms improve the number of gradient queries of classical samplers, such as Hamiltonian Monte Carlo (HMC) and Langevin Monte Carlo (LMC), in terms of dimension,  precision, and other problem-dependent parameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces quantum-enhanced stochastic gradient samplers that integrate unbiased quantum mean estimation with classical variance reduction techniques to accelerate approximate sampling from Gibbs distributions in finite-sum settings. The authors provide non-asymptotic analyses showing that their proposed QSVRG/QCV variants of HMC and LMC achieve improved gradient-query complexity over the current state-of-the-art classical methods under strong convexity and log-Sobolev assumptions. Overall, the work aims at establishing a theoretically grounded avenue for quantum speedups in sampling algorithms.

### Strengths
- The paper indeed tackles a crucial in many domains, i.e., sampling from Boltzmann-Gibbs type of distributions. 
- It does a thoroughly analysis of related works and also carries forward a rigorous mathematical analysis of the proposed method. 
- The paper is well structured, the related work section is very extensive and the preliminaries section exhaustive.

### Weaknesses
- The paper is very hard to follow at times: Many notions and concepts are mentioned in the introduction and not properly introduced. For example, the concept of oracle, is extensively mentioned in the first section without never being properly defined. This can represent a serious blocker for people without a strong background in quantum computing. 
- I have the impression the paper is too focused on quantum computing. For this reason, it may be hardly accessible to the broader audience normally targeted at top tier ML conferences. I believe a specialised quantum computing conference or journals are a better venue for publishing this work. 
- Again, also notation is sometimes not defined when appearing for the first time, thus making the paper not easy to follow. See for example eq (5) on page 2. 
- The bottom of page 2 from line 84 onwards also appears fairly verbose and hard to parse. Furthermore, algorithms are hereby introduced with their acronym without appropriate refs and or explanations. 
- Limitations are not discussed explicitly. 
- Conclusions are also missing thus making the paper appearing incomplete. 
- The experimental section is missing. While the authors admit that their theoretical works is based on fault-tolerant quantum devices, I still believe it is important to have a section that validates theoretical works. In this regards I wonder if the authors can simply use quantum computer simulators without any source of error to showcase the advantages of their approach in comparison to others. 
- Without an appropriate empirical analysis, I find it unfair to claim any practical advantage of the proposed approach over existing classical algorithms. While theoretically there should be an advantage, the classical algorithm can be tested and ran while the proposed approach, as it relies on fault tolerant quantum device cannot be compared. 
- I believe the authors should extend their work (if possible) by accounting for non fault tolerant quantum devices or error corrected qubits so that at least some sort of empirical evaluation can be carried out.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a new variance reduction approach for Hamiltonian Monte Carlo. The key idea is to replace the stochastic gradient estimation in SG-HMC via a quantum mean estimator. Since the quantum mean estimator has a quadratic improvement in $\epsilon$ compared to classical algorithms, the proposed idea can improve the gradient variance in HMC.

### Strengths
+ The idea of using a quantum mean error estimator in HMC is out-of-box. This idea can be transformative if it really works.
+ This paper also presents a set of theorertical proofs to show the potential advantage of this method in Section 4.
+ The introduction/tutorial about quantum computing is very intuitive, and it's easy to follow for generic machine learning researchers.

### Weaknesses
While the presented idea is very interesting, my main concern is that this paper draft looks so incomplete, and many key points are missing.
1.  No numerical/simulation results are provided to support the claimed benefit;
2.  No conclusion was made about the paper either
3. While this paper uses quantum mean estimator as a blackbox (which is OK), it did not provide the key details about quantum mean estimation. For instance, how a quamtum mean estimator will be implemented (algorithmically and in practical hardware)? Withoug such details, readers can hardly implement this idea.
4.  This paper didn't talk about the feasibility of implementing this idea either. Can this algorithm be impelmented using existing quantum hardware or quantum/classical hybrid processor? If not, how many quantum gates are needed in the future to enable real implementation?

### Questions
I have a few questions:
1. Can you explain how the quantum mean estimator will be implemented in the HMC context?
2. Can you explain how many quantum gates are needed to implement this framework? If a classical/quantum hybrid architecture is needed, how would this hybrid architecture look?
3. Measuring the results of a quantum computing framework is often very challenging in practical engineering implementation. Can you explain how you can measure the result of the quantum mean estimator for the gradient?

### Soundness
2

### Presentation
1

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
The paper introduces quantum algorithms that accelerate approximate sampling from Gibbs distributions $\pi(x) \propto e^{-f(x)}$, where $ f = \frac{1}{n}\sum_i f_i $. The authors integrate quantum mean estimation with classical variance reduction techniques (e.g., SVRG, CV) to design quantum analogues of stochastic-gradient samplers such as Langevin Monte Carlo (LMC) and Hamiltonian Monte Carlo (HMC). They analyze how to optimally balance quantum variance estimation costs with occasional full-gradient computations to preserve quantum advantage. The resulting algorithms achieve provable asymptotic reductions in gradient-query complexity over the best classical methods under both strong convexity and Log-Sobolev assumptions.

### Strengths
- Originality: Introduces the framework combining quantum mean estimation with classical variance reduction (SVRG, CV) for stochastic-gradient-based sampling, bridging two previously separate areas—quantum optimization and classical sampling theory.
- Quality: Provides detailed theoretical analysis, including variance control lemmas and nonasymptotic convergence bounds under both strong convexity and Log-Sobolev assumptions, leading to rigorously proven quantum query speedups.
- Clarity and structure: The paper is well-organized, with clear motivation, formal assumptions, and explicit algorithmic descriptions that parallel classical counterparts (LMC/HMC).
- Significance: Demonstrates asymptotic improvements in gradient-query complexity (e.g., from $\tilde{O}(n^{1/2}\varepsilon^{-1})$ to $\tilde{O}(n^{1/3}\varepsilon^{-1})$), clarifying where quantum advantages can arise in sampling—a central task in machine learning and statistical physics.

### Weaknesses
- Complexity accounting: The relation between query complexity (oracle calls) and total gate complexity is not specified, making direct comparison with classical cost a little ambiguous.

- Parameter dependence: Several results require knowledge of problem constants such as the Log-Sobolev constant $\alpha$, which are typically unknown in practice.

- A bit concern of novelty: quantum mean estimation is widely used to accelerate machine learning and optimization tasks. Could the author explain more about its specific technical novelty in combining them with the sampling framework?

### Questions
See the weakness part, the concern of novelty.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents quantum algorithms that accelerate the sampling from probability distributions $\pi \propto e^{-f}$, where $f = \frac{1}{n}\sum_i f_i$. By assuming access to individual gradients $\{\nabla f_i\}$, and leveraging quantum mean estimation techniques to existing variance reduction techniques in the classical literature, the new quantum algorithms achieve sub-quadratic speedups in key problem parameters such as dimension $n$ and accuracy $\epsilon$.

### Strengths
- Drawing ideas from quantum stochastic optimization methods, such as multi-dimensional quantum mean estimation and quantum gradient estimation, to improve large-scale, noisy sampling tasks.
- A non-trivial integration of quantum mean estimation to existing variance reduction techniques, leading to polynomial quantum speedups.
- Quantum speedups demonstrated for various sampling algorithms, including LMC and HMC. Three quantum speedups are identified in Table 1, two for strongly convex problems and one under the standard LSI assumptions.

### Weaknesses
All three reported quantum speedups are sub-quadratic: 
- QSVRG-HMC: $n^{1/2}\epsilon^{-3/4}$ v.s. classical SVRG-HMC: $n^{2/3}\epsilon^{-2/3}$.
- QCV-HMC: $\epsilon^{-3/2}$ v.s. classical CV-HMC: $\epsilon^{-2}$
- QSVRG-LMC: $n^{1/3}\epsilon^{-1}$ v.s. classical SVRG-LMC: $n^{1/2}\epsilon^{-1}$. 
It is noted that, to achieve the claimed speedups, fully fault-tolerant quantum computers are required. While theoretically non-trivial, such quantum speedups are of fairly limited value in practice. In fact, people commonly believe that a quadratic quantum speedup (in the asymptotic sense) may not be sufficient to yield a realistic performance gain due to the overhead of quantum error correction (QEC). 

Moreover, this paper does not discuss the limitations of quantum speedups for this class of problems. If it can be established (or at least argued) that this type of stochastic sampling problem cannot be further accelerated (for example, with some query/sample lower bounds), it would be of greater impact on the field of quantum computing.

### Questions
1. The assumption on the stochastic gradient oracle appears to be quite strong, as it requires a superposition of individual gradients. In most practical problems, the individual gradients are only available in an incoherent superposition (i.e., a classical ensemble, such as batched SGD). Can you give some concrete scenarios where this type of quantum oracle is achievable, meaning that their quantum implementation costs are not significantly higher than implementing the classical stochastic gradient oracle?
2. My understanding is that the proposed quantum algorithms are actually hybrid quantum-classical algorithms, since the iteration steps are still performed on classical computers, while only the gradient estimation steps are replaced using quantum variance reduction. If that's the case, it should be mentioned explicitly, as this approach is quite different from a number of existing quantum sampling algorithms (e.g., [Childs et al., 2022], [Ozgul et al., 2024]) that do not perform classical iteration steps but produce a quantum state that encodes the target measure $\pi$: 
3. Recently, there has been a new approach for probabilistic sampling using differential operators and QSVT: https://arxiv.org/abs/2505.05301. Is this a relevant approach for stochastic sampling? If so, how does the performance compare to the tabulated results in the paper?

### Soundness
3

### Presentation
3

### Contribution
2
