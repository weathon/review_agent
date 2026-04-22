# Accelerating Regression Tasks with Quantum Algorithms

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Regression is a cornerstone of statistics and machine learning, with
  applications spanning science, engineering, and economics.
  While quantum algorithms for regression have attracted considerable attention,
  most existing work has focused on linear regression, leaving many more complex
  yet practically important variants unexplored.
  In this work, we present a unified quantum framework for accelerating a broad
  class of regression tasks---including linear and multiple regression, Lasso,
  Ridge, Huber, $\ell_p$-, and $\delta_p$-type regressions---achieving up to a
  quadratic improvement in the number of samples $m$ over the best classical
  algorithms.
  This speedup is achieved by extending the recent classical breakthrough of
 Jambulapati et al. (STOC'24) using several quantum techniques, including
  quantum leverage score approximation (Apers &Gribling, 2024) and the
  preparation of many copies of a quantum state (Hamoudi, 2022).
  For problems of dimension $n$, sparsity $r < n$, and error parameter
  $\epsilon$, our algorithm solves the problem in
  $\widetilde{O}(r\sqrt{mn}/\epsilon + \mathrm{poly}(n,1/\epsilon))$
  quantum time, demonstrating both the applicability and the efficiency of
  quantum computing in accelerating regression tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
This paper introduces a unified quantum framework for accelerating sparsification in generalized linear models (GLMs), yielding quadratic speedups over classical methods across linear, multiple, Lasso, Ridge, Huber, $\ell_p$, and $\gamma_p$ regression. The authors extend the recent classical breakthrough of Jambulapati et al. (2024) by developing a Quantum Multiscale Leverage Score Overestimates (QMLSO) algorithm that leverages quantum leverage score approximation and quantum state preparation techniques. For problem dimension $n$, sparsity $r<n$, sample size $m$, and error parameter $\varepsilon$, their algorithm achieves $\widetilde{\mathcal{O}}!\left(\frac{r\sqrt{mn}}{\varepsilon}+\mathrm{poly}!\left(n,\frac{1}{\varepsilon}\right)\right)$ quantum time complexity, demonstrating quadratic speedup in m when $\varepsilon$ is constant and $m \gg n$.

### Strengths
1. Comprehensive theoretical framework with broad applicability. The paper successfully unifies quantum speedups for a diverse family of regression problems under a single GLM sparsification framework, extending beyond previous quantum work that focused primarily on linear regression. The introduction of proper loss functions and multiscale leverage score overestimates provides elegant abstractions that capture the essential structure needed for quantum acceleration across multiple problem variants.
2. Rigorous technical development with novel quantum subroutines. The QMLSO algorithm (Algorithm 1) and the quantum weight initialization procedure (Theorem 9) represent substantial technical contributions that carefully adapt classical contractive algorithms to the quantum setting. The complexity analysis is thorough, properly accounting for query complexity to different oracles and demonstrating clear quadratic speedups with explicit leading-order terms that dominate in the regime where sparsification is beneficial $\varepsilon=\Omega!\left(\sqrt{\tfrac{n}{m}}\right)$.

### Weaknesses
1. Limited discussion of QRAM requirements. While the paper acknowledges on page 5 (lines 217-220) that "QRAM serves as a natural quantum analogue of the classical RAM model" and mentions that "practical realization of scalable QRAM remains highly uncertain," there is insufficient critical analysis of how QRAM requirements scale with problem size and what this means for near-term quantum advantage.

2. Gap between stated contributions and actual novelty over prior quantum work. The paper claims on page 1 (lines 17-18) to address "broader and both theoretically and practically important regression tasks such as ℓp regression and Huber regression" beyond linear regression, but the relationship to Song et al. (2023) is understated. As acknowledged in Table 1, Song et al. already achieved $\widetilde{\mathcal{O}}!\left(\frac{\sqrt{m},n^{1.5}}{\varepsilon}\right)$ time for linear, multiple, and ridge regression, where the improvement here is from $n^{1.5}$ to $rn$, which is significant only when $r \ll n^{0.5}$.

### Questions
1. Can you provide explicit bounds on the QRAM size, access time, and circuit depth required for representative problem instances? 

2. Can you provide a revised complexity analysis that explicitly accounts for QRAM access time as a function of the data size, and identify at what point (if any) QRAM overhead eliminates the quantum advantage?

3. What exactly is the time complexity of your algorithms without QRAM? Does the quantum advantage survive this modification, and if so, under what parameter regimes?

### Soundness
2

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
3

### Summary
This paper presents a quantum framework for efficiently solving various regression and optimization problems by adapting Jambulapati et. al. into a quantum algorithm and noticing that many of the regression problems can be formulated as sparse generalized linear models. It achieves quadratic speedup comparing to previously best-known algorithms.

### Strengths
- This paper provides a unified optimization framework for many of the regression problems in optimization, which provides a convenient way for further quantum algorithm research.
- The paper itself is presented well, with easy-to-follow narratives.

### Weaknesses
- The paper does not highlight clearly the source of the quantum speedup, which might be hard to understand. Also the paper, although citing the Jambulapati et. al., does not discuss in detail how the classic algorithm is related to the quantum version.
- The contribution of the paper is a bit lacking. Although a unified speedup across different regression problems is nice, the contribution of the paper is limited to implementing the original algorithm with several quantum tricks (like Hamoudi's copy preparation trick, which is well-known and widely applied, for example in https://arxiv.org/abs/2402.12745)

### Questions
- Is it possible to prove a quantum lower bound on the regression problems in the paper?
- Is it possible to incorporate also infinite-norm regression (i.e., max) into the framework?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a quantum algorithm for improving the speed of finding sparsifiers (most commonly, weighted subset of training samples) that approximately preserve the value of the loss function over the parameter domain, allowing for faster training.

### Strengths
The paper applies recent work on quantum leverage score approximation by Apers and Gribling in the context of recent result showing existence of sparsifiers for a wide class of losses in Generalized Linear Models\ by Jambulapati et al. The resulting approach improves the complexity of finding the sparsifier: it shares the $\sqrt{m}$ term with existing approaches, but improves the dependence on $n$ from $n^{1.5}$ to $rn^{0.5}$, an advantage in sparse ($r<<n$) scenarios.

### Weaknesses
The result is an incremental advancement combining existing classical work (Jambulapati et al.) with prior quantum algorithms (Hamoudi, Li et al., Apers and Gribling). The presentation in the manuscript makes it unclear which results are prior work and which are novel (e.g. Def. 5. and Def. 6 should include reference to Jambulapati et al.). The main manuscript provides no clear description of the algorithms beyond listing the steps in Alg 1. and Alg. 2, and many of the key elements in these two algorithms are not properly explained. For example, in Alg. 1, ModLevApprox and WeightCompute are jointly described using one, very high-level sentence in the main manuscript; description in the appendix B.1. indicates these are not original contributions, but recapitulation of Apers and Gribling’s Theorem 4. 

The algorithm is aimed at a very practical problem of speeding up GLM, however, by working in a QRAM-based framework, it’s practical applicability on near-to-medium term hardware is heavily constrained.

### Questions
Are there application scenarios (e.g. data characteristics) where the constraint related to QRAM can be circumvented?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript presents quantum algorithms for many regression tasks, including linear, multiple, Lasso, Ridge, Huber, $\ell_p$, and $\delta_p$ regressions. These algorithms are based on a unified framework, where the core quantum technique is a fast algorithm for approximating leverage scores due to Apres & Gribling (2024).  On top of this, a generalized linear model sparsifier is constructed using the techniques from Cohen & Peng (2015) and Jambulapati et al. (2024).

Based on this GLM sparsification framework, the authors derived a quantum algorithm for various regression problems. These quantum algorithms are faster than best-known classical algorithms in the not-too-precise regime, and are faster than previous quantum algorithms by Song et al. (2023) when the linear model is sparse.

### Strengths
The manuscript is very well written. The statements are supported with rigorous proofs. Although this work relies on existing quantum subroutines, it incorporates quantum subroutines in a nontrivial way to develop multiscale leverage score overestimation and quantum importance sampling, which are key components of this framework. Overall, I think this submission makes a solid theoretical contribution to solving regression problems.

### Weaknesses
For linear, multiple, and Ridge-regressions, the quantum algorithms proposed in this work are faster than previous quantum algorithms by Song et al. (2023), mainly due to the sparsity $r \leq n$. It looks like this improvement is merely due to the fact that the authors considered sparse regression models. Is it the case, or is the improvement truly a consequence of the new techniques?

### Questions
This is related to my question in the previous section. If we consider sparse regression models with sparsity $r$ and use the quantum algorithm by Song et al. (2023) to solve it, what would the dependence on $r$ look like?

A minor typographical issue in lines 111-112: repeated word "constructing".

### Soundness
4

### Presentation
4

### Contribution
3
