# Accelerating Discrete Diffusion Models with Parallel Sampling

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Discrete diffusion models are widely used for learning and generating discrete distributions. As the generation process is inherently sequential, the acceleration of sampling is of significant importance. In this work, we parallelize the mainstream $\tau$-leaping algorithm for absorbing discrete diffusion in a Continuous-Time Markov Chain (CTMC) framework. By leveraging the continuous-time stochastic integral form of the $\tau$-leaping algorithm and the Picard iteration method, we achieve parallel-in-time sampling acceleration. We implement a predictor-corrector structure based on the Markov chain Monte Carlo (MCMC) method to control for additional errors and provide a proof of exponential convergence for our algorithm. We improve the overall time complexity of $\tau$-leaping from ${\mathcal{O}}(d^{2}\log^{2}d)$ to at most ${\mathcal{O}}(d^{3/2}\log^{5/2}d)$. In practice, our accelerated algorithm can achieve at most 5 to 8-fold sampling speedup over the traditional time-sequential $\tau$-leaping with respect to wall-clock time on convex, non-convex, and high-dimensional discrete distributions. Our research broadens the scope of leveraging discrete diffusion models in various challenging areas like molecular structure generation and Large Language Models (LLMs).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles slow, inherently sequential sampling in discrete diffusion by introducing a parallel-in-time $\tau$-leaping scheme based on Picard iteraiont. The key ideea is to replace stespwise dependencies with trajectory-wise updates inside time blocks, enabling parallel prefix oeprations. To control additional error, the method pairs with the predictor with MCMC correctors, including a Top-K barker variant suited for large vocabularies. Theoretical analysis claims a runtime improvement from roughly $O(d^2)$ to $O(d^{3/2})$ under explicit assumptions.

### Strengths
- The core algorithmic contributions, Picard $\tau$-leaping for discrete diffusion, is conceptually clean and well-motivated by parallel hardware.
- The paper provides a transparent error decomposition and parameter schedule leading to the stated complexity improvement.
- Empirical results on several synthetic distributions consistently demonstrate meaningful speedups at moderate Picard depth.

### Weaknesses
- All experiments use oracle scores, leaving performance with learned scores untested.
- The main complexity theorem excludes the corrector, so the analyzed system is not fully matching what is evaluated.

### Questions
- Despite the improvement in the exponent of $d$ from 2 to 3/2, the exponent of the log term $\log d$ increased from 2 to 5/2. Is there an intuitive explanation to this? By the way, is it possible to improve it from 5/2 to 1, drawing the technique from [1], which looks like a similar method for the continuous diffusion models?
- It seems the proposed algorithm requires a larger memory than the original inference scheme. In practice, is the cost of memory communication significant comparing to ordinary samplers?
- I noticed that two versions of the paper [2] is cited in the manuscript. To the best of my understanding, it seems the authors may have made a typo when trying to refer to the paper [3]. It would be great if the authors correct this in the revision.

[1] Zhou, Huanjian, and Masashi Sugiyama. "Parallel Simulation for Log-concave Sampling and Score-based Diffusion Models." Forty-second International Conference on Machine Learning.
[2] Ren, Yinuo, et al. How discrete and continuous diffusion meet: Comprehensive analysis of discrete diffusion models via a stochastic integral framework. The Thirteenth International Conference on Learning Representations.
[3] Ren, Yinuo, et al. Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms. arXiv preprint arXiv:2502.00234 (2025).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel integration method for continuous-time Markov chains by combining the tau-leaping integral with Picard iteration. To mitigate errors introduced during integration, the authors employ an MCMC corrector. While their approach is more memory-intensive than conventional tau-leaping integration, they theoretically demonstrate an improvement in time complexity. The algorithm is evaluated on several discrete toy distributions.

### Strengths
- The paper is well-written and clearly structured.
- The background and methodology are presented effectively, making the approach accessible.
- The experimental results convincingly show that the proposed method enhances sampling performance.

### Weaknesses
- **Clarity and presentation:** While the paper is generally well-written, minor improvements in clarity and flow could further enhance readability (see *Questions*).
- **Experimental details:** Some aspects of the experimental setup lack clarity (see *Questions*).
- **Scope of evaluation:** The evaluation is limited to toy distributions. It would be valuable to assess the method’s performance on more complex tasks, such as diffusion models trained for image generation.
- **Visualization:** Including visualizations (e.g., plots) of the chess and circle distributions (defined on grids) would aid reader comprehension.

### Questions
1. The variable **$K_p$** is first mentioned in Section 5.2 but appears undefined in earlier equations. Could the authors clarify its definition and usage?
2. The paper does not specify how the **KL divergence** is computed. Are samples from the target distribution available? If so, would the **Wasserstein distance** be a more appropriate metric?
3. Why was the method not evaluated on **diffusion models**, which are a relevant and challenging benchmark?
4. **Appendix C.2:**
   - In line 793, the term **$ds$** appears in the equation but is absent in the equation on line 799. Could the authors clarify this discrepancy?
   - The variable **$w_i$** is not defined anywhere in the section. Could its definition or role be provided?

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
2

### Summary
The authors proposed a novel approach of parallelizing a $\tau$-leaping algorithm for absorbing discrete diffusion. The algorithm leverages parallel structure in the Picard iterations as well as the additional tricks such a new predictor-corrector scheme to control additional errors. While the parallelization is only applied in one specific component of the algorithm (most of the algorithm keeps being sequential), the authors prove that their approach decreases the time complexity from O(d^2) to O(d^3/2) [this omits logarithmic terms], at the expense of increased space complexity however. They also demonstrate that their approach leads to increased runtime speeds in some practical problems.

### Strengths
The proposed algorithm is theoretically justified to bring a time complexity improvement over existing algorithms. The paper also has some empirical evidence of how the runtime improves compared to sequential approaches.

The paper overall is clearly written, all the claims are proven.

### Weaknesses
**Minor concerns**:

I find it a bit confusing that the authors chose to use time-complexity symbol which omits the logarithmic terms in the abstract. For a reader, it is misleading and paints slightly over-optimistic picture of the contribution.

**Major concerns**:

* The propose method actually is not of O(d^3/2) complexity but has some additional logarithmic terms added to time-complexity. Moreover, it requires a higher memory going from O(d) to O(d^3/2 log^1/2(d)). Given these increased constraints, it is important the authors provide a more careful empirical study of how these parameters can affect performance. In particular, can authors run experiments on real hardware (real GPUs with additional bottlenecks) and show the real memory consumed as well as runtimes for different number of GPUs and different values of dimensionality d (and the parameters of the algorithm)? I think it is important to understand the limitations of the approach, since in practice we could sometimes be bottlenecked by memory.

* The authors provide runtime comparison of their algorithm to its sequential counterpart,  but they do not, in fact, compare performance of their approach to methods listed in Table 1. I think it will be highly informative to see what is the performance of these baseline methods. The authors should vary (d), because while their algorithm has O(d^3/2) (omitting log terms) complexity, the constants can be high. It is therefore important to compare to baselines and see what runtimes are available in practice.

### Questions
1. How does performance depends real hardware and number of GPUs when we vary d and algorithm parameters? What are the memory requirements?

2. Can authors compare performance to baseline methods which are listed in Table 1?

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
The paper addresses the significant computational cost of sampling from discrete diffusion models, a process that is inherently sequential and time-consuming. It introduces the "Picard $\tau$-leaping" method, the first parallel-in-time sampling algorithm for absorbing discrete diffusion models. The method leverages the continuous-time stochastic integral form of the $\tau$-leaping algorithm and applies the Picard iteration method to achieve parallel-in-time acceleration. The paper demonstrates a theoretical improvement in time complexity for $\tau$-leaping from $\tilde{\mathcal{O}}(d^{2})$ to at most $\tilde{\mathcal{O}}(d^{3/2})$. In practice, the algorithm achieves a 5 to 8-fold sampling speedup in wall-clock time over the sequential method while maintaining sample quality on various discrete distributions.

### Strengths
The paper presents the first parallel-in-time $\tau$-leaping algorithm designed specifically for discrete diffusion models. This is a significant contribution, as previous parallel-in-time work focused almost exclusively on continuous diffusion models.

The practical experiments demonstrate a substantial 5 to 8-fold reduction in wall-clock runtime compared to the sequential baseline. This is achieved while maintaining comparable or identical sample quality (as measured by KL Divergence).

### Weaknesses
The paper's motivation cites challenging, high-dimensional applications like molecular generation and Large Language Models (LLMs). However, all experiments are conducted on low-dimensional, synthetic distributions (e.g., $8 \times 8$ Chessboard, 6-d Hypercube). There is no empirical evidence to support the claim that the speedup will translate to these complex, high-dimensional real-world tasks.

While $\tilde{\mathcal{O}}(d^{3/2})$ is an improvement over $\tilde{\mathcal{O}}(d^{2})$, this complexity is still computationally prohibitive for the high-dimensional problems (large $d$) that motivate the paper. The authors acknowledge that the block width $h$ cannot achieve $\mathcal{O}(1)$ with respect to $d$, which restricts the degree of parallelism.

The experiments are performed in a "best-case" scenario using an "Oracle Model". The study does not evaluate how the algorithm performs under realistic conditions where the score function ($\hat{s}_t^{\theta}$) is approximated by a noisy neural network. It is unknown how the parallelization error and the score estimation error will interact.

The theoretical convergence guarantees (Theorem 3.1) rely on a newly introduced Lipschitz continuity condition (Assumption 3.4). The justification for this assumption is brief, stating only that it "could be reasonable", and it is not as standard as other assumptions in the literature.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
