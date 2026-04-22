# Policy Newton Algorithm in Reproducing Kernel Hilbert Space

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Reinforcement learning (RL) policies represented in Reproducing Kernel Hilbert Spaces (RKHS) offer powerful representational capabilities. While second-order optimization methods like Newton's method demonstrate faster convergence than first-order approaches, current RKHS-based policy optimization remains constrained to first-order techniques. This limitation stems primarily from the intractability of explicitly computing and inverting the infinite-dimensional Hessian operator in RKHS. We introduce Policy Newton in RKHS, the first second-order optimization framework specifically designed for RL policies represented in RKHS. Our approach circumvents direct computation of the inverse Hessian operator by optimizing a cubic regularized auxiliary objective function. Crucially, we leverage the Representer Theorem to transform this infinite-dimensional optimization into an equivalent, computationally tractable finite-dimensional problem whose dimensionality scales with the trajectory data volume. We establish theoretical guarantees proving convergence to a local optimum with a local quadratic convergence rate. Empirical evaluations on a toy financial asset allocation problem validate these theoretical properties, while experiments on standard RL benchmarks demonstrate that Policy Newton in RKHS achieves superior convergence speed and higher episodic rewards compared to established first-order RKHS approaches and parametric second-order methods. Our work bridges a critical gap between non-parametric policy representations and second-order optimization methods in reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a method for utilizing second-order optimization for policies represented in Reproducing Kernel Hilbert Spaces (RKHS). Direct second-order optimization is not feasible due to the infinite-dimensional Hessian operator in RKHS. Hence, the authors introduce a finite-dimensional optimization problem, whose solution is equivalent to the Newton step. The authors compare their method to the vanilla Policy Gradient and second-order Policy Newton method, as well as the Policy Gradient in RKHS, demonstrating faster convergence (in terms of training iterations).

### Strengths
The authors provide extensive theory for their method, proving a quadratic convergence rate. The empirical evaluation results reflect the superior convergence rate.

### Weaknesses
1. The authors evaluate their method only on three tasks, which are all very low-dimensional, discrete, and relatively simple. I encourage the authors to add more tasks to the evaluation. Is the method also applicable to continuous control tasks?

2. While the proposed method achieves the highest reward out of the methods compared, it still does not seem to solve the LunarLander task consistently (the Gymnasium documentation specifies a reward threshold of 200 for an episode to be considered solved). Furthermore, all of the progress of the RKHS methods in the LunarLander task seems to happen in the first couple of iterations, which are not shown in the plot. For the rest of the training, the performance stagnates. What might be preventing the method from learning to solve the task?

3. The main drawbacks of second-order methods are the increased computation time and the limited scalability to larger models. The paper is lacking an evaluation of the computation time compared to first-order optimization. Furthermore, a comparison of the computation required for different policy sizes would be helpful.

### Questions
1. The introduction is relatively vague about the advantages of RKHS policy representations, simply stating that RKHS "offer a powerful non-parametric alternative, [...], valued for its representational flexibility, potential for improved sample efficiency, and capacity for dynamic adjustment during learning". Perhaps the introduction could be more explicit about what makes this representation more suitable, and in which kinds of tasks might benefit the most from these representations.

2. The description of plot 1b is too short. What exactly does the plot visualize? How is the PCA reduction done? There is no interpretation of the results. Also, the difference in reward between the optimal policy and suboptimal points is hard to assess, as large parts of the plot seem to have more or less the same color.

3. Lines 435-436 state that the "policy optimization in RKHS effectively leverages infinite-dimensional feature representations, enabling the optimization process to escape local optima". How does the feature representation help with escaping local optima?

4. What are the shaded areas in Figure 2?

5. Line 477 states that "Policy Newton in RKHS achieved significantly faster convergence to superior episodic rewards compared to first-order and parameteric Newton baselines", but some of the baselines in Figures 2(a) and (b) did not converge yet, so from the plot, it is not clear whether Policy Newton actually converges to superior episodic rewards.

Comments:

1. Specifying the training progress in "training iterations" in Figures 1 and 2 makes it hard to compare the convergence speed of the methods to other RL algorithms, consider changing the x-axis labels to environment steps.

2. Line 101 cites Maniyar et al. for the policy gradient method. The method, however, goes back to [1], which is not cited here.

[1] Ronald J. Williams "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3 (1992).

### Soundness
4

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
This paper studies second-order policy optimization (Newton) when the policy is modelled by a function belonging to a RKHS. It introduces a tractable method for computing (matrix-vector products with) the inverse Hessian operator, which is infinite-dimensional due to the RKHS assumption. It proves (quadratic) convergence rates for the proposed algorithm, and provides a few empirical evaluations.

### Strengths
The paper is written clearly, and seems to be correct. 
Results are novel and interesting, the proposed method may have a strong impact.

### Weaknesses
I do not see any major weaknesses, however I can highlight a couple of minor issues:

- Section 4.3 seems unnecessary, it’s just a re-statement of known results about convergence rate of Newton’s method on strongly convex losses. Perhaps this space could be used instead to extend Section 3, which represents the main contribution and it’s not very easy to grasp. 

- Line 94: “The objective of RL is to minimize…” It should be “maximize”. Similarly, the following equation should be “argmax”, not “argmin”

### Questions
- Why regularisation is cubic instead of quadratic in equation (3)?

- How does RKHS Policy Newton do in wall clock time? (Figure 1 and 2)

- How do RKHS policy methods do when state and/or actions are high-dimensional?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a policy iteration algorithm for reinforcement learning problems where the policies are formulated directly as elements of a reproducing kernel Hilbert space (RKHS). The method extends second-order optimisation algorithms to the RKHS setting by deriving computationally tractable approximations to the Hessian and the resulting optimal step direction. Theoretical guarantees are provided regarding the approximation error and convergence to an optimal solution, and experiments complement the theoretical results with demonstrations in practical settings, where the algorithm achieves superior performance in contrast to first-order methods and parametric policy iteration approaches.

### Strengths
* Paper is well written and follows a clear structure.
* Rigorous theoretical analysis with resulting guarantees.
* Experimental evaluations show significant performance improvements.

### Weaknesses
* A minimisation problem over $J(\pi_\theta)$ is introduced in Sec. 2.1. Yet, $J$ is formulated as the expected cumulative reward, which an agent should be seeking to maximise, instead of minimise. The result of the regularised Newton step in Eq. 5 also seems to be leading in a descent, instead of ascent, direction.
* Experimental evaluation is limited to a toy experiment and relatively simple classic RL problems (e.g., CartPole).
* Notation for temperature and trajectories set use the same symbol $\mathcal{T}$.
* Non-standard notation for gradient term in first expectation Eq. 4.
* The kernel for the numerical experiments is not specified. Was it a standard Gaussian or Matern kernel? The specific details should be stated in the paper, or at least in the appendix.

### Questions
* Was the objective $J(\pi)$ supposed to be written as the negative cumulative reward? How do you ensure the Newton step is leading in a direction that maximises the expected cumulative reward?
* Is there an alternative reference for the outer product kernel in Definition 3.1? Kubrusly and Vieira (2008) only introduce tensor products between general Hilbert spaces, not particularly RKHSs.

### Soundness
3

### Presentation
3

### Contribution
3
