## Human Reviewer 1

### Summary
In this submission, the authors tackle the challenge of black-box optimization. At present, the bulk of the field has concentrated on methods like Bayesian Optimization, which employ a _forward_ model $p(y|\mathbf{x})$ and an acquisition function $\alpha(\mathbf{x})$. Maximizing the latter at each iteration $t$ yields a candidate design $\mathbf{x}_t$ that will in turn give a noisy evaluation $y_t$.

Interestingly, here, the authors use an _inverse_ model $p(\mathbf{x}|y)$ together with an acquisition function $\tilde{\alpha}(y)$. As such, one can learn to generate designs $\mathbf{x}$ conditionally to an output value $y$, such that the evaluated design results in a noisy evaluation $y_t \approx y$. The precise value of $y$ that should be used for conditional generation is obtained by maximizing the acquisition function $\tilde{\alpha}$.

In this work, conditional generation is achieved through conditional diffusion models. While diffusion models have been used in black box optimization previously, building a suitable acquisition function to handle these models has not been done yet. The latter must adequately trade-off exploitation (high values $y$) and exploitation (values $y$ for which uncertainty model uncertainty $p(\mathbf{x}|y)$ is high). The paper's main contribution is a thorough study of uncertainty quantification for conditional diffusion models, leading to a decomposition between _aleatoric_ and _epistemic_ uncertainty. This ultimately leads to an acquisition function that trades off high function values and epistemic uncertainty.

Finally, the performance of the proposed acquisition function is theoretically grounded depending on some assumptions on the target function $f$, and the method itself is shown to outperform concurrent baselines on a number of continuous and discrete datasets.

### Strengths
- I found the paper to be well-organized and motivated. The technical novelty is light but seems to be theoretically grounded. 

- The proposed method consistently ranks among the best competitors on a benchmark involving multiple baselines and datasets, while maintaining computation times in the same order of magnitude as Gaussian Process-based alternatives.

### Weaknesses
- From a practical point of view, I am not sure that Theorems 2 and 3 are useful: they assume the $L$-smoothness of the function $f$. This function takes as input $\x$, which might be discrete, or an embedding of a discrete input like a molecule, and there is little to no chance to have $L$-smoothness in this case unless the embedding explicitly enforces that assumption. I believe this should at least be mentioned.

### Questions
- An interesting ablation study would be to add a factor weight $\beta$ in front of the epistemic uncertainty in Equation 8, and to vary this term. This would give an insight into how important uncertainty is in the acquisition process.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper proposes a novel approach for solving black-box optimization problems. Unlike traditional methods that focus on learning a surrogate to evaluate design decision quality, this approach employs a diffusion model to approximate the distribution within the design space conditioned on a target value. Unlike existing inverse methods that assume access to an offline dataset, this paper studies a dynamic setting where the exploration-exploitation tradeoff must be considered. To address this challenge, an uncertainty-aware exploration method is introduced. The effectiveness of the proposed approach is demonstrated through extensive numerical studies.

### Strengths
- Overall the paper was very well-written. The key concepts and challenges are clearly introduced, which makes it easy for me to appreciate the contribution.
    
- Black-box optimization is an important methodology that has a wide range of applications in engineering and science. 

- The numerical studies are comprehensive and convincing.

### Weaknesses
- **Soundness of the theoretical analysis**. While I appreciate the authors efforts to justify the proposed approach through a theoretical lens, I found some of the results unsatisfying. 
    
  - For example, in Theorem 2, the bound does not depend on the number of samples collected in each iteration $N$. Intuitively, a large $N$ might lead to over-conservative estimates and a small $N$ might renders the estimate too optimistic. Relating the this bound to $N$ may lead to insights into the choice of this important hyper-parameter. The current bound is independent of $N$, suggesting that it might be loose. Furthermore, this theorem assumes the existence of a perfect diffusion model. I suggest the authors add further discussion on the implication/validity of this assumption. This comment applies to Theorem 3 as well.
  - Under a similar Lipschitz assumption, is it possible to derive surrogate approximation guarantees for the forward-based approach? If so, how does the inverse bound compare to the forward bounds?


- **Acquisition function.** This is a minor point, but I was wondering if a weight should be assigned to  $\Delta$ in Equation (8) because (1) the two terms might be in different scales, and (2) the users may dynamically adjust the weight across different iterations to balance exploration and exploitation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
In traditional BBO a surrogate model $\hat{f}$ is learned to approximate the objective function and then helps optimize an acquisition function which yields the next $x_{k+1}$ where to evaluate $f$. To select the next query point $x_{k+1}$ Diff-BBO instead performs posterior inference in the parameter space $X$ conditioned on a specified target objective value $y$ obtained by maximising an introduced Uncertainty-aware Exploration (UaE) acquisition function.

Conditional diffusion model are trained to learn the conditional distribution $p(x | y)$, where $x$ represents feasible inputs in the parameter space.

$y$ is chosen with a proposed acquisition function, Uncertainty-aware Exploration (UaE), that prioritizes target values $y$ with high expected objective values while minimizing epistemic uncertainty. This acquisition function balances exploration and exploitation. The paper provides theoretical proofs demonstrating that UaE achieves a near-optimal solution for the BBO problem.

Numerical experiments to support the work are presented

### Strengths
The paper introduces a new approach supported by solid theoretical results and empirical validation across diverse tasks. Difference with existing methods is clearly presented.

### Weaknesses
An important part of the procedure is how is assembled the candidate set $\mathcal{Y}$ and its corresponding weights $w$. The paper does not specify a principled method for choosing or tuning these weights, which makes this important aspect somewhat empirical given that for too high weights, the model may focus excessively on unfeasibly high values of $y$ while too low weights might limit the search space.


A diffusion model requires a large dataset to effectively learn the data manifold in the design space. If the function $f$ is expensive to evaluate, building a large dataset may be computationally expensive. Most of the experiments are run with a relatively high number of evaluations. How would the method perform on a smaller dataset?

Diffusion models are usually susceptible to mode collapse, where generated samples fail to cover the full distribution of the data. Was this observed? This could cause Diff-BBO to overlook potentially optimal regions in the design space.

### Questions
Is there a systematic approach for choosing the weights $w$ for the candidate set $\mathcal{Y}$, or is this step largely empirical?

Could the method perform effectively with a smaller accumulated dataset, as opposed to the relatively high number of evaluations used in the experiments?

Was this issue of mode collapse observed in Diff-BBO? If so, how does it impact the model’s ability to explore potentially optimal regions in the design space?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
2

---

## Human Reviewer 4

### Summary
This work proposes an inverse approach leveraging diffusion models for online BBO problem. Specifically, this paper introduces a new acquisition function to propose objective function values, and employ a conditional diffusion model to generate samples. The authors conduct experiments in design-bench to verify the effectiveness of their method.

### Strengths
1. This paper is easy to follow. The structure of this paper is clear.
2. This paper provides a solution for online black-box optimization (BBO) called Diffusion-based inverse modeling for black-box optimization (DIFF-BBO), which uses objective function values to generate solutions. This approach is interesting and has the potential to improve the performance of online BBO.

### Weaknesses
1. The core difference between the main idea of the proposed method (the inverse method) and LLAMBO [1] should be explicitly discussed.  While there are differences in the specific implementation details compared to LLAMBO, this approach appears to be more of an aggregation of the previous methods. Specifically, the inverse approach was originally utilized in MINS [2], while the conditional diffusion model was incorporated in DDOM [3].

2. Why do the experimental tasks of online BBO methods use offline BBO benchmarks? Is it because there is no suitable benchmark for online BBO?

3. While the background on online BBO is quite solid, the paper does not sufficiently explore prior work in offline BBO. Expanding the discussion to include them could provide a more comprehensive literature review and motivation support.

4. Superconductor, Ant and D’Kitty and so on are high dimensional problem. So, it would be better if this paper could compare with more high dimensional Bayesian optimization in recent years rather than simple black-box optimization.

[1] Large Language Models to Enhance Bayesian Optimization. In Proceeding of the 12th International Conference on Learning Representations, Vienna, Austria, 2024.

[2] Model inversion networks for model-based optimization. In Advances in Neural Information Processing Systems 33, pp. 5126–5137, Virtual, 2020.

[3] Diffusion models for black box optimization. In Proceedings of the 40th International Conference on Machine Learning, pp. 17842–17857, Honolulu, HI, 2023.

### Questions
Please see Weaknesses part.

Besides, “They struggle with steering clear of out-of-distribution and invalid inputs” is often discussed in offline BBO instead of in online BBO. The reason why this paper presents it here for online setting should be discussed.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
5