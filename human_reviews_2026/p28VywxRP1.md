# Aligning Distributionally Robust Optimization with Practical Deep Learning Needs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 0, 6, 6

## Abstract
Deep learning (DL) models often struggle with real-world data heterogeneity, such as class imbalance or varied data sources, as standard training methods treat all samples equally. Distributionally Robust Optimization (DRO) offers a principled approach by optimizing for a worst-case data distribution. However, a significant gap exists between DRO and current DL practices. DRO methods often lack adaptive parameter updates (like Adam), struggle with the non-convexity of neural networks, and are difficult to integrate with group-based weighting in standard mini-batch training pipelines. This paper aims to bridge this gap by introducing ALSO -- Adaptive Loss Scaling Optimizer -- a novel optimizer that integrates an adaptive, Adam-like update for the model parameters with an efficient, principled mechanism for learning worst-case data weights. Crucially, it supports stochastic updates for both model parameters and data weights, making it fully compatible with group-based weighting and standard Deep Learning training pipelines. We prove the convergence of our proposed algorithm for non-convex objectives, which is the typical case for DL models. Empirical evaluation across diverse Deep Learning tasks characterized by different types of data heterogeneity demonstrates that ALSO outperforms both traditional DL approaches and existing DRO methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the distributionally robust optimization for deep learning problems. The Adam-like updates are conducted to the non-convex objectives. The authors provide the ALSO algorithms with two options and provide both theoretical and empirical results.

### Strengths
1. This paper introduces the ALSO framework, which employs the Adam algorithm to update parameters for non-convex loss functions. Additionally, the authors establish theoretical convergence results for the proposed method.

2. The authors present extensive empirical evidence to support and validate their proposed approach.

### Weaknesses
For distributionally robust optimization (DRO) and general minimax problems, there are some related works that should be discussed in this paper.
1. In line 117, the description of Qi et al. (2021) appears to be inaccurate. Qi et al. (2021) study the dual formulation of KL-DRO, where the objective is defined as $f=\lambda \log(\mathbb E(\exp(f_i)/\lambda))$. This objective is a compositional function, making it challenging to optimize. However, the current paper claims that Qi et al. (2021) studied  $f=\mathbb E(\exp(f_i)/\lambda)$ problem, which is much easier.
2. Although this paper argues that existing DRO methods often lack adaptive parameter updates and focuses on constrained DRO problems (as shown in equation (4)), there are indeed several personalized DRO approaches that incorporate adaptive updates, such as the Normalized SGD with Momentum in [1].

Furthermore, while this paper investigates the primal formulation of DRO problems, for Option 2 in the proposed ALSO framework, there does not seem to be a clear distinction between the proposed method and general non-concave–convex minimax optimization methods. In particular, several adaptive minimax methods already exist, such as [2] and [3]. The paper should discuss these related works, especially [2], which demonstrates that Adam-based minimax methods can be directly applied in this context.


[1] Jin, Jikai, et al. "Non-convex distributionally robust optimization: Non-asymptotic analysis." Advances in Neural Information Processing Systems 34 (2021): 2771-2782.

[2] Guo, Z., Xu, Y., Yin, W., Jin, R., & Yang, T. (2025). Unified convergence analysis for adaptive optimization with moving average estimator. Machine Learning, 114(4), 1-51.

[3] Yang, J., Li, X., & He, N. (2022). Nest your adaptive algorithm for parameter-agnostic nonconvex minimax optimization. Advances in Neural Information Processing Systems, 35, 11202-11216.

### Questions
Despite the aforementioned weaknesses, I also have several questions regarding the proposed algorithms.

This paper models the unknown distribution as a parameter with dimension equal to the number of samples n or the number of groups c, and formulates the DRO problem as a minimax problem. A well-known limitation of this approach is that the overall computational complexity depends on the dimensionality of the distribution parameter, making it challenging to handle large-scale DRO problems. Based on Theorem 4.5 in this paper, it appears that this issue also arises in the proposed method.

While Option 2 follows a straightforward and commonly used approach in minimax optimization, Option 1 employs the softmax (SM) function to update the distribution $\pi$. A natural question arises here: although the softmax output can be guaranteed to form a valid probability distribution, the distribution obtained through the projection operation in Option 2 explicitly lies within the uncertainty set. How can the authors ensure that the distribution produced by the softmax operation in Option 1 also belongs to the defined uncertainty set?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes an Adam-like algorithm for Distributionally Robust Optimization (DRO), which is a nonconvex minimax problem. Convergence rate has been established. Experiments have verified the effectiveness over non-DRO methods and non-adaptive DRO optimization.

### Strengths
1. The effectiveness of DRO has been verified on different tasks including on unbalanced data, tabular data, robust training under adversarial attacks, distributed training and split learning.  

2. The convergence results matches some literature in non-adaptive minimax optimization, though I will discuss some concerns later.

### Weaknesses
1. The contribution may be limited since it seems that the problem of concern has already been well studied in the literature. Particularly, the following highly-related literatures have been missed:

[1] Guo, Zhishuai, et al. "Unified convergence analysis for adaptive optimization with moving average estimator." arXiv preprint arXiv:2104.14840 (2021).

[2] Guo, Zhishuai, and Tianbao Yang. "Communication-efficient federated group distributionally robust optimization." Advances in Neural Information Processing Systems 37 (2024): 23040-23077.

[1] has studied Adam based algorithms for nonconvex minimax problems. [2] has studied Adam based algorithms for compositional-formulated DRO problems. Thus the novelty of this submission is questionable. 

2. The analysis seems problematic. Equation (47) does not always hold since the inner product could be negative.

3. The large batch size of $O(1/\epsilon^2)$ makes the algorithm non-practical.

### Questions
1. Given the two above literature, what are the novelty of this submission? 

2. Can the problem in (47) be addressed? Otherwise, the whole analysis would collapse.

### Soundness
3

### Presentation
3

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
The authors present a simple algorithm for distributionally robust learning adapted to deep learning: they use optimistic gradient steps with projected gradient ascent steps for the dual variables, and adam-like updates for the primal variables (i.e. the weights of the deep network). The authors consider a setup where the distributions are not necessarily on samples but on group of samples. For example it could be workers in a distributed setting such that the algorithm would be robust to some specific workers. The authors present a convergence proof to stationary points in a standard setup. Experiments demonstrate the performance of the approach on synthetic and real data.

### Strengths
- The algorithm ends up being rather simple. It is also quite intuitive from previous work. Adam can easily be replaced by other algorithms. 
- The theoretical proof may not reflect the actual practice (assumptions probably are not right for deep networks) but it still demonstrates the overall viability of the approach.
- The authors present 5 sets of experiments illustrating the relevance of the approach.
- The appendix presents numerous additional ablation studies

### Weaknesses
- On several experiments the gains of the method are quite small compared to Adam.
- The theoretical analysis may not help guide the practice since the assumptions may not match the reality of the tasks.

### Questions
- The authors use Adam in many of their experiments but they could exchange it with the best known algorithm for the workload. Typically, on ResNets SGD with momentum may work better. Did the authors try to replace Adam with another algorithm?
- Why do the authors use a regularization rather than weight decay?
- In Figure 1, aren't static weights the gold experiment? Wouldn't they offer the best possible value for the dual variables? Why are they performing less well?

### Soundness
3

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
2

### Summary
The paper proposes ALSO (Adaptive Loss Scaling Optimizer), an optimizer for distributionally robust training that integrates (i) an Adam-style adaptive update on model parameters ($\theta$) with (ii) a principled update of group/sample weights ($\pi$) on the probability simplex using a KL prior. The central goal is to bridge practical deep learning (DL) training pipelines—mini-batching, Adam, non-convexity, group-based weighting—with DRO formulations that are often impractical in DL.

### Strengths
- Well-motivated bridge from theory to practice. The problem framing correctly diagnoses the friction between existing DRO methods and DL practice (non-convexity, Adam-style training, batching, grouping).

- Simple, drop-in algorithm with practical details. Algorithm 1 is easy to implement ; the $\pi$ update (Option I) is a one-line mirror step (softmax on a shifted log-$\pi$), and the method slots into standard mini-batch training.

- Broad, convincing empirical coverage. Five diverse heterogeneity regimes: imbalance (Figure 1), tabular (Table 1), adversarial (Figure 2), distributed (Figure 3), split learning (Figure 4). Results consistently favor ALSO, especially for severe imbalance (uc ≥ 30 in Figure 1) and split learning (faster, smoother convergence in Figure 4).

### Weaknesses
- Mismatch between DRO objective and evaluation metrics. Across settings, evaluation often reports mean metrics (e.g., average accuracy over attacks in Section 5.3/Figure 2; overall F1 in Section 5.1/Figure 1), while DRO is about worst-case (or tail) risk.

-  Assumptions 4.1–4.3 (L-smoothness, Lipschitzness, unbiased variance-bounded oracles) and Theorem 4.5 adopt $\beta_2$ and batch size scalings tied to $\epsilon$ (e.g., $\beta_2=1-\epsilon^2$). It’s not clear how these map to default hyperparameters in practice.

- Figure 1 includes Upsampling/Static Weights and several DRO methods, but Focal Loss and Class-Balanced Loss are standard for imbalance.

### Questions
- Worst-case metrics: For Section 5, do results remain strong if we switch the evaluation to worst-group (or CVaR@α) rather than mean? Have you computed these metrics already?

- Beyond Section 5.1, what priors did you use in other sections?

### Soundness
3

### Presentation
3

### Contribution
2
