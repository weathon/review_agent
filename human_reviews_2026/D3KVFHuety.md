# Optimizer Selection Based On Function-Proxy Proximity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Many machine learning problems involve the challenging task of training a model to fit the training data; this task is especially challenging for nonconvex problems.  Many model training algorithms have been proposed, but it is often difficult to determine which algorithm is best suited to a given machine learning problem.  To contend with this challenge, we study the effect of loss function curvature shift on optimizers' proxies of the loss function, and obtain a bound on the rate at which a very large family of optimizers (including all of the most prevalent ones) descend towards the loss function's minimum, while only making relatively weak assumptions on the loss function.  Uniquely, our bound is tight even in parameter subspaces in which the loss function is concave, which have been shown to bear potential for fast descent while being neglected by existing convergence rate bounds.  We demonstrate the applicability of our bound by developing a meta-algorithm for optimizer selection based on it, and validate our meta-algorithm experimentally.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ELMO, a per-eigenspace minimax optimizer derived from cubic upper and lower models of local loss decrease under eigenspace-wise Lipschitz smoothness assumptions. Additionally, the authors introduce AMOS, a meta-optimizer that dynamically switches between first- and second-order updates based on estimated curvature-change rates. The idea is to separate directions where the curvature varies slowly (favoring second-order updates) from those where it changes rapidly (favoring first-order). Experiments on TinyShakespeare with NanoGPT suggest AMOS (combining SGD and Sophia) can match or slightly outperform AdamW while saving compute.

### Strengths
- S1: The paper introduces an original eigenspace-based formulation of curvature smoothness, replacing the usual global Hessian Lipschitz constant with per-direction Lipschitz parameters. This perspective is conceptually fresh and seems like it could influence future optimizer theory.
- S2: The derivation of the minimax ELMO step is mathematically clear and appears internally consistent. Its PSD quasi-Newton interpretation provides a robust theoretical link between cubic-regularization methods and practical quasi-Newton updates.
- S3: The presentation is generally clear, with well-structured definitions and pseudo-code for both ELMO and AMOS, the motivation for each design choice is also easy to follow.
- S4: AMOS demonstrates a practical direction for hybrid optimization: dynamically routing parameters between first- and second-order methods based on curvature-change estimates. In the small-scale tests provided it shows potential for compute-efficient optimization without loss of accuracy.

### Weaknesses
- W1: The eigenspace Lipschitz assumption (and its estimation) is strong and not easily verifiable in practice.
- W2: Experiments are limited to a single small-scale setting, with no comparison to established second-order or hybrid baselines, it would be great to include e.g. K-FAC, Shampoo or cubic-regularization methods as a baseline.
- W3: The computational overhead and scalability are not quantified, making it unclear whether AMOS is viable for large models.
- W4: Experimental evaluation of ELMO is not included.
- W5: The paper is missing some important ablations on key hyperparameters such as the Lipschitz threshold and Hessian computation frequency.

### Questions
- Q1: In Eq. (1), if the gradient happens to align with one eigenvector, all orthogonal directions have zero projection leaving their sign ambiguous. Does this ambiguity introduce any theoretical or practical issues from temporal inconsistencies?
- Q2: How often is the Hessian computed, and by what method (full vs. low-rank)? Please report the runtime and memory overhead relative to AdamW or Sophia.
- Q3: How stable is AMOS on larger or noisier settings e.g., with curvature estimated from stochastic gradients? Are there any observations on how noisy curvature estimates affect optimizer routing?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new optimizer, ELMO (Eigenspace-Lipschitz Minmax Optimizer), and a related meta-algorithm AMOS for dynamic optimizer selection. ELMO is derived from a theoretical analysis of loss function curvature using per-eigenspace Lipschitz constants, yielding a minmax-optimal update rule that generalizes quasi-Newton methods. Building on this theory, AMOS adaptively switches between first- and second-order optimizers (e.g., SGD and Sophia) during training based on estimated local curvature change rates.

### Strengths
The theoretical formulation is interesting and quite ambitious. The idea of analyzing optimizer behavior through eigenspace-specific Lipschitz parameters is novel and could deepen understanding of curvature-aware optimization. The distinction between convex and concave subspaces and the corresponding analysis of their Lipschitz distributions is also very interesting.

### Weaknesses
The work remains largely theoretical. ELMO itself is not evaluated experimentally, and the only empirical results are small-scale tests of AMOS on NanoGPT with the TinyShakespeare dataset, which again seems limited. The proposed approach is computationally heavy:computing or approximating per-eigenspace Lipschitz constants and Hessian eigendecompositions is not practical for modern large-scale models.

### Questions
1. The experimental section does not clearly describe how the Hessian and parameter partitions are computed in practice. From the description, it seems you may use the Hessian estimator from Sophia and then aggregate per-tensor averages. Could you clarify the exact procedure? In particular, how is this implemented for block-diagonal Hessian structures?
2. In Figure 5, the final training loss appears to vary noticeably with different values of the hyperparameter​, which seems larger than the improvement shown in Figure 1 when comparing optimizers. Could you provide the exact final loss values for corresponding experiments in both Figure 1 and Figure 5?
3. Could you provide a wall-clock time comparison between AMOS and the baseline optimizers to illustrate the practical computational overhead of your approach?

### Soundness
2

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
2

### Summary
This paper proposes a new perspective on optimizer selection in machine learning by analyzing optimizer performance in relation to the local Lipschitz continuity of the Hessian in each eigenspace of a non-convex loss function. The authors develop ELMO, a minmax-optimal algorithm based on a refined cubic approximation, and show theoretically how descent rates can be tightly bounded even in regions of concavity. Building on this, they construct AMOS, a meta-algorithm that dynamically selects between first- and second-order optimizers based on per-parameter Hessian Lipschitz estimates, and empirically validate its performance—particularly on training language models with NanoGPT. The paper provides theoretical insights, a generalized optimizer regret bound, and empirical investigations of Hessian spectrum behavior during training.

### Strengths
**1. Strong theoretical contribution**: The paper presents a careful derivation of descent rate bounds for quasi-Newton methods by leveraging a per-eigenspace Hessian Lipschitz continuity assumption.

**2. Empirical validation with strong visual results**: The experimental evaluation is comprehensive, covering from the CIFAR dataset to the modern transformer model nanogpt, supporting the paper’s main claims.

**3. Comprehensive Appendix coverage**: The appendices include substantial mathematical details, ablations, connections to prior metrics, and reproducibility details.

### Weaknesses
I am not an expert in optimization, from my perspective, the paper appears well-integrated. I will finalize my rating after reading other reviewers’ feedback and the authors’ rebuttal.

### Questions
**Q1.** Could the authors provide a comparison of time complexity across different methods? The current figures only report results in terms of iteration counts, which may be unfair to simpler algorithms that have lower per-iteration costs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies how curvature shifts in nonconvex loss functions affect the behavior of optimization algorithms. The authors derive local upper and lower approximation models using an eigen decomposition of the hessian at the current point $\theta_t$. The prvoide a upper bound bound on a function progress using derived models. The paper also proposed two method ELMO and AMOS; ELMO uses eigen decomposistion of the hessian at the current point, while AMOS accumulate information from previous decompositions.

### Strengths
1. The paper provide a local upper bound model using eigenvalue decomposition of the hessian matrix. This bounds can be used for both convex and non-convex functions.
2. Seems like the proposed method performs well on practice.

### Weaknesses
1. The does not provide a thorough review and comparision with existing results on nonconvex optimzation, preconditioned method, and meta-optimizers. I believe it would be beneficial to extend the literatrure review of this paper, and make a comparison of this work in the context of previous results. I don't think that the reference provided in paper are aligned with the topic of interest of the paper. For nonconvex optimization and preconditioned method please see "Spectral Preconditioning for Gradient Methods
on Graded Non-convex Functions" by Doikov et al., 2024, and references there. For meta optimizer literature please see "MADA: Meta-Adaptive Optimizers through hyper-gradient Descent" by Ozkara et al. 2024 and the references there. For nonconvex optimization please see "Deterministic Nonsmooth Nonconvex Optimization" by Jordan et al, 2023. and minimization of majorization function please see Section 2 in Lectures on convex optimization by Nesterov, 2018.
2. The assumption 2 and standard Lipschitz continuous Hessian assumptions are not compared property. I think it is important to highlight the relation between Assumption on Lipschitzness of the hessian with constant $L_H$ and assumption 2. is it if and only if relation? how does $L_H$ from Hessian Lipschitzness correspond to $L_H$ from assumption 2?
3. On lines 234-237, it is said "We also note that our bound is on the rate at which the model’s performance increases (as measured by the loss function). This is in contrast to previous works, ...". I don't think it is true. Most of the results on nonconvex optimization (smooth) use bounds on a function decrease: $f(y) \leq f(x) + \langle \nabla f(x), x - y \rangle + \frac{L}{2}\|y - x\|^2$. By plugging $y=x_{k+1} = x_k - \frac{1}{L} \nabla f(x_k)$ and $x = x_k$ we obtain a bound on "the model's performance": $f(x_{k+1}) - f(x_k) \leq -\frac{1}{2L} \|\nabla f(x_k)\|^2$. 
4. Regarding results on lines 244-247, it seems that similar results can/was derived for second order smooth function. It is well known upper approximation of the function using second order information. The only difference, is that you use eigenvalue decompose of the hessian to provide this bound for all the eigenvectors directions, which in my opinion is not insightful. It would be beneficial to include more reference and comparison with a previous results on nonconvex optimization and the function's upperbounds. Can you please elaborate more on the contribution of this work in the context of existing literature on nonconvex optimization?
5. ELMO method does not define EIGEN function, which find $L_i$ for all i, which might be difficult to compute or estimate.
6. The convergence guaranties are not tractable from the classical optimization point of view and is not compared with previous results.
6. Numerical experiments does not provide enough details. Could you please provide these details, and also present wall-time plots?

### Questions
Please see question in the weaknesses section. Additionally, I list several questions here:
1. Can you please elaborate on "Existing bounds are also rarely applicable to novel optiimzer, ..." on lines 47-49. Can you please provide more details on this claim?
2. Can you please add more details on the meaning on "local concavity of the parameter space", on lines 51-53?
3. On lines 137-138, what does "this" in "that this is a generalization of ARC" refers to? ALso, I believe the timeline in this sentence is confusing.
4. On lines 151-154 can you please elaborate more on the meaning of the sentence?
5. On lines 231-233, "since the negative-definite Hessian...". Could you please provide more details on this claim or reference, or mathematical statement? Consider $f(x, y) = x^2 +xy - y^2$ its hessian has both negative and positive eigenvalue and is concave in one direction, and convex in another direction.
6. Could you please define LIPSCHITZ function/algorithm?
7. On lines 269-271 can you please elaborate on what does "equal applicability" means?
8. How would you find $L^i_t$? It seems like a difficult computational task to find exact $L^i_t$. Are you fining an upper estimation of real $L^i$s?

### Soundness
2

### Presentation
2

### Contribution
2
