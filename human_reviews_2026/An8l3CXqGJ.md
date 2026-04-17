# Towards Permutation Invariant Learning with High-Dimensional Particle Filters

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Sequential learning in deep models often suffers from challenges such as catastrophic forgetting and loss of plasticity. This effect is largely due to the permutation dependence of gradient-based algorithms, where the order of training data affects the learning outcome. In this work, we introduce a novel learning framework based on high-dimensional particle filters that yields approximately permutation-invariant results. We theoretically demonstrate that particle filters are approximately invariant to the sequential ordering of training minibatches or tasks, offering a principled solution to mitigate catastrophic forgetting and loss-of-plasticity. Next, we develop an efficient particle filter for optimizing high-dimensional models, combining the strengths of Bayesian methods with gradient-based optimization. Finally, through extensive experiments on continual supervised and reinforcement learning benchmarks, including SplitMNIST, SplitCIFAR100, and ProcGen, we empirically demonstrate that our method consistently improves performance, while reducing variance compared to standard baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel learning framework that uses high-dimensional particle filters to achieve approximate permutation-invariance in sequential learning tasks. The authors argue that this approach can mitigate common challenges in deep learning, such as catastrophic forgetting and loss of plasticity, which are often exacerbated by the order of training data. They provide a theoretical framework for analyzing permutation invariance with particle filter with provable bonds. It further proposes a gradient-based particle filter to make the theoretical framework align with the practical gradient descent optimization. The flexibility of the proposed method is demonstrated through comprehensive empirical validation across diverse sequential learning benchmarks, including both continual supervised learning and lifelong reinforcement learning tasks.

### Strengths
- Novel Theoretical Framing. The paper provides a novel theoretical lens in analyzing permutation invariant learning. It offers a formal theoretical framework to justify the particle filter approach, which is a significant contribution beyond purely empirical work.

- The paper is well-written, and the core idea is easy to follow. It clearly motivates the problem of permutation dependence and presents its proposed solution in a logical, understandable manner.

- The method is designed as a modular component that can be combined with existing techniques. The empirical results show that this combination can lead to improved performance.

### Weaknesses
- How can permutation invariance directly help mitigate forgetting? The best case I suppose is that it can match the best-order gradient descent performance.

- Weak empirical results. The reported accuracies on SplitCIFAR100 are very low compared to modern continual learning benchmarks, making it difficult to assess the method's true value. On this more complex dataset, the performance gain of the proposed WPF over the standard Gradient Descent baseline is minimal. This suggests the method's benefits diminish with task complexity. So I’m doubtful on its scalability to larger models and datasets.

- The theoretical guarantees (Theorems 1 and 2) are loose to the point of being impractical. The bounds depend on terms that grow exponentially with the number of training steps $T$.

### Questions
Assumptions in Theorem 2 look impractical. First, continual learning task shift leads to a large increase in loss. Second, proper controls on the gradient descent should be needed?

### Soundness
3

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
1

### Summary
This paper introduces a learning framework based on high-dimensional particle filters that yields approximately permutation-invariant results. The main contributions can be summarized as:

a. Theoretically demonstrate that particle filters are approximately invariant to the sequential ordering of training minibatches or tasks.

b. Develop a particle filter for optimizing high- dimensional models.

c. Conduct extensive experiments on continual supervised and reinforcement learning benchmark.

### Strengths
a. This paper seems to be technically solid.

b. I appreciate that this paper focuses on task ordering and tries to study continual learning with permutation invariance.

### Weaknesses
a. Confusing notations. This paper denotes $x$ as the model parameters, which is not regular. I recommend to use $\theta$ or other notations. 

b. This paper is not well written and hard to follow.

### Questions
a. Please deeply discuss the meaning of $P(L_t|x)=e^{-L_t(x)}$.

b. Please provide the motivation of eq.3 and explain it.

c. What is the definition of particle filters?

d. My major concern is about the motivation. I don't figure out how to connect particle filters with continual learning or catastrophic forgetting or plasticity. Please explain the motivation in details.

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
3

### Summary
This paper proposes an approach to achieving permutation-invariant learning using a gradient-based high-dimensional particle filter (WPF). The key contribution is linking Bayesian filtering with sequential optimization to mitigate catastrophic forgetting and loss of plasticity in continual and lifelong learning. The idea is both conceptually interesting and practically relevant.

### Strengths
1. It provides an interesting idea connecting particle filters to permutation-invariant optimization.
2. Provides a formal treatment with several theorems and proofs.
3. Attempts to evaluate across multiple continual learning and RL benchmarks.

### Weaknesses
1. The “gradient-based particle filter” closely resembles an ensemble of SGD trajectories with reweighting, which may not justify the Bayesian framing.
2. The theoretical results rely on restrictive assumptions and do not clearly translate to observed improvements.
3. The empirical gains are moderate and could be explained by model ensembling effects rather than permutation invariance.
4. The proposed method scales linearly with the number of particles, which limits real-world applicability in large-scale deep learning.

### Questions
1. In Theorem 1, you assume constants $C \approx 1$ and $\epsilon$ small for permutation invariance. How realistic are these conditions for the proposed gradient-based particle filter in practice?
2. Can you provide empirical evidence (e.g., measured C and $\epsilon$ values) to support that the theoretical assumptions hold?
3. Is the permutation invariance effect here essentially due to ensembling multiple gradient trajectories?
4. Could you quantify the degree of permutation invariance empirically?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel learning framework that achieves CL by approximately achieving permutation-invariant learning. The proposed algorithm implements a gradient-based particle filter that approximates the Bayesian particle filter, which has the property of approximate permutation invariance to the order of loss function. The algorithm is validated on CL datasets including Split MNIST and Split CIFAR100, as well as CRL tasks including multiple Procgen games. With rather a simple implementation, the algorithm consistently beat some standard CL baselines on these tasks.

### Strengths
- The benchmark tasks, especially the CRL experiments on the Procgen games, are enough to demonstrate the ability of the proposed algorithm.
- The work is closely related to the CL literature. Practical CL algorithms include EWC, LwF, SI that the paper has compared with them, as well as the other CL algorithms that I mentioned previously. This work would also be inspiring to the meta learning and curriculum learning field.

### Weaknesses
This was an ICML 2025 submission, and I was one of the reviewers. At that time, it got a unanimous borderline that leans toward rejection from reviewers and AC.
I do not spot enough changes that address the concerns raised at that time.

Some concerns on the experimental design:

It would be better if the authors could provide more comparisons of the proposed algorithm with other CL methods, including replay-based methods like A-GEM [1], ER-Reservoir [2] and projection-based methods like GPM [3] on the existing tasks. 

[1] Chaudhry, A., Ranzato, M. A., Rohrbach, M., & Elhoseiny, M. (2018). Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420.

[2] Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H., & Ranzato, M. A. (2019). On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486.

[3] Saha, G., Garg, I., & Roy, K. (2021). Gradient projection memory for continual learning. arXiv preprint arXiv:2103.09762.

### Questions
I restate my questions in the ICML review batch. My questions lie mainly on the pseudocode in Algorithm 1.

Q1: How is $\{x_0^{(i)}\}_{i=1}^n$ initialized?

Q2: How is the variance $\sigma^2$ set?

Q3: How does Algorithm 1 affect the training process of $L_t$? I do not see where the model weight is in the pseudocode.

Did you address these theoretical and practical concerns raised previously?

### Soundness
2

### Presentation
2

### Contribution
2
