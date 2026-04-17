# Problem-Parameter-Agnostic MAML

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Meta-learning aims to equip artificial intelligence systems with the ability to learn how to learn. Among its methods Model-Agnostic Meta-Learning (MAML) is particularly effective for enabling rapid task adaptation. However, vanilla MAML suffers from a critical drawback: its performance is highly sensitive to carefully tuned hyperparameters, especially learning rates. Since theoretically these learning rates depend on problem-specific factors (e.g., task heterogeneity and loss smoothness) that are typically unknown, this reliance hinders training stability and limits adaptation performance.
To address this challenge, we propose TFMAML, a tuning-free MAML algorithm that integrate adaptive stepsize and momentum techniques. 
TFMAML offers two key advantages: (i) it eliminates dependence on problem-specific parameters, allowing stepsizes to be pre-set without costly manual tuning or additional training process;
ii) it guarantees convergence, unlike vanilla MAML, which lacks convergence guarantees to first-order stationary points.
We provide rigorous theoretical analysis showing that TFMAML achieves the state-of-the-art convergence rate of $\mathcal{O}(\epsilon^{-4})$ to reach FOSP. Furthermore, we prove that the its first-order variant, TFFOMAML, avoids Hessian computations while retaining the same $\mathcal{O}(\epsilon^{-4})$ convergence rate. 
Unlike standard First-Order MAML, which suffers from a constant error floor, TFFOMAML eliminates this bias and converges reliably to stationary points.
Extensive experiments validate our theory, demonstrating the clear superiority of TFMAML and TFFOMAML over existing benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the problem of optimization of the model-agnostic meta-learning objective, which is challenging for two reasons: 1) the nested structure and 2) the stochasticity of the inner update. Previous analyses focus on stochastic gradient descent, which rely on problem-specific learning rates and thus are difficult to apply in practice. This work proposes to use the normalized stochastic gradient descent with momentum, which enables convergence guarantees that 1) do not rely on problem-specific learning rates, step-sizes, etc. and 2) has bounds that converge to zero as the number of optimizations goes to infinity. Experimental results on Omniglot with a 4-layer CNN show improved performance over SGD and Adam after a small number of iterations.

### Strengths
- The paper is well-written, with the main ideas clearly emphasized.
- The proposed algorithm is simple and straightforward to apply in practice.
- The theoretical analyses appear rigorous and sound (although I did not check the math).
- The experiments validate the theoretical results and the effectiveness of the algorithm over previous works.

### Weaknesses
I believe that the main weakness of the paper is in its novelty.
- The proposed algorithm is a naive application of the normalized SGD with momentum framework from Cutkosky (2023).
- The same bounds to lemmas 1 and 2 are found in Fallah et al. (2020), but the paper does not make this clear. It is not clear what the main theoretical innovations are in this paper, so I would appreciate if that is clarified.

### Questions
- How does the dependency on T in theorems 1 and 2 compared to SGD?
- FO Adam appears quite good compared to Adam, which is not true of TFMAML and SGD. Is there any intuition why?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the author proposes a new optimizer for meta-learning that incorporates normalized gradient updates. The work identifies that, during meta-learning updates, momentum can become unstable and introduce systematic bias. Additionally, variability in gradients makes selecting an appropriate step size challenging. To address these issues, the author introduces a framework called TFMAML, which integrates momentum aggregation with gradient normalization to better control the step size. The paper further provides theoretical analysis and empirical results demonstrating the effectiveness of the proposed method.

### Strengths
1: The author identifies a key challenge in optimizing meta-learning: the large variance of gradients, which makes stable updates difficult.

2: The author provides theoretical analysis that helps address the update challenge and demonstrates the importance of the proposed approach.

### Weaknesses
1: The paper’s presentation is difficult to follow. It frequently switches between prior work and the proposed method, making the narrative unclear. A clearer structure separating background, method, and contributions is needed.

2: The main contribution is not clearly stated. It appears the paper proposes an outer-loop optimizer for MAML using gradient normalization for step-size control, while momentum aggregation is already common in optimizers like Adam. The novelty and distinction from existing methods should be clarified.

3: Experiments rely mainly on Omniglot. Standard benchmarks such as miniImageNet and tieredImageNet should be included. Moreover, the reported Omniglot accuracy (70%–80%) is far below commonly reported MAML results (98.7%), which requires explanation.

### Questions
1: Could you more clearly describe your method? Specifically, where are momentum and gradient normalization applied — in the inner loop or the outer loop?

2: Typically, the outer loop of MAML uses Adam, which already incorporates momentum and adaptive step sizes. How does your approach differ from Adam in this context?

3: Could you clarify how your method addresses the limitations discussed in the paper, especially in comparison to Adam?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method that removes the need to tune the learning rate of MAML in a multitask learning setting. MAML and its subsequent variant FOMAML are well known as representative methods in meta learning, yet theory has shown that the inner-loop step size must be tuned to problem-specific factors such as task heterogeneity and loss smoothness. The paper shows that, under certain assumptions, convergence of the proposed method can be guaranteed by scheduling the step size using quantities observable from the dataset and the problem setup.

### Strengths
- The logical flow is clear and well written. In particular, the problem setup and assumptions are stated explicitly, which makes the argument easy to follow.
- It is meaningful that the paper show the global-gradient estimator becomes biased due to the estimation error of the inner-loop gradient.
- The theoretical results appear sound.
- Prior work on step-size scheduling from a theoretical viewpoint required approximations using quantities such as gradient norms, whereas the proposed method uses only quantities that are evident from the problem setup. This is a solid contribution from a practical perspective.
- On Omniglot, the numerical experiments show that the proposed method achieves higher generalization performance with fewer iterations than MAML or FOMAML with naive SGD. It also achieves performance comparable to MAML or FOMAML that use Adam, which is common in practice.

### Weaknesses
- **Insufficient empirical validation:** While the theoretical contribution is valuable, there are concerns about whether the theoretical claims truly align with empirical facts, as follows.
    - The theoretical claims mainly discuss convergence guarantees of the proposed method, which is distinct from generalization performance. In contrast, the experiments compare the proposed method with baselines in terms of generalization, so the theory was not directly verified.
    - Lemma 2 shows that the original MAML has a bias in its global-gradient estimate. How problematic is this bias in practice was not shown. Although the paper provides an upper bound on the norm of the bias, that bound suggests the bias could be negligible in realistic regimes depending on the loss, gradient smoothness, and the choice of step size. A numerical study that measures the magnitude of gradient estimation bias under realistic settings would strengthen the paper. A similar study is also needed to show how momentum alleviates the bias.
    - From a practical viewpoint, the main theoretical contribution appears to be an algorithm that does not require hyperparameter tuning. However, the experiments essentially compare the proposed algorithm and baselines essentially on a single problem setup. To substantiate the claim that the method is truly problem-parameter-agnostic, the paper should show robust effectiveness across multiple datasets and/or model architectures.
- **Concerns about consistency in the narrative:** The introduction highlights two main issues of MAML:
    1. the presence of bias in the estimation of the global gradient, and
    2. the fact that the appropriate step size depends on problem-specific factors.
    
    The paper provides a clear discussion for the second point and explains why the proposed approach can resolve it. For the first point, however, it does not sufficiently explain how introducing momentum removes the estimation bias, especially the kind described in Lemma 2. The paper cites prior work to claim that momentum is helpful, but it should give a more concrete logical explanation of how momentum addresses the bias.

### Questions
- Reading the problem setup in Section 2, it appears that both training and test tasks are sampled from the same finite set of tasks. In the meta learning scenario targeted by MAML, one typically aims to infer unseen tasks at test time that were not provided during training. Am I correct that the problem setting in this paper is multitask learning rather than meta learning? Since the abstract and other parts describe the setting as meta learning, I found this somewhat misleading.
- Related to the previous question, meta learning is essentially about generalization to unseen tasks. From that viewpoint, how do guarantees on the time-averaged gradient norm converging over iterations contribute to generalization performance?
- In line 112, since $\mathcal{I}$ is a set, should it not be $\mathcal{I} = \\{1, \ldots, I\\}$? It seems the curly braces are missing.
- As noted in the experimental section, practitioners typically use Adam rather than plain SGD. Can any discussion or extensions be derived from the theoretical results to cover such optimizers?

### Soundness
2

### Presentation
3

### Contribution
4
