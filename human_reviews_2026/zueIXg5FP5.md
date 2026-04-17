# Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Given a training dataset, the goal of dataset distillation is to derive a synthetic dataset such that models trained on the latter perform as well as those trained on the training dataset. In this work, we develop and analyze an efficient dataset distillation algorithm for supervised learning, specifically regression in $\mathbb{R}^d$, based on matching the losses on the training and synthetic datasets with respect to a fixed set of randomly sampled regressors without any model training. Our first key contribution is a novel performance guarantee proving that our algorithm needs only $\tilde{O}(d^2)$ sampled regressors  to derive a synthetic dataset on which the MSE loss of any bounded linear model is approximately the same as its MSE loss on the given training data. In particular, the model optimized on the synthetic data has close to minimum loss on the training data, thus performing nearly as well as the model optimized on the latter. Complementing this, we also prove a matching lower bound of $\Omega(d^2)$ for the number of sampled regressors showing the tightness of our analysis.

Our second contribution is to extend our algorithm to offline RL dataset distillation by matching the Bellman loss, unlike previous works which used a behavioral cloning objective. This is the first such method which leverages both, the rewards and the next state information, available in offline RL datasets, without any policy model optimization. We show similar guarantees: our algorithm generates a synthetic dataset whose Bellman loss with respect to any linear action-value predictor is close to the latter’s Bellman loss on the offline RL training dataset. Therefore, a policy associated with an action-value predictor optimized on the synthetic dataset performs nearly as well as that derived from the one optimized on the training data. We conduct extensive experiments to validate our theoretical guarantees and observe performance gains on real-world RL environments with offline training datasets and supervised regression datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes a dataset distillation algorithm by matching the losses between training and synthetic datasets without introducing extra model training. The authors develop and analyze a method for supervised learning and offline reinforcement learning, where only $\hat{\mathcal{O}}(d^2)$ regressors are sufficient to ensure the MSE loss of any bounded linear model is approximately preserved. They also provide a matching lower bound of  $\Omega (d^2)$, establishing tightness. Furthermore, the authors extend the algorithm beyond supervised learning, i.e., offline RL, by leveraging the next state and reward information in the dataset to match the Bellman loss rather than the behavior cloning loss. Extensive theoretical analysis and supplementary experiments prove that the proposed method efficiently distills the synthetic dataset from the training dataset without relying on auxiliary techniques-additional classifier training.

### Strengths
- The paper is well structured and addresses comprehensive details.
- Extensive theoretical proofs concretely support the main claim.
- Experiments demonstrate that the proposed method shows a clear margin compared to other baselines, recovering near-optimal or even outperforming performance.

### Weaknesses
- The experimental section appears relatively narrow, focusing on small regression settings with Gym control tasks. Comparing with other baselines (Lei et al. 2024, Light et al. 2024) for generating synthetic datasets would improve the soundness of the suggested method.

### Questions
- What would be the major bottleneck when extending the proposed method to a non-linear function approximator (i.e., neural network) with offline RL?

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
This paper introduces a dataset distillation method for supervised regression and offline RL that matches the loss on the original and synthetic datasets with respect to a fixed set of randomly sampled models, avoiding bi-level optimization. The key contribution is a theoretical guarantee that only $O(d^{²})$ sampled linear regressors are needed for the supervised case, with a matching lower bound. The experiments demonstrate its effectiveness.

### Strengths
1. The paper provides theoretical guarantees for dataset distillation, an area where such analysis is often lacking. The upper and matching lower bounds for the supervised case are particularly compelling.
2. The experiments adequately support the theoretical claims, showing that the method works well in practice, even with non-linear neural networks, and outperforms baseline approaches like random subsampling.

### Weaknesses
1. The empirical evaluation is limited to relatively small-scale datasets and standard RL benchmarks. A more extensive evaluation on larger-scale or more complex datasets would strengthen the claims of practical efficacy.

### Questions
1. Could the authors discuss potential pathways for extending the theoretical guarantees to non-linear function approximators, such as neural networks? Note that I'm not asking for additional experimental results, just want to have a discussion, since making a more constructive theoretical analysis on non-linear functions is more feasible and practical for most real-world cases.
2. For the offline RL setting, how restrictive is the decomposable feature map assumption in practice?

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
3

### Summary
The paper tackles supervised regression and offline RL settings by constructing synthetic datasets that preserve the original objective of training with original training dataset. It introduces a loss-matching method using randomly sampled linear regressors/Q-value predictors achieving Õ(d^2) sampling guarantees for regression (alongside a matching Ω(d^2) lower bound) and exp(O(d log d)) in offline RL (furthermore relaxed Õ(d^2) under decomposable feature maps). The approach shows solid empirical performance on standard regression datasets and classic offline RL benchmarks, and offers a timely theory-first approach for distillation.

### Strengths
- The use of loss-matching method using randomly sampled linear regressors/Q-value predictors is particularly important tool that is used in RL literature. 
- Two results for each setting look comprehensive as contributions to the learning community.
- Experiments demonstrate that small synthetic sets and few sampled models can perform competitively on standard regression datasets and classic offline RL benchmarks.

### Weaknesses
- Even though experiments demonstrate that small synthetic sets match the performance when trained with entire training dataset, some issues pop up:
- - This phenomena cannot be explained by the current theory. The distillation problem is only interesting for the case $n>>m$. This is also the focus of the literature (Light et al. 25, Lei et al. 24) cited by this work.
- - Lemma C.4 does provide net arguments for the choice of synthetic dataset size. But it seems that both $m$ and $n$ are proportional to $d$, whereas the dependence on $\epsilon$ for $m$ is unclear.
- - The training size for toy sequential decision problems are high. I suggest authors to really use established benchmarks like D4RL to demonstrate their distillation behavior.

### Questions
- na -

### Soundness
3

### Presentation
3

### Contribution
3
