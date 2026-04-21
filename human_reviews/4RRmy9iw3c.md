# AutoAL: Automated Active Learning with Differentiable Query Strategy Search

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 3, 3, 6

## Abstract
As deep learning continues to evolve, the need for data efficiency becomes increasingly important. Considering labeling large datasets is both time-consuming and expensive, active learning (AL) provides a promising solution to this challenge by iteratively selecting the most informative subsets of examples to train deep neural networks, thereby reducing the labeling cost. However, the effectiveness of different AL algorithms can vary significantly across data scenarios, and determining which AL algorithm best fits a given task remains a challenging problem. This work presents the first differentiable AL strategy search method, named AutoAL, which is designed on top of existing AL sampling strategies. AutoAL consists of two neural nets, named SearchNet and FitNet, which are optimized concurrently under a differentiable bi-level optimization framework. For any given task, SearchNet and FitNet are iteratively co-optimized using the labeled data, learning how well a set of candidate AL algorithms perform on that task. With the optimal AL strategies identified, SearchNet selects a small subset from the unlabeled pool for querying their annotations, enabling efficient training of the task model. Experimental results demonstrate that AutoAL consistently achieves superior accuracy compared to all candidate AL algorithms and other selective AL approaches, showcasing its potential for adapting and integrating multiple existing AL methods across diverse tasks and domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper attempts to tackle the "generalization problem" of active learning (AL) algorithms across data scenarios. I believe this is a core issue in the current active learning field. This paper proposes AutoAL, a differentiable AL strategy search method to select the most effective AL sampling strategies in each iteration. It consists of two neural nets, named SearchNet and FitNet, which are optimized concurrently under a differentiable bi-level optimization framework. The experiments on multiple datasets validate the effectiveness of the proposed approach.

### Strengths
1. The problem studied in this paper is valuable. This paper presents the first differentiable AL strategy search method. 
2. The proposed AutoAL approach is interesting and easily followed.
3. The paper is well organized. 
4. The experiments validate the effectiveness of the proposed approach, and the ablation study in Figure 5 is insightful.

### Weaknesses
1. My only concern is the efficiency of the AutoAL algorithm. Although more efficient solutions have been proposed to solve second-order optimization problems, I cannot find any relevant experiments to verify them.

### Questions
See the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
1

### Summary
This paper proposes an active strategy query algorithm where the optimal query strategy is selected by a bi-level optimization network. In particular, the authors aggregates the query strategies by a scoring function implemented as a Gaussian Mixture Model. Then, the authors split out a validation set from the labeled samples to guide scoring function calculation. Experimental results show that the proposed method supasses the baselines.

### Strengths
1. The studied strategy selection problem for active learning is important.
2. The bi-level optimization strategy is rational.

### Weaknesses
1. There is still room for improvement in the paper writing.

   1.1. It's unnecessary to name the two networks "fitnet" and "searchnet," as it seems intended to make people think this is a significant innovation. However, in meta-learning, this kind of separated network design and bi-level optimization paradigm is very common.

   2.1. The notations are somewhat confusing. For example, the authors didn't clearly define the output of the search net in Section 3.2, making it hard to understand Sec 3.2. It wasn’t until I finished reading the method section that I realized the output is actually a sample-wise score, forming an aggregation of scores for different queries.

2. The novelty of this paper is relatively limited. The proposed meta-learning/bi-level optimization has been applied to AL [1,2]. Also, I think the algorithm design is too complicated.

3. The motivation for modeling the scores by GMM distributions is unclear. Why is the score function of each strategy distributed as a Gaussian Distribution? Why is the final score function a linear weighted aggregation of different strategies? The authors should provide a concrete application or example.

4. The comparison methods are too outdated, with the latest ones being LPL and BADGE from 2019. Additionally, the datasets are quite limited; despite the complexity of the method design, only CIFAR and MNIST datasets were used. Validation should be conducted on the ImageNet dataset (at least Image100). Otherwise, given that the algorithm design is much more complex than the baselines, its effectiveness cannot be convincingly demonstrated.

[1] Kunkun Pang, Mingzhi Dong, Yang Wu, and Timothy Hospedales. 2018. Meta-learning transferable active learning policies by deep reinforcement learning. International Workshop on Automatic Machine Learning (ICML AutoML 2018).

[2] https://grlplus.github.io/papers/96.pdf

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents AutoAL, a framework for automated active learning that optimizes query strategy selection using differentiable methods. Traditional active learning approaches often rely on predefined strategies like uncertainty sampling or diversity sampling, which may not perform optimally across different datasets or tasks. AutoAL addresses this limitation by integrating existing active learning strategies into a unified framework. It employs two neural networks, SearchNet and FitNet, within a bi-level optimization structure to automate the selection process. By relaxing the discrete search space of active learning strategies into a continuous domain, AutoAL enables gradient-based optimization, enhancing computational efficiency and adaptability. Experimental results demonstrate that AutoAL consistently outperforms individual strategies and other selective methods across various natural and medical image datasets, highlighting its effectiveness and versatility.

### Strengths
- A new approach that automates active learning strategy selection through differentiable optimization, surpassing manual and non-differentiable methods.
- Effective integration of strategy selection and data modeling via the bi-level optimization of SearchNet and FitNet.
- Flexibility and adaptability, allowing incorporation of multiple existing strategies and tailoring to specific tasks and data distributions.

### Weaknesses
- Increased complexity and computational overhead due to the additional neural networks and bi-level optimization, potentially challenging scalability on large datasets.
- Dependence on a predefined pool of candidate strategies, which may limit performance if optimal strategies are not included.
- Lack of in-depth theoretical analysis explaining the method's effectiveness and the conditions under which it performs best, possibly affecting generalizability.

### Questions
- How does AutoAL perform in terms of computational efficiency compared to traditional methods on large-scale datasets?
- What mechanisms ensure robustness against convergence issues and local minima in the bi-level optimization?
- Can AutoAL be extended to generate new active learning strategies dynamically rather than relying solely on a predefined candidate pool?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces AutoAL, a differentiable active learning (AL) strategy search method that builds on existing AL sampling strategies. AutoAL contains two neural networks, SearchNet and FitNet, which are co-optimized through a differentiable bi-level optimization framework to identify optimal AL strategies for different tasks. Experimental results show that AutoAL outperforms individual AL algorithms and other selective approaches.

### Strengths
- The work handles an important ML problem; Active Learning (AL) with differentiable strategy search.
- The proposed method is technically sound
- Writing is clear and easy-to-follow

### Weaknesses
- Hybrid AL methods that combine uncertainty and diversity have been demonstrated to perform effectively in a variety of situations. It would be beneficial to include examples where the proposed AutoAL approach is particularly necessary or advantageous for specific applications.
- The bi-level optimization within AutoAL relies on labeled data. How does the algorithm perform if the labeled data is skewed or imbalanced? For instance, if the initial labeled set suffers from class imbalance, might this severely impair the algorithm? The assumption of a randomly selected initial set, as used in the current experiments, appears to be less practical.
- Similarly, is there a guarantee that the AutoAL approach, trained with labeled data from the current AL round, will identify the most informative samples from the unlabeled pool in the subsequent AL round? A more detailed analysis of the algorithm's guarantees is necessary.
- The approach of training an additional network for sample selection shows similarities to [1] employing meta-learning with an additional network for querying.
- Can the proposed method be applied to the open-set AL problem [1]?
- The datasets used in the experiments are of small scale. It is imperative to validate the performance on large-scale datasets, such as ImageNet.

---
[1] Meta-Query-Net: Resolving Purity-Informativeness Dilemma in Open-set Active Learning, NeurIPS, 2022

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
