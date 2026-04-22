# Labeled TrustSet Guided: Batch Active Learning with Reinforcement Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
Batch active learning (BAL) is a crucial technique for reducing labeling costs and improving data efficiency in training large-scale deep learning models. Traditional BAL methods often rely on metrics like Mahalanobis Distance to balance uncertainty and diversity when selecting data for annotation. However, these methods predominantly focus on the distribution of unlabeled data and fail to leverage feedback from labeled data or the model’s performance. To address these limitations, we introduce TrustSet, a novel approach that selects the most informative data from the labeled dataset, ensuring a balanced class distribution to mitigate the long-tail problem. Unlike CoreSet, which focuses on maintaining the overall data distribution, TrustSet optimizes the model’s performance by pruning redundant data and using label information to refine the selection process. To extend the benefits of TrustSet to the unlabeled pool, we propose a reinforcement learning (RL)-based sampling policy that approximates the selection of high-quality TrustSet candidates from the unlabeled data. Combining TrustSet and RL, we introduce the \textbf{B}atch \textbf{R}einforcement \textbf{A}ctive \textbf{L}earning with \textbf{T}rustSet (\textbf{BRAL-T}) framework. BRAL-T achieves state-of-the-art results across 10 image classification benchmarks and 2 active fine-tuning tasks, demonstrating its effectiveness and efficiency in various domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper combines batch active learning and reinforcement learning. It introduces TrustSet, a subset of the labeled data that is particularly informative and balanced to mitigate long-tail class issues and redundancy. An RL policy is trained to select batches from the unlabeled pool that approximate high-quality samples as indicated by the TrustSet. Empirical results are reported on 10 image classification benchmarks and 2 active fine-tuning tasks with SOTA performance.

### Strengths
The manuscript presents a hybrid approach that integrates a TrustSet drawn from the labeled data with an RL-based sampling policy. The focus on addressing long-tail and class-imbalance issues through label distribution awareness and model feedback from the labeled pool is interesting. The evaluation is quite extensive.

### Weaknesses
(W1) The paper doesn’t provide clear theoretical or empirical evidence to support the benefits of TrustSet, such as improving class balance or reducing redundancy. Since TrustSet is a key contribution, I think its effectiveness should be explicitly verified through dedicated experiments or analysis.
(W2) The paper replaces accuracy-based rewards with a TrustSet-based one, which seems even less measurable, making it unclear how this simplifies or improves policy training. Given the importance of the reward function in RL, its motivation and value need stronger clarification.
(W3) It would be valuable to evaluate the proposed method on large-scale datasets such as ImageNet to better assess its performance and computational efficiency in an image classification setting.

### Questions
Please refer to the Weaknesses section. Specifically explain how you will improve the paper (as reviewers we are not interested in just private education but in improvements of the manuscripts).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes to combine the benefits of distance-based active learning methods and reinforcement learning-inspired active learning framework (learning to approximate the optimal selection with sampling policy) to improve batch active learning. The paper introduces TrustSet to balance informativeness and diversity for the selected labeled set, and combines with the RL-based sampling policy to form BRAL-T. The proposed method is compared with existing AL baselines on multiple datasets in various settings.

### Strengths
1. The writing of the paper is clear. Key technical elements are well presented within a limited space. The connection from the problem definition to the detailed components in Sections 3 and 4 is clear.

2. Although the connection between active learning and reinforcement learning has been studied, the formulation using a more powerful selection framework (which is distance-based and considers the balance between informativeness and representativeness) and in the batch setting is a novel contribution.

3. The results are evaluated on various tasks, including comparison with important AL baselines that are more recent. The AUBC metric is helpful.

### Weaknesses
1. The biggest concern is about the RL process. Learning a policy by sampling L and U from $L_{i+1}$ can be risky. There is not enough theoretical support or analysis to back up this decision. The selection bias in AL and the limited size of $L_{i+1}$ make the effectiveness of this RL process questionable. Combining with the complexity of batch AL, there is no evidence that the policy can approximate optimal selection.

2. Some components, like the curriculum learning part, add more heuristics to the proposed framework. The ablation study on hyperparameters is not too detailed.

3. In some cases, the performance of the proposed method does not show too much advantage compared to stronger baselines such as WAAL. The starting point of AL curves is unclear, causing trouble in interpreting the AUBC metric. The F-acc metric might indicate some not yet converged results.

### Questions
1. How can the RL process be justified considering the bias in labeled set and limited labeled data size?

2. What is the starting point for AL curves in Figure 4 and is AUBC a fair comparison?

### Soundness
2

### Presentation
3

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
The paper presents a new active learning method called BRAL-T that finds class-balanced TrustSets that balance between representation and hard examples. This method uses GradNd score to find hard data from the labeled pools and limit them to n for class. The loss function employs Super Loss from Curriculum Learning to shift the focus on finding more representative data. A Reinforment Learning model is then used to simulate the TrustSet for less resource.

### Strengths
The method introduces a new way to gather both representative and hard examples by making an example set called TrustSet on labelled data. 

The method improves upon coreset methods which only find representative examples and uncertainty methods which focus on hard examples.

The method is very straightfoward and easy to understand

### Weaknesses
The method seems to rely heavily on the initial labeled data pool to create good TrustSet. However, the specifications of the size or diversity of the labeled pool are not discussed. It seems like the labelled pool nees to have a good representation of the true data distribution for the method to be effective.

The paper claims that using Super Loss with GradNd will select a more diverse and representative subset. This claim should be investigated further with experimental backings.

The experimental section is not entirely convincing. The datasets are very simple such as cifar and mnist. Other real world datasets should be tested on such as Imigenet and Celeb A. More recent benchmarks should be used such as TAILOR (Zhang et al., 2023)

### Questions
Are there any specification for the labelled pool to create good TrustSet?

How does this this method select imbalance examples in extreme rare classes where there are >5 examples for each class?

### Soundness
2

### Presentation
3

### Contribution
3
