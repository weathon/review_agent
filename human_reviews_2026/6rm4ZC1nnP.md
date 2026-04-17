# LDC-MTL: Balancing Multi-Task Learning through Scalable Loss Discrepancy Control

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Multi-task learning (MTL) has been widely adopted for its ability to simultaneously learn multiple tasks. While existing gradient manipulation methods often yield more balanced solutions than simple scalarization-based approaches, they typically incur a significant computational overhead of $\mathcal{O}(K)$ in both time and memory, where $K$ is the number of tasks. In this paper, we propose LDC-MTL, a simple and scalable loss discrepancy control approach for MTL, formulated from a bilevel optimization perspective. Our method incorporates two key components: (i) a bilevel formulation for fine-grained loss discrepancy control, and (ii) a scalable first-order bilevel algorithm that requires only $\mathcal{O}(1)$ time and memory. Theoretically, we prove that LDC-MTL guarantees convergence not only to a stationary point of the bilevel problem with loss discrepancy control but also to an $\epsilon$-accurate Pareto stationary point for all $K$ loss functions under mild conditions. Extensive experiments on diverse multi-task datasets demonstrate the superior performance of LDC-MTL in both accuracy and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates MTL from the perspective of loss balancing. Specifically, it proposes a bi-level loss balancing paradigm to dynamically adjust the task weights and promote the balanced optimization. Besides, it also proposes a first-order alternative for efficiency purposes. Extensive experiments have demonstrated the proposed method's ability to achieve competitive performance across multiple mainstream MTL datasets.

### Strengths
1. Employing a bi-level paradigm for MTL becomes popular recently, thus this paper is timely.
2. Developing efficient MTL algorithms is crucial in the era of large foundation models, and this paper addresses a timely and important issue.
3. Extensive evaluation demonstrates excellent performance across mainstream MTL datasets.

### Weaknesses
1. It remains unclear how Eqn. (1) facilitates balanced task learning speeds, especially given that $\tau$ is fixed at 1 in most experimental settings.
2. The paper lacks sufficient details regarding the router model, which plays a critical role in the proposed framework.
3. The proposed method adopts a different training schedule (e.g., learning rate, batch size) compared to prior MTL approaches [1][2]; please provide a discussion on GPU memory usage under this setting.
4. It is recommended to report another widely adopted MTL metric, Mean Rank (MR), for a more comprehensive evaluation [1][2].
5. Please specify which version of LDC-MTL was used for evaluation.

Reference:

[1] Fair Resource Allocation in Multi-Task Learning. ICML 2024.

[2] FAMO: fast adaptive multitask optimization. NeurIPS 2023.

### Questions
Please refer to the Weaknesses.

### Soundness
3

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
This paper introduces LDC-MTL, a scalable multi-task learning method that tackles the persistent issue of loss discrepancy across task objectives. Motivated by the difficulty of balancing tasks whose losses operate on different scales or progress at different speeds, LDC-MTL formulates the learning challenge as a bilevel optimization problem: the lower-level optimizes a weighted sum of task losses, and the upper-level adjusts the weights to minimize their discrepancies. The authors propose a single-loop, first-order algorithm that claims $\mathcal{O}(1)$ memory and time overhead per iteration, compared to $\mathcal{O}(K)$ for previous gradient manipulation techniques. Experiments across standard MTL benchmarks demonstrate the efficiency and competitive accuracy of LDC-MTL.

### Strengths
1. The bilevel formulation provides a clear and principled avenue for directly controlling loss discrepancies while aiming for balanced performance across tasks (Section 4).

2. The authors conducted extensive experiments to verify the effectiveness of the proposed LDC-MTL.

3. The authors provide a thorough theoretical analysis, showing that the algorithm achieves $\epsilon$-Pareto stationarity under standard Lipschitz and PL conditions.

### Weaknesses
1. The authors claim that LDC-MTL achieves $\mathcal{O}(1)$ memory and time overhead per iteration. However, this property has already been stated in prior work such as FAMO. In fact, all loss-based MTL methods (e.g., UW, DWA, and more recently GO4Align) inherently possess this characteristic, as they aggregate task losses directly rather than computing gradients for each task separately—thus requiring only a single backward pass per iteration. Therefore, this advantage cannot be regarded as a distinctive contribution of the proposed method.

2. The idea of employing bilevel optimization for multi-task learning is not novel. Recent works, including GO4Align and ConsMTL [2], have already adopted similar formulations, making the conceptual contribution of this paper relatively incremental.

3. In Figure 1, the authors present the performance of LDC-MTL and other methods on a toy example. However, since there is no dominating solution along the Pareto front, convergence to the same Pareto stationary point cannot be considered a meaningful advantage in the absence of any prior preference among tasks.

4. The proposed method involves several hyperparameters, such as the learning rate $\alpha$ and the penalty coefficient $\lambda$. The authors determine these through grid search, which contradicts the claimed advantage of computational efficiency, as it implies multiple runs are needed to find suitable values. More thorough ablation studies and analyses are necessary to demonstrate the robustness of the method and to provide practical guidance for hyperparameter selection.

5. Confusing supplementary material. The authors appear to provide code in the supplementary material, yet upon inspection, there seems to be no actual implementation corresponding to the proposed LDC-MTL method. This raises concerns about reproducibility and transparency; the authors should ensure that the provided materials are complete and directly related to the presented work.

6. Missing references. The paper overlooks several key related works [1,2,3], some of which report superior empirical performance (though this does not affect my evaluation of the paper). In the MTL field, there are no universally accepted quantitative metrics for superiority—measures such as $\Delta m\\%$ and MR are only proxy indicators, and it is well known that $\Delta m\\%$ tends to favor baselines with lower initial performance. Therefore, I do not insist that the authors include these methods in their experiments, but they should at least acknowledge and differentiate them properly in the Related Work section.

[1] Independent component alignment for multi-task learning. CVPR 2023

[2] Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters. CVPR 2025

[5] Rep-MTL: Unleashing the Power of Representation-level Task Saliency for Multi-Task Learning. ICCV 2025

### Questions
Refer to weaknesses.

### Soundness
2

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
This work proposed a first-order bi-level optimization approach to solve the multi-task learning problem. The proposed method mainly contains two components: (i) a bilevel formulation for fine-grained loss discrepancy control, and (ii) a scalable first-order bilevel algorithm that requires only constant order time and memory.

### Strengths
1.  The proposed method avoids solving the complex bi-level structure in BLO. It also achieves good performance in MTL. 
2. The paper structure is clear and easy to follow.
3. A theoretical proof to support the effectiveness of the proposed method.

### Weaknesses
1. Lacking $\mathcal{O}(1)$ baselines such as "Smooth Tchebycheff Scalarization for Multi-Objective Optimization, ICML 2024."

2. Using bi-level optimization to solve the MTL problem is not new. The "related work" part does not mention those works, such as "Multi-objective meta-learning, AIJ" and "A first-order multi-gradient algorithm for multi-objective bi-level optimization, ECAI 2024".

3. The proposed approach is quite like a penalty-based bi-level optimization approach, such as "On Penalty-based Bilevel Gradient Descent Method". Though the author cites some papers  (Kwon et al., 2023; Yang et al., 2023), but does not discuss the differences and how similar these approaches are. This is important for evaluating the theoretical contributions of this paper. In my opinion, the theoretical result does not show an advantage.

### Questions
See the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes LDC-MTL, a scalable loss discrepancy control method for multi-task learning (MTL). It formulates MTL as a bilevel optimization problem, where the lower level minimizes a weighted sum of task losses and the upper level minimizes the discrepancies among those losses. A first-order, single-loop approximation is introduced, leading to an $\mathcal{O}(1)$ time and memory algorithm that scales with the number of tasks. Theoretical analysis establishes convergence both to a stationary point of the bilevel problem and to an $\epsilon$-accurate Pareto-stationary point. Experiments on CelebA ($40$ tasks), QM9 ($11$ tasks), NYU-v$2$ ($3$ tasks), and Cityscapes ($2$ tasks) show competitive or superior $\Delta m\\%$ compared to scalarization and gradient-manipulation baselines, with runtime close to linear scalarization (LS).

### Strengths
The paper's main strengths lie in its clear bilevel formulation for controlling task-wise loss discrepancies, which provides a principled alternative to existing loss balancing methods. The proposed approach is both scalable and lightweight, featuring a single-loop update that avoids the typical $\mathcal{O}(K)$ gradient storage overhead found in many multi-task algorithms. Theoretical analysis further reinforces the work by providing convergence guarantees that ensure Pareto stationarity under mild assumptions. Empirically, the paper presents comprehensive experiments across four benchmark datasets and compares against competitive baselines, demonstrating consistent improvements. Moreover, the additional analyses on gradient conflict reduction and loss variance offer valuable insights into the behavior and interpretability of the proposed method.

### Weaknesses
1. The claimed $O(1)$ efficiency critically depends on the empirical observation that $||\nabla_W g(W^t,z^t_N)||$ remains small. While this is illustrated for the Cityscapes dataset $(K = 2)$, providing similar empirical evidence for other datasets, particularly CelebA $(K = 40)$, would strengthen the generality of this claim.

2. The experimental analyses on loss discrepancy and gradient conflict (Table $3$, Figures $6$ and $7$) are conducted only against linear scalarization. For a fairer assessment, comparisons should also include the strongest loss balancing baselines, such as GO4Align, which achieves top performance in the main experiments.

### Questions
1. Could you provide a performance comparison between Algorithm $1$ and Algorithm $2$? Additionally, what do you think of a hybrid strategy that runs Algorithm $2$ until $||\nabla_W g(W^t, z^t_N)||$ drops below a threshold relative to $||\nabla_W g(W^t, x^t)||$, and then switches to Algorithm $1$ for the remainder of training?

2. Could you include a comparison with ConsMTL $[1]$, a recent state-of-the-art gradient-manipulation method that formulates MTL as a bi-level optimization over shared and task-specific parameters? This would strengthen the claim that LDC-MTL consistently outperforms gradient-based baselines.

3. In Appendix A-$7$, it is stated that the scatter plots in Figures $5 (a)$ and $5 (b)$ follow the same setting as Figure $8$ in $[2]$. Does this imply that the axis labels in Figures $5 (a–b)$ might have been mislabeled? Additionally, could you include FAMO and GO4Align (each trained with distinct random seeds) in this Pareto front analysis for a more comprehensive comparison?

$[1]$ Qin, Xiaohan, Xiaoxing Wang, and Junchi Yan. "Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

$[2]$ Xin, Derrick, et al. "Do current multi-task optimization methods in deep learning even help?." Advances in neural information processing systems 35 (2022): 13597-13609.

### Soundness
3

### Presentation
3

### Contribution
3
