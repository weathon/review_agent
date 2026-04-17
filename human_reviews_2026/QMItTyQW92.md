# DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation

- Decision: Accept (Oral)
- Scores: 8, 6, 6

## Abstract
Knowledge Distillation (KD) is a widely adopted framework for compressing large models into compact student models by transferring knowledge from a high-capacity teacher. Despite its success, KD presents two persistent challenges: (1) the trade-off between optimizing for the primary task loss and mimicking the teacher's outputs, and (2) the gradient disparity arising from architectural and representational mismatches between teacher and student models. In this work, we propose Dynamic Trade-off Optimization for Knowledge Distillation (DTO-KD), a principled multi-objective optimization formulation of KD that dynamically balances task and distillation losses at the gradient level. Specifically, DTO-KD resolves two critical issues in gradient-based KD optimization: (i) gradient conflict, where task and distillation gradients are directionally misaligned, and (ii) gradient dominance, where one objective suppresses learning progress on the other. Our method adapts per-iteration trade-offs by leveraging gradient projection techniques to ensure balanced and constructive updates. We evaluate DTO-KD on large-scale benchmarks including ImageNet-1K for classification and COCO for object detection. Across both tasks, DTO-KD consistently outperforms prior KD methods, yielding state-of-the-art accuracy and improved convergence behavior. Furthermore, student models trained with DTO-KD exceed the performance of their non-distilled counterparts, demonstrating the efficacy of our multi-objective formulation for KD.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces DTO-KD (Dynamic Trade-off Optimization for Knowledge Distillation), a multi-objective optimization framework that improves how student models learn from teachers in knowledge distillation. Traditional KD methods face two main issues:

1. A trade-off between optimizing the primary task loss and imitating the teacher’s outputs.
2. Gradient disparity, caused by mismatches in architecture and representation between teacher and student models.

DTO-KD dynamically balances these objectives at the gradient level using gradient projection techniques to resolve:

1. Gradient conflict — when task and distillation gradients point in conflicting directions, and
2. Gradient dominance — when one objective overwhelms the other.

By adapting trade-offs per iteration, DTO-KD ensures balanced and constructive learning updates. Experiments on ImageNet-1K (classification) and COCO (object detection) show that DTO-KD achieves state-of-the-art accuracy, faster convergence, and produces student models that outperform non-distilled baselines, validating the effectiveness of its multi-objective formulation.

### Strengths
1. The proposed framework addresses trade-off optimization between task and distillation losses at a gradient level. By doing this, the proposed framework avoids hyperparameters for the distillation and task losses and enables dynamic learning for model training.

2. The proposed KD algorithm is a simple but effective method to align the gradients from the task and distillation losses for solving gradient conflict and gradient domainace issues in KD settings.

3. The paper demonstrates the effectiveness of the proposed framework through comprehensive experiment evaluations. The proposed framework outperforms the state-of-the-art KD methods in both classification and detection tasks.

### Weaknesses
1. The proposed framework can only be used for a scenario where training data is available. The proposed framework is not suitable for data-free KD paradigms. In other words, the proposed framework is not scalable to other KD paradigms.

### Questions
No question for this paper.

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
4

### Summary
This paper proposes DTO-KD, a new framework that formulates knowledge distillation (KD) as a multi-objective optimization problem. It dynamically balances the task loss and distillation loss at the gradient level, eliminating the need for fixed or manually tuned loss weights.
DTO-KD addresses two critical issues—gradient conflict (GrC) and gradient dominance (GrD)—by using gradient projection and adaptive weighting to ensure Pareto-optimal updates.

### Strengths
- Applying multi-objective optimization theory to KD with closed-form update rules represents a novel and mathematically grounded extension beyond prior heuristic balancing methods.
- The paper is well-structured and clearly motivates the link between gradient conflict/dominance and suboptimal KD.
- DTO-KD provides a generic, architecture-agnostic optimization framework

### Weaknesses
The experimental validation is limited, as the paper does not include results on small datasets or across a broader range of teacher–student architectures.

### Questions
- The paper describes Stage 2 as a dual problem that mathematically subsumes Stage 1 (Eq. 7–9). If Stage 2 provides the final optimization, it is unclear why Stage 1 is still experimentally isolated in Table 3. The authors should clarify how ablation on Stage 1 remains meaningful if Stage 2 inherently replaces or dominates it.
- The study mostly focuses on transformer-based students, although CNNs are still used widely.
- The authors can compare the proposed algorithm with many more benchmark algorithms on small-scale datasets (e.g. CIFAR-100).
- The paper lacks a statement on releasing the source code.
- Several references appear malformed.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a practical algorithm to address gradient conflict and gradient dominance issues in knowledge distillation. The core idea is to dynamically balance the task and distillation losses at the gradient level at each optimization step. Experiments show consistent improvement.

### Strengths
1.	This work tries to tackle a common issue in knowledge distillation, which offers it potentially strong impact.
2.	The technical part is generally solid, with detailed theoretical and empirical analysis.
3.	The experimental gain is persuasive.

### Weaknesses
In general, the presentation is a major limitation of this paper.
1. (Major) This work draws inspiration heavily from Liu et al. (2023). Although the authors do discuss the difference between the two studies and the unique contribution of this work, such discussion should be placed earlier and more concentratedly, and beyond simply stating “first time applying the methodology to knowledge distillation field”. Do the authors provide some novel theoretical contribution than Liu et al. (2023)? This should be highlighted in the paper. In the worst case, this can raise concerns about the novelty of this study.
2. (Minor) In the early part of the paper, the proposed method is only described as dynamically balancing losses. However, exactly how such balancing is done remains unclear until the main method section. Furthermore, the main method section has only formal narrative, lacking a more heuristic description to help readers quickly grab the idea. Some intuitive explanations in the introduction would help.
3. (Minor) The introduction figure only shows experimental improvement. Adding some illustrations of either the two gradient issues or how the proposed method works would help.
4. (Minor) The majority (left) part of Figure 2 does not show enough information, as this is generally a generic framework for feature-based knowledge distillation. On the other hand, the right part is with too small font sizes. The authors should re-balance the organization of this figure.
5. (Minor) The 4th paragraph of the introduction feels out of place. It does not connect closely to the previous and the following content.
6. (Minor) It appears that the authors are not using \citet and \citep correctly for in-text and parenthesis citations.
7. (Minor) Some related work is recommended to be discussed such as [1, 2]

[1] ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence. ICML 2025

[2] f-Divergence Minimization for Sequence-Level Knowledge Distillation. ACL 2023.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
