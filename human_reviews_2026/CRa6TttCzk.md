# Multi-Task Learning as Stratified Variational Inequalities

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Multi-task learning (MTL) provides a powerful paradigm for jointly optimizing multiple objectives, yet real-world tasks often differ in maturity, difficulty, and importance. Naively training all tasks simultaneously risks premature updates from unstable objectives and interference with high-priority goals. We introduce SCD-VIO---Stratified Constraint Descent via Variational Inequalities and Operators---a new operator-theoretic paradigm for hierarchy-aware MTL. Rather than heuristic reweighting, SCD-VIO formulates training as a stratified variational inequality, where task feasibility is defined relative to its own cumulative performance and enforced through Yosida-regularized soft projections. This self-calibrated gating (SC Gate) requires no extra hyperparameters and ensures that lower-priority tasks are activated only after higher-priority ones have stabilized, aligning optimization flow with natural task dependencies.

SCD-VIO is model-agnostic and integrates seamlessly with standard MTL backbones. Experiments on three large-scale recommendation benchmarks---TikTok, QK-Video, and KuaiRand1k---show that it consistently boosts prioritized objectives while maintaining or improving overall performance. Taken together, these results position SCD-VIO as both a principled theoretical formulation and a practical, plug-and-play solution for hierarchy-aware multi-task learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SCD-VIO, a new framework for hierarchical multi-task learning (MTL). It reformulates MTL as a stratified variational inequality, where each task’s optimization is constrained by the stability of higher-priority tasks. The paper proposes a self-calibrated gating mechanism that uses smooth projections to activate lower-priority tasks only after higher-priority ones are stable. The paper provides a set of experiments on recommendation benchmarks and shows gains in prioritized objectives and overall performance.

### Strengths
- SCD-VIO recasts hierarchical MTL as a stratified variational inequality, giving formal grounding to task prioritization.
- Evaluation results show consistent gains on prioritized objectives. Also, the paper provides full component ablation studies (masking, violation weighting, self-calibration), justifying the framework’s design.
- The paper is clear and easy to follow (however theoretical section needs major revision, see below).

### Weaknesses
- The theoretical section in its current state does not support or contribute to the paper. I suggest restructuring the paper by stating the main results and theorems in the main text, and providing proof sketches while deferring the full results and proofs to the Appendix.
- The paper’s operator-theoretic and variational inequality framing introduces fairly heavy formalism without yielding clear practical or conceptual benefits. The core method is essentially a soft gating and reweighting scheme that could be explained without this abstraction. In its current form, the theory may obscure rather than clarify the intuition.
- The method assumes a full, correct ordering of task priorities is available, but this may not be the case in real-world multi-task settings. Mis-specification harms performance (as shown in ablations), so some automatic or data-driven hierarchy discovery mechanism may improve applicability for real-world scenarios.
- Evaluation is limited to recommendation datasets. This narrow scope limits confidence in generalization to other domains such as computer vision, NLP, or reinforcement learning, where task dependencies differ.
- Baseline coverage is limited. Table 4 evaluates only three baselines on a single dataset, focusing on task scheduling rather than strong recent multi-task learning or auxiliary-learning methods (e.g., Pareto MTL or dynamic reweighting). This weakens empirical claims.
- Minor: Line 262, the word “text” appears out of place.

### Questions
- How does the method scale with the number of tasks, both in terms of computational overhead and stability of training?
- Since the residuals $r_i$ scale as $\tilde{r}_i = \beta r_i$ when $\tilde{\mathcal{L}}_i = \beta \mathcal{L}_i$, 
the scheduling depends on the magnitude of the losses. Would it not be more appropriate to make the gating invariant to such scaling (i.e., $\tilde{r}_i = r_i$)?
- Does the framework naturally support tasks with equal or overlapping priorities, and if so, how are such cases handled?
- What motivated the replacement of $p_i$ with the softplus transformation? 
- How sensitive is the method to mis-specified or noisy task hierarchies, and can it adapt or learn the hierarchy automatically during training?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SCD-VIO, a novel framework that integrates a dynamic, priority-aware mechanism for hierarchy-aware multi-task learning. Motivated by the observation that real-world tasks differ in difficulty, maturity, and importance, the method explicitly addresses the challenge of preserving performance on high-priority tasks. SCD-VIO formulates priority-aware MTL as a stratified variational inequality, defining task feasibility in terms of cumulative historical performance. By employing Yosida-regularized soft projections and a self-calibrated gating mechanism, lower-priority tasks are activated only once higher-priority tasks have stabilized, aligning optimization trajectories with natural task hierarchies. Unlike prior approaches, SCD-VIO enforces hierarchical constraints directly at the gradient level while remaining modular, differentiable, and model-agnostic. The paper presents experiments on standard recommendation benchmarks and provides a theoretical analysis that further supports the framework’s principled, scalable, and practical design.

### Strengths
The paper is overall well-written and well-motivated within the context of existing literature. The proposed method is innovative and relevant to the field, offering a conceptually sound approach that could have a meaningful impact on the multi-objective learning community. This work presents an interesting and original idea for a priority-aware mechanism for hierarchy-aware multi-task learning. In addition, the approach is theoretically grounded.

### Weaknesses
The experimental results are not sufficiently convincing. 
(a) In the original STEM paper, the reported results differ from those in this manuscript. For example, comparing Tables 2 and 3 in STEM with Tables 1 and 2 here, the Like AUC for the QK-Video task using STEM without SCD-VIO is reported as 0.9426 in STEM, whereas it is 0.8373 here. Is there an explanation for this significant discrepancy? 
(b) There is no discussion of runtime, training duration, or number of iterations. 
(c) Table 2 omits the OMoE model — why? Table 3 omits the ESMM model — why? 
(d) It would be valuable to see “inverse” experiments. For example, in Table 3, it would be interesting to swap the priorities of task 1 and task 2 and observe whether the results are preserved. 
(e) In Table 4, comparisons are made using STEM as the base method. However, in Table 1, it appears that the largest improvement of the proposed algorithm occurs with the STEM model. It would be informative to see results across more than a single base model.

The empirical section of the current version lacks sufficient evidence to fully substantiate that this method achieves improvements. Expanding the experimental results would considerably enhance the paper’s impact and credibility.

### Questions
Please address the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
Traditional multi-task learning often fails when tasks have different priorities, causing interference from less important or unstable objectives. This paper introduces SCD-VIO, a new method that reframes the training process as a stratified variational inequality to enforce a task hierarchy. Using a hyperparameter-free "self-calibrated gate," SCD-VIO ensures high-priority tasks stabilize before activating lower-priority ones, demonstrating improved performance on large-scale recommendation benchmarks.

### Strengths
1. The paper organization is clear and easy to follow.
2. Extensive experimental results have verified the effectiveness of the proposed algorithm.

### Weaknesses
1. Lacking comparison with other highly efficient MTL methods, such as FAMO, and "Smooth Tchebycheff Scalarization for Multi-Objective Optimization".

2. The algorithm presented in this paper is more of an empirical result from engineering (In particular, the conclusion in Section 3.4) than a conclusion drawn from Yosida regularization. The approach, from combining loss functions across multiple tasks to using Yosida regularization, lacks clear logic. (Section 3.2-3.3) No reference to support the author's claim in Section 3.3.

### Questions
See the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies multi-task learning through a hierarchical optimization perspective. It considers designing a method that optimizes higher-priority tasks first. Specifically, it proposes Stratified Constraint Descent via Variational Inequalities and Operators. It defines priority constraints that higher-priority task losses not exceeding their historical averages. To deal with the nonsmoothness introduced by the constraints, they use Yosida-regularized soft projections to smooth the constraints.

Theoretical convergence to stationarity is provided. Empirical studies on some multi-task recommendation benchmarks such as TikTok, QK-Video, and KuaiRand1k are provided.

### Strengths
1. This paper studies multi-task learning through a hierarchical optimization perspective. It considers designing a method that optimizes higher-priority tasks first. This is an important problem and highly relevant to the community.

2. The paper provides both theoretical analysis of the convergence of the algorithm and empirical studies on some benchmarks.

### Weaknesses
1. The motivation for defining the constraints in Eq (3) is unclear. Why the current loss should not exceed its historical average?

2. Insufficient discussion of related works. The paper basically casts the multi-task learning problem as a hierarchical constrained optimization problem. Using (hierarchical) constrained optimization to solve multi-task learning problems is not new. See a few relevant works listed below. A detailed comparison to the prior works [1-4] should be discussed.

3. A comparison to prior multi-task learning approaches on the benchmark [5] would be preferred to better understand the effectiveness of a hierarchical formulation.

4. Some parts of the paper are not clear, see Questions.


[1] Automatic and Harmless Regularization with Constrained and Lexicographic Optimization: A Dynamic Barrier Approach, NeurIPS 2021

[2] Pareto Multi-Task Learning, NeurIPS 2019

[3] FERERO: A Flexible Framework for Preference-Guided Multi-Objective Learning, NeurIPS 2024

[4] Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization, ICML 2020

[5] LibMTL: A Python Library for Multi-Task Learning, JMLR 2023

### Questions
1. After formulating the problem through hierarchical constrained optimization Eq. (3)-(4), why not use other methods, such as penalty, barrier, projected gradient to solve the constrained optimization?

2. The difference of this formulation compared to the epsilon-constrained formulation or multi-level/lexicographic formulation should be fully discussed.

3. How does Yosida regularization approximate the original formulation? It would be better if theoretical or empirical sensitivity analysis can be provided.

4. The stationary solution that the proposed algorithm converges to is not defined clearly. Is it Pareto stationary? Does it require feasibility of the other tasks? More discussion should be provided.

### Soundness
3

### Presentation
2

### Contribution
2
