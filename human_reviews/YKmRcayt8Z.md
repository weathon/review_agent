# Finite-Time Analysis for Conflict-Avoidant Multi-Task Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5

## Abstract
Multi-task reinforcement learning (MTRL) has shown great promise in many real-world applications. Existing MTRL algorithms often aim to learn a policy that optimizes individual objective functions simultaneously with a given prior preference (or weights) on different tasks.  However, these methods often suffer from the issue of gradient conflict such that the tasks with larger gradients dominate the update direction, resulting in a performance degeneration on other tasks. In this paper, we develop a novel dynamic weighting multi-task actor-critic algorithm (MTAC) under two options of sub-procedures named as CA and FC in task weight updates. MTAC-CA aims to find a conflict-avoidant (CA) update direction that maximizes the minimum value improvement among tasks, and MTAC-FC targets at a much faster convergence rate. We provide a comprehensive finite-time convergence analysis for both algorithms. We show that MTAC-CA can find a $\epsilon+\epsilon_{\text{app}}$-accurate Pareto stationary policy using $\mathcal{O}({\epsilon^{-5}})$ samples, while ensuring a small $\epsilon+\sqrt{\epsilon_{\text{app}}}$-level CA distance (defined as the distance to the CA direction), where $\epsilon_{\text{app}}$ is the function approximation error. The analysis also shows that MTAC-FC improves the sample complexity to $\mathcal{O}(\epsilon^{-3})$, but with a constant-level CA distance. Our experiments on MT10 demonstrate the improved performance of our algorithms over existing MTRL methods with fixed preference.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper studies the multi-task reinforcement learning problem from a theoretical perspective. Different from many of the existing literature, the paper focus on addressing the issue of gradient conflict brought by the imbalanced gradients from different tasks. In this paper, the author proposes a formulation that is able to dynamically weight the task importance, using two updates, named as CA and FC. For the convergence results, the paper demonstrates that the proposed MTAC-CA method enjoys a sample complexity of $O(\epsilon^-5)$ and MTAC-FC method has a complexity of $O(\epsilon^-3)$ to find a pareto stationary policy, up to function approximation errors. The paper also uses numerical examples to demonstrate the effectiveness of the algorithm.

### Strengths
* The paper studies a theoretically sound problem, and the theoretical results derived look correct from my brief examination.
* The paper structure is complete, and the presentation is good. The problem formulation and the results in the paper are easy to follow.
* I appreciated that the author has numerical experiments in the paper and also provides the code for the experiment, which are sometimes not the case for many RL theory paper.

### Weaknesses
I mainly have the following two concerns
* The analysis for the TD learning part looks standard. Yet it seems neither did the author claim this analysis as original contribution nor did the author cite any reference in the proof (please point me to the place if I missed that).
* The overall novelty of the analysis remains questionable. The research problem sounds good to me. However, given (Xiao et al., 2023), which studies the general multi-task learning problem, I am not sure how much does this paper contributes to the community. If the contribution is only about the function approximation/gradient estimation from the RL side, I feel like this adds-on is not enough for ICLR.

### Questions
I hope the author could provide further clarifications on the two points that I mentioned in the weakness section. If they can be addressed, I  will consider adjusting my score.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The paper presents a solution to Multi-Task RL, where we wish to solve more than one task with the same policy. In such cases, it is possible that the gradients we obtain from different tasks are conflicting and as such we would like to ensure we find a Pareto optimal solution. The paper proposes two algorithms for this problem setting, one that focuses on fast convergence and the other that keeps the CA-distance (a measure of disagreement between different tasks' gradients). The authors also present empirical results on the MT10 benchmark.

### Strengths
- The paper is generally well-written, easy to follow and focuses on important problem in the RL community

### Weaknesses
- It seems like the ideas for the proposed algorithms are borrowed directly from Xiao et al. (2023). The only novelty seems to be the explicitly obtain the bounds for when the previously developed algorithms are applied to Actor-Critic style RL. However, I believe the convergence of Actor-Critic algorithms has been extensively studied before, e.g. Barakat et al. (2022). As such, I do not see much novelty in the proposed approach. It seems like a simple combination of two existing analyses with no added value (an A + B type of paper).
- The SDMGrad algorithm proposed by Xiao et al. (2023) has been applied to RL problems and in fact, one of the benchmark problems they present is MT10, also considered by the authors of this submission. However, for some reason, the authors do not compare it with SDMGrad. If we, however, look at the values reported by Xiao et al. (2023), it seems like SDMGrad achieves better values than the algorithm presented in the submission. As such, I believe there is also no empirical novelty, as the proposed algorithm performs worse than existing algorithms.

### Questions
- Can authors clarify what are the exact differences compared to the work of Xiao et al. (2023)? Are there any unique challenges, when using their results in the RL setting? Is there anything beyond a simple A + B type combination of existing results?
- Why is the previous, most similar work of SDMGrad not shown in the experimental evaluation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper studies a multi-task RL problem where there is no prior weighting for each task. The paper proposes an algorithm class (with variants in dynamic weighting method) for finding a Pareto stationary policy and establishes the sample complexity. Illustrative numerical simulations are presented which show the superior performance of the proposed algorithm on an averaged multi-task objective.

### Strengths
The paper is well written with the background intro, algorithm development, and proof sketch clearly presented. The problem being studied is well motivated and important -- it is natural to consider multi-task RL where we do not in advance know how much weight to put on each task. The algorithms proposed are novel, with rigorous complexity analysis. I did not go through the complete proof but find the main claim and argument credible. Despite the simplicity, the simulation results do verify the superior algorithm performance.

### Weaknesses
1) As the weight is not given a priori, the optimization objective is very unclear. Eq.(1) is obviously not an valid objective since it is multi-dimensional. The theoretical results establish convergence to a Pareto stationary policy, defined as a point minimizing $\mathbb{E}[\min_{\lambda}||\lambda^{\top}\nabla J(\pi)||^2]$. However, this does not seem a valid metric for multi-task learning, either, as it can be made small as long as any of the tasks is well solved. In the simulations, Table 1 compares the algorithm on the averaged success rate. I suppose an averaging over all tasks is made here with equal weight for each task. This is certainly a valid objective. The issue is that, if the objective is known to be equally weighted, it defeats the purposes of dynamically adjusting the weights. In this sense, the paper fails to evaluate the proposed MTAC-CA and MTAC-FC algorithms either theoretically or in simulation.

2) The technical innovation of the work is unclear. While the paper claims "addressing this question is not easy, primarily due to the difficulty in conducting sample complexity analysis for dynamic weighting MTRL algorithms" in line 065-066, what actually makes the analysis challenging should be concretely discussed (mathematically). The analysis of actor-critic algorithms on single-task RL is well studied and this paper essentially follows the existing analysis to treat the policy and critic iterates. On the other hand, the dynamic weight adjusting schemes seem to have been studied in the CA multi-task supervised learning setting. Does adapting the analysis from supervised learning to reinforcement learning create significant challenges here? If so, the authors should highlight such challenges and how they are overcome.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
