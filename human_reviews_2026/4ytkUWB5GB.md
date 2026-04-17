# Simple Optimizers for Convex Aligned Multi-Objective Optimization

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
It is widely recognized in modern machine learning practice that access to a diverse set of tasks can enhance performance across those tasks. This observation suggests that, unlike in general multi-objective optimization, the objectives in many real-world settings may not be inherently conflicting. To address this, prior work introduced the Aligned Multi-Objective Optimization (AMOO) framework and proposed gradient-based algorithms with provable convergence guarantees. However, existing analysis relies on strong assumptions, particularly strong convexity, which implies the existence of a unique optimal solution. In this work, we relax this assumption and study gradient-descent algorithms for convex AMOO under standard smoothness or Lipschitz continuity conditions -- assumptions more consistent with those used in deep learning practice. This generalization requires new analytical tools and metrics to characterize convergence in the convex AMOO setting. We develop such tools, propose scalable algorithms for convex AMOO, and establish their convergence guarantees. Additionally, we prove a novel lower bound that demonstrates the suboptimality of naive equal-weight approaches compared to our methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies aligned multi-objective optimization under convex assumptions.
The authors assume that all objectives share a common global minimizer and propose simple gradient-descent–type algorithms (PAMOO and MG-AMOO) that adapt the update direction based on the relative “gap” between each objective’s current value and its optimum.
They provide convergence guarantees in a newly defined Maximum-Gap metric, claiming an $\mathcal{O}(1/K)$ rate without requiring strong convexity. Section 6 extends the analysis to approximately aligned objectives, and a brief toy-function experiment is presented to illustrate convergence.
Overall, the paper is theoretically clean but focuses on a highly idealized problem setup with minimal empirical validation.

### Strengths
The theoretical analysis is mathematically sound, and the proofs appear correct.

### Weaknesses
The paper’s presentation is rather weak, and the fundamental assumption of the framework appears unrealistic.
In the introduction, the authors state that “the set of objectives is assumed to have a shared optimal solution” and then design gradient-descent algorithms under this assumption.
However, such a condition is almost never satisfied in practical multi-objective or multi-task learning settings—especially when objectives are derived from distinct data distributions, tasks, or losses.
Under this restrictive assumption, extending results from the non-convex to the convex case does not make the problem any closer to real-world relevance.

Although the authors attempt to generalize their theory in Section 6 by introducing an ε-aligned extension, this addition does not convincingly strengthen the practical relevance of the framework.\
The commonly adopted Pareto-front formulation in multi-objective optimization already naturally accommodates small deviations among optimal values; hence, the proposed “approximate alignment” assumption seems redundant.\
The authors should clearly justify why such a separate theoretical extension is necessary and provide a comparison with standard Pareto-based MOO algorithms under this relaxed condition.

Furthermore, the experimental section reinforces these concerns: the paper only provides synthetic toy-function demonstrations, without even a minimal empirical test on any machine-learning task.
This lack of connection between the theoretical framework and realistic ML optimization scenarios is concerning and casts serious doubt on the overall relevance of the work to the ML community.

### Questions
See weaknesses above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the convex multi-objective optimization problem. Compared with the existing work, this paper removes the strongly convex assumption and provide two methods with convergence guarantees.

### Strengths
1. Compared with existing studies, this paper adopts weaker assumptions.

2. The paper introduces a new metric, Maximum Gap (MG), to evaluate convergence.

3. It proposes two methods to address the formulated problem, which, compared with the EW baseline, eliminate the constant term in the convergence rate.

### Weaknesses
1. The writing and presentation can be improved. For example, the symbol $\lambda$ in line 132 is used before being defined, and Lemma 3, although presented in the appendix, is referenced in the main text without any brief introduction or explanation.

2. The main concern of this paper lies in its problem formulation. The paper focuses on a non-conflicting setting in multi-objective optimization (MOO). However, gradient conflict is one of the most challenging aspects of MOO and typically requires non-trivial efforts to address, as seen in algorithms such as MGDA. Moreover, general MOO aims to approximate the entire Pareto set, whereas this paper’s results are limited to specific points. Although the non-conflicting setting significantly simplifies the general MOO problem, the motivation for studying this particular case is unclear. Are there theoretical or empirical studies on this setting besides Efroni et al. (2025)?

3.  While the non-conflicting setting simplifies the general MOO problem—and as shown in Theorem 1, the baseline EW algorithm can already solve it, and the proposed PAMOO and MG-AMOO methods improve the upper bound only by a constant factor. However, these methods require access to the optimal function value for each individual objective. Is it practical to assume such information is available, and is it necessary to obtain it merely to achieve an improvement by a constant factor, especially when the number of objectives 
m is typically small?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a method for multi-objective optimization that relaxes the strong convexity assumption, focusing on convex functions with conditions like Lipschitz continuity or smoothness. The authors propose the Maximum-Gap (MG) metric to evaluate convergence in cases where a common optimal solution does not exist. Two algorithms, PAMOO and MG-AMOO, are presented with new convergence guarantees in the convex setting. MG-AMOO reduces multi-objective optimization to single-objective optimization for efficiency. The paper includes experiments comparing these algorithms to existing methods, demonstrating their performance under various settings.

### Strengths
This paper provides a theoretical foundation to support its conclusions. The authors develop the mathematical framework for aligned multi-objective optimization.

### Weaknesses
**Question 1** The paper's central assumption that all objectives share a common minimizer seems very strong and rarely satisfied in realistic multi-objective learning . If there are domains where this assumption is reasonable, please document them with additional references and empirical evidence from realistic settings.

**Question 2** Much of the development, including the MG metric converging to zero and the need to know $\{f_i^{\star}\}$, critically depends on this alignment. While Section 6 discusses an $\varepsilon$-approximate variant, the empirical and theoretical treatment still assumes access to optimal values and does not fully characterize the non-aligned regime where $C^{\star}=\varnothing$. This limits applicability.

**Question 3** Multi-objective optimization becomes more challenging when objectives conflict, especially in non-convex settings. While some works [1-2] have addressed this in non-convex cases, this paper focuses solely on convex optimization under the assumption of non-conflicting objectives. This seems overly restrictive, as real-world problems often involve conflicting objectives and non-convex landscapes. Would it be possible to extend the analysis to non-convex settings, or justify why the convex case alone is sufficient ?
**Question 4** The experimental comparison is limited to EW and PAMOO. Please consider recent multi-task/MOO solvers: MGDA, GradNorm, PCGrad/Grad Surgery. Even if theoretical regimes differ, empirical comparisons matter for readers choosing optimizers.

[1]Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance.

[2] Mitigating gradient bias in multi-objective learning: A provably convergent stochastic .

### Questions
And some minor issues:

1.Typos: alighned $\rightarrow$ aligned; suprising $\rightarrow$ surprising; analgous $\rightarrow$ analogous. 

2. State explicitly how $\{f_i^{\star}\}$ are obtained in experiments (exactly known or estimated surrogates).

3. Briefly restate Eq. (3)'s inequality intuition in the main text (one line) to help readers not flipping to the appendix.

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
1

### Summary
While many machine learning tasks can be optimized together without conflict, the "Aligned Multi-Objective Optimization" (AMOO) framework used to analyze this previously relied on an unrealistic strong convexity assumption. This paper relaxes that requirement, studying AMOO under standard convexity and smoothness conditions that are far more common in modern deep learning. The authors develop new analytical tools and scalable gradient-descent algorithms for this setting, proving their convergence and demonstrating their superiority over naive equal-weighting methods.

### Strengths
1. Well-motivated problem and novel approach. 
2. The improved theoretical results and detailed analysis demonstrate a significant improvement over previous methods.

### Weaknesses
1. Compared to the detailed theoretical results presented in this paper, the experimental verification is very weak, as experiments are only conducted on 2-layer neural networks. While this is helpful for verifying the algorithm's theoretical results, it is insufficient for validating the algorithm's performance in real-world training scenarios. I think it would be better to conduct experiments on multi-objective training tasks, such as multi-task learning or multi-objective recommendation system training.

2. Why did the author only compare EW instead of some gradient-based multi-objective optimization algorithms, such as MGDA and its variant, which also claim a certain convergence speed?

### Questions
See the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
