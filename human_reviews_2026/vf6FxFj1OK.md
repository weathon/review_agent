# Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion

- Decision: Reject
- Scores: 6, 4, 6, 4, 2

## Abstract
Multi-Objective Reinforcement Learning (MORL) extends standard RL by optimizing policies over multiple and often conflicting objectives. Although max-min scalarization has emerged as a powerful approach to promote fairness in MORL, it has limited applicability, especially when incorporating constraints. In this paper, we propose a unified framework for constrained MORL that combines the max-min criterion with constraint satisfaction and generalizes prior formulations such as unconstrained max-min MORL and constrained weighted-sum MORL. We establish a theoretical foundation for our framework and validate our algorithm through a formal convergence analysis and experiments in tabular environments. We extend our framework to practical applications, including simulated edge computing resource allocation and locomotion control. Across these domains, the method demonstrates strong handling of fairness and constraint satisfaction in multi-objective decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a unified framework for constrained MORL that simultaneously addresses the max–min criterion and constraint satisfaction, while also establishing a solid theoretical foundation for the framework. The authors further develop an iterative algorithm with a theoretical analysis of its convergence rate. Empirically, they demonstrate that the proposed method performs well not only in tabular settings but also on more practical tasks.

### Strengths
1. The proposed framework for constrained max–min MORL is impressive. In establishing its theoretical foundation, the authors effectively combine reinforcement learning and optimization techniques to derive a tractable constrained optimization formulation of the primal problem, which serves as the basis for the practical algorithm.

2. The experimental section goes beyond tabular settings and includes trials on more practical environments, showing the method’s applicability.

### Weaknesses
1. The proposed method appears to perform well only on small-scale MORL tasks, where the number of objectives remains limited. It would strengthen the paper to extend the evaluation to settings with a larger number of objectives, thereby demonstrating the method’s practical scalability and value.

2. Although the paper provides a theoretical analysis of convergence, it does not include learning curves to empirically illustrate convergence behavior. Including such results would make the empirical section more convincing.

### Questions
1. The paper mentions heterogeneous objectives. Could the authors clarify how the proposed design effectively handles these heterogeneous objectives?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- The authors propose an algorithm that maximizes the minimum of multiple objectives (fairness) while satisfying constraints.
- To formulate the constrained multi-objective RL (CMORL) problem as a convex optimization, they convert the problem to a dual problem using the occupancy measure.

### Strengths
- The authors transform the CMORL problem into a convex dual problem, thereby reducing its complexity.
- They derive the update rules for the dual variables (v, w) using value functions from standard RL frameworks, lowering the implementation difficulty.
- They prove the convergence of the proposed method.

### Weaknesses
- Insufficient survey of prior CMORL work.
    - The authors focus on resolving the issue that linear weights fail to reflect user preferences for heterogeneous objectives.
    - CoMOGA [1] was proposed to address the same problem, yet it is neither cited nor compared experimentally, which is a notable omission.
- The introduction needs more explanation.
    - It does not clarify how fairness leads to max-min optimization.
    - At line 54, a simple example illustrating the power of max-min optimization would be helpful.
    - Even with heterogeneous objectives, scale-invariant algorithms like CoMOGA can resolve the issue by taking uniform relative importance as user preference; the need for max-min over this alternative should be explicitly addressed.
- The claim that the proposed method can handle heterogeneous objectives is not strongly supported, as the proposed method is only evaluated on homogeneous objectives.
- Equations 4–8 are similar to derivations in prior work but lack proper citation.
    - COptiDICE [2], a prior method that similarly reformulates constrained RL in occupancy measure space, derives analogous steps during dual transformation. These should be cited.
- Updating dual variables (u, w) and primal variables (policy, value) simultaneously is not feasible, resulting in increased computational cost.
    - As shown in Equation 15 of Theorem 3.6’s proof, the optimality gap is proportional to $\epsilon$.
    - This requires fully converging the policy for fixed (u, w) before updating (u, w), and repeating the process.
    - Consequently, convergence demands far more iterations than standard RL training process.

[1] Kim, Dohyeong, et al. "Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning." The Thirteenth International Conference on Learning Representations.

[2] Lee, Jongmin, et al. "COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation." International Conference on Learning Representations.

### Questions
- In Table 2 (tabular setting results), the optimality gap for Gaussian smoothing is large.
    - Was the smoothing factor for Gaussian smoothing set to sufficiently small?
    - If not, the comparison may not be fair.
- All objectives in the experimental environments are homogeneous.
    - If heterogeneous objectives are converted into constraints, how should appropriate thresholds be determined?
    - Setting thresholds too high would prevent meaningful performance, while setting them too low would ignore the constraints.
    - Without clear guidelines, empirically tuning thresholds introduces an additional iterative process outside the learning algorithm, increasing overall complexity despite not being reflected in the reported computational cost.
- Why was the proposed method not compared with CMORL algorithms (e.g., LP3 [1], CoMOGA) in Section 5.2?

[1] Huang, Sandy, et al. "A constrained multi-objective reinforcement learning framework." Conference on Robot Learning. PMLR, 2022.

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
3

### Summary
This paper rewrites the constrained max-min multi-objective RL problem as a convex occupancy-measure program. It then gives an equivalent dual with nonnegative constraint weights u and simplex weights w, making a soft value iteration operator. With this change, the gradients of the objective with respect to the $(u,w)$ are characterized. The theoretical analysis covers convergence and sample complexity.

### Strengths
-	The paper clearly identifies applications where the max-min criterion is important, emphasizing scenarios where standard weighted-sum scalarization cannot fully deal with.
-	In structured MOMDPs, the method gives less optimality errors than Gaussian smoothing and is easier to update.

### Weaknesses
-	Core validation is tabular, and the two application studies are small-scale. There lacks large continuous-control benchmark. 
-	Comparisons focus on a modified Gaussian smoothing max-min method, and stronger constrained baselines such as CPO, PCPO, and recent MORL methods are missing.
-	The analysis is limited, for example, missing analyses on number and tightness of constraints, fixing and learning $w$ in the setting. Moreover, the assumption is strong, for example, relying on strict feasibility.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work is based on max-min scalarization in MORL and does not achieve a clear improvement over the earlier version in some of the experiments. A combination of earlier work is presented that combines the max-min criterion with constraint satisfaction. Constrains are, however, treated in a similar way as the multiple objectives used here as a contribution to the loss function. A very good theoretical analysis is shown, but a critical analysis of the assumption as needed for practical applicability is not attempted.

### Strengths
It seems that here an aspect of MORL is considered that aims at reward shaping in the sense of finding good combinations of several rewards. This is interesting but different from other approaches and should therefore be explained more clearly in the introduction already before starting to work with the formalism.

Scalarization-based approaches can be very useful, especially if the limitations are discussed rather than removed by suitable assumptions, so that practical applicability remains an unquestionable priority of the proposed method (as promised in the abstract).

The manuscript is well prepared, and can even gain if not disconnected from other literature on MORL that  works with the concept of Pareto optimality (not mentioned here apart from App. G).

### Weaknesses
The manuscript is well prepared, but seem a bit disconnected from other literature on MORL because it does not use the concept of Pareto optimality which is not even mentioned (apart from App. G) although it is  the dominant concept in other work on MORL, thus reasons for a different approach should be discussed. Likewise, a reader might want to know haw the concepts of “fairness” can be related evaluation of solutions in the Pareto optimization framework. This is particularly critical here,  as fairness is emphasized several times in the manuscript, but none of the available fairness measures is discussed explicitly, so that it remains questionable this goal is actually approached. 

Nevertheless, a scalarization-based approach can be useful, but if its limitations are removed from the discussion by suitable assumptions, the practical applicability of the proposed methods remains limited, even so it is promised in the abstract.

Pareto-optimization is a widely-used concept in MORL, thus reasons for a different approach need to be discussed more openly. Likewise, a reader might want to know haw the concepts of “fairness” can be related evaluation of solutions in the Pareto optimization framework. This is particularly critical here, because fairness is emphasized several times in the manuscript, but none of the available fairness measures is discussed explicitly, so that it remains questionable this goal is actually approached. 

The examples could be more indicative if not only performance is studied, but also the particular advantages are discussed that should become relevant “especially when handling heterogeneous objectives or incorporating constraints”.

In addition to its intended message, Table 1 also implies that the work is combining to existing threads rather than presenting a conceptual progress. Considering also the some of the proofs do not change  or which are similar to the earlier work because soft constraints are considered here that are largely complying with the multi-objective framework. The proofs included here are similar in complexity to earlier work, i.e. largely formal and little effort has been taken to increase relevance for practical applications where e.g. smoothness is often not easily decidable unless the problem is already solved or where the effort to realize local methods by suitable initialization of sometimes comparable to an ad-hoc solution of the problem.

### Questions
The parameter $\beta$ occurs in equ. 1 in a role analogous to a temperature, whereas in physics $\beta$ typically used to denote an inverse temperature. As $\beta$ occurs here mostly as inverse,would it be possible to avoid unnecessary confusion and to consider using a parameter $\frac{1}{\beta}$ instead of $\beta$?

What is the justification for introducing the dimension $L$ in Section 2.?

Would it be possible to abbreviate the word “constrained” by “constr.” rather than by “const.”?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the problem of multi-objective reinforcement learning (MORL) with a max-min objective. Given K + L objectives, L of which must obey some constraints, they define an entropy regularized objective to set up a convex optimization problem that they then solve with mirror descent.

### Strengths
Constrained MORL is an important problem and the empirical evaluation of the results in this paper suggest their algorithm has practical application to, e.g., edge computing resource allocation.

### Weaknesses
This work omits several related works that I believe imply the results in this paper [1,2,3]. The main difference compared to [2] appears to be the presence of constraints, but why can’t these can be handled as in [1]? Moreover, how do the results in this work improve upon [1]?

The max-min objective is treated as a unique challenge for MORL, but [1] already shows how to reduce max-min fairness to general constrained RL, where the minimum reward for any objective is constrained to be above some alpha, the optimal value of which can then be identified through binary search. Moreover, the authors claim Park et al [4] is the only prior work to address max-min RL, which is false. [1,2,5] all address this problem explicitly. 

[1] “Intersectional Fairness in Reinforcement Learning with Large State and Constraint Spaces” Eaton, Hussing, Kearns, Roth, Sengupta, Sorrell

[2] “Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach” Byeon, Park, Chae, Leshem 

[3] “Reinforcement learning with convex constraints” Miryoosefi, Brantley, Daume, Dudik, Shapire

[4] “The max-min formulation of multi-objective reinforcement learning: From theory to a model-free algorithm” Park et al

[5] “On welfare-centric fair reinforcement learning”  Cousins, Asadi, Lobo, Littman

### Questions
Please see comments in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
