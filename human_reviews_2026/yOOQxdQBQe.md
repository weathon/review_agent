# Lipschitz-Guided Monte Carlo Tree Search with Knowledge Transfer across Sequential Tasks

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Monte Carlo Tree Search (MCTS) has proven highly effective in solving complex planning tasks by balancing exploration and exploitation using Upper Confidence Bound for Trees (UCT). However, existing works have not considered MCTS-based lifelong planning facing a sequence of MDPs -- e.g., each MDP with varying transition probabilities and rewards from previous ones -- throughout the operational lifetime. This paper presents LiZero for Lipschitz lifelong planning using MCTS. We propose a novel concept of adaptive UCT (aUCT) to transfer knowledge from previous tasks to the exploration/exploitation of a new task, depending on both the Lipschitz continuity between tasks and the confidence of knowledge in Monte Carlo action sampling. We analyze LiZero's acceleration factor in terms of improved sampling efficiency and also develop efficient algorithms to compute aUCT in an online fashion by both data-driven and model-based approaches, whose sampling complexity and error bounds are also characterized. Numerical results show that LiZero significantly outperforms existing MCTS and lifelong learning baselines in terms of much faster convergence (3$\sim$4x). Our results highlight the potential of LiZero to advance decision-making and planning in dynamic environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes LiZero, an MCTS variant for sequential task transfer. The key idea is an adaptive UCT (aUCT) bound that reuses past tasks’ Q-estimates with a Lipschitz-controlled penalty for task dissimilarity.

### Strengths
1. The problem is clearly formulated, and the scope is focused: an MCTS/UCT variant for lifelong task transfer.
2. Results across the task sequence show the effect cleanly with little gain on Task 1 and increasing gains by Task 10.
3. The high-level idea (planning-time transfer via Lipschitz control) is interesting and well-motivated.

### Weaknesses
1. Assumption in Theorem 4.3 (NN distance). The bound assumes a Lipschitz link between model outputs and a parameter-space metric. This is strong for modern networks; the constant L may be loose, and it is hard to verify. The usefulness of the method in higher-dimensional or continuous settings may depend on how valid this assumption is in practice.
2. Scalability. (1) Estimating d for many prior tasks can be expensive, especially with sampling-based estimators. This raises the need for an efficient and stable estimator. (2) Overall efficiency depends on the effectiveness of Theorem 4.3.
3. Given the above concerns, the paper would be more convincing if it reported results in an environment with a continuous state space (e.g., CartPole). It does not need to use continuous action spaces, since handling them is an orthogonal challenge for MCTS. Furthermore, it would help to demonstrate that the NN-based approach satisfies a Lipschitz relationship in a more complex environment with richer dynamics.

### Questions
1. Is the data-driven distance estimator the sampling-based approach? 
2. Do MCTS baselines have access to true transition functions?  
3. As described in the experiment setting, is the transition function non-deterministic? If so, how do you handle it during planning? 
4. What is the number of MCTS iterations per decision in each experiment?

Minor issues / typos: 
1. Can you fix the citep/citet in latex across the paper? 
2. Duplicate citation of “Lipschitz lifelong reinforcement learning” 
3. data-driven distance “estimater” -> estimator

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
3

### Summary
This paper introduces LiZero that applies Monte Carlo Tree Search  to lifelong planning problems across a sequence of distinct MDPs. The core theoretical contribution is a new adaptive Upper Confidence Bound for Trees (aUCT), which enables knowledge transfer by balancing the similarity between tasks and the confidence of the stored knowledge. To make this practical, the paper presents efficient online algorithms to compute the aUCT bound using both data-driven and model-based approaches. Experimental results show that LiZero significantly outperforms existing MCTS and lifelong RL baselines, achieving faster convergence and greater sample efficiency.

### Strengths
- This paper has strength in theoretical development. Based on the definition of Lipschitz regularity across MDPs, the task distance d(M,M'),  and the uncertainty correction term P(N,N'), is directly instantiated in the design of the aUCT bound and the final action-selection rule. This mapping from theorem to implementation ensures that every part of the algorithm is mathematically grounded, enhancing the paper’s internal consistency and conceptual clarity
 
- In a sequence of ten gridworld tasks, LiZero consistently converges faster than both UCT-based MCTS variants (MCTS-R, MCTS-O, pUCT) and lifelong RL baselines (RMax, LRMax). The performance gain emerged in early tasks, indicating effective reuse of prior task knowledge. Quantitatively, LiZero achieves faster convergence speed and achieves early reward improvement. These results empirically validate the theoretical acceleration predicted by Theorem 3.4, confirming that tighter aUCT bounds facilitate faster and more stable planning across sequential tasks.

### Weaknesses
- While the authors demonstrate LiZero’s effectiveness by transferring knowledge across several related tasks, all experiments are performed within a single environment. This limited setup makes it difficult to assess the method’s generalizability. Moreover, despite repeatedly emphasizing the algorithm’s practicality (particularly in Section 4), the experiments are confined to a small 25 × 25 gridworld, which cannot be considered a practical or realistic scenario for lifelong planning with continuous states and/or actions. Detailed description of the experimental setup is missing. 

- There is no sensitivity analysis for critical hyperparameters such as the exploration constant, the Lipschitz scaling coefficient, or the discount factor. The paper also does not analyze how search-level parameters that determine MCTS behavior (e.g., tree depth, branching width, rollout budget) affect performance. This absence of ablation and hyperparameter analysis prevents a clear understanding of which factors drive LiZero’s performance and whether the observed gains are robust across settings.

### Questions
(see the points of weakness above)

### Soundness
2

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
This paper introduces LiZero, an extension of Monte Carlo tree search (MCTS) for transferring learned action-values from one task to another, motivated by lifelong planning. The paper provides theoretical results showing that the difference in the estimated action-values of two MDPs is upper-bounded by a distance between the MDPs and by a decreasing function of the visit counts. Based on this, they derive an upper bound for UCT, and they show that incorporating this upper bound into MCTS can speed up convergence. They also provide multiple practical methods for estimating the distance between MDPs in practice. Finally, they evaluate LiZero experimentally on a sequence of 10 tasks (benchmark from existing work), showing that LiZero's transfer provides a significant improvement over MCTS and lifelong RL baselines.

### Strengths
* LiZero is an interesting contribution, introducing the idea of transfer for lifelong learning in MCTS.
* The algorithm is supported by thorough theoretical analysis.
* The numerical results are positive.

### Weaknesses
While the paper introduces an interesting contribution supported by theoretical results, its presentation suffers from issues, some of which (e.g., inconsistent notation) make it difficult to verify the results of the paper:
* Proof of Theorem 3.2: Notation $n_1$ and $n_2$ seem to be undefined. Are these the same as $N$ and $N'$? In the same proof, $L_2$ also seems to be undefined. Is this the same as $P(N, N')$? In the same proof, the second line of Equation (18) seems to replace $\bar{G}$ with $G$ in the expectation (first term). Why is this an equality?
* Definition 3.1: The distance $d(\mathcal{M}, \mathcal{M}')$ also depends on $\kappa$. It would be helpful if this was clear from the notation.
* Theorem 3.2: Related to the previous point, the formulation of this theorem in the main text should include the $\kappa$ factor for the distance (which can only be inferred from the appendix).
* Theorem 3.4: On line 246, the definition of $\mathcal{S}_1$ includes $i$, but this symbol does not appear anywhere else in the formula, so it is not clear what the point of $\exists i$ is. Also, this definition is extremely different from the definition of $\mathcal{S}_1$ in Equation (20) in the appendix; not clear if these are equivalent. Further, this theorem in the main text uses notation $\mathcal{S}_0$, but this notation is undefined in the main text; it is only introduced in the proof. Without knowing what $\mathcal{S}_0$ is, the claim $\Gamma > 1$ on line 254 would not be clear.
* The paper does not seem to specify any hyperparameters used in the experiments (knowing these would be important for both the baselines and the proposed approach).

It would also be beneficial if the paper provided a few examples of practical applications.

A couple minor issues:
* Source code is available, but it has no comments or documentation, making it difficult for other to run it and reproduce the results.
* Section 5 should be broken up into subsection; or at the very least, add some paragraph headers, so that it is easier to find baselines, environment, etc.
* References should be formatted with \citep instead of \cite for readability. Quotation should be typeset as ``...''
* There are also a number of typos (e.g., line 077: "performance. faster", line 267: "MDPS") and editing mistakes (e.g., line 163: "(From Eqn 18)" seem misplaced).
 * The proposed approach is evaluated on a single series of tasks. It is not obvious how well the proposed approach will work on another set of tasks.

### Questions
Please clarify the notation (see weaknesses).

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes LiZero, a lifelong Monte Carlo Tree Search (MCTS) framework that transfers knowledge across sequential tasks via an adaptive UCT (aUCT) rule. The aUCT bound jointly depends on (i) the Lipschitz distance between tasks and (ii) the confidence of previously learned Q-values. Theoretical analysis shows improved sample efficiency with an acceleration factor > 1, and experiments on sequential grid-world tasks demonstrate 3–4 times faster convergence and ~31% higher early rewards compared to MCTS and lifelong RL baselines.

### Strengths
- Novel direction: First to formalize lifelong MCTS with cross-task transfer.
 
- Solid theory: Derives Lipschitz-continuity–based bounds with provable convergence and acceleration guarantees.
 
- Algorithmic contribution: Introduces both data-driven and neural distance estimators with theoretical error analysis.
 
- Empirical gains: Consistent, large improvements over MCTS, pUCT, and lifelong RL baselines.

### Weaknesses
- Restrictive assumptions: Relies on shared state/action spaces and Lipschitz-similar tasks, limiting generality beyond simple settings.
 
- Narrow evaluation: Only tested on discrete grid-worlds; scalability to continuous or high-dimensional domains remains unverified.
 
- Missing runtime analysis: The computational cost of distance estimation and maintaining multiple task models is not discussed.
 
- Weak Lipschitz enforcement: Theorem 4.3 assumes globally Lipschitz neural dynamics without explicit constraints (e.g., spectral norm control), which may render the bound loose in practice.
 
- Unvalidated distance metrics: The quality of the proposed distance estimators is never measured directly, leaving unclear whether performance gains truly arise from meaningful task similarity.
 
- Questionable IS practicality: The importance-sampling correction in Eq. (6) requires a known reference distribution U and full support of $\pi$; potential variance explosion and normalization issues are unaddressed.
 
- Unclear definition of U(s,a): Although a “normalized uniform distribution” is referenced, its concrete form and normalization method are unspecified, especially for large or continuous state spaces—raising reproducibility concerns.

### Questions
- How does LiZero behave when tasks are dissimilar or violate Lipschitz continuity—can it avoid negative transfer?
 
- What is the actual computational overhead of the distance estimation in LiZero-P/N?
 
- Could the framework integrate into MuZero-style model-based planning?
 
- How would performance scale with longer task sequences or partially overlapping state spaces?
 
- How are the importance weights $w_i$ obtained and stabilized in practice?
 
- How is the “normalized uniform distribution” U(s,a) implemented—over what domain, and are clipping or normalization strategies applied to control variance?

### Soundness
2

### Presentation
2

### Contribution
2
