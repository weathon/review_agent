# Contextual Multi-Armed Bandits with Minimum Aggregated Revenue Constraints

- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
We examine a multi-armed bandit problem with contextual information, where the objective is to ensure that each arm receives a minimum aggregated reward across contexts while simultaneously maximizing the total cumulative reward. This framework captures a broad class of real-world applications where fair revenue allocation is critical and contextual variation is inherent. The cross-context aggregation of minimum reward constraints, while enabling better performance and easier feasibility, introduces significant technical challenges—particularly the absence of closed-form optimal allocations typically available in standard MAB settings. We design and analyze algorithms that either optimistically prioritize performance or pessimistically enforce constraint satisfaction. For each algorithm, we derive problem-dependent upper bounds on both regret and constraint violations. Furthermore, we establish a lower bound demonstrating that the dependence on the time horizon in our results is optimal in general and revealing fundamental limitations of the free exploration principle leveraged in prior work.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the contextual multi-arm bandit problem with minimum aggregated revenue constraints, i.e., the average revenue of each arm should be above a given threshold. This problem is subsumed by the more general problem of contextual bandits with linear constraints [Slivkins et al. 2024], but the authors go beyond their results by providing two algorithms OLP and OPLP: OLP achieves polylog regret and square root constraint violation, while OPLP has the reversed guarantee. Further, there is a lower bound regret of $\Omega(\log T)$ and a lower bound sum of regret and violation of $\Omega(\sqrt{T})$.

### Strengths
+ The paper studies a sub-problem of a more general problem which is of independent interest, and obtains a better upper bound result than the general problem. 
+ The authors show the possibility of applying re-solving-based techniques (though not emphasized) in the contextual bandits with linear constraints, which I feel really interesting.

### Weaknesses
- I feel like the "contextual" term should appear in the title. 
- The authors do not discuss the literature on the re-solving technique which is widely used in contextual decision-making problems, e.g., [Chen et al., NeurIPS 2024]. 
- Though the main results are already interesting, I still encourage the authors to justify the following points, which would be better for the understanding of the whole community:
  1. There is underlying hardness for re-solving based techniques to work in the CBwK problem. Intuitively, samples for certain (context, arm) pairs can be unattainable because the probability of selecting such a pair could be zero in the LP [Chen et al., NeurIPS 2024]. But this problem does not seem to happen for the considered problem. Can the authors explain an intuition? 
  2. Why is the sub-Gaussian reward assumption necessary?
  3. Do you have a beyond-the-worst-case result, which says that the algorithm still has a performance guarantee when the non-degeneracy condition does not hold? 
  4. What will happen if the fluid LP is infeasible? This may happen in real life, and is there any way to detect this without losing too much?
  5. What does $\rho^*$ mean intuitively? And can you give an intuitive explanation of why the regret relies on $\rho^*$?
- The author(s) suppose that the context prior is known. This assumption should be relaxed for a general CB paper. 


Reference:

Chen et al. Contextual Decision-Making with Knapsacks Beyond the Worst Case. NeurIPS 2024.

### Questions
See the above "weaknesses" part. Feel free to answer those questions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies a new constrained bandit setting where the learner must not only maximize cumulative reward but also ensure that per arm revenue constraints are satisfied. Unlike Bandits with Knapsacks (BwK), which constrain resource consumption, here the constraints are on revenue accumulation. The learner’s decisions must generate sufficient revenue over time for all arms, not just maximize expected reward. The paper proposes modified UCB-style algorithms that solve a linear program (LP) that accounts for both reward maximization and constraint satisfaction. Based on using UCB only or UCB and LCB as the surrogate for reward and revenue parameters in the LP formulation of the optimal allocation problem, the proposed algorithms achieve different tradeoffs between reward maximization and constraint satisfaction. These tradeoffs are explained via regret bounds. The paper also presents an interesting lower bound that refutes the free exploration property used in prior works in non-contextual setups. Simulations depict that the proposed algorithms work reasonably well.

### Strengths
Revenue-based constraints are practically relevant (e.g., ensuring minimum revenue in ad allocation, subscription systems, or online platforms). 

Theorems are cleanly stated, with clear regret definitions and decomposition. The lower bound that refutes free exploration is interesting and strengthens the theoretical contribution of the work. 

The paper is overall well-written (except for some small typos and notational inconsistencies, which can be addressed by carefully going over the entire paper). 

The comparison with related work is adequate. There are Works that address more general constraints, but they have suboptimal regret bounds for the specific problem considered in this work because they do not utilize the problem structure specific to this work. This paper's analysis is based on a new suboptimality gap based on saturated constraints of the LP and can be regarded as an important theoretical contribution.

### Weaknesses
Algorithms are intuitive and clearly explained. Their design principle is quite standard. Use UCB or combine LCB or UCB, which is the two things that are expected to achieve the tradeoffs between reward maximization and constraint satisfaction. The design process of the algorithms are clear and intuitive. However, they do not require out-of-the-box thinking. This is not a significant weakness, but it makes the novelty of the algorithms somewhat incremental.

### Questions
1. Please provide further insight on Assumption 3. Line 366 says that inner maximization in OLP is feasible under Assumption 3. Line 253 says that strict feasibility is required only for OPLP. 

2. How can the algorithms be extended to the case with infinite contexts? 

3. Is “known context probabilities” assumption be still reasonable in the infinite context setting?

### Soundness
3

### Presentation
3

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
This paper makes meaningful contributions to the multi-armed bandit (MAB) field by proposing the novel MAB-ARC (Multi-armed Bandit with Adaptive Resource Constraints) problem, filling a gap in the intersection of bandit optimization and dynamic resource constraint management. The two tailored algorithms (OLP for performance priority, OPLP for constraint priority) and their complete theoretical bounds (regret and constraint violation upper/lower bounds) demonstrate rigorous academic thinking and practical problem-solving awareness. However, the work is limited by restrictive finite space assumptions and high computational overhead, which affect its generalization and real-world applicability.

### Strengths
The paper introduces the MAB-ARC problem, a new research direction that integrates adaptive resource constraints into traditional multi-armed bandits. This fills a gap in existing literature—most MAB studies either ignore resource constraints or assume static constraints, while MAB-ARC focuses on dynamic resource allocation scenarios (e.g., real-time budget adjustment in advertising, resource-limited industrial control). The problem is not only theoretically innovative but also highly aligned with practical needs, providing a new research paradigm for constraint-aware bandit optimization.

### Weaknesses
The paper assumes that both the context space (decision-making feature space) and arm space (optional action set) are finite. This assumption simplifies theoretical derivation but severely limits generalization:​
In many practical scenarios (e.g., continuous context features like user age/income in recommendations, high-dimensional arm spaces like multi-channel advertising combinations), the context or arm space is infinite or high-dimensional.​
The algorithm cannot be directly applied to these scenarios, reducing its practical value and narrowing its target audience (e.g., excluding researchers/engineers working on continuous-space bandits). The algorithm requires constructing and solving a linear programming (LP) problem at each time step. As the number of time steps increases, the cumulative computational cost grows linearly. For large-scale scenarios (e.g., online advertising with daily impressions, real-time control systems with millisecond-level time steps), this overhead will cause decision delays or even system unresponsiveness.​
The paper does not analyze computational complexity in depth, nor does it propose optimization strategies (e.g., approximate LP solving via heuristic methods, batch processing of time steps to reduce LP calls, or leveraging dual decomposition for faster computation). This lack of engineering consideration hinders the algorithm’s practical deployment.

### Questions
Extend the MBA-ARC model to infinite context spaces (e.g., continuous features) or high-dimensional arm spaces, and adjust the algorithms/ theoretical analysis accordingly (e.g., using kernel methods to map continuous contexts to finite feature spaces, or adopting dimensionality reduction techniques for high-dimensional arms).​ Quantify the time/memory cost of solving the LP problem per time step, and compare it with baseline algorithms to highlight the overhead issue. Add a section discussing practical deployment considerations, such as how to set hyperparameters for OLP/OPLP (e.g., constraint violation tolerance, LP solving precision) based on scenario requirements, and how to handle edge cases (e.g., sudden resource shortages, context distribution shifts).

### Soundness
3

### Presentation
3

### Contribution
3
