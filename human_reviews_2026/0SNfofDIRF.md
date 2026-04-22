# Bandits with Single-Peaked Preferences and Limited Resources

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
We study an online stochastic matching problem in which an algorithm sequentially matches $U$ users to $K$ arms, aiming to maximize cumulative reward over $T$ rounds under budget constraints. Without structural assumptions, computing the optimal matching is NP-hard, making online learning computationally infeasible. To overcome this barrier, we focus on single-peaked preferences---a well-established structure in social choice theory, where users' preferences are unimodal with respect to a common order over arms. We devise an efficient algorithm for the offline budgeted matching problem, and leverage it into an efficient online algorithm with a regret of $\tilde O(UKT^{2/3})$. Our approach relies on a novel PQ tree-based order approximation method. If the single-peaked structure is known, we develop an efficient UCB-like algorithm that achieves a regret bound of $\tilde O(U\sqrt{TK})$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work studies an online stochastic matching problem involving $U$ users and $K$ arms, aiming to maximize the cumulative reward over $T$ rounds under budget constraints. Without assuming any structural property of user preferences, the problem is NP-hard, rendering online learning computationally infeasible. To address this, the authors focus on the case of single-peaked preferences. They first propose an efficient algorithm that achieves a regret of $\tilde{O}(UKT^{2/3})$. When the single-peaked structure is known in advance, they further develop an algorithm that attains a regret of $\tilde{O}(U\sqrt{TK})$.

### Strengths
This is the first work to study the matching bandit setting under single-peaked preferences, where an efficient algorithm can be designed. The proposed method achieves a tight regret bound when the preference structure is known.

### Weaknesses
The main concern lies in the tightness of the regret bound under the unknown structure. The authors propose an ETC-based algorithm that achieves a regret of order $T^{2/3}$, which appears to be suboptimal. Establishing a tight regret bound for the unknown-structure case would be a meaningful goal, as the current result seems incomplete. If achieving a tighter bound is not possible, the authors should at least provide a corresponding lower bound to justify that the $T^{2/3}$ rate is optimal. Moreover, the ETC and UCB algorithms themselves do not appear to be novel.

### Questions
Is there a specific reason for using the term ``preference'' here? To me, preference suggests a Bradley--Terry--Luce--type model, where the reward of each arm is defined relatively, depending on the assigned counterpart.

### Soundness
3

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
This paper studies an online stochastic matching problem where a learner must sequentially match users to arms under a budget constraint, with the goal of maximizing cumulative reward. The general problem is NP-hard, but the authors circumvent this computational barrier by imposing a single-peaked preference structure, a well-known concept from social choice theory where each user's utility is unimodal with respect to a common ordering of the arms. For the offline problem, they develop SP-Matching, an efficient dynamic programming algorithm that finds the optimal budgeted matching. Leveraging this, they propose two online algorithms: EMC, an explore-then-commit algorithm for the challenging case of unknown preference structure, which achieves a regret of $\tilde{O}(UKT^{2/3})$ by extracting an approximate order using PQ trees; and MVM, an efficient UCB-like algorithm for the case of known structure, which achieves a tighter regret of $\tilde{O}(U\sqrt{TK})$ by leveraging a novel maximal matrix within its confidence sets. This paper also gives a regret lower bound analysis for both known and unknown peaks cases.

### Strengths
1. The single-peaked preference assumption is grounded in social choice theory and circumvents NP-hardness, enabling efficient algorithms with standard regret bounds instead of weaker α-regret.
2. It provides a complete landscape by offering efficient algorithms for both unknown (EMC) and known (MvM) structure settings, while rigid theoretical analysis are given on both algorithms.
3. The PQ-tree-based order extraction method and the concept of a maximal matrix, which enables optimistic planning while preserving structural constraints, are novel.

### Weaknesses
1. While lower bounds are provided, gaps remain between upper and lower bounds in some settings (e.g., for EMC). The paper does not conclusively determine if the attained rates are optimal for polynomial-time algorithms in the unknown structure case. To my knowledge, the ETC-based algorithm is not an order-optimal algorithm in the classical stochastic MAB problem, so it is not surprising that there exists a gap between the upper and lower bounds. Even though the proposed EXTRACT-ORDER algorithm used in EMC provides insight into the construction of an ASP order.
2. The setting that different users can be matched to the same arm simplifies this problem, while a more common setting is the bipartite matching between users and arms.  


Minor
1. Title of Section 4.3: written as ETC instead of EMC.

### Questions
See weakness.

### Soundness
3

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
4

### Summary
This paper addresses an online stochastic matching problem where a learner sequentially matches users to arms under budget constraints, aiming to maximize cumulative reward over $T$ rounds. The key innovation is leveraging single-peaked (SP) preferences to overcome computational intractability, where user preferences are unimodal with respect to a common arm order. This paper studies 1) offline setting, 2) online setting with known preference structure, and 3) online setting with unknown structure.

### Strengths
1. ​​The paper bridges combinatorial bandits and social choice theory by incorporating SP preferences, transforming an NP-hard problem into a tractable one while maintaining standard regret bounds instead of approximate regret.
2. ​​Both offline and online algorithms are designed with theoretical guarantees. The maximal matrix construction for MvM and PQ-tree-based order recovery in EMC demonstrate creative problem-solving.
3. ​​Experimental results demonstrate theoretical regret rates.
4. The budget constraint modeling and SP assumption aligns well with real-world applications like content recommendation.

### Weaknesses
My major concern is the gaps in bounds​​ of online setting with unknown structure. The gap between EMC's $O(UKT^{2/3})$ regret and the $\Omega{\sqrt{KT}}$ lower bound suggests potential for improved algorithms or tighter analysis.

### Questions
1. Could the SP assumption be relaxed to more general structures while retaining computational efficiency?
2. What is the hardness in designing algorithm attaining $O(\sqrt{T})$ regret for online setting with unknown structure? Is it possible to improve the lower bound by constructing instance with this hardness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies a budget-constrained matching problem. The authors first point out that, under general preference structures, the problem is NP-hard even in the offline setting. To overcome this computational barrier, the paper introduces the assumption of single-peaked (SP) preferences, which theoretically ensures tractability. For the offline case, the authors propose a dynamic programming algorithm, SP-MATCHING, which computes the optimal matching in polynomial time under a known SP order. This algorithm provides the theoretical and computational foundation for the subsequent online methods. In the online setting, the paper considers two cases:

1.	When the SP structure is known, the authors propose an optimistic algorithm MvM (Match-via-Maximal) based on maximal matrices, which uses SP-MATCHING each round to compute the optimal matching and achieves sublinear regret $O(\sqrt{T})$.
2. When the SP structure is unknown, the authors propose an explore-then-commit (EMC) method that first learns an approximate preference order from data collected in a round-robin manner via an EXTRACT-ORDER process based on PQ trees, and then executes this policy based on the approximate preference order, achieving regret $O(T^{2/3})$.
The paper further establishes corresponding regret lower bounds for the known and unknown peak settings. Experiments on synthetic data validate the theoretical results.

### Strengths
1. The paper introduces the single-peaked (SP) structural assumption for budget-constrained matching, turning an otherwise NP-hard problem into one solvable in polynomial time. This may offer a principled blueprint for simplifying other computationally hard online learning settings.
2. The algorithmic design is well structured: SP-MATCHING yields a polynomial-time optimal solution offline; the online setting presents MvM (known SP) and EMC (unknown SP), both built on SP-MATCHING. The theory is tight and carefully argued, with detailed proofs of regret bounds and supporting lemmas in the appendix. The results are complete with both upper bounds and lower bounds provided.

### Weaknesses
1.	Strong assumptions.
The paper relies on the single-peaked (SP) preference assumption, which may be overly idealized in complex real-world environments. Although the paper states that this preference structure is common in various domains, such as recommendation systems (Line 049-052), no reference is provided for justification. The additional requirement of known SP order and user peaks for the MvM algorithm is unrealistic, which limits its applicability. 

2. Insufficient experiments.

This paper introduces a new matching setting with a budget constraint, which appears to have not been studied before. The experiments are also restricted to synthetic data. No real-world applications/datasets have been tested. Moreover, the paper lacks comparisons with general combinatorial bandit baselines, particularly those without SP assumptions but with α-regret guarantees, which would better contextualize the claimed advantages.

### Questions
1. For the UCB-based algorithm, the authors claim that solving the optimization problem requires a known SP order. How about using the estimated order from the estimated means and UCB/LCBs? Specifically, though the total order is initially unknown,  the algorithm can construct a partial ordering based on estimations. Can the algorithms solve some optimization problem corresponding to partial ordering to avoid the strong assumption and get an $O(\sqrt{T})$ regret? The authors can discuss this point. 
2. Writing suggestion (optional): When two algorithms are available, usually the paper first presents an algorithm that requires strong assumptions to convey some intuition and then presents a more general one.

### Soundness
3

### Presentation
3

### Contribution
3
