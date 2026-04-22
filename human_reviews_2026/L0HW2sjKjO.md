# Sparse Policy Space Response Oracles

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
In multi-agent non-transitive games, the Policy Space Response Oracles (PSRO) framework approximates Nash Equilibrium by iteratively expanding policy populations. However, the framework suffers from severe policy redundancy in the processes of policy generation and policy population construction, thereby leading to a substantial increase in computational complexity. To address these limitations, this paper proposes Sparse PSRO, a novel framework that overcomes policy redundancy through two key innovations: (1) Sparsity Metric, which quantifies the dissimilarity between candidate strategies and existing populations via convex combination residual constraints, guiding the algorithm to explore underrepresented payoff regions while suppressing redundant policy generation; (2) Policy Space Sparsification, which constructs the Policy Hull backbone via intensive early exploration and admits only geometrically distinct strategies through threshold control, effectively reducing the number of policies and lowering computational complexity. Theoretical analysis proves that Sparse PSRO maintains a finite policy population with guaranteed separation distances, preventing exponential population growth while ensuring convergence to the Nash Equilibrium. Experiments across diverse environments (including RGoS, AlphaStar888, Blotto, and Kuhn Poker) demonstrate that Sparse PSRO significantly outperforms six baseline methods in terms of exploitability and policy population size, thus validating its effectiveness in efficiently approximating Nash Equilibrium with reduced computational costs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Sparse PSRO for multi-agent non-transitive games. Sparse PSRO tries to improve the inefficiencies of the traditional PSRO method, in terms of low exploration efficiency and redundant strategies. To this end, sparse PSRO employs policy sparsity regularization, which quantifies and encourages diversity among strategies. Afterwards,  a threshold-controlled strategy addition rule is introduced for the addition of new strategy to the population, within the traditional framework of PSRO. Experimental results on normal form games show that Sparse PSRO achieves lower exploitability compared to recent PSRO variants.

### Strengths
- The paper is well written. 
- Baseline methods are relatively comprehensive. 
- The performance improvement seems significant.

### Weaknesses
•	huge resemblance to previous diversity-regularized PSRO (in particular PSD-PSRO): the motivation of the method, the definition of sparsity metric (a different name to the diversity metric), the way of regularization, proofs, etc. After double-checking the method PSD-PSRO, considering the discussion in Section 4.4, to a large extent, I think Sparse PSRO can be viewed a reduced version of PSD-PSRO to the normal-form games, where strategies can be represented in coefficients of pure strategies.

•	The sparsity metric in Equation 6 only applies to simple normal-form games, where strategies can be represented in coefficients of pure strategies. For larger games, where strategies are usually represented in neural networks, strategies cannot be represented in coefficients of pure strategies, Sparse PSRO is currently not applicable.

•	insufficient guidance of how to appropriately set the hyperparameter sparsity threshold.

•	insufficient ablation study of the hyperparameter sparsity threshold in Equation 8.

### Questions
- Is Sparse PSRO able to extend to larger games where coefficient representation of strategies (Equation 6) is not applicable?

- Is there some guideline as how to properly set $\mu$ in Equation 8?

- What is the most significant difference of  Sparse PSRO, compared to previous diversity-enhanced PSROs, especially PSD-PSRO?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the severe policy redundancy issue within the PSRO framework during policy generation and population construction. It proposes two measures: a Sparsity Metric and Policy Space Sparsification. Through experiments across diverse environments, the proposed methods are validated to efficiently approximate Nash equilibria at lower computational cost.

### Strengths
1. To address the severe policy redundancy issue in policy generation and population construction within the proposed PSRO framework, the authors conducted a thorough review of related work, enabling readers to clearly recognize the shortcomings of existing approaches.
2. Theoretically, the authors demonstrate that sparse PSRO preserves the finite size of the policy pool while ensuring the algorithm converges to Nash equilibria. This provides a solid foundation for the claimed low computational cost.
3. The authors tested the proposed algorithm in a variety of different environments, thereby demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The authors did not explicitly specify how to select an appropriate $\mu$. Consequently, this hinders the widespread application of sparse PSRO across diverse scenarios. The authors also acknowledge this as a limitation.
2. In the experimental section, although this work conducted experiments across multiple different games, the algorithm's effectiveness in complex games such as Google Soccer remains to be examined. 
3. This work primarily focuses on symmetric two-player zero-sum games, though the title does not reflect this.
4. The paper contains several instances of imprecise phrasing. For example, on line 216, the author states: “Policy Hull is the convex combination of policies in the population.” This formulation is incorrect and conflicts with the preliminaries section.

### Questions
1. A-PSRO is also an algorithm that employs threshold control to generate policies. Although the authors mention this method in the related work section, why was it not included as a baseline algorithm? 
2. The size of the policy pool is insufficient to demonstrate that your approach resolves the issue of redundant policies. Based solely on the policy pool size, sparse PSRO does not exhibit a significant advantage over naive PSRO. Why not conduct a more detailed statistical analysis of the existence and quantity of redundant policies?
3. This paper does not address how to represent an agent's policy compared to other studies. Is the policy defined as a row in the payoff matrix? Or is it a representation based on state-action distributions and occupancy measures? If a policy is represented by a row in the payoff matrix, how does the diversity measure proposed in this paper differ from the “Response Diversity” measure in “Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games”?
4. Based on the ablation experiments, empirically speaking, the results of Sparsification-PSRO should not be as poor as shown in the figures. Could you provide further explanation?
5. In Algorithm 1, why is it necessary to “Sample $K$ Policies $\{\pi^k_i \}^K_{k=1}$ from Policy Hull $\Pi_i^t$”? This sampling step appears abrupt and lacks connection to the other steps in Algorithm 1.
6. What is “relative population performance” in line 161? The authors have not explained this concept.

### Soundness
2

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
4

### Summary
This paper addresses the **policy redundancy** issue in the PSRO algorithm by proposing a metric to measure whether a newly trained policy lies within the convex hull formed by the existing set of policies. Based on this metric, the authors introduce the **Sparse PSRO** algorithm.

### Strengths
- This paper is easy to follow

### Weaknesses
The proposed method **does not appear to be significantly different** from existing approaches. Furthermore, the paper does not provide a detailed explanation of **how the proposed sparsity metric is used for policy optimization**. Specifically, I cannot discern from the equations how the **gradient propagation** is performed using this metric, and I would appreciate an explanation from the authors.

Additionally, the concept of **"policy redundancy"** is not clearly defined in the paper. Moreover, the analysis of how existing algorithms handle this issue is **contradictory**: the authors initially state that existing algorithms address this problem by optimizing for diversity, but then later claim that current algorithms neglect policy redundancy. This inconsistency is confusing to me.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new, improved version of PSRO (policy space response oracles). The method, Sparse PSRO, has two components: (1) a Sparsity metric, which guides best-responses to be different from existing policies in the population, and (2) Sparsification, in which a new policy is not added to the population unless it is suitably different than existing policies in the population.

### Strengths
The research direction is interesting. 

The empirical results look good.

### Weaknesses
1. The theorems and proofs are strange.

The first theorem (Theorem 4.1) states that the policy population remains "finite" for any t>1. This statement seems silly and vacuous to me -- of course if we add at most 1 policy each iteration, the cardinality of the set of policies will be finite at any iteration. The paper then goes on to say that this theorem avoids "exponential growth", which is not the same as finiteness.

The proof of second theorem (Theorem 4.3) also seems questionable to me. It is stated in the proof that expanding the policy hull reduces population exploitability -- I don't see why this was the case. Intuitively, it seems like expanding the policy hull *doesn't increase population exploitability*, but I don't see why it necessarily reduces it. 

Also, not sure how big of an issue it is, but it is stated that the game is a symmetric zero-sum game. Are we only studying *symmetric* zero-sum games in this work? But the experiments are not all on symmetric games, right? (e.g. Kuhn poker is not symmetric)

2. It's not immediately clear what the x-axis on the empirical results (Figure 1, Figure 2, Figure 4). I'm assuming that for Sparse-PSRO and Sparsity-PSRO, an "iteration" is only when a new policy is actually added to the population. If so, this mismatch between iteration and computation/time is a bit swept under the rug, and it would be nicer if this was mentioned in the paper or if graphs were included with other choices of x-axis as well.

3. I don't understand the main metric (sparsity).
It is defined in Equation 6. Crucially, it includes an arithmetic operation on policies: ${\pi_i - \Pi^t_i}^\top$. It's not clear to me how we are supposed to add or subtract policies. The most obvious interpretation is that we are performing vector addition and subtraction where e.g. the pure strategies are the vectors <0,0,0,1>, <0,0,1,0>, <0,1,0,0>, and <1,0,0,0>, and the mixed strategies lie on the simplex between these points. But then I don't see how the methods in the paper differ from normal PSRO/double oracle, in that any novel pure strategy should be equidistant from the existing policy hull induced by a population of pure strategies.

### Questions
1. Why do we need to mention Caratheodory's theorem to justify Equation 5? Does it not suffice to define a mixed strategy? Or are we specifically interested in extensive-form games? If so, then Kuhn's Theorem seems like the more appropriate citation.

### Soundness
2

### Presentation
2

### Contribution
2
