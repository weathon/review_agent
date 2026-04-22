# Bicriteria Algorithms for Submodular Cover with Partition and Fairness Constraints

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
In many submodular optimization applications, datasets are naturally partitioned into disjoint subsets. These scenarios give rise to submodular optimization problems with partition-based constraints, where the desired solution set should be in some sense balanced, fair, or resource-constrained across these partitions. While existing work on submodular cover largely overlooks this structure, we initiate a comprehensive study of the problem of Submodular Cover with Partition Constraints (SCP) and its key variants.  Our main contributions are the development and analysis of scalable bicriteria approximation algorithms for these NP-hard optimization problems for both monotone and nonmonotone objectives. Notably, the algorithms proposed for the monotone case achieve optimal approximation guarantees while significantly reducing query complexity compared to existing methods.
 Finally, empirical evaluations on
real-world and synthetic datasets further validate the efficiency and effectiveness
of the proposed algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Addressing the limitation of traditional submodular cover problems that overlook partition structures and fairness constraints, this paper systematically investigates three constrained submodular cover variants for the first time: non-monotonic SCP (Submodular Cover with Partition Constraints), monotonic SCKP (Submodular Cover with Knapsack Partition Constraints), and monotonic SCF (Submodular Cover with Fairness Constraints). Its core contributions are as follows: (1) proposing a "submodular maximization-cover conversion framework" that transforms cover problems into more tractable maximization problems (SMP/SMKP), breaking the bottleneck of traditional methods that directly design cover algorithms; (2) designing bicriteria approximation algorithms (nonmono-bi, greedy-knapsack-bi, Block-Fair-Bi) for the three problem types, achieving optimal approximation ratios in monotonic scenarios while significantly reducing query complexity; (3) validating the proposed algorithms on both real-world (Corel5k, ImageNet_50) and synthetic datasets, demonstrating their superiority over baselines such as STREAM and GREEDY-Fair in terms of function value achievement rate, budget optimization, and fairness balance.

### Strengths
It is the first systematic study of the submodular cover problem with partition constraints, covering practical scenarios such as fairness and budget allocation, and has strong application value.

The proposed "block greedy" strategy effectively overcomes the limitations of traditional greedy algorithms under partition constraints, balancing the approximation ratio and query efficiency.

All algorithms provide strict approximation ratio guarantees and are detailedly derived through methods such as induction and concentration inequalities.

### Weaknesses
I did not identify any major weaknesses in this paper. However, there is still room for improvement in the writing and presentation. For example, at the end of Chapter 1, where the contributions are summarized, the authors could more explicitly clarify whether the SCP and SMKP problems have been previously studied, and highlight how this work improves upon existing results in terms of approximation ratio or computational complexity.

In addition, since the proposed algorithm is derived by transforming a submodular maximization algorithm into a submodular cover problem, the paper should include a more thorough discussion of related works on submodular maximization that are closely connected to this approach, such as

* *Fast adaptive non-monotone submodular maximization subject to a knapsack constraint*
* *Fairness in Streaming Submodular Maximization Subject to a Knapsack Constraint*
* *Fair Submodular Maximization over a Knapsack Constraint*
* *Linear-Time Algorithms for Representative Subset Selection From Data Streams*

### Questions
Please refer to the Weaknesses

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
The paper studies Submodular Cover with Partition constraints and variants (SCP for non-monotone, SCKP for monotone with knapsack-partition costs, and SCF for “fair” matroid constraints). It introduces a unifying block-greedy framework and several conversion theorems to obtain bicriteria guarantees.

### Strengths
1. The paper propose Unified technique (block-greedy) that works across non-monotone, knapsack-partition, and fairness constraints, and explicitly relates partition constraints to cardinality style reasoning.

2. Tight/tighter bicriteria bounds in monotone settings and a clear conversion blueprint (Theorem C.3) that is reusable. 

3. The paper contextualizes the 0.305 barrier and the 1/2 feasibility impossibility for submodular cover (Crawford 2023). This frames an interesting gap (0.305–0.5). 

4. Table 1 in the appendix, which states (α,β), query complexity, and assumptions for each problem (SCP/SCKP/SCF), is helpful.

### Weaknesses
1. The presentation could be improved. 

E.g. (1) Early in §2.1, explicitly formalize the optimization objective (minimize v s.t. ) and keep that exact form visible;

(2) Minor wording/punctuation: in Appendix C  “stated. in Theorem C.3” and a few others.

2. It seems that most of the results proposed in this work are quite straightforward and based on known algorithms / techniques. 

3. For SCKP, the authors compare BLOCK-G to GREEDY and GREEDY-Knapsack and show smaller budget at similar f value. Would you provide variance/error bars across multiple random seeds and report the impact of α and δ sweeps (0.1→0.01) on queries/time and feasibility?

### Questions
See the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies the submodular cover problem, where the ground set is partitioned into disjoint sets: $U_1, ..., U_N$.  The authors study various optimization problems
1. Cardinality Constraint: Find a set $S$ such that $f(S) \geq \tau$, under the constraint that $S$  can not over-represent a member of the partition.  I.e, |S \cap U_i| is bounded by a specified parameter.
2. Knapsack;  Find $S$, so tyat $f(S) \geq \tau$ and C(S \cap U_i) \lee specified threshold.
3. Fairness: There is an upper and lowerbound on $|S \cap U_i|.

The authors proposed an unified framework of bi-criteria approximation algorithms for these constraints. The approach build on a "block-greedy" algorithm for submodular maximization under partition-type constraints, together with conversion theorems that reduce maximization to cover problems. The work provide empirical validation of the results.

### Strengths
1. The work presents a unified framework for multiple constraints.
2. Theoretically sound bi-criteria approximation algorithms are presented.
3. The proposed algorithms operate per partition rather than an element-by-element greedy approach, leading to improved query complexity.
4. Empirical results show the practical viability of the proposed algorithms

### Weaknesses
1. Some proof ideas and algorithm design are mainly borrowed from existing works such as Chen et al 25 and Chen and Crawford 24b. Can you explain the new ideas and differences from these works.
2. For SCKP, query complexity depends in c_max/c_min which, in the worst case, can be arbitrarily large.  Is there a way to address this or can it be shown the it is needed?

### Questions
Please see weakness

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
This paper studies submodular cover under partition constraints. The problem consists of finding a subset of sufficiently large value (measured in terms of a given submodular objective), while minimizing a notion of costs and adhering to some partition constraints. To some extent, this problem can be thought of being the dual of submodular maximization with partition constraints. 

In particular, the authors study and propose bi-criteria approximation results for three problems: 
- submodular cover with partition constraints, where the objective can be non-monotone. 
- monotone submodular cover with knapsack, where the partition constraint is in terms of some cost function,
- submodular cover with fairness constraints

### Strengths
- Submodular maximization is an important topic in ML, with a vast body of work in NeurIPS, ICML, and ICLR
- The problem is practically motivated and non-trivial
- The authors provide positive results in various settings, also improving the running time of an ICLR 25 paper

### Weaknesses
- The theoretical results are not tight
- From the main body, it is fairly difficult to get a complete idea of the algorithmic contribution and its novelty. In particular, this block-greedy is presented as one of the main contribution of the paper, but it is hard to get a complete idea about it by reading the main body

Minor: 
- Consider using \citep instead of \cite when the citation is not part of the sentence
- Consider uniforming and updating the bibliography, for instance, the paper cited in line 561 and 562 has appeared at FOCS a couple of years ago.

### Questions
What is the role of $\alpha$ in the statements of the results in the intro? I understand that $\varepsilon$ is to be intended as a precision parameter that can be tuned by the algorithm designer, while the role of $\alpha$ is unclear.

### Soundness
3

### Presentation
2

### Contribution
2
