# Boolean Satisfiability via Imitation Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
We propose ImitSAT, a branching policy for conflict-driven clause learning (CDCL) solvers based on imitation learning for the Boolean satisfiability problem (SAT). Unlike previous methods that predict instance-level signals to improve CDCL branching indirectly, or rely on reinforcement learning and insufficient CDCL information to enhance branching, ImitSAT learns from expert KeyTrace that collapses a full run into the sequence of surviving decisions. Replaying a KeyTrace on the same instance is nearly conflict-free, providing dense decision- level supervision and directly reducing propagations—the dominant contributor to wall-clock time. This prefix-conditioned supervision enables ImitSAT to reproduce high-quality branches without exploration, yielding faster convergence, stable training, and seamless integration into CDCL. Extensive experiments demonstrate that ImitSAT reduces propagation counts and runtime, outperforming state-of-the-art learned approaches. We released the source code and trained model at https://github.com/zewei-Zhang/ImitSAT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies SAT problem. As claimed by the authors, previous learning-based method were limited. In this work, they propose a branching policy for CDCL SAT solvers based on imitation learning. The authors conduct experiments to evaluate the effectiveness of their approach. They claim that their proposed method reduces propagation counts and outperforms previous learning-based method.

### Strengths
1. The paper is generally well-written, with clear explanations.

2. The description of the dataset construction method is detailed and reasonable.

3. Experimental results are satisfactory.

### Weaknesses
1. What is the SAT solver integrated by SATformer in the experiment? According to the SATformer paper, combining it with CaDiCaL and Kissat can produce performance improvements.

2. MiniSAT, used by ImitSAT, is relatively outdated. How does ImitSAT perform compared to current SOTA solvers (such as Kissat and CaDiCaL)? Can combining ImitSAT with these solvers bring further performance improvements?

### Questions
Please reply to my comments listed in ‘Weaknesses’.

### Soundness
3

### Presentation
3

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
The paper introduces ImitSAT, a framework for learning a CDCL decision policy by imitation learning on condensed solver runs. This works by post-processing solver trails to eliminate intermediate conflicts and backtracking steps, which results in compact trails that only contain the key decisions made by the solver. In doing so, the model learns to imitate an improved version of the underlying solver, which can then be used to guide the solver decisions under a certain query budget. The authors compare their method to two other learned hybrid solvers, namely GQSAT and SATformer, in terms of solver-relative performance and wall-clock time on both in-distribution and out-of-distribution benchmarks. Results show that ImitSAT considerably outperforms both baselines on almost all benchmarks under a much lower runtime.

### Strengths
1. The paper is clear and easy to follow. 
2. The key idea is fairly interesting: distilling a post-processed solver policy into a neural network. 
3. The authors compare their method against two somewhat recent baselines. 
4. The paper evaluates the method on both in-distribution and out-of-distribution instances.
5. The method outperforms the reported baselines while having a lower runtime.

### Weaknesses
1. The authors do not evaluate their method on industrial instances, so it's unclear how well it scales to real-world settings. 
2. The attempted query budgets are quite low (up to 5 decisions), so it's unclear how well the method scales with higher budgets. 
3. Wall clock time comparisons are missing the pure VSIDS heuristic baseline (i.e., without the model), which would be the ultimate test to see if the proposed method speeds up the CDCL solver. 
4. It is unclear in Section 5.1 whether the SATformer and Graph-Q-SAT baseline models were pertrained on their original papers' dataset or on the same random 3-SAT dataset that ImitSAT was trained on. If it's the former, then the comparison in Table 1 is quite unfair since the baselines were trained on a different dataset. 
5. Using a text model is arguably not ideal for embedding formulas compared to GNNs as it fails to capture key properties such as invariance to clause permutation and variable renaming. However, I do understand that it might be more convenient to work with text.

### Questions
1. Have you tried applying your method to industrial benchmarks like SATCOMP? It does not have to outperform the baseline solver, but it would be good to know where ImitSAT stands in real-world settings. 
2. How does the performance look if you only use the model decisions? 
3. Have you experimented with higher query budgets than 5? It would be interesting to plot performance over query budget. 
4. Equation (10) does not index $F_{DIMACS}$ by $t$, so is it always the original input formula? That is, does it not include the learned clauses? If so, wouldn't that compromise the decision optimality since the model does not see the clauses that were learned during the run? 
5. When the model's top answer is illegal, you fall back to the VSIDS heuristic. Why not mask out the illegal variable decisions from the model's output logits/probabilities? In doing so, you could use the model's top legal decision. 
6. In Figure 4, can you also include the time of the baseline un-augmented MiniSAT solver? 
7. Have you considered bootstrapping the training process by using the model to improve the trails dataset? For example, whenever the model produces a trail with fewer propagations, you could use the produced model trail for subsequent epochs instead of the KeyTrace one. 
8. Were the two baselines trained on the same 3-SAT dataset on which ImitSAT was trained, or were they trained on their original papers' datasets? 
9. Can you also include the KeyTrace ratios in Tables 1 & 2? 
10. To see the time scaling with query budgets, can you include the 5-call ImitSAT to Figure 4, possibly accompanied by higher query budget variants?

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
3

### Summary
Conflict Driven Clause Learning (CDCL) is the state of the art approach for solving boolean satisfiability (SAT) problems. During CDCL solving, the majority of the solving time is spent on unit propagation, the efficiency of which is heavily influenced by the branching decisions quality.

This paper introduces ImitSAT, which learns to branch in a CDCL solver using imitation learning. To do this, key traces are first extracted by collapsing a CDCL branching search tree into a single sequence of decisions. Then, an autoregressive learner is trained to predict the key traces sequentially through imitation learning. By learning to make better quality branching decisions, ImitSAT is able to cut down solving time. Experiments show that ImitSAT outperforms SATFormer and Graph-Q-SAT.

### Strengths
The paper is easy to follow and the idea is simple but effective. 

The method is well-motivated by the run-time analysis that propagation is the most time-consuming part of CDCL solving.

The algorithm introduces a budget for querying the model to ensure efficiency.

The approach shows good performance, beating Graph-Q-SAT and SATFormer on structured SAT families outside of training distribution.

### Weaknesses
The idea of using imitation learning for branching is not new. For example, it has been successfully applied to guide branch-and-bound for mixed integer programming [1,2].

The algorithm consumes the query budgets greedily, which, given the limited budget could potentially be far from optimal for instances larger than those in Appendix E. Are there simple heuristics that can improve this?

The experiments are on very small sized SAT problems, which provides limited insight to how ImitSAT can be used in practice.

Training is computationally expensive, since CDCL solvers are used to solve a problem entirely to produce a single key trace as a training instance. Therefore, the generalization ability of the method is crucial and calls for a deeper look. 


[1] He H, Daumé H, Eisner J. Learning to search in branch and bound algorithms. Advances in neural information processing systems. 2014;27.

[2] Khalil E, Le Bodic P, Song L, Nemhauser G, Dilkina B. Learning to branch in mixed integer programming. InProceedings of the AAAI conference on artificial intelligence 2016 Feb 21 (Vol. 30, No. 1).

### Questions
How does the efficiency of ImitSAT maintain as the size of the SAT problem changes? To what sized instance can it generalize?

How is the query budget determined?

### Soundness
2

### Presentation
3

### Contribution
2
