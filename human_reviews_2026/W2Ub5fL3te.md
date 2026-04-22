# HTS-Adapt: A Hybrid Training Strategy with Adaptive Search Region Adjustment for MILPs

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
Mixed Integer Linear Programming (MILP) problems are essential for optimizing complex systems but are NP-hard, posing significant challenges as the problem scale and complexity increase. Recent advances have integrated machine learning to predict partial solutions by exploiting structural patterns in MILP instances. However, existing methods often suffer from inaccurate and infeasible predictions, limiting their practical utility. In this work, we improve the Contrastive Predict-and-Search (ConPaS) framework by introducing a Hybrid Training Strategy with Adaptive Search Region Adjustment mechanism (HTS-Adapt). HTS selectively applies label-based learning and contrastive learning based on the structural properties of variables, improving prediction accuracy. Adapt dynamically adjusts the search space to mitigate infeasible predictions, thereby reducing computational overhead. Experiments demonstrate that our approach achieves a notable performance enhancement by improving prediction accuracy and reducing the search space, proving its effectiveness in addressing real-world MILP challenges. Compared to the MILP solver SCIP, our method achieves an average reduction of more than 50\% in the solution gap across four MILP datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HTS-Adapt, an approach that integrates two innovative techniques to enhance the predict-and-search pipeline. For the prediction stage, a hybrid loss combining BCE and contrastive loss is introduced to better identify high-quality solutions. For the search stage, an IIS-based procedure advances the PaS framework by dynamically adjusting the trust region. Experimental results demonstrate the superiority of HTS-Adapt over both the standard PaS and contrastive PaS methods.

### Strengths
1,  The hybrid training strategy is insightful. It narrows the scope of "bad" variables via contrastive loss, while directly applying BCE loss to "good" variables—which are less likely to cause infeasibility—is both efficient and effective.
2. The experimental design is sound. The inclusion of solution trajectory visualizations and ablation studies enables a more thorough analysis of the results.

### Weaknesses
Baselines: Although Section 2 provides a detailed review of L2O literature, the experimental comparisons are limited to the "PaS → ConPaS → HTS-PaS" pipeline. The authors should consider a broader set of baselines, including learning-based branch-and-cut and learning-based LNS methods.
Novelty: While the HTS component is well-designed, it essentially combines two existing techniques. The IIS component, on the other hand, appears more rooted in operations research than machine learning, and also seems an employment of existing approaches. Thus, the novelty of the paper may be limited.
Clarity on IIS: The description of the IIS procedure is somewhat unclear, and I do not fully understand. For example, in Line 4 of Algorithm 1, what does "fix" entail when $\Delta\neq 0$—hard fixing (as in LNS) or soft fixing (as in local branching)? Additionally, does the second constraint set $C_2$ contain at most one element (i.e., the trust region constraint)?

### Questions
1. There is an assumption that "some variables consistently exhibit identical values across high-quality solution sets." This makes sense, but to be more critic, what is the underlying intuition for this phenomenon? Furthermore, what would it imply if this assumption does not hold—would the HTS method reduce to contrastive learning, thereby invalidating the first contribution?
2. Algorithm 1 involves repeatedly solving MILPs within a 1000-second time limit. Since the total time limit in experiments is also set to 1000 seconds, does this imply that the repetition occurs only once?
3. In Figure 2, for MIS and MVC instances, the blue curve (HTS-Adapt) is significantly lower than others from the beginning, suggesting better initial prediction. However, for CA and IP instances, all methods perform comparably in the first 200 seconds, after which HTS-Adapt gradually outperforms the rest. How can this difference be explained? Should it be attributed to better prediction or more effective search?

### Soundness
3

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
2

### Summary
To efficiently solve mixed integer linear programming (MILP), machine learning techniques are used to predict partial solutions, but often suffer from inaccurate and infeasible predictions. Therefore, this work extends the current Contrastive Predictand-Search (ConPaS) framework by two-fold. First, introducing a Hybrid Training Strategy (HTS) to achieve more accurate predictions. Second, proposing an Adaptive Search Region Adjustment mechanism (Adapt) to ensure feasibility and reduce computational overhead.

### Strengths
The experiments are sufficient, and the proposed method performs well.

### Weaknesses
1. The formatting needs improvement. For example, many citations are missing brackets, which makes the paper difficult to read. The plots in Figures 2 and 3 are too small, while the legends are too large. In addition, the tables should be resized, and the placement of Table 1 is odd—it is introduced in Section 5 but appears at the beginning of Section 4.
2. The preliminaries are insufficient. The work is largely based on ConPaS, but the paper only introduces PaS; the introduction and explanation of ConPaS are lacking.
3. There are several typos. For example, the transpose symbol in Equations (1) and (3). Also, should the c in Equation (2) be bolded? It is inconsistent with Equations (1) and (3).

### Questions
1. Why were only MIS and MVC chosen to evaluate generalization?
2. The experiments use SCIP and Gurobi. Given that CPLEX is also a popular MILP solver, why wasn’t CPLEX included?

### Soundness
2

### Presentation
1

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
This paper targets two difficulties of the “predict-then-search” paradigm for Mixed-Integer Linear Programs: (i) learned predictions are often infeasible, and (ii) the search radius is fixed in advance. The authors propose HTS-Adapt, which contains: 1) Hybrid Training Strategy (HTS): supervised cross-entropy loss for “stable” variables that rarely change in near-optimal solutions, and contrastive InfoNCE loss for “volatile” variables; 2) Adaptive Search Region Adjustment (Adapt): whenever the predicted partial assignment is infeasible, an Irreducible Infeasible Subsystem (IIS) is computed to identify the culprit variables and their domains are enlarged selectively instead of naïvely expanding the whole trust region.
Experiments on four classic combinatorial benchmarks (MIS, MVC, CA, IP) show that coupling HTS-Adapt with SCIP reduces the average primal gap by more than 50 % compared with SCIP alone and with previous PaS/ConPaS baselines, while predicting a larger fraction of variables without degrading feasibility.

### Strengths
1) Clearly identifies the feasibility and static-radius issues of prior PaS methods.
2) Novel combination of cross-entropy and contrastive learning tailored to variable behaviour, together with targeted expansion of the search region via IIS.
3) Comprehensive empirical evaluation across four datasets; code and hyper-parameters are promised to be released.

### Weaknesses
1) Only SCIP is used as the back-end solver; no comparison with Gurobi or CPLEX to demonstrate solver-agnostic benefits.
2) Contrastive component uses plain InfoNCE; more advanced graph-contrastive losses are not explored.
3) No runtime breakdown (prediction / IIS / solver) is given, so the computational overhead of Adapt is unclear.
4) Missing theoretical analysis, e.g., worst-case number of IIS calls needed to regain feasibility.

### Questions
1) How much latency does the IIS computation introduce at each node, and is there a lightweight approximate IIS routine for large instances?
2) The threshold for “stable” variables is empirical—does it remain valid across problem types, and could it be learned automatically?
3) When scaling to 10^6 variables, can the GNN still fit in GPU memory, and does the IIS routine remain tractable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper improves the predict-and-search framework for solving mixed integer linear programmings (MILP). It mainly addresses two issues, namely prediction accuracy and static search regions that might be missing feasible solutions or unnecessarily large for the search space. To address the first issue, it uses a hybrid training strategy that it separates treatment for stable and unstable variables. Stable variables have identical values across different solutions and are trained with cross-entropy loss for precision. Unstable variable are trained with contrastive loss. This is a hybrid combination of two previous works (PaS and ConPaS). Secondly, it employs adaptive search region adjustment for expanding the search space only when infeasibility arises, guided by irreducible infeasible subsystem (IIS). Experimental results show good performance over four MILP benchmark instances, where the proposed method is better than the baselines in terms of primal solution quality and primal gap, and also faster convergence to good solutions.

### Strengths
1. The paper has two main contributions, one is to use the hybrid loss based on variable types for training, and the other is using adaptive search region adjustment. 
    2. Experimental results show that the new method outperforms several strong baseline such as PaS and ConPaS in terms of objective values and primal gaps.

### Weaknesses
1. The contributions are incremental  It is ok to be incremental if the methods work well with good insights provided either theoretically or empirically. However, the paper mainly describes the methods and shows that it works, but without good explanation or analysis. For example, you could show the precisions of your prediction of the stable/unstable variables compared to PaS and ConPaS
2.  The value additivity of each of the two contributions are not clear. I was looking for a more thorough ablation study (on more MILP problems) to understand the value of each components. 
3. The clarity of Section 4.2 is bad. It hurts the overall clarity of the paper since this part is one of the main contributions. I wasn’t able to understand how C1, C2 are computed and how r_max is chosen. Can you provide an algorithmic description of line 11 in Algorithm 1?

### Questions
1. In table 4, Adapt-only has smaller values for the best k0, k1 and delta than HTS-only, how do you explain this? How are these parameters determined?
2. How sensitive is your method to parameter r_max? How do you determine its value?

### Soundness
3

### Presentation
2

### Contribution
2
