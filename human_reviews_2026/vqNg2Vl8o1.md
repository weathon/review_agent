# Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Model reduction, which aims to learn a simpler model of the original mixed integer linear programming (MILP), can solve large-scale MILP problems much faster. Most existing model reduction methods are based on variable reduction, which predicts a solution value for a subset of variables. From a dual perspective, constraint reduction that transforms a subset of inequality constraints into equalities can also reduce the complexity of MILP, but has been largely ignored. Therefore, this paper proposes a novel constraint-based model reduction approach for MILPs. Constraint-based MILP reduction has two challenges: 1) which inequality constraints are critical such that reducing them can accelerate MILP solving while preserving feasibility, and 2) how to predict these critical constraints efficiently. To identify critical constraints, we label the tight-constraints at the optimal solution as potential critical constraints and design an information theory-guided heuristic rule to select a subset of critical tight-constraints. Theoretical analyses indicate that our heuristic mechanism effectively identify the constraints most instrumental in reducing the solution space and uncertainty. To learn the critical tight-constraints, we propose a multi-modal representation that integrates information from both instance-level and abstract-level MILP formulations. The experimental results show that, compared to the state-of-the-art MILP solvers, our method improves the quality of the solution by over 50\% and reduces the computation time by 17.47\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript proposes a method that learns to predict tight constraints for Mixed-Integer Linear Programs, which are then used to construct a reduced sub-problem. The goal is to enable faster solution search compared to the original MILP. The paper provides both theoretical and empirical evidence supporting the proposed method. However, the theoretical results are somewhat trivial and rely on a strong assumption, while the baselines considered for comparison are insufficient.

### Strengths
1. The manuscript is well-written with a clear and logical flow.
2. The idea of constraint reduction is well-motivated, and it holds potential for practical impact.
3. Empirical results demonstrate the advantages of the proposed method over the considered baselines.

### Weaknesses
1. **Confusing definitions**:  
    - In Definition 1, $C_i$ is defined as a constraint type with unspecified parameters a,b and c. However, the phrase "C_i(x) is satisfied" is unclear. How should the satisfaction of a constraint type be interpreted? Clarifying this would improve understanding.
    - In Definition 2, the term "fixing" appears early in the description of properties of the problem. Isn’t "fixing" typically an action performed during the reduction step? Could this be clarified?
2. **Strong assumption**: The local decoupling assumption is quite strong and its applicability in practice remains unclear. It would be useful to provide statistics or empirical evidence demonstrating the proportion of constraints that satisfy this assumption in realistic scenarios. This would help assess the gap between theory and practice.
3. **The main Theorem 4 is trivial and meaningless**: The main result in Theorem 4, which suggests that fixing tight constraints leads to a sub-problem containing the optimal solution, is trivial and lacks meaningful insight for the learning task, considering there is no guarantee that the model will accurately predict all tight constraints.
4. **Inadequate baselines**: The authors mention prior works that also learn tight constraints for model reduction, yet fail to compare their method against these methods. I recommend incorporating at least one of these works as a baseline to enhance the comparison and demonstrate the advantages of the proposed method.

### Questions
1. The tight constraints are defined based on the optimal solution. What happens if there are multiple optimal solutions? How does the proposed approach handle such cases? Also, what if obtaining the optimal solution is computationally expensive?
2. An IP dataset is used in both of the considered learning baselines but is excluded from the experiments in this work. Could the authors clarify the reasoning behind this exclusion?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel constraint-based model reduction framework for Mixed-Integer Linear Programs (MILPs). The core idea is to identify and fix a subset of "critical" tight constraints to reduce the problem's complexity, complementing the more common variable-fixing approaches. Experiments on standard benchmarks and a real-world dataset show improvements in solution quality and solving time.

### Strengths
1. The paper's focus on constraint reduction is a valuable direction. The idea that not all tight constraints are equally valuable for acceleration is intuitive, and providing a data-driven method to identify them is a meaningful contribution to the ML-for-optimization community.
2. The proposed GNN architecture that incorporates both the instance-specific bipartite graph and an abstract graph with textual embeddings is a sophisticated and well-motivated approach. Fusing information from the problem category level is a sensible way to improve generalization and prediction accuracy for constraints.

### Weaknesses
1. While the information-theoretic motivation for the TCP heuristic is appealing, its practical implementation relies heavily on the "Local Decoupling Assumption" (Assumption 1). This assumption is a significant simplification, as constraints in MILPs are inherently coupled. The paper acknowledges that this is a heuristic, but the leap from the theoretical to its application for selecting constraints within a specific instance requires further justification. How does the pre-computed ρ for a constraint type reliably indicate the criticality of a particular constraint instance whose variables interact with many other constraints? A more detailed discussion or empirical validation of this link's robustness would strengthen the method's foundation.
2. ​Baseline comparison is lacking in details, since Gurobi already has a lot of presolving techniques used to reduce variables and constraints. A fairer comparison needs to turn these presolving methods off.

### Questions
1. What is the setting used in the experiments for the MILP Solver?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a novel framework for accelerating Mixed Integer Linear Programming (MILP) solving by introducing the concept of Critical Tight Constraints (CTCs) — a subset of constraints that are most influential in determining the final optimal solution. The key idea is that by identifying and reducing redundant or weakly active constraints, one can substantially decrease solver time without compromising solution accuracy.

To efficiently identify CTCs, the authors design a novel framework that combines multi-dimensional MILP representation learning and constraint-level embeddings with pretrained language model (PLM) augmentation. This hybrid modeling enables the system to capture both structural and semantic relationships among constraints. Once CTCs are identified, the solver performs targeted reduction to simplify the MILP instance before solving. Extensive experiments conducted on large-scale MILP benchmarks, using Gurobi as the backend solver, demonstrate significant runtime reductions while maintaining high solution fidelity.

### Strengths
1. The notion of Critical Tight Constraints (CTCs) represents a fresh and impactful contribution to the MILP optimization literature. Unlike prior works focusing on branching, cut selection, or presolving heuristics, this paper introduces a new perspective centered on constraint-level importance.

2. Constraint reduction directly improves solver efficiency, making the method immediately relevant to industrial-scale applications such as logistics, scheduling, and network design.

3. The proposed neural framework integrates multiple components — structural MILP encoding, PLM-guided feature enhancement, and supervised CTC prediction — in a cohesive and well-justified pipeline. 

4. The experiments are comprehensive and clearly demonstrate performance improvements over baseline solvers. The consistent gains on Gurobi are particularly convincing, showing both scalability and generalization.

5. The paper is well-written and logically organized. The motivation for focusing on constraint reduction is clearly articulated and supported by both intuition and empirical evidence.

### Weaknesses
1. While the intuition behind identifying CTCs is compelling, the paper would benefit from a theoretical discussion or empirical analysis showing why certain constraints consistently dominate others. Some sensitivity or ablation studies on the constraint structure could strengthen the justification.

2. The integration of PLM embeddings is an interesting choice, but it would help to quantify their actual contribution via ablation — e.g., selection of different PLM. It would be helpful to provide additional implementation details on how CTC reduction interacts with Gurobi and whether the approach is solver-agnostic.

3. The paper is quite dense, covering multiple modeling, training, and experimental components. While this reflects solid effort, the narrative could benefit from streamlining or reorganization to emphasize the core technical contributions better.

### Questions
1. It would be helpful to include a sensitivity analysis of the hyperparameter Δc, which controls the constraint search or reduction threshold. This experiment could clarify how robust the proposed method is across different datasets and problem scales, and whether performance degrades significantly under suboptimal Δc values.

2. The current experiments are convincing, but primarily conducted on moderate-scale benchmarks. Could the authors provide additional results or analysis on large-scale MILP instances with more complex and heterogeneous constraints? Such evaluation would strengthen the claim that the proposed reduction framework generalizes well beyond the tested benchmarks.

3. The proposed method identifies Critical Tight Constraints through a neural model. It would be interesting to see an analysis of what kinds of constraints are typically recognized as critical—e.g., are they associated with certain structural or semantic patterns? Such insights could improve the interpretability and trustworthiness of the model’s predictions.

4. Since modern MILP solvers (like Gurobi) already include powerful presolve and constraint simplification modules, could the authors clarify how their learned reduction interacts with, or complements, the built-in presolver? A discussion or ablation comparing the two would help isolate the true contribution of the learned component.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes predicting tight constraints to perform constraint reduction in MILP problems, effectively accelerating MILP solving. To better predict tight constraints, in addition to the existing bipartite graph representation, the paper introduces an abstract-level representation of MILP and a TCP module.

### Strengths
1. Every component proposed or used in the paper is well-motivated.
2. The definition and use of the fixed constraint strength $\rho$ are quite interesting.

### Weaknesses
1. Introducing a pretrained language model and the abstract-level GNN will bring additional computational overhead.
2. I have doubts about Theorem 4. If an error occurs during the process of fixing constraints, would the resulting problem become infeasible, or would its optimal value differ from that of the original problem?
3. In Table 8, I noticed that the hyperparameters used for different problem types vary significantly. This implies that for a new type of problem, the proposed method would require manual hyperparameter tuning. I would like to know how sensitive the method’s performance is to these hyperparameters.

### Questions
1. When the lower or upper bounds of constraints in MILP problems are no longer 0 or 1, how does the proposed constraint reduction method handle such cases, and how does this affect performance in practice?
2. Could the paper provide experimental results on more complex real-world problems (such as those from MIPLIB)? For a real-world problem of an unknown type, how is its text description obtained?
3. How significant is the performance impact when using different pretrained language models?

### Soundness
3

### Presentation
3

### Contribution
3
