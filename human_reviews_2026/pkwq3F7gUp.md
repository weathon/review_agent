# MILPnet: A Multi-Scale Architecture with Geometric Feature Sequence Representations for Advancing MILP Problems

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
We propose MILPnet, a multi-scale hybrid attention framework that models Mixed Integer Linear Programming (MILP) problems as geometric sequences rather than graphs. This approach directly addresses the challenge of Foldable MILP instances, a class of problems that graph-based models, specifically Graph Neural Networks (GNNs), fail to distinguish due to expressiveness limits imposed by the Weisfeiler-Lehman test. By representing MILPs through sequences of constraint and objective features, MILPnet captures both local and global geometric structure using a theoretically grounded multi-scale attention mechanism. We theoretically prove that MILPnet can approximate feasibility, optimal objective value, and optimal solution mappings over a measurable topological space with arbitrarily small error. Empirically, MILPnet outperforms graph-based methods in feasibility prediction accuracy and convergence speed on Foldable MILPs, while using significantly fewer parameters. It also generalizes effectively among problem scales and demonstrates strong performance on real-world MILP benchmarks when integrated into an end-to-end solver pipeline. Our code is available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes MILPnet, a multi-scale hybrid-attention model that represents MILP instances as sequences of geometric features (constraints/objective), rather than graphs. The motivation is that message-passing GNNs are limited by 1-WL expressivity and can fail on families of “Foldable MILPs” that are indistinguishable under WL-type symmetries. MILPnet builds multi-scale views of these sequences and applies cross-scale attention to capture local and global structure. The authors prove universal approximation properties (feasibility, optimal value, solution mapping) on a measurable topological space and report large empirical gains over graph-based methods on Foldable MILPs, with fewer parameters and better convergence; they also show generalization across scales and integration into a solver pipeline.

### Strengths
1. Compelling motivation: explicit WL-type failure mode for GNNs on MILPs; sequence modeling avoids that bottleneck. 
2. Multi-scale design: principled way to capture local/global geometric structure in constraint/objective sequences. 
3. Theoretical guarantees beyond typical empirical claims. 
4. Practical angle: improved convergence and smaller models; integration into solver workflows.

### Weaknesses
1. The comparison is not comprehensive. Please compare to modern ML-augmented B&B (e.g., branching/cuts/node-selection learned policies) and clarify where MILPnet plugs into the B&B stack (pre-solve heuristic? branching score? feasibility classifier?). 
2. While the paper compares against standard GNN-based MILP representation methods, to convincingly establish state-of-the-art performance, it should also include or discuss recent baselines such as RL-based MILP solvers (e.g., RL-MILP Solver 2024), partial-assignment reduction frameworks (e.g., ConPaS 2023/24), solver-configuration learning methods (e.g., Separator-Learning 2023), and model-reduction methods (e.g., Learning Model Reduction 2024). Including these would strengthen the empirical claim of the universality of your sequence-based representation.
3. More ablation studies are required. Provide ablations on (i) feature choices per token, (ii) number and granularity of scales, (iii) attention depth vs. performance, and (iv) effect on different MILP families (set-covering, capacitated facility location, routing, scheduling).
4. Generalization tests are needed to demonstrate the practical potential for the work. Beyond Foldable MILPs, evaluate on large real-world sets and report time-to-first-feasible and optimality gap closed vs. SCIP/CPLEX.

### Questions
1. Where exactly in a production solver would you place MILPnet (branching score predictor, primal heuristic, node ranking, cut selection)? Please quantify wall-clock speedups and gap closure.
2. Is MILPnet permutation-invariant to constraint/variable orderings? If not, what data augmentation or canonicalization is used?
3. Can the theory be tightened to show sample complexity or stability under small perturbations of coefficients?
4. How sensitive is performance to scale hyper-parameters (window sizes, strides) and to sequence length (very large models)?
5. Could you hybridize with recent high-capacity GNNs (e.g., higher-order/WL-beyond) to reap graph-topology benefits while preserving sequence expressivity?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MILPnet, a multi-scale hybrid attention framework that departs from traditional graph-based models to address the Mixed Integer Linear Programming problem. Instead of modeling MILPs as bipartite graphs, MILPnet represents each MILP instance as a geometric feature sequence. The paper proves that MILPnet can approximate MILP feasibility, optimal objective value, and optimal solution mappings with arbitrarily small error in a measurable topological space, and empirical results demonstrate that it outperforms graph-based methods by multiple orders of magnitude in feasibility prediction accuracy and convergence speed on Foldable MILPs.

### Strengths
1) By modeling MILPs as geometric feature sequences, it overcomes the Weisfeiler-Lehman test’s expressive limit, enabling effective distinction of Foldable MILP instances that graph-based models fail to differentiate. 

2) It provides strict mathematical proofs that guarantee arbitrary-precision approximation of MILP feasibility, optimal objective value, and optimal solution mappings in a measurable topological space, ensuring reliability. 

3) Empirically, it outperforms graph-based baselines by multiple orders of magnitude in feasibility prediction accuracy and convergence speed on Foldable MILPs.

### Weaknesses
1. The maximum window size in its multi-scale attention mechanism does not follow a monotonic pattern—smaller windows accelerate convergence on simple instances (e.g., FOLD(20,6)) but degrade performance on complex ones (e.g., FOLD(100,20)), requiring careful tuning for different problem scales. 

2) The paper provides limited empirical validation on extremely large-scale MILP instances, leaving uncertainty about its scalability in industrial-grade scenarios with massive constraints/variables. 

3) Although padding ensures topological equivalence, the framework’s performance on MILP instances with drastically varying constraint counts lacks in-depth analysis, raising questions about its stability across highly heterogeneous problem sizes.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MILPnet, a multi-scale hybrid attention model that represents MILP problems as geometric sequences instead of graphs to address GNN limitations on "Foldable MILPs." It encodes constraints, objectives, and variables into sequences, uses shifted-window attention for local-global feature capture, and proves approximation capabilities for feasibility, optimal values, and solutions.

### Strengths
The paper shifts from graph-based to sequence-based representations for MILPs, addressing GNN limits in expressiveness for Foldable instances. It provides proofs of universal approximation results. The numerical results report improvements in accuracy and efficiency on Foldable instances.

### Weaknesses
1. In Theorem 2-4, only a finite dataset $D$ is considered. Since the dataset is finite, I do not think the notation $P$ is needed and the $\epsilon$ in equations (9) and (10) is not needed. The authors are basically borrowing proof ideas from the work of Chen et al but the proof presentation and notation have much room for improvement.
2. The sequential representation breaks the permutation-invariance property of MILP.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
