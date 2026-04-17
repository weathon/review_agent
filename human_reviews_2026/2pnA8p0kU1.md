# ML-Guided Primal Heuristics for Mixed Binary Quadratic Programs

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Mixed Binary Quadratic Programs (MBQPs) are an important and complex set of problems in combinatorial optimization. As solving large-scale combinatorial optimization problems is challenging, primal heuristics have been developed to quickly identify high-quality solutions within a short amount of time. Recently, a growing body of research has also used machine learning to accelerate solution methods for challenging combinatorial optimization problems. Despite the increasing popularity of these ML-guided methods, a large body of work has focused on Mixed-Integer Linear Programs (MILPs). 
MBQPs are challenging to solve due to the combinatorial complexity coupled with nonlinearities. This work proposes ML-guided primal heuristics for Mixed Binary Quadratic Programs (MBQPs) by adapting and extending existing work on ML-guided MILP solution prediction to MBQPs. We introduce a new neural network architecture for MBQP solution prediction and a new training data collection procedure. Moreover, we extend existing loss functions in solution prediction and propose to combine contrastive weighted cross-entropy losses. We evaluate the methods on standard and real-world MBQP benchmarks and show that the developed ML-guided methods significantly outperform existing primal heuristics and state-of-the-art solvers. Furthermore, models trained with our proposed extension with combined losses outperform other ML-based methods adapted from MILPs and improve generalization in cross-regional inference on a real-world wind farm layout optimization problem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ML-guided primal heuristics for mixed binary quadratic programs by encoding each instance as a tripartite graph  and using Graph attention to predict assignments to binary variables. Training combines cross-entropy and contrastive losses with a Randomized Relax-Search data generator; inference performs Neural Diving by fixing high-confidence variables before solving the reduced problem.

### Strengths
1. The Randomized Relax-Search framework for solution generation is novel and practical. It could enable efficient sampling of large, diverse candidate sets for challenging MBQPs and yielding richer supervision signals.
2. The empirical performance against baselines is relatively strong, indicating that the proposed pipeline translates into effective primal heuristics.

### Weaknesses
Although the performance of the proposed framework is promising, the significance of this work remains questionable from the following aspects:
1. Excluding the data-generation component, the method is more like a straightforward combination of [1], [2], and [3]. It would be nice to see the author explaining about what new capability or insight emerges from this integration beyond the sum of parts?
2. Since tripartite QP encodings is not new (e.g., hypernode constructions in [4]), a short comparison would help: what is distinct in the author's design, and how do those choices benefit MBQPs?
3. Baselines are limited to conventional algorithms. Methods like [1] can be adapted to MBQPs given sufficient data. Is it possible to include such learning-based baselines or justify their exclusion with a discussion of expected performance.

[1] Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).
[2] Han, Qingyu, et al. "A gnn-guided predict-and-search framework for mixed-integer linear programming." arXiv preprint arXiv:2302.05636 (2023).
[3] Huang, Taoan, et al. "Contrastive predict-and-search for mixed integer linear programs." Forty-first International Conference on Machine Learning. 2024.
[4] Wu, Chenyang, et al. "On representing convex quadratically constrained quadratic programs via graph neural networks." arXiv preprint arXiv:2411.13805 (2024).

### Questions
Besides of those mentioned in the weakness section, there are some other questions.
1. Why are commercial solvers (e.g., Gurobi) excluded from the baseline? SCIP is often weaker on nonconvex MIQPs; including Gurobi (and others) would provide a more balanced baseline set.
2. How many solutions were collected by the Randomized Relax-Search procedure? Is it possible to report the total counts to contextualize the dataset sizes and training signal diversity.

### Soundness
3

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
This work applies a combination of established Learning-to-Optimize (L2O) techniques to the MIQP domain. While the integration is competently executed, the paper would benefit from a clearer articulation of its core novel contribution, as the current approach appears to be a straightforward application of existing concepts rather than a significant algorithmic innovation.

### Strengths
This paper is fairly easy to follow, it combines the state-of-the-art L2O techniques.

### Weaknesses
- While the paper competently integrates several existing techniques—such as graph representations for optimization problems, distribution learning, and contrastive learning—the core novelty of the overall framework appears limited. The application seems to be a straightforward combination of these methods without significant customization for the MIQP structure itself. This concern is compounded by the experimental design: the absence of a state-of-the-art commercial solver like Gurobi as a baseline makes it difficult to assess the method's practical utility. 

- Furthermore, benchmarking on a more challenging and diverse dataset, such as QPLIB [1], would provide a more rigorous validation of the approach's robustness and generalizability.

[1] https://qplib.zib.de

### Questions
- Regarding the graph representation, did the authors consider the alternative of creating direct edges between variable nodes instead of introducing quadratic nodes? This simpler representation, which uses significantly fewer nodes, has been successfully applied to Mixed-Integer Quadratic Programs (MIQPs) in prior work [1].

[1] Chen, Ziang, Xiaohan Chen, Jialin Liu, Xinshang Wang, and Wotao Yin. "Expressive Power of Graph Neural Networks for (Mixed-Integer) Quadratic Programs." In Forty-second International Conference on Machine Learning.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops ML-guided primal heuristics for MBQPs by representing an instance as a tripartite graph (constraints, variables, quadratic-term nodes), processing it with a four-round GAT to predict a binary assignment, and solving a reduced sub-MBQP via Neural Diving at inference. Besides, it introduces Randomized Relax-Search to collect diverse positive/negative solutions for training, and combines CL with a CE/WCE term (CL+WCE). 

In my opinion, two biggest concerns with L2O methods remain: (1) the expensive training cost for supervised learning, and (2) the lack of feasibility guarantees. In this paper, I do not see convincing evidence that either issue is meaningfully discussed. These gaps materially limit the practical impact of the approach.

### Strengths
The tripartite representation and GAT pipeline explicitly model quadratic couplings through Q–V edges and C–V interactions, which is a sensible structural upgrade over MILP-only predictors. The theoretical observation behind Proposition 1 motivates a combined CL+WCE loss that empirically dominates pure CL/WCE across benchmarks and improves feasibility and solution quality in short time budgets. The randomized relax-search procedure yields more and better positive samples than simply logging solver trajectories, strengthening the supervised signal.

### Weaknesses
1. The experimental deck emphasizes SCIP and classical heuristics under a 60s limit, but it does not benchmark against tuned commercial MBQP solvers, e.g., CPLEX or Gurobi, or MIQP linearization pipelines, lacking fairness. 
2. The training data collection is computationally expensive (11,000s per instance), and the paper does not present a cost–benefit accounting that includes training and maintenance when compared at matched accuracy/latency. 
3. Feasibility is not guaranteed and controlled by the fixing ratio 𝑝. Although sensitivity is reported, failure modes for dense/near-infeasible cases are not charted. 
4. Finally, scalability beyond n=1000 and different quadratic densities/condition numbers lacks a systematic Pareto analysis.

### Questions
1. Can you provide an ablation showing how often the model’s fixed assignments are infeasible before refinement, and how much each mechanism reduces infeasibility across densities and near-infeasible regimes? Also, if the above ratio is very high, what would you do to improve it?
2. Any ideas to address the problem of expensive training data costs? BTW, given such high data and training costs, is supervised learning truly a suitable mode for L2O, especially when dealing with practical large-scale industrial problems?

### Soundness
2

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
2

### Summary
This work proposes an ML-guided primal heuristic for Mixed Binary Quadratic Programs (MBQPs), based on solution prediction and extended from ML-guided methods for Mixed-Integer Linear Programs (MILPs). The network takes an MBQP as input, represented as a tripartite graph, and outputs predicted solutions for the binary decision variables. Compared to MILPs, MBQPs face the combined challenges of combinatorial complexity and nonlinearities. To address these, the authors:
1. propose a neural network design that includes a tripartite graph representation of MBQP instances, a custom feature set, and a graph attention network architecture;
2. introduce a loss function that combines weighted cross-entropy with a contrastive learning term;
3. develop a data collection procedure, Randomized Relax-Search, which generates high-quality solutions as ground-truth training data for large-scale MBQPs.

### Strengths
Overall, the writing is clear, particularly the introduction of the proposed methodology.

### Weaknesses
The formatting needs improvement. For example, some citations (e.g., “RENS Berthold (2014), Undercover Berthold & Gleixner (2014), and Relax-Search Huang et al. (2025)” in Section 2.1) are missing brackets, and some citations (e.g., “Nair et al. (Nair et al., 2020) and Han et al. (Han et al., 2023)” in Section 2.2) are redundant.

### Questions
1. The authors state that the adapted CL-based method performs worse in Table 2 because it fails to produce feasible solutions for many instances. However, why does this method fail to produce feasible solutions for QMKP, CQKP, and WFLOP but perform well on CBQP? Why do the adapted WCE and extended CE+WCE methods can produce feasible solutions?
2. The experimental comparison with other ML-guided methods is lacking. Are there any ML-guided methods for MBQPs, or even specifically for the chosen problems?

### Soundness
3

### Presentation
2

### Contribution
2
