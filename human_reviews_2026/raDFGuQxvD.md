# Towards Efficient Constraint Handling in Neural Solvers for Routing Problems

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Neural solvers have achieved impressive progress in addressing simple routing problems, particularly excelling in computational efficiency. However, their advantages under complex constraints remain nascent, for which current constraint-handling schemes via feasibility masking or implicit feasibility awareness can be inefficient or inapplicable for hard constraints. In this paper, we present Construct-and-Refine (CaR), the first general and efficient constraint-handling framework for neural routing solvers based on explicit learning-based feasibility refinement. Unlike prior construction-search hybrids that target reducing optimality gaps through heavy improvements yet still struggle with hard constraints, CaR achieves efficient constraint handling by designing a joint training framework that guides the construction module to generate diverse and high-quality solutions well-suited for a lightweight improvement process, e.g., 10 steps versus 5k steps in prior work. Moreover, CaR presents the first use of construction-improvement-shared representation, enabling potential knowledge sharing across paradigms by unifying the encoder, especially in more complex constrained scenarios. We evaluate CaR on typical hard routing constraints to showcase its broader applicability. Results demonstrate that CaR achieves superior feasibility, solution quality, and efficiency compared to both classical and neural state-of-the-art solvers. Our code, pre-trained models, and datasets are available at: https://github.com/jieyibi/CaR-constraint.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Construct-and-Refine (CaR), a framework designed to improve constraint satisfaction in neural combinatorial optimization solvers. Existing constraint-handling approaches can be broadly categorized into feasibility masking and feasibility awareness, but both tend to struggle with hard-to-satisfy constraints.
The proposed framework introduces two decoders: one for constructing an initial feasible solution and another for refining it iteratively. The authors evaluate CaR on TSPTW and CVRPLTW, demonstrating that it achieves higher constraint satisfaction rates for difficult instances and better objective values for larger-scale problems compared to existing methods.

### Strengths
- The proposed framework achieves superior constraint satisfaction and objective performance across multiple constraint types, demonstrating that the construct-and-refine approach can lead to more effective constraint handling.

- The paper includes a well-designed ablation study that helps clarify which components of the framework contribute most to performance improvements.

### Weaknesses
- The range of constraints for which the proposed framework is effective remains unclear. For example, constraints such as time-window or route-length can often be relaxed smoothly, but it is not evident whether CaR is equally effective for non-continuous or discrete constraints, such as precedence or ordering constraints.

### Questions
- What types of constraints are better handled by existing constraint-handling methods, and which types particularly benefit from the proposed CaR framework?

- What characteristics must existing methods, such as POMO, have in order to integrate with CaR?

- In what situations does the proposed method fail to find a constraint-satisfying solution? Are there any common patterns or characteristics in such cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a neural combinatorial optimization (NCO) solver for vehicle routing problems (VRPs) with hard constraints.
Existing NCO approaches can be categorized into (1) construction solvers that sequentially build solutions, (2) improvement solvers that iteratively refine them, and (3) hybrid methods that combine both.
However, these approaches struggle to effectively handle hard constraints such as time windows or multiple interdependent resource limits.
To address this issue, the authors introduce a new framework called Construct-and-Refine (CaR), which explicitly refine a constructed solution to recover feasibility.
CaR jointly trains construction and refinement modules with a shared encoder, enabling fast and effective recovery of feasibility while maintaining solution quality.
Experiments on several constrained VRP variants (TSPTW, CVRPBLTW) demonstrate that CaR achieves superior feasibility, optimality, and computational efficiency compared to both classical heuristics and recent neural baselines.

### Strengths
- This paper is well-motivated. It addresses a fundamental challenge in neural combinatorial optimization (NCO) — effectively satisfying hard constraints such as time windows constraints. This limitation has been a core weakness of existing NCO solvers, and this paper clearly explains the importance of effectively handling such constraints.
- The proposed approach is conceptually sound. Introducing an explicit feasibility refinement module to recover infeasible solutions is a natural and intuitive. 
- Though the proposed approach is simple, it works well in practice. Experiments demonstrate that it consistently finds feasible solutions more reliably than other NCO solvers and classical heuristics on hard-constrained routing problems such as TSPTW and CVRPBLTW, while also achieving higher solution quality. The experimental evaluation is comprehensive, covering multiple benchmarks, ablation studies, and generalization tests that convincingly support the method’s effectiveness.

### Weaknesses
While the paper’s empirical contributions are strong, there is no theoretical discussion of why the joint learning of construction and refinement works well. In particular, questions regarding learnability, convergence, or generalization bound of the proposed method remain open. Formal analysis would strengthen the work’s long-term impact.

### Questions
- The first question is how CaR could be analyzed from a theoretical standpoint (possibly as a future work). Possible directions include learnability, convergence, or generalization bound.
- The second question is about applicability of the proposed method. It appears general and could be extended to other structured optimization domains where constraint handling is non-trivial, such as scheduling problems or SAT.

### Soundness
3

### Presentation
2

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
This paper addresses the challenge of applying neural combinatorial optimization (NCO) solvers to VRPs with complex constraints. The authors introduce "Construct-and-Refine" (CaR), a hybrid framework that jointly trains a construction and refinement module. The construction module is trained to produce diverse and high-quality solutions. These are then fed to a refinement module that operates for a short horizon to rapidly find feasible, high-quality solutions. Experiments on hard-constrained VRPs (TSPTW, CVRPBLTW) and others (CVRP, TSPDL) show that CaR can achieve state-of-the-art feasibility and solution quality.

### Strengths
(1) The paper is well-written and clearly structured. 

(2) The paper tackles a critical problem. The failure of NCO solvers to handle complex constraints is arguably the barrier to their practical adoption. 

(3) The experimental evaluation is comprehensive, demonstrating strong results on different VRPs with hard constraints.

### Weaknesses
(1) This paper proposes CaR to efficiently handle complex constraints in VRPs and subsequently produce high-quality solutions. However, the core contribution appears to combine PIP and existing hybrid methods (LCP and NCS) for solving hard-constrained VRPs.

(2) The idea of combining construction and improvement is not new. NCS also employs a joint training paradigm with a shared component. The distinction of the joint training paradigm is not strong. 

(3) What is the fundamental difference between the approach used in this paper for handling constraints and the essence of PIP?

(4) The ablation study of the effectiveness of the joint training is unconvincing. A much stronger ablation would be jointly training PIP + NeuOpt, but with separate encoders and without the $L_{SL}$ feedback loss, and compare with (1) add the $L_{SL}$ loss, (2) add the  shared encoder

(5) The arguments in this paper "In CVRPBLTW, for instance, strict masking filters out more than 60% of nodes, severely limiting the search space (Figure 6) and hindering RL convergence." Why does limiting the model search space hinder RL convergence?

### Questions
(1) Can the authors more clearly distinguish the conceptual difference between CaR's joint training and that of NCS? Both seem to jointly train a constructor and improver with a shared component.

(2) Could the authors provide results for CaR compared against NeuOpt-GIRE (or another SOTA improvement solver)  that is re-trained to optimize its performance for a short rollout?

(3) Why was the ablation in Table 5 compared against a "Random" constructor? Could the authors provide results for CaR compared with a jointly trained constructor and improver with a separate encoder and the same model, but not use the $L_{SL}$ loss during training?

(4) How sensitive is the framework to the choice of p (top-p solutions)?

### Soundness
2

### Presentation
2

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
This paper presents Construct-and-Refine (CaR), a neural framework for efficiently handling hard constraints in vehicle routing problems (VRPs). By integrating construction and refinement paradigms via joint training and shared representations, CaR achieves significant improvements in feasibility, solution quality, and computational efficiency. The method demonstrates strong performance across diverse constrained VRP variants, outperforming both classical and neural baselines while maintaining minimal inference overhead.

### Strengths
1.	The paper is highly mature, with extensive experiments covering multiple VRP variants (TSPTW, CVRPBLTW, CVRP, TSPDL), scales, and constraint settings. The implementation details are thorough, including hyperparameters, baseline adaptations, and reproducibility measures.

2.	The paper provides detailed ablations validating key components, e.g., joint training, diversity loss, shared representations, and their individual impacts on performance. These analyses offer valuable insights into understanding the framework.

### Weaknesses
The technical contribution of this paper is limited. Although it addresses important constrained problems, CaR primarily adapts the existing construct-and-refine framework (e.g., NCS) to constrained domains. The shared representation and lightweight refinement design are incremental and do not introduce fundamentally novel ideas.

### Questions
1.	How does CaR achieve effective refinement in just 10 steps, while prior learn-to-improve methods require thousands? Is this solely due to the short-horizon design, or does the joint training framework play a critical role?

2.	Why are TSPDL results omitted from the main experiments? Additionally, Appendix results for TSPDL100 are missing. Could the authors clarify this gap?

### Soundness
3

### Presentation
3

### Contribution
2
