# RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Primal heuristics play a crucial role in quickly finding feasible solutions for NP-hard integer linear programming (ILP). Although $\textit{end-to-end learning}$-based primal heuristics (E2EPH) have recently been proposed, they are typically unable to independently generate feasible solutions. To address this challenge, we propose RL-SPH, a novel reinforcement learning-based start primal heuristic capable of independently generating feasible solutions, even for ILP involving non-binary integers. Empirically, RL-SPH rapidly obtains high-quality feasible solutions with a 100% feasibility rate, achieving on average a 44× lower primal gap and a 2.3× lower primal integral compared to existing start primal heuristics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RL-SPH, a two-phase reinforcement-learning heuristic that aims to produce feasible solutions for ILPs without relying on solver-side repair. The method separates the search into a feasibility-first stage and a quality-improvement stage, uses a Transformer-GNN backbone to encode problem structure and current solution state, and restricts actions to small integer moves over a selected subset of variables. Experiments cover several classic ILP families and one non-binary integer dataset (NBI). Reported results indicate near-perfect feasibility rates and improvements in primal gap/integral, and the method can be used standalone or as a front-end to SCIP/Gurobi. Overall, the direction is interesting and relevant: pushing learning methods to reliably generate feasible ILP solutions is practically important. However, the experimental comparisons against recent learning-based baselines are limited, the evaluation on general integer variables is too narrow, and the paper omits crucial training curves and wall-clock analyses needed to judge compute cost and convergence behavior.

### Strengths
1. The paper provides a concrete demonstration that an RL policy can be trained end-to-end to produce feasible ILP solutions without solver repair, which is a nontrivial and interesting.

1. Unlike many learning-based works that focus only on binary variables, the method is designed for general integer domains and includes a non-binary benchmark.

### Weaknesses
1. The considered learning-based baselines are too limited. The paper mainly compares against PAS but does not include more recent learning approaches such as ConPS [1] and methods that explicitly generate feasible solutions such as [2] and [3]. Notably, [3] also trains in an unsupervised manner and reports high feasibility. Specifically, [3] adopts a gradient-based method, and the paper should discuss and compare the difference between RL-based and gradient-based methods.

1. The claim of handling general integer variables is under-evaluated. Only a single non-binary dataset (NBI) is used to support this point. More diverse general-integer tasks, or MIPLIB instances, are necessary to support this claim.

1. There are no training curves and no direct comparison of training compute with that of other learning baselines. Likewise, the paper should report test-time solving curves and wall-clock primal integral (PI) under identical hardware and thread settings.

1. The paper’s metric definitions and timing protocols are under-specified. In particular, PI depends on how time is discretized; if “timestep” refers to MDP steps for RL but wall-clock for non-RL baselines, the comparison is not sound.

[1] Huang et al., Contrastive Predict-and-Search for Mixed Integer Linear Programs. ICML 2024.

[2] Zeng et al., Effective Generation of Feasible Solutions for Integer Programming via Guided Diffusion. SIGKDD 2024.

[3] Geng et al., Differentiable Integer Linear Programming. ICLR 2025.

### Questions
1. Why choose Actor-Critic rather than more usually used algorithms such as SAC or PPO?

1. How exactly is PI computed in your experiments? Is “timestep” defined as MDP steps or wall-clock time? If it is MDP steps for RL but something else for baselines, the metric is not comparable. If it is wall-clock, please clarify the time interval.

1. In Algorithm 1, is the search routine used only during training to collect trajectories, or is the same search policy executed during testing as the deployed heuristic?

1. At test time, do you iterate the MDP for a fixed horizon, or do you use a termination rule? Please detail the stopping criteria, and how it interacts with solver calls when RL-SPH is used as a front-end.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the challenging problem of learning-based primal heuristics for integer linear programming (ILP), where existing end-to-end approaches often fail to produce feasible solutions independently. The authors propose RL-SPH, a reinforcement learning–based start primal heuristic that can generate feasible solutions—even for non-binary ILPs—with a 100% feasibility rate. Empirically, RL-SPH achieves significantly better performance than existing heuristics, demonstrating its practical promise for rapidly obtaining high-quality feasible solutions.

### Strengths
- The work tackles an important and underexplored problem: using learning-based methods to construct initial feasible solutions for general ILPs—a task known to be highly challenging but with substantial practical impact.  
- The related work section is thorough and, to the best of my knowledge, covers nearly all recent relevant literature, which is commendable.  
- The proposed method is conceptually simple yet highly effective. While the technical novelty may not be groundbreaking, the contribution lies in successfully navigating a difficult and sparsely explored direction, supported by key insights and theoretical justification.  
- The experimental evaluation is comprehensive, including comparisons with classical heuristics, cross-problem generalization, and compatibility across different solvers, which collectively strengthen the empirical claims.  
- The appendix is well-structured and contains valuable supplementary material and extended experiments. (That said, due to time constraints, I was unable to review it in full; a brief summary of appendix content in the main text would be helpful.)

### Weaknesses
- Several technical details are insufficiently explained, particularly in Section 3.2:  
  + Figure 4 is unclear; it would help to explicitly illustrate the inputs and outputs of the actor and critic in each phase, and clarify which components constitute the Transformer’s sequence input.  
  + The integration of structural information (e.g., from the constraint matrix) with variable features needs elaboration—how is this multi-dimensional structure encoded and concatenated?  
  + It is ambiguous what exactly serves as the Transformer input: are variable nodes treated as a sequence (as in standard Graph Transformers)? Are phase indicators, objective coefficients, or other global features injected, and if so, in what form (e.g., as tokens, positional encodings, or side inputs)?  
  + The ablation study shows the Transformer’s benefit, but the source of this advantage—e.g., long-range dependency modeling, attention over constraints, etc.—is not analyzed. Clarifying the key mechanism would greatly aid future work.  
- The paper’s overall clarity and organization could be significantly improved. For instance, Section 3.3 appears to describe a component that logically precedes Figure 2, yet is placed afterward. The subdivision of Section 3.1 is overly granular, and Section 3.1.4 (training) is oddly positioned before Section 3.2. The text and figures also feel crowded, likely due to page limits; the authors should consider moving less critical details to the appendix to prioritize core content in the main body.  
- The reward design in Section 3.1.3 is complex and central to the RL training, yet lacks motivation. A clear rationale—e.g., design principles, trade-offs considered, or inspiration from optimization theory—would help readers better appreciate the method’s novelty.  
- The feasibility-aware search strategy in Section 3.3 appears somewhat heuristic. Given its empirical importance, the authors should clarify whether it draws from classical methods or is newly designed, what its computational cost is, whether it is a core contribution, and whether there is room for further optimization.

### Questions
- Could the authors clarify the technical details raised in the first weakness point, especially regarding the Transformer input construction and feature integration?  
- The method achieves 100% feasibility even on non-binary ILPs, which is impressive given the vast search space. Were any special techniques employed to handle large or unbounded integer domains?  
- The overall framework—iteratively setting and updating variable values—is relatively simple. In the authors’ view, what is the key enabler of its success: the reward design, the Transformer architecture, the feasibility-aware search strategy, or their combination?  
- In Section 4.3, what happens if RL-SPH fails to find a feasible solution within 5 seconds? Is there a fallback mechanism, or is the run simply counted as a failure?  
- The results on selected MIPLIB instances are compelling. Would it be possible to report performance on a broader set of instances to further strengthen the empirical claims?  
- What do the authors see as the main bottleneck limiting scalability to very large-scale ILPs? Is it the Transformer architecture (e.g., quadratic attention), the RL training overhead, or something else?

**Overall, I consider this a high-quality contribution. While it may not introduce a paradigm-shifting algorithm, it demonstrates a viable and effective learning-based path in a notoriously difficult area—feasible solution generation for general ILPs—which I believe constitutes a meaningful advance. I will actively participate in the rebuttal discussion and look forward to a more comprehensive and objective evaluation based on the authors’ responses.**

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
3

### Summary
The paper introduces RL-SPH, a reinforcement-learning-based start primal heuristic that can independently produce feasible solutions for ILPs (including non-binary integers). Key components:
1) Bipartite graph encoding of the ILP; a transformer-based GNN outputs per-variable actions {−1, 0, +1}.
2) Two-phase reward: phase 1 maximises feasibility improvement; phase 2 maximises objective improvement after the first feasible solution.
3) Feasibility-aware variable selection: only variables appearing in currently violated constraints are made “changeable”, reducing the action space.
Trained with Actor-Critic on five NP-hard benchmarks, RL-SPH attains 100 % feasibility rate, 44 × smaller primal gap and 2.3 × lower primal integral than built-in start heuristics of SCIP, and generalises to larger, denser and out-of-distribution instances as well as to MIPLIB.

### Strengths
1) Guarantees feasibility even from random initial solutions, overcoming a major limitation of previous ML-based heuristics.
2) Creative use of transformer-GNN plus phase-separated actor-critic to capture long-range variable-constraint interactions.
3) Strong empirical performance across five problem classes and robust generalisation to size/density/distribution shifts and MIPLIB instances.

### Weaknesses
1) Action magnitude is restricted to ±1; preliminary results show larger moves improve objective but are not pursued, leaving optimality on the table.
2) Training requires thousands of episodes on 64 parallel instances; total GPU-hours and carbon footprint are not reported.
3) Only compares against SCIP’s internal heuristics; no comparison with recent large-neighbourhood-search or diffusion-based heuristics.
4) The polynomial-time guarantee relies on the assumption that the agent always improves feasibility; in practice exploration may stall—no discussion of timeout handling.

### Questions
1. What is the computational cost (GPU-hours and energy) of training RL-SPH on the largest benchmark, and how does it compare with collecting near-optimal labels for supervised methods?
2. Could dynamically increasing the action magnitude (e.g., ±5) when the agent is stuck improve final objective value without harming feasibility?
3. How would RL-SPH perform as a standalone solver on very large instances (10^6 variables) where LP relaxation is expensive, and could problem decomposition be integrated?

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
3

### Summary
This paper proposes RL-SPH, a novel end-to-end framework for learning to generate feasible and high-quality solutions for ILPs. RL-SPH can operate independently, though it optionally allows LP-relaxation initialization. The method introduces a two-phase reward system that explicitly separates feasibility recovery (Phase 1) and objective optimization (Phase 2). A key component of the algorithm is its feasibility-aware variable selection, where in each step the agent selects a subset of variables and decides whether to increment, decrement, or keep them unchanged. The selection heuristic prioritizes variables that contribute most to constraint violations, effectively narrowing the search space and improving learning stability. Extensive experiments on five benchmark datasets show that RL-SPH achieves 100\% feasibility rate, substantially reduces Primal Gap and Primal Integral.

### Strengths
The design of the reward is novel and optimization-inspired, and the two-phase reward system (feasibility → optimization) is one of the paper’s main contributions. It provides dense and directional feedback in regions where classical RL methods often suffer from sparse or unstable signals. It provides dense and directional feedback in regions where classical RL approaches suffer from sparse or misleading signals.

The “feasibility-aware variable selection” (Algorithm 3) is also a well-motivated component. By allowing the agent to modify only a small subset of variables (≈ 2 log n) that appear most frequently in violated constraints, the search remains interpretable and computationally manageable. This strategy reduces unnecessary exploration and empirically accelerates the recovery of feasible states.

Another strength is that RL-SPH can start from either LP-relaxation or pure random initialization, with the latter achieving nearly identical performance. This shows that the model learns feasibility patterns directly rather than relying on solver-provided starting points, which makes the approach more self-contained and practical.

Finally, the experimental evaluation is broad and convincing. The paper tests on five benchmark datasets, including a non-binary integer (NBI) case, demonstrating that the approach generalizes beyond 0–1 ILPs. RL-SPH consistently attains full feasibility and yields clear improvements in Primal Gap and Primal Integral compared to established heuristics.

### Weaknesses
The reward formulation involves several manually chosen hyperparameters (e.g., $\alpha$, $p$, $q$, etc). While the method performs well, it is unclear how sensitive the results are to these hyperparameters. A small sensitivity analysis would help confirm whether the learning dynamics are robust or heavily dependent on manual tuning.

The baselines used (Feasibility Pump, Diving, Rounding, PAS) are all classical heuristics. However, several recent learning-based feasibility or primal heuristics (e.g., GNN-guided predict-and-search methods or learning-to-repair approaches) could provide a fairer context for comparison. Including at least one recent end-to-end learning method, particularly an RL-based method, would better position RL-SPH within current literature.

Although ablation studies are included, they focus primarily on architectural components. The paper would benefit from a clearer quantitative analysis of the two-phase reward system (e.g., measuring feasibility rate after Phase 1 alone, or visualizing constraint violation) over time. This would help validate the intended role of each reward component and make the learning process more interpretable.

While the paper reports Primal Gap and Primal Integral metrics, it does not discuss actual wall-clock runtime. These metrics only indirectly reflect temporal progress and are normalized by iteration counts, not by real elapsed time. As a result, the practical efficiency of RL-SPH relative to classical heuristics or solver-based methods remains unclear. Reporting average runtime per instance would make the comparison more fair. This omission significantly weakens the empirical evaluation, as efficiency is a core claim of the paper.

### Questions
1. The −100 penalty effectively discourages idle behavior ($x_{t+1} = x_t$), but it cannot prevent oscillations between distinct but equivalent states (loops). Did the authors observe such cyclic behavior during Phase 1, and if so, how was it mitigated? Would adding a short-term memory or visited-state penalty help stabilize the feasibility recovery process?
2. The paper reports PG/PI/FR but no wall-clock runtime. Could the authors provide the average runtime per instance? This would make comparisons fairer.
3. The paper claims that “to the best of our knowledge, RL-SPH is the first end-to-end primal heuristic that explicitly learns feasibility for ILPs with a theoretical guarantee.” However, Tang, Khalil & Drgoňa [1] have already proposed a learning-based optimization framework with explicit feasibility guarantees for MINLPs, which subsumes ILPs as a special case. Could the authors clarify how RL-SPH differs conceptually or theoretically from this prior work, and in what sense it constitutes the “first” method with feasibility guarantees?
4. The paper shows that RL-SPH(Random) performs comparably to RL-SPH(LP) for 0–1 problems, but no corresponding experiment is reported for the non-binary integer (NBI) dataset. Was random initialization more difficult or unstable in this setting (e.g., due to unbounded variable ranges or larger search space)? Providing results or discussion for the NBI case would clarify whether the Random Initialization generalizes beyond binary ILPs.

[1] Tang, B., Khalil, E. B., & Drgoňa, J. (2024). Learning to Optimize for Mixed-Integer Non-linear Programming with Feasibility Guarantees. arXiv preprint arXiv:2410.11061.

### Soundness
3

### Presentation
2

### Contribution
2
