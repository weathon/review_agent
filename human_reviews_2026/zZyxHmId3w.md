# MIRACLE: Model-free Imitation and Reinforcement Learning for Adaptive Cut-Selection

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Mixed-Integer Programming (MIP) solvers rely heavily on cutting planes to tighten LP relaxations, but traditional approaches generate thousands of cuts that consume gigabytes of memory while providing minimal benefit. We present an intelligent cut selection framework that achieves a 98.1\% reduction in memory usage while maintaining competitive solving with an objective gap of approximately 0.08\%. Within this RL framework, we use Proximal Policy Optimization (PPO) to learn a behavioral model that imitates the expert solver’s decisions. The adversarially imitated behavioral model drives an agent comprising these key innovations: (i) a cut-selection policy trained via curriculum learning; and (ii) adaptive inference that dynamically adjusts computational budgets. Through comprehensive evaluation across SetCover and diverse MIPLIB problems, we demonstrate consistent speedups (3.78$\times$ average on MIPLIB) and achieve a 100\% success rate on instances where traditional SCIP fails 47-53\% of the time. Our method also reduces peak memory consumption from 3.03GB to 46 MB, enabling optimization in previously inaccessible and other resource-constrained environments where traditional solvers face fundamental limitations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents an intelligent cut selection framework for Mixed-Integer Programming (MIP) solvers that drastically reduces peak memory usage by 98.1% while maintaining competitive performance. The method employs Proximal Policy Optimization (PPO) within an adversarial imitation learning setup to mimic an expert solver’s decisions, featuring a curriculum-trained cut-selection policy and adaptive inference for dynamic computational budgeting. Evaluated on SetCover and diverse MIPLIB instances, it achieves consistent speedups and a 100% success rate on problems where SCIP fails 47–53% of the time, demonstrating strong potential for deployment in memory-constrained settings.

### Strengths
- Cut selection is indeed a critical research direction, and improving it can substantially boost solver efficiency.  
- The focus on reducing peak memory consumption is particularly interesting and, to the best of my knowledge, has been rarely emphasized in prior literature—making it a compelling practical attempt.  
- The method shows stability across different hyperparameter configurations, which is encouraging and suggests the system may be close to deployment-ready.  
- The use of adversarial imitation within the reinforcement learning framework is interesting and offers a fresh perspective on reward design, with detailed theoretical analysis, which, in my view, constitutes a meaningful contribution.

### Weaknesses
- The paper overall feels somewhat rushed and leaves several important aspects underdeveloped.  
- The core motivation—why reducing peak memory consumption is essential—is not sufficiently justified. As a central claim of the paper, this requires clearer explanation: in what real-world scenarios does memory become a bottleneck, and which components of the proposed method specifically target this issue?  
- The paper completely omits a related work section and fails to acknowledge or compare against several highly relevant prior works on learning-based cut selection, including:  
  [1] Wang Z, Li X, Wang J, et al. Learning Cut Selection for Mixed-Integer Linear Programming via Hierarchical Sequence Model. ICLR 2023.  
  [2] Paulus M B, Zarpellon G, Krause A, et al. Learning to cut by looking ahead: Cutting plane selection via imitation learning. ICML 2022.  
  [3] Tang Y, Agrawal S, Faenza Y. Reinforcement learning for integer programming: Learning to cut. ICML 2020.  
  Not only are these works not cited, but no experimental comparison is provided. This omission is significant and, in my view, unfair to prior contributions.  
- Despite extensive mathematical formalism, many crucial implementation details remain unclear:  
  1) It is not specified on what data the discriminator network is trained, what its training procedure is, or how it integrates with the PPO-based RL loop—making the overall training pipeline ambiguous.  
  2) The choice of SetCover and MIPLIB as testbeds lacks justification; other standard MIP benchmarks (e.g., MaxCut, TSP) exist, and the selection criteria for the 300 test instances are unexplained.  
  3) While Appendix Algorithm 1 offers partial clarification, key questions persist—for example, where the Phase 1 policy comes from and whether Phases 1–3 are trained sequentially.  
  4) The heavy notation would greatly benefit from a summary table or diagram; as is, the exposition is unnecessarily difficult to follow.

### Questions
- The policy ultimately imitates SCIP’s cut selection strategy—so why does it achieve noticeable speedups, especially in solving time? Is the gain due to fewer cuts, better timing, reduced overhead, or another factor?  
- What data was used for training? How were the 300 test instances selected, particularly the 150 from MIPLIB? Given that MIPLIB contains far more instances, what was the selection criterion (e.g., difficulty, size, feasibility)?  
- In Table 1, what does “early stop” refer to, and what is the meaning of “Max Iterations”?  
- How were the 600-second time limit and 12 GB memory cap determined? Many MIPLIB instances require significantly longer to solve—could this experimental setup bias results toward easier cases?  
- Why introduce a custom “success rate” metric instead of using widely adopted, more direct measures like primal gap or final objective value?  
- Could the authors clarify: on what data is the discriminator trained, what is its training workflow, and how does it coordinate with the RL training? Understanding the full pipeline is essential for evaluating the method’s soundness and reproducibility.  
- Why choose SCIP’s cut selection as the expert policy? Given that it relies on heuristic rules, are there higher-quality expert strategies (e.g., from stronger solvers or human-designed oracles) that could further improve performance?  
- What is the design rationale behind the curriculum learning scheme? What performance gain does it provide, and could an ablation study quantify its contribution?

I am eager to participate in the discussion phase and look forward to a thorough and objective assessment of this work based on the authors’ responses.

### Soundness
2

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
4

### Summary
This paper presents MIRACLE, a reinforcement learning framework that addresses memory inefficiency in Mixed-Integer Programming (MIP) solvers by learning intelligent cut selection policies through Proximal Policy Optimization (PPO) combined with Generative Adversarial Imitation Learning. The framework trains a neural policy that learns from expert solver (SCIP) demonstrations to select high-impact subsets of cutting planes, using a discriminator network to provide dense reward signals and curriculum learning across four difficulty phases. Evaluated on 300 instances spanning SetCover and MIPLIB benchmarks, MIRACLE achieves 86.3% average memory reduction, 3.78× average speedup, and 100% success rate on challenging MIPLIB instances where traditional SCIP fails 47-53% of the time. The method employs adaptive inference that automatically adjusts cut budgets and iteration limits based on problem difficulty classification, with ablation studies showing robust performance across hyperparameter configurations. The framework addresses three key limitations of existing approaches: treating MIP solvers as black boxes, myopic decision-making without environmental modeling, and treating memory efficiency as an afterthought rather than a primary objective.

### Strengths
1. The paper follows a logical flow from problem formulation through methodology to evaluation, with helpful visual aids like Figure 1 showing the complete MIRACLE framework and Figure 2 providing comprehensive performance summaries. 
2. The framework achieves a remarkable 98.1% memory reduction on SetCover benchmarks (from 3.03GB to 46MB) and maintains 86.3% average reduction across all 300 test instances, while simultaneously delivering 3.78× average speedup on MIPLIB problems. Most impressively, MIRACLE achieves a 100% success rate on challenging MIPLIB instances where traditional SCIP fails 47-53% of the time, demonstrating that memory efficiency directly translates to improved solver reliability.
3. The evaluation spans 300 instances from established benchmarks, including SetCover and diverse MIPLIB datasets across three difficulty levels, with the largest instances containing up to 1000 operations. The method demonstrates consistent performance across fundamentally different problem types and scales, with detailed ablation studies showing robustness across hyperparameter configurations in real-world deployment scenarios.

### Weaknesses
In general, this paper show very impressive results on memory reductions and even speed up. However, I have the following concerns which may largely impact the paper’s significance.

While the paper demonstrates impressive results under a 12GB memory limit, it lacks compelling evidence for why this specific constraint is realistic or necessary in modern computing environments where servers often have 64-256GB RAM. The authors don't provide concrete examples of deployment scenarios (edge computing, embedded systems, or cloud resource billing) where such strict memory limits are critical, making the 12GB restriction appear artificially constraining rather than addressing a well-motivated practical need.

The paper only compares against SCIP-default and SCIP-aggressive, omitting recent learning-based cut selection methods from the ML4CO literature that likely also achieve memory reductions as a byproduct of selective cut management (e.g., methods that use GNNs for cut ranking or reinforcement learning for adaptive cut generation). More critically, the speed comparisons in Table 3 are conducted under the artificial 12GB memory limit which causes SCIP to fail on 47-53% of instances. This fundamentally biases the evaluation since traditional solvers are designed to trade memory for speed and would perform differently without this constraint. The paper would greatly benefit from: (a) performance comparisons both with and without memory limits to fairly assess the speed-memory tradeoff, (b) inclusion of recent ML-based cut selection baselines, and (c) analysis of how much memory reduction other selective cut approaches naturally achieve without explicitly optimizing for it.

The evaluation is limited to SetCover and MIPLIB instances, missing other important combinatorial optimization problems such as Maximum Independent Set, and Facility Location problems that are equally relevant in practice and used in previous works. This narrow scope makes it difficult to assess whether MIRACLE's approach generalizes beyond the two tested problem types.

### Questions
See the weakness part.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the memory explosion caused by thousands of generated cuts in branch-and-cut. Uses GAIL to learn a dense reward function from SCIP expert trajectories, avoiding hand-crafted rewards; Trains a lightweight policy with PPO to select a budget-limited, high-impact subset of cuts; Employs curriculum learning (easy → hard) and an adaptive inference mechanism that sets cut budget B and early-stop thresholds according to predicted instance difficulty.
On 150 SetCover and 150 MIPLIB instances MIRACLE attains 98 % peak-memory reduction (46 MB vs 3 GB), raises success rate from 47 % to 100 % under 12 GB RAM limit, and yields 3.78 × mean speed-up over SCIP with large effect sizes and statistical significance.

### Strengths
1) Novel paradigm: demonstrates that intelligent cut selection can reduce memory by > 95 % while improving speed and reliability.
2) Solid methodology: adversarial reward learning avoids manual engineering; curriculum and adaptive inference remove hyper-parameter tuning at test time.
3) Strong empirical evidence: 300 instances, statistical tests, ablation studies, and theoretical bounds all align with the claimed benefits.

### Weaknesses
1) Policy network is a small MLP (≈ 17 K parameters); capacity limits for capturing complex cut interactions are not discussed.
2) Only compares against SCIP default and aggressive modes; no comparison with other recent ML-based cut selectors.
3) GAIL training requires a large set of expert trajectories (1000 instances); data collection cost is not reported.
4) Theoretical analysis assumes bounded rewards and Lipschitz policies but does not verify these assumptions hold in the MILP domain.

### Questions
1) How does the small MLP policy scale to instances that generate 50,000+ cuts, and would deeper architectures or graph networks help?
2) What is the computational cost of collecting the 1000 expert trajectories, and could the method work with fewer demonstrations?
3) Have you experimented with other policy-gradient algorithms (e.g., TRPO, SAC) and does PPO remain the best choice?

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
4

### Summary
This paper proposes MIRACLE, an RL-based framework for adaptive cut selection in mixed-integer programming (MIP). The authors formulate cut selection as a Markov Decision Process (MDP) and employ a PPO-based policy combined with Generative Adversarial Imitation Learning (GAIL) for reward shaping. The approach explicitly optimizes for memory efficiency, achieving dramatic reductions in memory usage (up to 98%) while maintaining comparable or better solving performance compared with SCIP baselines. The authors also provide theoretical analysis (sample complexity, convergence, and memory reduction guarantees) and extensive experiments on SetCover and MIPLIB benchmarks.

### Strengths
1. The paper highlights memory efficiency in MILP research. This shift in focus is both original and practically valuable.

2. Combining PPO with GAIL and curriculum learning is a thoughtful design that balances imitation and reinforcement. The adaptive inference strategy further increases practicality.

3.The evaluations across hundreds of MILP instances are detailed and statistically validated, with significant memory and reliability improvements.

### Weaknesses
1. It remains unclear whether MIRACLE generalizes to other MIP components (e.g., branching, node selection) or large-scale industrial solvers beyond SCIP.

2. The paper would benefit from more insight into what the learned policy captures—e.g., are the selected cuts similar to expert strategies or exploiting new patterns?

3. It would be helpful if the authors can provide some training details of RL and analysis the covergence.

4. Although the experiments are extensive, most evaluations focus on SetCover and MIPLIB, leaving out other industrial problem classes. It would help to report on training/inference cost or model sensitivity to solver variants.

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
4
