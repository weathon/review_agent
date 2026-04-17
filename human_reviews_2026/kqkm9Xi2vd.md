# Core Advantage Decomposition for Policy Gradients in Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
This work focuses on the credit assignment problem in cooperative multi-agent reinforcement learning (MARL). Sharing the global advantage among agents often leads to insufficient policy optimization, as it fails to capture the coalitional contributions of different agents. Existing methods mainly assign credits based on individual counterfactual contributions, while overlooking the influence of coalitional interactions. In this work, we revisit the policy update process from a coalitional perspective and propose an advantage decomposition method guided by the cooperative game-theoretic core solution. By evaluating marginal contributions of all possible coalitions, our method ensures that strategically valuable coalitions receive stronger incentives during policy gradient updates. To reduce computational overhead, we employ random coalition sampling to approximate the core solution efficiently. Experiments on matrix games, differential games, and multi-agent collaboration benchmarks demonstrate that our method outperforms baselines. These findings highlight the importance of coalition-level credit assignment and cooperative games for advancing multi-agent learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tackles credit assignment in cooperative multi-agent reinforcement learning (MARL). It proposes a core-guided advantage decomposition method grounded in cooperative game theory, where individual agent advantages are derived to satisfy a $\epsilon$-core solution concept. To reduce computational overhead, the method approximates the core via random coalition sampling. The authors provide lower bounds on coalition policy improvement and evaluate on matrix games, VMAS, and cooperative MuJoCo.

### Strengths
1. Provides theoretical analysis, including lower bounds on coalition policy improvement.

2. Uses a principled cooperative game-theoretic framework for advantage decomposition.

3. Demonstrates empirical effectiveness on matrix games, VMAS, and cooperative MuJoCo tasks.

### Weaknesses
1. Related work and baselines appear outdated.

2. Reported improvements over baselines are modest and not clearly statistically significant.

3. Missing evaluations on widely used benchmarks such as SMAC/SMACv2 and Google Research Football.

4. Assumes additivity (sum of individual advantages equals joint advantage), which may not hold in highly non-linear interactions or with strong coordination requirements.

5. Coalition advantages are estimated for counterfactual coalitions that the critic may not have seen, raising out-of-distribution estimation concerns.

6. Sensitivity to the $\epsilon$-core hyperparameter may affect stability and reproducibility; its selection criteria are unclear.

7. Scalability  of random coalition sampling with increasing agent counts are not fully characterized.

### Questions
1. How are counterfactual coalition advantages computed when the critic has not seen those joint actions in the replay buffer? Are there measures to mitigate out-of-distribution bias (e.g., constraints, regularization, uncertainty)?

2. How sensitive is performance to the $\epsilon$ parameter of the $\epsilon$-core? 

3. Do you evaluate on SMAC/SMACv2 and Google Research Football? If not, what prevents running on these benchmarks, and how do you expect the method to scale there?

4. How does computational cost scale with number of agents and coalition samples (training/inference wall-clock, GPU hours)? What is the per-update overhead relative to standard CTDE methods?

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
3

### Summary
The paper introduces CORA, a credit-assignment wrapper for cooperative MARL policy gradients. CORA (i) computes coalition-level advantages for every subset of agents, (ii) allocates individual credits by solving a quadratic program that respects the strong ε-core (coalition rationality + global budget), and (iii) approximates the exponential set of constraints via random coalition sampling. Theoretically, CORA guarantees that beneficial coalitions receive a lower-bound policy improvement even when global advantage is negative. Empirically, CORA consistently outperforms MAPPO, HAPPO and Shapley-based baselines on matrix games, differential games, VMAS and Multi-Agent MuJoCo.

### Strengths
1. New granularity: First work to embed core solution (cooperative game theory) inside policy-gradient updates; bridges coalitional stability and MARL credit assignment.  
2. Theoretical substance: Novel lower bounds on log-policy improvement for any coalition; shows ε-core ensures provable incentives for valuable sub-teams.  
3. Scalable approximation: Random sampling reduces 2ⁿ QP constraints → O(n/δ²) with controllable δ-probable core guarantee (Theorem 4).  
4. Strong empirical record: SOTA or near-SOTA on 12 tasks spanning discrete, continuous, dense and sparse-reward settings; ablations verify sample-efficiency and robustness to small coalition budgets.  
5. Reproducibility: Complete pseudocode, hyper-parameters, seeds and anonymized code provided; experiments use public benchmarks.

### Weaknesses
1. Computational Footprint  
   Even with random coalition sampling, CORA requires hundreds of extra Q-value evaluations per step, significantly increasing training cost. The paper does not report wall-clock overhead relative to MAPPO for n = 6, 10, 15, limiting its practical deployment at scale.
2. Scalability Ceiling  
   The algorithm’s complexity is O(m·|C| + QP). Current experiments only go up to n = 6 agents, with no results for n ≥ 20, leaving unclear how performance degrades in larger systems.
3. Insufficient Baseline Comparison  
   The paper does not compare with recent Shapley-based policy gradient methods (e.g., SHAQ, SCCA) or role decomposition methods like RODE, limiting a full assessment of CORA’s relative strengths.
4. Variance Regularization Issues  
   Experiments show that “CORA w/o std” outperforms full CORA in some tasks (e.g., differential games), suggesting that variance regularization may suppress exploration. There is no adaptive mechanism or analysis of when to enable/disable this term.
5. Strong Theoretical Assumptions  
   - The provided lower bounds on policy improvement rely on compatible linear critics and small step size α, but no discussion is given for deep neural network critics or non-linear policies.  
   - Theorem 4 gives a δ-probable core guarantee, but no rate is provided for how ε decreases with sample size m, lacking insight into the trade-off between approximation quality and sampling efficiency.

### Questions
1. Computational Cost & Real-Time Feasibility  
   What is the per-step training time and GPU memory usage of CORA compared to MAPPO when n = 10 or 15? Can further parallelization or approximation reduce this cost?
2. Performance at Scale  
   How does CORA perform degrade as n ≥ 20? Does the policy improvement lower bound still hold? Are there QP solver failures or insufficient sampling issues?
3. Comparison with Recent Credit Assignment Methods  
   How does CORA compare with latest Shapley-based PG methods (e.g., SHAQ, SCCA) or RODE? Under what task structures does CORA offer clear advantages?
4. Adaptive Variance Regularization  
   Can an adaptive mechanism be designed to dynamically adjust the strength of the variance regularization term based on task exploration difficulty or training stage?
5. Theory Extension to Deep Critics  
   Do CORA’s policy improvement bounds still hold under deep neural network critics and non-linear policies? Can compatible function approximation or other techniques extend the theory?
6. Empirical ε vs. m Relationship  
   How does ε decrease as sample size m increases in practice? Can empirical curves or tighter theoretical bounds be provided to guide sampling strategy?

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
4

### Summary
This paper proposes CORA (Core Advantage Decomposition), a novel method for credit assignment in multi-agent reinforcement learning. CORA formulates advantage decomposition as an $\epsilon$-core problem from cooperative game theory, ensuring coalition rationality and balanced credit allocation among agents. Each agent’s advantage $A_i$ is obtained by solving a quadratic program constrained by coalition-level advantages. The method is theoretically justified under natural policy gradient updates and empirically evaluated on matrix games, differential games, VMAS, and MaMuJoCo tasks, showing improved stability and performance over MAPPO and related baselines.

### Strengths
- The paper introduces a principled connection between cooperative game theory ($\epsilon$-core) and multi-agent credit assignment, offering a novel theoretical perspective.
- Cross-agent credit assignment is a fundamental problem in multi-agent reinforcement learning, and the paper provides a novel solution to this problem.

### Weaknesses
- The method appears computationally expensive, yet the paper does not provide quantitative analysis or discussion on runtime efficiency or scalability.  
- The experiments are limited to small and medium-scale environments; more complex benchmarks such as SMAC or Google Research Football are not tested.  
- (Minor) Several figures have small fonts and are difficult to read.

### Questions
1. Please compare CORA’s coalition-based advantage decomposition with the implicit credit assignment mechanisms in COMA and VDN, as well as with HAPPO’s explicit but globally shared advantage decomposition. How do these different decomposition paradigms differ in terms of stability, scalability, and theoretical grounding?

2. How would you position CORA relative to recent explicit credit assignment approaches such as  
   - She et al. (2022) “Agent-Time Attention for Sparse Rewards Multi-Agent Reinforcement Learning,” and  
   - Chen et al. (2023) “STAS: Spatial-Temporal Return Decomposition for Multi-Agent Reinforcement Learning”?

3. Given that CORA currently shows promising results mainly in small to medium-scale environments, is there a principled way to improve its computational efficiency so that it can scale to larger agent populations while maintaining its sample-efficiency advantage? If the approach remains limited to few-agent settings, its practical impact might be constrained.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Core Advantage Decomposition (CORA), a credit-assignment scheme for cooperative MARL that decomposes the global advantage into per-agent credits by solving, at each update, a strong 
$\epsilon$-core optimization over coalitions. Concretely, the authors define a coalitional advantage and allocate individual advantages via a quadratic program that enforces core constraints and penalizes deviation from uniform sharing. They also give a sampling-based approximation with a PAC-style guarantee using VC-dimension arguments. CORA is integrated into an actor-critic (PPO-style) training loop with two critics, and is evaluated on matrix games, differential games, VMAS, and MA-MuJoCo. Empirically, CORA reportedly improves returns over MAPPO/HAPPO/COMA in several tasks.

### Strengths
1. Casting per-agent advantage allocation as a strong $\epsilon$-core program (Eq. 7) is neat and leads to interpretable coalition rationality constraints and a variance-regularized objective.

2. Theorems 2 and 3 relate NPG updates to (coalitional) improvement, clarifying when beneficial coalitions are amplified even if the global advantage is negative. 

3. Theorem 4 provides a simple sample-complexity bound for entering a 𝛿-probable core using VC-dimension, which is rarely discussed in MARL credit assignment.

4. Experiments span matrix/differential games, VMAS, and MA-MuJoCo, with ablations on coalition sampling and discussions of runtime/constraint violations in a synthetic setting.

### Weaknesses
1. There is extensive prior work using cooperative-game concepts for credit (e.g., Shapley-based SQDDPG, Shapley Counterfactual Credits, SHAQ). The paper does not compare against these or thoroughly argue why the core yields better learning dynamics than Shapley-style alternatives. The theoretical pieces largely restate core feasibility rather than showing tighter improvement bounds or variance reductions over Shapley. Empirically, none of these Shapley baselines are included.

2. The lower-bound results assume natural policy gradient updates with compatible function approximation, while the implementation uses PPO with clipping and two critics. The paper asserts first-order relations, but does not quantify how clipping, advantage normalization, or off-policy bootstrap in Q affect the guarantees. Rhis leaves the theoretical relevance to the practical algorithm unclear.

3. While MAPPO/HAPPO/COMA are included, other strong contemporary PPO-style baselines and trust-region variants on MA-MuJoCo (e.g., HAPPO/HATRPO references) suggest nuanced performance differences. The paper does not situate its results in that evolving landscape or evaluate on common benchmarks like SMAC or MPE where credit-assignment is heavily studied.

4. The baseline action is chosen as the modal/mean policy output. This may bias estimates and reduce exploration, yet the paper reports that removing the std term helps on differential games, suggesting sensitivity that is not analyzed. Moreover, solving Eq. 7 requires many Q evaluations per step. The cost is discussed only in a random-advantage toy experiment, not the real tasks.

### Questions
1. Can you include direct comparisons to SQDDPG, Shapley Counterfactual Credits, and/or SHAQ on at least one VMAS and one MA-MuJoCo task? If not, please justify why these are out of scope and provide a discussion on when core vs. 

2. For VMAS and MA-MuJoCo, what is the average per-update time and memory overhead vs. MAPPO/HAPPO when (i) using all coalitions and (ii) using sampled coalitions at your recommended 
$m$? Please also report the number of 𝑄 forward passes per batch and the resulting throughput. 

3. Have you tried expectation baselines $Q(s,a_C,\pi_{N\backslash C})$ or action-masking variants (as mentioned in Remark 1)? A targeted ablation isolating the effect of the baseline actioncould clarify whether the gains stem from the core constraints or the baseline choice.

### Soundness
2

### Presentation
2

### Contribution
2
