# Correlated Policy Optimization in Multi-Agent Subteams

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
In cooperative multi-agent reinforcement learning, agents often face scalability challenges due to the exponential growth of the joint action and observation spaces. Inspired by the structure of human teams, we explore subteam-based coordination, where agents are partitioned into fully correlated subgroups with limited inter-group interaction. We formalize this structure using Bayesian networks and propose a class of correlated joint policies induced by directed acyclic graphs . Theoretically, we prove that regularized policy gradient ascent converges to near-optimal policies under a decomposability condition of the environment. Empirically, we introduce a heuristic for dynamically constructing context-aware subteams with limited dependency budgets, and demonstrate that our method outperforms standard baselines across multiple benchmark environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers subteams in cooperative multi-agent RL as a means to balance coordination and scalability. First, under the tabular setting, the authors establish theoretical results investigating convergence and suboptimality for bayesian networks policy gradients under decomposability assumptions. Second, the authors empirically validate their theoretical results in a tabular setting. Third, the authors propose a practical approach for constructing subteams in MARL tasks and test it in three MARL environments, as well as investigate how it can be used for value factorization in centralized training.

### Strengths
The papers features a good discussion of related work in Sec. 2. I believe the key contributions of this work are related with the analysis of gradient-based methods to optimize BN policies - the paper seems to contribute with a novel analysis of gradient-based methods to optimize BN policies in the tabular setting. The paper is clear and well-organized. The experiments seem to support the theoretical findings.

### Weaknesses
The contribution of the paper in the empirical part seems a bit marginal. In particular, I have two key concerns (take a look at my last question below for additional details): (i) are the baselines considered representative of the multiple previously proposed algorithms discussed in Sec. 2?; and (ii) the empirical algorithms analysed in Sec. 6.2 do not seem directly linked to the algorithm analyzed in the theoretical part of the paper. The theoretical results also rely on some assumptions that are a bit restrictive (access to the underlying state and Asssumption 1), but I believe this should not be a reason to reject the paper alone.

### Questions
- The authors consider access to the full underlying state of the environment (Markov game). What about the case of Dec-POMDPs?
- While assumption 1 may be used by previous works, I still believe it is a bit restrictive. How hard would it be to extend the theoretical results while removing Assumption 1?
- Maybe the authors could discuss after presenting Theo. 1 what are the key differences between the result in Theo. 1 and the equivalent single-agent result.
- line 288 - "1. Therefore, the suboptimality bound in (6) reveals a tradeoff when choosing the fineness/coarseness of the decomposition" - Is it possible to find the optimal tradeoff between the two terms in the bound of Lemma 3?
- How many runs were performed to get the empirical results (e.g., Fig. 1)?
- Figure 1: What do the shaded areas in the plots represent?
- Secs. 6.2 and 6.3 of the paper seem a bit disconnected from the theoretical part of the paper. I'm particularly referring to the algorithmic aspect (not to the problem of finding good partitions). The gradient ascent algorithm proposed and analysed by the authors in (2) and (3) considers a regularized objective. What would be the closest "function-approximation" variant of such an algorithm? Why did you choose MAPPO and MADDPG? Can these algorithms be seen as some kind of "function-approximation-counterparts" of the tabular algorithm in (2)-(3)?
- No other previous works (e.g., those discussed in Sec. 2) propose ways to perform subteams decomposition? If so, I believe such methods should also be included as baselines in the empirical part of the work.

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
3

### Summary
This paper studies correlated policy optimization in cooperative multi-agent reinforcement learning (MARL) using subteam-based coordination. The authors represent joint policies as Bayesian networks (BNs), where agents within a subteam are fully connected and thus act in a correlated manner, while inter-subteam dependencies are limited.

Theoretical results extend previous work on BN policy gradients by establishing finite-time convergence to near-optimal policies under a decomposability assumption on the environment’s reward and transition functions. The analysis reveals a trade-off: finer partitions of agents yield faster convergence but potentially higher suboptimality due to decomposition errors.

Empirically, the authors validate their theory in a tabular coordination game that precisely satisfies the theoretical assumptions, and then introduce a heuristic for constructing context-aware subteams based on pairwise dependency scores. This heuristic is integrated into deep MARL algorithms such as MAPPO and MADDPG, where it consistently improves learning speed and final performance across multiple benchmark environments.

### Strengths
- The problem is well-motivated. The work addresses a central challenge in MARL—scaling coordination to large agent populations—by introducing a principled way to exploit structured correlations.
- The exposition is clear, and the theoretical intuitions are well communicated.
- The paper has a strong theoretical contribution.  The finite-time convergence and near-optimality guarantees under decomposability assumptions extend prior results (e.g., Chen & Zhang, 2023) in a meaningful way.
- The experiments are well-aligned with the theory. The tabular experiments are well designed to exactly satisfy the theoretical setup, lending credibility to the claims.
- The empirical results are convincing. The proposed heuristic for subteam construction yields consistent performance improvements in deep MARL tasks, as shown in Figures 2–3.

### Weaknesses
Dependence on domain-specific priors: The heuristic requires a priori pairwise dependency scores dij , which rely on domain knowledge. This may limit generality and make comparisons to methods that learn dependency structures less direct.

### Questions
1. How exactly are the dij dependency scores computed in each environment? Are they purely distance-based, or do they involve any learned or empirical statistics?
2. In Section 6.2.1 you mention that “parent actions are detached from the computation graph to prevent backpropagation.” Could you elaborate on this?
3. Have the authors seen or considered the following closely related paper? Kapoor, A., Freed, B., Schneider, J., & Choset, H. (2025). Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization. Reinforcement Learning Journal, 1, 380–399. This work also decomposes agents into subgroups with partially decoupled rewards and might provide a useful empirical or conceptual comparison.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper utilizes Bayesian Networks (BNs) to model the structured correlations among multi-agent subteams and proposes a class of correlated joint policies induced by directed acyclic graphs (DAGs). Building upon prior work, it establishes a convergence rate for tabular softmax BN policy gradient ascent under fixed DAG structures. Furthermore, the authors prove that, for BNs aligned with the context of multi-agent subteams, regularized policy gradient ascent converges to a policy with bounded suboptimality. Finally, this paper relaxes certain assumptions and proposes a heuristic method, which is evaluated across multiple benchmark environments.

### Strengths
1. The paper is well-structured, with a clear and well-defined motivation.

2. The description of the background and the theoretical analyses are detailed, with clear explanation of the underlying thoughts.

3. This is the first work that establishes optimality guarantees for BN policies without requiring full independence among agents.

4. This paper conducts sufficient experiments to demonstrate the effectiveness of the method.

### Weaknesses
1. The assumptions required for the proof section (full observability, fully correlated with a sub-team and fully independent across sub-teams, fixed DAG topology) are too strict.

2. Practical methods relax some assumptions in the proof section, but some may deviate significantly from the original proof (such as global observability).

3. Experimental scenarios are too simple to demonstrate the upper limit of the method's advantages.

### Questions
1. If possible, please answer the questions mentioned in the Weaknesses Section first.

2. What impact will partial observability have on the theoretical conclusions of this paper?

3. Why is maximizing the average pair-wise dependency score between the agents considered a reasonable criterion for merging two subteams? What are the underlying motivations or theoretical considerations behind this method?

4. What are the considerations for selecting baseline algorithms (MAPPO/MADDPG) in different experimental environments?

5. This is my second time reviewing your submission. As you can see, I continue to hold a generally positive view of your work. I appreciate that you have added new experimental scenarios in response to the previous round of reviewers’ comments — that is definitely a constructive improvement. However, it is somewhat disappointing that the additional experiments are still based on the VAST (2021) scenario, which is relatively dated and not particularly convincing as an evaluation benchmark. Therefore, the new results do not substantially strengthen the empirical evidence of your claims. I strongly encourage you to follow the suggestion of Reviewer QYoJ from NeurIPS25 and include experiments on large-scale scenarios in SMAC with more than 20 agents, which would make your framework’s advantages much more persuasive. For this round, I maintain my overall positive inclination toward the paper, but I must note that I cannot guarantee other reviewers will share the same perspective — some may weigh empirical validation more heavily than theoretical contribution. I therefore reserve the right to align my final assessment with the consensus on other reviewers' evaluation on your experiment.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the critical challenge of scalability in cooperative multi-agent reinforcement learning (MARL), which is often hindered by the exponential growth of the joint action space. The authors draw inspiration from human teams to propose a "subteam-based coordination" structure. This structure is formalized using Bayesian networks (BNs) , where agents within a subteam are fully correlated (modeled as a fully connected subgraph ) while interactions between subteams are limited (modeled as conditionally independent ).

The paper's contributions are threefold:

- It provides a finite-time convergence rate for tabular softmax BN policy gradient ascent under a fixed DAG, strengthening prior asymptotic results.

- It proves that for this subteam-based BN structure, regularized policy gradient ascent converges to a near-optimal policy (not just an equilibrium) under a "decomposability condition" on the environment's reward and transition functions. The resulting suboptimality bound is explicitly characterized by the decomposition errors.


- It introduces a practical heuristic for dynamically constructing these subteam DAGs based on "dependency scores" and an edge budget. This heuristic is integrated with deep MARL algorithms (MAPPO/MADDPG) and value-factorization methods (VAST) , demonstrating superior performance over baselines across several benchmarks.

### Strengths
**Clear Motivation and Formalism**: The paper's premise is intuitive and well-motivated. The core idea of partitioning agents into highly-coordinated subteams to reduce complexity is a natural one. The use of Bayesian networks to formalize this—capturing full correlation within teams and independence between them—is an elegant and clean way to model the problem .

**Theoretical Contribution**: The main theoretical result (Theorem 2) is a strong contribution. It moves beyond typical convergence guarantees (to an equilibrium, like in Theorem 1) to provide a bound on the policy's suboptimality. Crucially, it connects this bound directly to a structural property of the environment: the decomposability errors. This provides a concrete, formal basis for why and when subteam factorization should work.

- **Identification of a Key Trade-off**: The theory and experiments jointly expose a fundamental trade-off in subteam design. Coarser partitions (i.e., fewer, larger subteams) lead to smaller decomposition errors and thus a lower asymptotic suboptimality bias. Finer partitions (i.e., more, smaller subteams) may converge faster but suffer from larger errors. The tabular experiments in Section 6.1, which fit the decomposition errors (Table 1) and plot the final suboptimality (Figure 1), provide a clear and compelling validation of this theoretical insight.

- **Empirical Validation and Practical Heuristic**: The paper successfully bridges the theoretical insights to a practical setting. The theory motivates the search for partitions with low decomposition error, and the authors propose a sensible heuristic (Algorithm 1) to approximate this by merging subteams with high "dependency" . The empirical results are comprehensive, showing the heuristic's effectiveness in:

  - Tabular settings (validating the theory).

  - Deep MARL actor-critic methods (outperforming 'full', 'product', and 'random' DAGs) .

  - Centralized training (CTDE) value-factorization methods (improving VAST) .

### Weaknesses
- **On the "Decomposability" Assumption**: The main result hinges on the environment's decomposability (Definition 2). The authors correctly state that any environment can be decomposed this way, as the errors can absorb any discrepancy. However, the utility of the bound in Theorem 2 depends on these errors being small. 

- **Dependency Scores for the Heuristic**: The practical heuristic (Algorithm 1) requires pre-defined "dependency scores" to guide the merging process. In all experiments, these scores are based on domain-specific spatial proximity (e.g., Euclidean or Manhattan distance, or binary state position) . This seems to be a form of privileged information. 

- ** Strictness of Assumption 3(iii)**: The theoretical framework (specifically Assumption 3(iii)) requires no edges between subteams. This is noted as a technical requirement for the proof. This hard partitioning seems somewhat rigid, as real-world teams often have sparse but important cross-team coordination. 

- **"Line" DAG Baseline**: In the tabular results (Section 6.1), the "line" DAG is used as a baseline and performs quite well (Figure 1). The paper notes this DAG does not satisfy Assumption 3  (as it's a single chain, not a partition of fully-connected subteams). This is an interesting result.

### Questions
- (W1) Could the authors comment on the generality of low-error decomposability? The tabular experiments show this holds for the Coordination Game, but how restrictive is this assumption for more complex, non-obvious benchmarks? Is it possible to estimate or bound these errors a priori to determine if an environment is suitable for this method?

- (W2) How sensitive is the heuristic to the quality of these scores? What would happen in a task where the primary dependency is functional (e.g., "passer" and "shooter") rather than spatial? Could these dependency scores be learned dynamically, perhaps by analyzing value function contributions or action correlations, to make the approach more general?

- (W3) How challenging would it be to relax this assumption? For example, if a small number of inter-team edges were permitted, would this completely break the value function decomposition (Lemma 5), or could it be handled by introducing additional, bounded error terms into the analysis?

- (W4) Does this suggest that the 'fully-correlated subteam' structure is a sufficient, but perhaps not strictly necessary, condition for good performance? It implies that other sparse, structured DAGs might also be effective, even if they don't fit the specific subteam partition model.

### Soundness
4

### Presentation
4

### Contribution
4
