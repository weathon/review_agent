# When Is Diversity Rewarded in Cooperative Multi-Agent Learning?

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 10

## Abstract
The success of teams in robotics, nature, and society often depends on the division of labor among diverse specialists; however, a principled explanation for when such diversity surpasses a homogeneous team is still missing. Focusing on multi-agent task allocation problems, we study this question from the perspective of reward design: what kinds of objectives are best suited for heterogeneous teams? We first consider an instantaneous, non-spatial setting where the global reward is built by two generalized aggregation operators: an inner operator that maps the N agents’ effort allocations on individual tasks to a task score, and an outer operator that merges the M task scores into the global team reward. We prove that the curvature of these operators determines whether heterogeneity can increase reward, and that for broad reward families this collapses to a simple convexity test. Next, we ask what incentivizes heterogeneity to emerge when embodied, time-extended agents must learn an effort allocation policy. To study heterogeneity in such settings, we use multi-agent reinforcement learning (MARL) as our computational paradigm, and introduce Heterogeneity Gain Parameter Search (HetGPS), a gradient-based algorithm that optimizes the parameter space of underspecified MARL environments to find scenarios where heterogeneity is advantageous.  Across different environments, we show that HetGPS rediscovers the reward regimes predicted by our theory to maximize the advantage of heterogeneity, both validating HetGPS and connecting our theoretical insights to reward design in MARL. Together, these results help us understand when behavioral diversity delivers a measurable benefit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a principled framework for understanding when and why behavioral diversity among cooperative agents leads to improved collective performance. The authors introduce the notion of heterogeneity gain, defined as the performance difference between heterogeneous and homogeneous teams, and formally analyze it within a general two-level reward aggregation model. Using majorization theory and Schur-convexity analysis, they prove that the curvature of the inner and outer aggregation functions determines whether diversity enhances or hinders performance — specifically, concave-outer and convex-inner structures favor heterogeneity. Building on this theoretical foundation, the paper proposes HetGPS (Heterogeneity Gain Parameter Search), a differentiable environment co-design algorithm that discovers reward parameterizations maximizing or minimizing heterogeneity gain through bilevel optimization. Empirical studies across analytic matrix games and embodied MARL tasks (multi-goal capture, tag, and football) validate the theory, showing that HetGPS consistently rediscovers curvature conditions predicted to reward diverse behaviors. The work provides a unifying explanation for when diversity benefits cooperative learning and establishes a new paradigm linking reward curvature, environment design, and emergent specialization in multi-agent systems.

### Strengths
1. The paper introduces a mathematically precise metric $\Delta R = R_{\text{het}} - R_{\text{hom}}$​ to quantify when heterogeneous (specialized) teams outperform homogeneous (identical) teams. It proves that the sign of ΔR depends solely on the Schur-convexity or concavity of the inner and outer reward aggregators: (1) Schur-convex inner operator leads to heterogeneity gain; (2) Schur-concave inner operator leads to no gain; (3) Schur-convex outer operator (under constant-sum normalization) leads to no gain. This gives a simple convexity test for when diversity helps — a clear theoretical advance over heuristic assumptions in prior MARL work.
2. This paper provides closed-form and bounded results for Softmax, Power-Sum, and {min, mean, max} families, creating a general mathematical map of reward structures that promote or suppress diversity.
3. A new gradient-based environment co-design algorithm that directly optimizes environment parameters to maximize or minimize $\Delta R$: (1) It operates via differentiable simulators (back-propagates through environment); (2) It avoids inefficiencies and instability of RL-based environment design (e.g., PAIRED); (3) It can either promote or suppress heterogeneity by switching gradient ascent/descent. The algorithm alternates MARL policy training and environment updates efficiently — roughly 25 % overhead — and robustly rediscovering the theoretical curvature configurations.
4. Tests across (i) analytical matrix games, (ii) embodied long-horizon MARL tasks (Multi-Goal Capture, 2v2 Tag, Football). All experiments confirm theoretical predictions: **concave-outer + convex-inner** reward structures yield the largest heterogeneity gains. The algorithm autonomously learns the same curvature conditions predicted by theory (inner convex, outer concave) for both Softmax and Power-Sum parameterizations — a strong consistency check.
5. This paper links convex analysis and majorization theory to concrete MARL reward design, turning diversity from a heuristic into a controllable design variable. Also, it provides a principled answer to an open question in cooperative MARL — under what reward structures does behavioral diversity outperform homogeneity?
6. The contribution of this paper is applicable beyond RL to any cooperative optimization setting expressible as nested aggregations (e.g., task allocation, team composition, distributed robotics).
7. Complete code and YAML configurations provided; all mathematical proofs and assumptions detailed in appendices. Figures (e.g., Fig. 2–5) clearly demonstrate the alignment between theoretical curvature analysis and empirical $\Delta R$ outcomes. Writing is logically structured, connecting intuition, theory, and experiment in a coherent flow.

### Weaknesses
1. While elegant, the use of Schur-convexity assumes symmetric and differentiable reward functions. Many practical rewards in MARL are asymmetric, piecewise, or sparse — outside the smooth function classes covered by majorization theory.
2. Though claimed to be “25% higher,” the method still requires dual policy training (heterogeneous vs homogeneous teams) and outer-loop environment gradient updates. The bilevel optimization may become prohibitively expensive for large-scale, high-dimensional MARL tasks.
3. Empirical validation is confined to simple matrix games and small-scale MARL benchmarks (Multi-goal Capture, 2v2 Tag, Football mini-games). There are no large-scale or high-dimensional tasks (e.g., SMACv2, MPE with >10 agents, or real robotic control) to demonstrate robustness.

### Questions
> Please answer the following questions:

1. The theory presumes identical agents differing only in behavior (i.e., “behavioral heterogeneity”). What if agents differ in capacity, observation space, or action bounds? Would Schur-convexity still capture heterogeneity gain?
2. The method performs bilevel optimization: inner MARL loop + outer gradient update of environment parameters. Does the paper analyze or guarantee convergence, or could gradient coupling cause oscillations?
3. HetGPS successfully rediscovers concave-outer/convex-inner curvature. Would it also succeed if reward functions were implemented via neural networks instead of analytic forms?

> The possible suggestions:

A prior work [1] also found that the reward structure (even encoding the same goal) in a real-world problem could impact the performance of MARL algorithms a lot, though it does not consider the multi-task scenarios. This can be used as an example to motivate the research goal of this paper.

[1] Wang, Jianhong, Wangkun Xu, Yunjie Gu, Wenbin Song, and Tim C. Green. "Multi-agent reinforcement learning for active voltage control on power distribution networks." _Advances in neural information processing systems_ 34 (2021): 3271-3284.

The global reward structure that aggregates reward functions embedding different subtasks/roles is highly relevant to payoff allocations/credit assignment. Especially, in prior work MARL algorithms with Shapley values [2] also demonstrate that agents with heterogenous behaviors is corresponding to different payoffs assigned by Shapley values. This paper aims to automatically search for a design of local rewards which is an reverse process against the prior work, that design local rewards inspired by cooperative game theory leading to heterogeneous behaviors. Thus, this relevant prior work is encouraged to be discussed in related work.

[2] Wang, Jianhong, Yuan Zhang, Tae-Kyun Kim, and Yunjie Gu. "Shapley Q-value: A local reward approach to solve global reward games." In _Proceedings of the AAAI conference on artificial intelligence_, vol. 34, no. 05, pp. 7285-7292. 2020.

### Soundness
3

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
3

### Summary
This paper addresses the critical and previously unresolved question of when behavioral heterogeneity is inherently beneficial in cooperative multi-agent task allocation problems. The authors provide a principled theoretical answer by demonstrating that the potential advantage of specialized policies is determined by the curvature of the global reward function, which is modeled as a double aggregation of agent efforts. Specifically, positive heterogeneity gain is predicted when the inner, task-level aggregator is Schur-convex and the outer, team-level aggregator is Schur-concave. To validate and extend these insights to complex, time-extended environments, the paper introduces HetGPS (Heterogeneity Gain Parameter Search), a gradient-based bilevel optimization algorithm. HetGPS efficiently searches the parameter space of rewards and successfully rediscovers the theoretically optimal reward regimes in various MARL settings, thereby connecting the abstract theoretical predictions directly to reward design in practical environments5. The results ultimately offer clear criteria for diagnosing and designing missions where agent diversity is essential for maximizing team performance.

### Strengths
1. This paper is well-written and easy to follow. The author provide sufficient supplementary material, making the conclusions of this paper clearer and more convincing.
2. This paper provides a rigorous, formal theory for predicting when behavioral heterogeneity is advantageous in multi-agent task allocation problems. This theoretical framework, based on the curvature of reward aggregation operators (Schur-convexity/concavity), moves the selection of diversity from ad-hoc heuristics to a principled design dimension.
3. The introduction of Heterogeneity Gain Parameter Search (HetGPS) is a significant algorithmic contribution. This gradient-based bilevel optimization method efficiently searches the reward parameter space to find configurations that maximize or minimize the empirical heterogeneity gain.
4. The extensive experiments, ranging from abstract matrix games to complex embodied MARL scenarios (Multi-goal-capture, Tag, Football), successfully demonstrate that the theoretical predictions derived from reward curvature reliably transfer to long-horizon settings. HetGPS further validates the theory by automatically discovering the predicted optimal reward regimes.

### Weaknesses
1. The core theoretical criterion for heterogeneity gain is based solely on the curvature of the reward function (Schur-convexity/concavity). This analysis is inherently restricted to the reward structure and does not formally integrate the complexity of environment dynamics.
2. The high efficiency and tractability of the HetGPS algorithm fundamentally rely on the assumption of an end-to-end differentiable simulator. This is required to compute the exact environment gradients via backpropagation.

### Questions
1. The two issues mentioned in the Weaknesses section.
2. The curves in Figure 4 overlap significantly, making them difficult to distinguish clearly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a theoretical analysis of the heterogeneous vs homogeneous credit assignment in MARL, which provides insights for the reward shaping problem in different MAS tasks. It presents HetGPS to optimize the parameter space of an underspecified MARL environment. The experiments showcase that it can discover new reward regimes to maximize the advantage of heterogeneity.

### Strengths
It is novel to formulate diverse reward allocation choices to a mathematical curvature question via Schur-convex/concave tools. The theorems/constructive counter-examples are clean with explicit assumptions. The algorithm description and experiment analysis are clearly presented. This work has significance in influencing environment/reward design and architecture choices in MARL.

### Weaknesses
1. Results hinge on symmetry/coordinate-wise monotonicity and near constant-sum task scores. It would be good to tabulate common benchmarks that violate these assumptions and provide bounds or heuristics for the reward difference when constant-sum fails.
2. Longer-horizon Dec-POMDP dynamics may interact with curvature in nontrivial ways; more systematic ablations or counterexamples would strengthen the claim
3. Figures all consist of 9 cases, making it difficult to distinguish the lines

### Questions
1. What are the conservative bounds $\Delta R$ when constant-sum is violated?
2. Will there be any cases where curvature is only a part of the total reward yet remains predictive?
3. In non-differentiable simulators, how do sample complexity and final performance compare when using score-function/ES estimators?
4. For non-symmetric or non-monotone aggregators (e.g., capacity/bottleneck constraints), will the results admit first-order approximations or sharp counter-examples?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper investigates aggregation functions in a multi-task/multi-objective multi-agent RL setting, particularly answering the question of for which aggregations heterogeneous behavior is advantageous over homogeneous behavior.
A theoretical analysis shows that this is connected to the schur-convexity of the aggregation functions.
Experiments confirm the theoretical analysis.
Further, a method to optimize environment parameters such that they favor heterogenity is proposed and validated.

### Strengths
* The problem setting is interesting and relevant. The question of whether or not to share parameters in Multi-Agent RL is relevant and the analysis of aggregation functions in this paper provides a useful step towards answering it.
 * The theoretical analysis and its presentation are very clear and well explained
 * The experiments nicely confirm the theoretical predictions.

### Weaknesses
Minor points:
 * The assumption of normalized inner aggregators could be justified better. It's not entirely clear to me whether this is justified in practice

### Questions
*  See question about assumption in weaknesses
 * In HetGPS, it is not entirely clear to me why an approach that alternates between policy and environment improvement was chosen. Intuitively, it could be posed as an entirely bi-level process, in which the policies are trained from scratch for each environment configuration


Nitpicks:
 * L71 It may be better to introduce the symbols T and U for agent-wise and team-wise aggregation here already, on a first read the two addition symbols are a bit unclear.
 * L172 Similar issue, the abuse of notation is initially confusing and (in my opinion) unnecessary
 * L177 Introducing the allocations under the close simplex, i.e. allowing for sum < 1, and then excluding this case in L207 seems unnecessary
 * L351 "continues to be and informative"

Overall I really enjoyed reading this paper, great work!

### Soundness
4

### Presentation
4

### Contribution
3
