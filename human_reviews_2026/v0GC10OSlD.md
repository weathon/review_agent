# Learning Reward Functions for Cooperative Resilience in Multi-Agent Systems

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Multi-agent systems often operate in dynamic and uncertain environments, where agents must not only pursue individual goals but also safeguard collective functionality. This challenge is especially acute in mixed-motive multi-agent systems. This work focuses on **cooperative resilience**—the ability of agents to anticipate, resist, recover, and transform in the face of disruptions—a critical yet underexplored property in Multi-Agent Reinforcement Learning. We study how reward function design influences resilience in mixed-motive settings and introduce a novel framework that learns reward functions from ranked trajectories, guided by a cooperative resilience metric. Agents are trained in a suite of social dilemma environments using three reward strategies: (i) traditional individual reward; (ii) resilience-inferred reward; and (iii) hybrid that balance both. We explore three reward parameterizations—linear models, hand-crafted features, and neural networks—and employ two preference-based learning algorithms to infer rewards from behavioral rankings. Our results demonstrate that hybrid strategy significantly improve robustness under disruptions without degrading task performance and reduce catastrophic outcomes like resource overuse. These findings underscore the importance of reward design in fostering resilient cooperation, and represent a step toward developing robust multi-agent systems capable of sustaining cooperation in uncertain environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on reward or mechanism design for the mixed-motive game Harvest, where two agents can collect apples with a state- and location-dependent regrowth rate. It proposes a reward learning method based on preference-based inverse reinforcement learning (IRL), consisting of two steps:
1. The experience trajectories are ranked w.r.t. a cooperative resilience metric, which is specifically designed for the Harvest domain, including the number of consumed apples, available apples, inequality of consumption, etc.
2. A reward function is learned based on the ranking using preference-based learning via handcrafted features, linear function, and neural networks, as approximation variants.

The approach is evaluated on small instances of the Harvest domain, consisting of an 8x8 grid and 2 agents (or 16x16 and 4 agents), and compared with baselines, such as a Random policy, PPO, and QMIX.

### Strengths
The paper has a clear focus on reward design for mixed-motive games. It is mostly well-written and easy to follow.

The proposed method is interesting and novel.

### Weaknesses
**Novelty**

While the proposed method seems novel, the paper misses important discussions (and experimental comparisons) about prior work, which also shapes the behavior of self-interested agents via rewards [1,2,3,4,5].

**Soundness**

There are some flaws in its definition and assumptions:
- The paper formulates the problem as an MDP, despite focusing on multi-agent settings. It is unclear if the MDP refers to a single agent (which would be flawed due to violating the Markov property [6]) or the joint multi-agent view (which would represent a multi-agent MDP or MMDP [7]).
- The paper assumes full observability, contradicting the original setting of Harvest, which is defined as a partially observable stochastic game [8,9].
- The paper assumes information sharing between the agents, e.g., the experience trajectories containing the joint actions, which is unrealistic in mixed-motive scenarios, where **(1)** agents are assumed to be independent [8,9,10], and **(2)** global and perfect communication is required, thus limiting scalability.

The proposed method is specifically designed for the Harvest domain, as the cooperative resilience metric consists of four indicators (the paper states *"five"* but I could not find the fifth one):
1. Cumulative consumption of apples
2. Resource/Apple availability
3. Gini index (distribution of apple consumption)
4. Hunger index (number of time steps without any consumption)

I wonder how these metrics would map to other common social dilemmas, such as Cleanup, Public Goods, Coin, and Wolfpack.

To further strengthen the contribution, a theoretical analysis would have been helpful, e.g., where the concept is shown to work in the iterated prisoner's dilemma.

**Significance**

The introduction of the paper puts a strong emphasis on environmental disruptions. However, throughout the paper and experimental section, I do not find such disruptions that would validate the claims regarding cooperative resilience (the metric is reported in the experiments, but I am unaware of the actual disruptions, if any). I recommend testing variations, e.g., where the regrowth rate of apples is varied [8,9] or communication channels are noisy [5]. Without such an evaluation, I cannot confirm its significance.

The experiments are somewhat preliminary, as they focus on very small instances of the Harvest domain with four agents at most. Prior work evaluates with 8-12 agents in the Harvest domain [1,5,8], as well as other domains, such as Cleanup or Coin [2].

Due to the lack of scaling and variety of test environments, I am concerned about the generality and scalability, and consequently, the broader relevance of the approach.

**Literature**

[1] Lupu et al., "Gifting in Multi-Agent Reinforcement Learning", AAMAS-20

[2] Yang et al., "Learning to Incentivize Other Learning Agents", NeurIPS-20

[3] Schmid et al., "Stochastic Market Games", IJCAI-21

[4] Vinitsky et al., "A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings", Collective Intelligence 2023

[5] Phan et al., "Emergent Cooperation from Mutual Acknowledgment Exchange in Multi-Agent Reinforcement Learning", JAAMAS-24

[6] Littman et al., "Markov Games as a Framework for Multi-Agent Reinforcement Learning", ICML-94

[7] Boutilier et al., "Planning, Learning and Coordination in Multiagent Decision Processes", TARK-96

[8] Perolat et al., "A multi-agent reinforcement learning model of common-pool resource appropriation", NeurIPS-17

[9] Leibo et al., "Multi-agent Reinforcement Learning in Sequential Social Dilemmas", AAMAS-17

[10] Foerster et al., "Learning Opponent-Learning Awareness", AAMAS-18

### Questions
Regarding the cooperative resilience metric: Cumulative consumption is potentially unbounded, whereas the Gini index has strict bounds. How does the metric ensure that these indicators are weighted fairly?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a preference-based inverse reinforcement learning (IRL) framework aimed at discovering reward functions that promote cooperative resilience in multi-agent systems. The approach first quantifies resilience for full trajectories using indicators such as consumption, resource availability, inequality, and hunger. Trajectories are ranked by these resilience scores, and pairwise preferences are used to train reward models via either a margin-based or probabilistic preference-learning objective. The learned rewards are then used—alone or in combination with standard individual rewards—to train PPO-based multi-agent policies. Experiments on Commons-Harvest–style environments and a larger 16×16 scenario show that hybrid rewards reduce resource depletion and increase sustainability compared with Random, PPO, and QMIX baselines.

### Strengths
The paper’s main strengths are its clear and timely problem framing—learning reward functions that encode cooperative resilience rather than handcrafting sustainability terms—coupled with a well-structured pipeline that converts trajectory-level resilience scores into pairwise preferences and trains reward models (margin-based or probabilistic) that plug seamlessly into standard MARL; it explores multiple reward parameterizations (handcrafted, linear, neural) and a practical hybrid objective (individual + resilience) that consistently improves long-horizon outcomes such as reduced depletion and longer sustainable operation; the environment setup and evaluation protocol are described with care, including statistical testing and ablations on preference generation strategies, and the presentation emphasizes reproducibility with implementation details and organized appendices, making the contribution both conceptually meaningful and practically usable by the community.

### Weaknesses
1. The resilience metric is manually constructed as a harmonic mean over several indicators with fixed weights and failure/recovery windows. The paper does not study how these design choices affect trajectory rankings or learned rewards. Because the metric defines the training signal, its sensitivity is a critical missing analysis.

2. Rewards are learned from single-shock episodes (one disruption at step 500) but tested on triple-shock long runs. The authors claim generalization to unseen disruptions, yet they never quantify how well ranking-based rewards transfer to new disturbance regimes. This leaves it unclear whether improvements stem from genuine resilience learning or from overfitting to the training scenario.

3. Only Random, PPO, and a lightly tuned QMIX variant are compared. More recent cooperative MARL algorithms(MAPPO,HAPPO,COMA) or better-tuned decomposers(QTRAN,QPLEX,VDN) could provide stronger baselines. Without tuning or wall-clock comparisons, improvements may partly reflect hyper-parameter asymmetry rather than reward-learning advantages.

4. The trajectory pairs come from random policies; possible noise, transitivity violations, or ranking bias are not analyzed. Since both MPL and PPL depend on clean ordinal information, unverified ranking noise could distort the learned reward landscape.

5. The larger-scale 16×16 experiment shows mixed or statistically weak resilience gains, and only 50 evaluation episodes are run. The paper also claims interpretability for handcrafted or linear rewards but does not show what the learned weights actually represent or how they correlate with indicators.

### Questions
1. How sensitive are results to indicator selection, normalization, and harmonic aggregation?

2. How does performance change if the disruption frequency or type differs between training and testing?

3. What are the exact tuning budgets and compute times for PPO, QMIX, and the preference-learning stages?

4. How stable are learned rewards when preference noise or inconsistent pair rankings are introduced?

5. In the larger-scale setting, which element—preference model, reward parameterization, or hybridization—drives the observed gains?

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
This paper proposes a framework a model-free approach to promoting group resilience by learning reward functions that encourage cooperative resilience in multi-agent systems facing disruptions. The authors define a cooperative resilience metric that combines several system-level indicators (resource availability, equality, hunger, and sustainability) and use it to rank trajectories. These rankings are then used in a preference-based inverse reinforcement learning setup (both margin-based and probabilistic) to infer a reward function that favors resilient behavior. Agents trained with this learned reward are shown to sustain shared resources and recover more effectively after disruptions in a Commons-Harvest gridworld. The paper argues that this method generalizes beyond the specific environment and can be combined with any MARL algorithm to automatically shape cooperative incentives.


The paper’s motivation is strong and the conceptual framing is original, but the evidence provided doesn’t convincingly support the central claim that the system learns resilience rather than merely overfitting a predefined metric. The use of the same resilience score for both supervision and evaluation introduces a causal ambiguity that seriously limits interpretability. Combined with noisy ranking data from random trajectories, narrow metrics, limited environment diversity, and unorthodox PPO configurations, the current results feel preliminary. The hybrid reward setup also weakens the narrative that this approach automates reward design. For these reasons, I would lean to reject. The idea is promising, but it needs stronger experimental grounding, more rigorous validation, and better evidence that the learned reward captures generalizable, causal resilience.

### Strengths
- Clear and coherent methodology. The proposed pipeline of ranking trajectories, learning preferences, and inferring rewards is logically structured and mathematically sound.
- Novel IRL formulation for resilience.
- Reproducibility. The paper provides detailed appendices, configurations, and discusses reproducibility assets and ethical considerations, which is commendable.
- Practical potential. The idea of learning system-level incentives from ranked behaviors could, in principle, be applied in many cooperative domains where manual reward design is difficult.

### Weaknesses
- Metric-evaluation circularity. The same cooperative-resilience metric used for ranking trajectories is also used to evaluate success. This makes it impossible to tell whether agents actually learned to be resilient or simply optimized the evaluator. The fact that disruptions occur at the same fixed timestep in both training and testing further amplifies this problem.

- Uninformative supervision data. Rankings are generated from random-policy trajectories, which are likely dominated by stochastic noise rather than meaningful cooperation. There is no noise or variance analysis to show that the ranking signal is informative.

- Narrow evaluation focus. Despite defining multiple indicators, the main evaluation metric is “last-apple consumption,” which reflects only sustainability. Other aspects of resilience, like recovery, fairness, and stability, are buried in the appendix.

- Limited generalization and overstated claims. The experiments are confined to a two-agent discrete gridworld with PPO as the sole training algorithm. The “method-agnostic” claim is not empirically demonstrated.

- Missing related work on group resilience.  Shraga et al. Collaboration Promotes Group Resilience in Multi-Agent RL. - RLC 2025.

### Questions
- How do you justify metric–evaluation coupling? 

- What were the variance results over resilience scores across seeds.

- How does this work relate to Shraga et al 2025 ?

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
3

### Summary
The paper tackles the significant challenge of designing reward functions for Multi-Agent Reinforcement Learning (MARL) that foster cooperative resilience in mixed-motive environments subject to disruption. Traditional reward structures often fail here, promoting individual gains at the expense of collective system persistence. The authors introduce a framework leveraging Inverse Reinforcement Learning (IRL) to infer a collective reward component directly from trajectories ranked by a cooperative resilience metric.
The methodology employs preference-based IRL (MPL and PPL) to learn rewards parameterized by handcrafted features, linear models, or neural networks. The central experimental result confirms that a hybrid reward strategy—which balances the learned resilience reward with standard individual consumption incentives—significantly improves robustness. Tested in a Commons Harvest social dilemma under a comprehensive, generalized disruption protocol, the hybrid strategy achieved higher cooperative resilience, extended system sustainability, and drastically mitigated catastrophic resource depletion events (last-apple consumption dropped to 13.2% in testing, compared to over 60% for baselines). The framework is presented as a general, method-agnostic approach to reward design that complements existing MARL algorithms.

### Strengths
- Originality

The primary original contribution is the robust methodology for grounding reward inference in a quantitative, system-level metric of cooperative resilience. While Inverse Reinforcement Learning (IRL) is not new, using preference-based IRL (MPL/PPL) derived from trajectories ranked explicitly by their recovery and failure profiles under stress is a novel application pathway for incentive design in MARL. This approach circumvents the conventional IRL dependence on near-optimal expert demonstrations by utilizing quantitative rankings derived from a resilience score, which captures the complex dynamic, temporal, and distributed nature of recovery under disruption. The paper successfully operationalizes the abstract concept of resilience (anticipating, resisting, recovering, and transforming) into a practical learning signal. Crucially, the introduction of the hybrid strategy demonstrates an important, original insight into practical reward alignment in mixed-motive settings. The result is an emergent reward function that leads to specialized, non-overlapping spatial behaviors (one agent exploring, one anchoring/harvesting) that maximize joint welfare, illustrating the power of this incentive structure beyond simple aggregated consumption maximization. This framework provides a structured means to inject long-term persistence goals into existing MARL systems.


- Quality

The authors tested 27 configurations across two learning algorithms, various sampling strategies, and three reward parameterizations. Critically, the reward inference used trajectories generated under a single resource removal disruption, while the final evaluation used a complex protocol featuring three distinct, temporally separated disruptions: resource removal, regrowth rate reduction, and agent failure simulation. This methodology provides strong evidence that the learned rewards result in generalized robustness rather than overfitting to a single failure mode present in the training data. The results are quantified using statistical tests (Mann–Whitney U test with Benjamini–Hochberg correction), confirming that the hybrid strategy significantly outperforms Random, standard PPO, and QMIX in terms of cumulative consumption and episode length. Furthermore, the paper includes a non-trivial scalability test in a larger 16x16 environment with four agents and permanent resource depletion, confirming the core benefits persist in increased complexity. The transparency provided by interpreting the learned weights for the best-performing handcrafted model (Section B) significantly enhances the quality of the analysis, offering clear causality between incentive structure and emergent cooperative behaviors (e.g., incentivizing proximity distance while rewarding local density).


- Clarity

The technical presentation is straightforward and professional. The problem formulation, the definition of the MDP, and the two-step methodology (ranking trajectories, then learning the reward via optimization/probabilistic modeling) are clearly laid out. The mathematical structure defining the resilience score (using failure and recovery profiles via integrals and the harmonic mean aggregation) is detailed enough to follow the underlying mechanism.


- Significance

The paper addresses a fundamental limitation in MARL: how to automatically design incentives for long-term collective welfare under uncertainty. The success of the hybrid strategy demonstrates that resilience and individual productivity are not necessarily a zero-sum game; the method simultaneously achieves the highest average consumption and the lowest social dilemma failure rate (13.2% last-apple consumption) compared to baselines. This is a powerful proof of concept for applications in domains like environmental resource management or decentralized infrastructure control.

### Weaknesses
- Quality

The experimental quality suffers from weaknesses in the baseline selection and the dependence on parameterization. The paper compares performance against basic PPO and QMIX and explicitly notes the omission of more recent, high-performing cooperative MARL algorithms. This leaves a significant open question regarding the necessity of the complex two-stage IRL process compared to simpler, modern reward shaping or decentralized planning techniques. Furthermore, the QMIX baseline required a 10x manual reward increase (+10 instead of +1) just to prevent agents from converging to suboptimal areas, suggesting the baseline policies were highly fragile and perhaps not optimally representative of competitive cooperative MARL standards. The overwhelming success of the Handcrafted parameterization compared to the Linear and Neural Network models (e.g., Last Apple % of 1.75% for MPL-M1 Handcrafted vs. 8.75% for MPL-K1 NN Hybrid, Table A.5 vs A.7) suggests a severe limitation in the generalization capacity of the learned models when forced to work from raw state inputs. This heavily implies that the quality of the system’s performance is primarily driven by the expert’s choice of six input features, rather than the ability of the preference learning pipeline to automatically identify resilience-aligned signals in complex, non-linear state spaces.

- Clarity

the paper does not fully clarify the mechanisms behind the selection of the best performing configuration (Handcrafted MPL-M1 Hybrid) over other high-performing variants, particularly the PPL models which often yielded comparable resilience scores with drastically different average rewards (e.g., Table A.2). This suggests that the nuanced trade-offs captured by margin definition and sampling strategy were not fully analyzed or clearly presented.

### Questions
1. The resilience metric calculation relies on defining the time of worst degradation ($t_f$) and the recovery endpoint ($t_r$). Given that five system indicators are tracked (e.g., consumption, Gini index, resource availability), which specific indicator defined $t_f$ and $t_r$ in practice, or was a rule established based on the harmonic mean score itself to estimate these critical integration points for trajectory ranking?

2. The Handcrafted model was significantly more successful than the Linear and Neural Network parameterizations in achieving optimal resilience and low selfishness (1.75% last-apple consumption for Handcrafted MPL-M1 Hybrid). Does this heavy reliance on six expert-designed features mean the method fundamentally requires high-quality domain expertise, or is it expected that non-linear models would outperform the handcrafted features if given more training data or different architectures?

### Soundness
3

### Presentation
3

### Contribution
3
