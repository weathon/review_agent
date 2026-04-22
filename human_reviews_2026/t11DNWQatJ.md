# The Cell Must Go On: Agar.io for Continual Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Continual reinforcement learning (RL) concerns agents that are expected to learn continually, rather than converge to a policy that is then fixed for evaluation. Such an approach is well suited to environments the agent perceives as changing, which renders any static policy ineffective over time. The few simulators explicitly designed for empirical research in continual RL are often limited in scope or complexity, and it is now common for researchers to modify episodic RL environments by artificially incorporating abrupt task changes during interaction. In this paper, we introduce AgarCL, a research platform for continual RL that allows for a progression of increasingly sophisticated behaviour. AgarCL is based on the game Agar.io, a non-episodic, high-dimensional problem featuring stochastic, ever-evolving dynamics, continuous actions, and partial observability. Additionally, we provide benchmark results reporting the performance of DQN, PPO, and SAC in both the primary, challenging continual RL problem, and across a suite of smaller tasks within AgarCL, each of which isolates aspects of the full environment and allow us to characterize the challenges posed by the different aspects of the game.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors presented a new environment that is suitable for continual RL. It is based on a game and it supports a quite complex amount of states and actions. They benchmarked their environment over standard RL algorithms showing they are struggling to learn fixed policies even over simplified versions of the game.

### Strengths
- The paper is clear, and the description of the environment is detailed enough
- The advantage of the proposed framework over classical RL benchmarks is clear
- The information present in the main paper and in the appendix are complete and exaustive

### Weaknesses
- The relationships between the environment and the continual learning tasks are not stated explicitly until the experimental section
- The audience interested in this environment is really narrow
- The fact that most of the existing CL methods cannot be applied to this setting suggests that the research is currently on a different path than the one proposed by the authors

### Questions
1) You mentioned the time for training with SAC should be contextualized with information on the architecture you used in the main paper. You should also add some information about this also in the main paper.
2) Most of the experimental results are deferred to the appendix. I know there are space constraints, but at least some of the results from the continual learning setting should be included in the main paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces AgarCL, inspired by the Agar.io online game, positioned as an environment for studying continual RL. It features pixel-based observations, hybrid actions, and a non-episodic, long-horizon, partially observable setting. The objective is to grow by collecting pellets and consuming smaller opponents, while avoiding viruses and larger opponents. The authors evaluate standard RL algorithms (DQN, PPO, SAC) and find that none achieve strong performance, highlighting the environment’s difficulty. They also propose a suite of mini-games to isolate sub-challenges such as exploration and mass dynamics.

### Strengths
1. The rewards and hybrid action space (mimicking cursor control with discrete actions) are well designed and align with the gameplay dynamics.
2. The accompanying video provides a clear, intuitive overview of the environment, helping readers unfamiliar with [Agar.io](http://agar.io/) quickly grasp the core mechanics and objectives.
3. The paper is clearly written, visually well-organized, and supported by detailed figures and appendices that make the environment’s design, components, and experiments easy to follow.
4. The authors conduct numerous experiments across different settings and algorithms, providing a thorough empirical assessment and insightful analyses of learning behavior.
5. The introduction of mini-games is a useful contribution. These tasks serve as controlled, diagnostic environments that are useful not only for AgarCL but also for studying general RL behavior and subproblem difficulty.

### Weaknesses
1. **Continual?** The environment is not truly *continual* in the RL sense. In continual RL, the world itself changes while the agent’s policy persists: new objectives emerge, opponent distributions evolve, and goals shift. In AgarCL, the apparent “change” stems entirely from the agent’s own state: as mass increases, movement slows and the field of view expands, altering the interaction dynamics. These effects are endogenous and fully captured by a single stationary MDP, while the transition and observation functions remain fixed. The game is rather *continuing* (non-episodic) but not *continual*: the underlying rules never drift independently of the agent. Consequently, the collapse of frozen policies reflects poor generalization or behavioral distribution shift, not genuine environmental non-stationarity. Likewise, while it is clear that standard RL baselines lack CL capabilities, their low performance only indicates that AgarCL is a complex non-stationary environment. For AgarCL to be a continual RL environment, it would need exogenous drift, e.g., evolving opponent strategies, changing spawn rates, decaying resource yields, or irreversible world modifications. Without such dynamics, the task remains a single stationary environment with long horizons and internal variability. Although, while a great environment for regular RL, it lacks the continual component.
2. **Framing**. The paper implies that environments with gradual, endogenous shifts are inherently more realistic or valuable than those with abrupt task changes. However, I find this claim unconvincing. The relevance of abrupt versus smooth change depends entirely on the application domain. Think of scenarios, such as a warehouse robot deployed in a new facility with unseen layouts. The agent faces sudden distribution shifts, since the warehouse does not gradually evolve into a different one. The robot’s existing policy may fail to generalize, yet retraining from scratch is impractical. It should thus adapt to the new layout without forgetting past ones. The authors’ own observation that few existing continual RL methods work “out of the box” in AgarCL because they are tailored to the sequential tasks setting indicates that this CL setting remains highly relevant.
3. **Baselines**. Despite positioning the work as a continual RL benchmark, no actual CL methods are evaluated. All reported results come from standard RL algorithms (DQN, PPO, SAC). As a result, the paper does not demonstrate whether the environment meaningfully distinguishes CL capabilities.
4. **Deterministic opponents**. AgarCL relies on deterministic, hand-crafted bots, which introduces the risk of overfitting and exploitation rather than genuine learning. RL agents are well known for discovering loopholes with fixed opponents [1]. Although the evaluated RL baselines fail to obtain meaningful performance, given that AgarCL’s opponents are rule-based, the agent may simply learn to exploit their patterns instead of acquiring generalizable strategies. This undermines the claim that performance improvements reflect continual adaptation.
5. **Metrics**. Many continual RL benchmarks evaluate conventional CL metrics such as transfer and forgetting [2, 3, 4, 5, 6]. In contrast, this paper reports only cumulative reward as the primary metric.
6. **Derivative Design**. Even if the work were positioned purely as a new RL environment rather than continual, its contribution would not be groundbreaking, as there are existing Agar.io-style implementations for RL [7, 8, 9]. While the mini-games are useful tasks themselves, the pixel observations provide a new layer of complexity, and the notable simulation speed-up can reduce the runtime burden, the core mechanics and dynamics remain largely unchanged.

### Minor points

1. Using smoothing would improve the readability of Figures 4, 5, 16, and 17.
2. The frame skip is set to 4 in the environment specifications. This should be mentioned only in the experiments section, since this is generally up to the user to define.

[1] Delfosse, Quentin, et al. "Deep Reinforcement Learning Agents are not even close to Human Intelligence." *arXiv preprint arXiv:2505.21731* (2025).

[2] Powers, Sam, et al. "Cora: Benchmarks, baselines, and metrics as a platform for continual reinforcement learning agents." *Conference on Lifelong Learning Agents*. PMLR, 2022.

[3] Tomilin, Tristan, et al. "Coom: A game benchmark for continual reinforcement learning." *Advances in Neural Information Processing Systems* 36 (2023): 67794-67832.

[4] Johnson, Erik C., et al. "L2explorer: A lifelong reinforcement learning assessment environment." *arXiv preprint arXiv:2203.07454* (2022).

[5] Wołczyk, Maciej, et al. "Continual world: A robotic benchmark for continual reinforcement learning." *Advances in Neural Information Processing Systems* 34 (2021): 28496-28510.

[6] Tomilin, Tristan, et al. "MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning." *arXiv preprint arXiv:2506.14990* (2025).

[7] Zhang, Ming, et al. "Gobigger: A scalable platform for cooperative-competitive multi-agent interactive simulation." *The Eleventh International Conference on Learning Representations*. 2023.

[8] Ansó, Nil, et al. "Deep reinforcement learning for pellet eating in agar. IO." *The 11th International Conference on Agents and Artificial Intelligence*. SciTePress, 2019.

[9] Wiehe, Anton Orell, et al. "Sampled policy gradient for learning to play the game Agar. io." *arXiv preprint arXiv:1809.05763* (2018).

### Questions
1. Can pretrained policies be used as opponents?
2. How does the agent perceive the area outside the outer wall of the environment with the pixel-based observations? Are the pixels outside the area padded with some value?
3. Why do the pixel observations have separate channels for in-game objects? Is it necessary to semantically separate the input rather than use the RGB channels that a human player would see?

### Soundness
2

### Presentation
4

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
The paper introduces AgarCL, a new benchmark environment for continual reinforcement learning (RL) based on the game Agar.io. Unlike traditional episodic RL tasks, AgarCL is non-episodic, partially observable, and non-stationary, requiring agents to adapt continuously as the environment evolves.
AgarCL features hybrid actions (continuous movement + discrete split/eject), pixel-based observations, and mass-based rewards, creating smooth but persistent changes in dynamics as agents grow or shrink. The authors benchmark DQN, PPO, and SAC, showing that none can learn stable or effective policies in the full environment.
They also design mini-games isolating specific challenges (exploration, credit assignment, non-stationarity) and show PPO performs best but still struggles. Fixed policies degrade over time, demonstrating the need for continual adaptation.

### Strengths
The paper presents a clear and well-structured formalization of the problem, with solid motivation and coherent methodology. The presentation is generally good, and the theoretical framing is appealing

### Weaknesses
The contribution lacks strong novelty, as it mostly adapts an existing game setup rather than introducing new concepts. The analysis of opponent policies could be expanded, for example by addressing non-stationary behaviors. While the focus is not on continual learning (CL), it would be valuable to highlight the method’s compatibility with existing CL frameworks. Finally, a few figures and tables would benefit from clearer legends for better readability.

### Questions
While the work is not explicitly about continual learning, do you plan to make AgarCL compatible with existing continual learning frameworks or benchmarks (e.g., through defined task boundaries, curriculum setups, or standardized evaluation metrics)?

Do you plan to include or analyze non-stationary opponent behaviors to better reflect continual adaptation challenges and multi-agent dynamics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript introduces AgarCL, a continual reinforcement learning research platform derived from the Agar.io game. The environment is non-episodic, high-dimensional, partially observable, and features continuous actions and endogenous non-stationarity due to evolving dynamics and other agents. The authors position AgarCL as a testbed that avoids artificial task switches commonly used to induce non-stationarity in episodic benchmarks. They provide baseline results for DQN, PPO, and SAC on both the full environment and a suite of mini-games designed to isolate specific challenges. The results suggest that fixed policies struggle to maintain stable performance and that standard deep RL methods face considerable difficulty in this setting. The manuscript emphasizes problem formulation and environment contribution over new algorithmic solutions, and acknowledges substantial computational demands

### Strengths
1. The manuscript targets a gap in CRL benchmarks by providing a non-episodic environment with smooth endogenous non-stationarity, moving beyond artificial task switches common in prior work.
2. The environment’s combination of partial observability, continuous control, high-dimensional observations, and potentially infinite horizon is well aligned with realistic continual learning challenges.
3. The manuscript provides a detailed description of the platform and its dynamics, which can help researchers understand and instrument experiments in this setting.
4. Due to well-engineered and accessible, AgarCL may serve as a common platform that encourages standardized evaluation of RL approaches in non-episodic settings.

### Weaknesses
1. The originality of the platform is limited, as it largely adapts Agar.io for RL without clear evidence of novel environment design beyond configuration.
2. The evaluation does not include algorithms specifically designed for CRL (or methods targeting non-stationarity, online adaptation, memory consolidation, or meta-learning), making it hard to assess whether the environment differentiates among approaches intended for this setting.
3. The baselines (DQN, PPO, SAC) are standard and not state-of-the-art for the reported setup; tuning procedures and fairness across methods are insufficiently detailed, and the results provide limited actionable guidance for algorithm development.
4. The manuscript’s positioning of AgarCL as a CRL benchmark is blurred by extensive experiments in non-continual configurations and by the strong influence of other agents; it risks being better framed as a multi-agent platform without providing corresponding multi-agent protocols or analyses.
5. The structure reads more like a technical report than a concise research manuscript; core contributions and key takeaways are not sharply distilled, and many content appears relegated to appendices without synthesis in the main text.
6. The evaluation protocol lacks standard continual learning metrics (e.g., forgetting, forward/backward transfer, stability–plasticity trade-offs) and does not establish clear, reproducible benchmarks for long-horizon continual performance.
7. Practical considerations (compute requirements, scalability, parallelization, runtime) are acknowledged but not resolved; heavy resource demands limit accessibility and may impede community adoption.
8. Although the authors deserve credit for their work in developing this platform, the manuscript does not contribute new knowledge and sufficient value to the ICLR community. This is perhaps the wrong venue for this work. The contribution is primarily infrastructural and may be better suited to a specialized track.

### Questions
Please refer to the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2
