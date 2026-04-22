# MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 4

## Abstract
Building general-purpose graphical user interface (GUI) agents has become increasingly promising with the progress in vision language models. However, developing effective mobile GUI agents with reinforcement learning (RL) remains challenging due to the heavy-tailed distribution of task difficulty and the inefficiency of large-scale environment sampling. We present an online agentic reinforcement learning framework MOBILERL to enhance GUI agents in mobile environments. Its core component is the Difficulty-Adaptive GRPO (ADAGRPO) algorithm. In ADAGRPO, we design difficulty-adaptive positive replay and failure curriculum filtering to adapt the model to different task difficulties. We introduce the shortest-path reward adjustment strategy to reshape rewards concerning the task length in multi-turn agentic tasks. Those strategies jointly stabilize RL training, improve sample efficiency, and generate strong performance across diverse mobile apps and tasks. We apply MOBILERL to two open models (Qwen2.5-VL-7B-Instruct and GLM-4.1V-9B-Base). The resultant MOBILERL-9B model achieves state-of-the-art results in terms of success rates on both AndroidWorld (80.2%) and Android-Lab (53.6%). The MOBILERL is open-sourced at https://github.com/THUDM/MobileRL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents MobileRL, a framework for training mobile GUI agents using online reinforcement learning that addresses three key challenges: complex instruction following under sparse rewards, heavy-tailed task difficulty distributions, and sampling bottlenecks in large-scale mobile environments. The framework consists of a two-stage warm-up (reasoning-free SFT followed by reasoning SFT) and the ADAGRPO algorithm, which extends GRPO with three novel strategies: Shortest-Path Reward Adjustment (SPA) that rewards shorter successful trajectories, Difficulty-Adaptive Positive Replay (AdaPR) that maintains a buffer of challenging successful trajectories, and Failure Curriculum Filtering (FCF) that down-weights persistently unsolvable tasks. MobileRL-9B (based on GLM-4.1V-9B-Base) achieves good results.

### Strengths
- Well-motivated approach: The paper clearly identifies real challenges in mobile GUI agent training (sparse rewards, heavy-tailed difficulty, sampling inefficiency) and proposes targeted solutions.

- Strong empirical results: Substantial improvements over prior work across multiple benchmarks (e.g., +15.8% on AndroidWorld, +12.4% on AndroidLab compared to previous best).

- Comprehensive ablation studies: The paper systematically evaluates each component (reasoning-free SFT, reasoning SFT, and each part of ADAGRPO), demonstrating their individual contributions.

### Weaknesses
- Dependence on reward model for AndroidLab: The AndroidLab training relies on a VLM-based reward model with only 86% accuracy, introducing potential bias. The paper acknowledges this but doesn't deeply explore its impact. The smoothness difference in training curves (Figure 11) suggests this is a significant limitation.

- Limited evaluation on out-of-domain tasks: All benchmarks (AndroidWorld, AndroidLab, AndroidControl) use similar Android apps. Generalization to truly novel apps or real-world public applications is not demonstrated.

### Questions
1. Circularity in Complexity Measurement

> Question: The paper defines task complexity based on the number of steps taken, but this creates a circular dependency problem. If an agent gets stuck or takes inefficient paths, does a simple task become "complex"? Conversely, if an agent finds a shortcut, does a complex task become "simple"?
> Whose trajectories define the step count - expert demonstrations, the agent being evaluated, or averaged across all agents?
If different agents take vastly different numbers of steps for the same task, how is complexity consistently defined?


2. Confusion Between Task Difficulty and Solution Length

> Question: Steps taken seems to conflate intrinsic task difficulty with solution length. A task requiring 25 careful, straightforward steps (e.g., filling a long form) might be less cognitively complex than a 10-step task requiring creative problem-solving or error recovery. How do you justify steps as a proxy for complexity? Consider requesting:
> 1. Analysis using alternative complexity metrics (e.g., number of different apps involved, branching factor in state space, required reasoning depth)
2. Breakdown of what actually makes tasks "complex" beyond length

3. Technical Questions

- 3.1 SPA mechanism: How sensitive is performance to the choice of α in Equation 2? Have you explored other reward shaping functions beyond the linear penalty? Does SPA risk encourage premature termination (early "Finish" actions)?
- 3.2 AdaPR buffer dynamics: How does the replay buffer composition evolve during training? What percentage of training data comes from replay vs. on-policy sampling at different stages? Is there a risk of overfitting to replayed trajectories?
- 3.3 FCF design choices: The three-epoch cooldown seems arbitrary. How was this chosen? Have you compared against more sophisticated curriculum learning approaches (e.g., prioritized sampling based on learning progress)?
- 3.4 Reward model bias: Can you quantify the impact of the 86% accuracy reward model on AndroidLab? What types of errors does it make, and how do these propagate through RL training?

### Soundness
3

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
This paper introduces MobileRL, a framework for training mobile GUI agents through online reinforcement learning. The authors combine several mechanisms, Shortest-Path Reward Adjustment (SPA), Difficulty-Adaptive Positive Replay (AdaPR), and Failure Curriculum Filtering (FCF), under an algorithm called ADAGRPO, and deploy it in a large-scale distributed Android simulation environment. Experiments on AndroidWorld and AndroidLab show clear gains over previous baselines.

### Strengths
1. The problem setting is realistic and important. Training interactive mobile agents remains difficult, and the authors provide a complete end-to-end RL pipeline that improves performance.
2. The engineering effort is great. Building a distributed Android environment with scalable parallel sampling is non-trivial.

### Weaknesses
1. Overall, while the paper addresses a practically interesting problem, I find its academic contribution is relatively limited. Most of the proposed techniques have clear analogues in existing RL literature, trajectory length penalty, prioritized replay, and curriculum learning, and are mainly repurposed for the GUI agent setting. From a conceptual standpoint, the paper’s novelty lies more in the system design and implementation scale than in introducing fundamentally new RL ideas or algorithms.
2. I have some reservations about the mixture sampling strategy. The historical “successful” trajectories stored in the buffer are highly off-policy, while the underlying training objective (a clipped PPO-style GRPO variant) is fundamentally on-policy. As ADAGRPO employs a clipping mechanism, these off-policy samples may often fall outside the acceptable distribution range and thus get heavily clipped, thereby limiting their effective gradient contribution. This raises a question about sample efficiency and the real utility of replayed data. It would be helpful if the authors could provide some experiments to show the clip ratio of the off-policy samples or effective sample contribution.
3. There is no training recipe in the provided codebase, which is a core component of the proposed framework. Did I miss it?

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors introduce MobileRL, an online agentic reinforcement learning framework for mobile GUI control. They highlight large-scale environmental interaction and heavy-tailed task difficulty as current challenges in scaling up RL for GUI control. To address this, they build on top of GRPO with three main components: (1) off-policy sampling from a buffer of previously successful trajectories, (2) a curriculum that downweighs and filters out low-success rate tasks, and (3) reward shaping to encourage a shortest-path solution. After performing SFT using reasoning and non-reasoning trajectories, they apply MobileRL to achieve, at the time of writing, the state of the art in native model (non-agent-system-based) GUI control.

### Strengths
Originality: To the best of my knowledge, prior work has not applied oversampling of successful trajectories, reward shaping to encourage shortest paths, and curriculum learning to the GUI and multi-turn agent-LLM setting. The reward shaping is particularly interesting, since encouraging shortest paths has been underexplored in this setting, and the specific formulation is a good contribution.

Quality: The work is of high quality. The core contributions are sufficiently explained, and the experimental analysis is comprehensive and supportive.

Clarity: The work is generally very clear.

Significance: Given the rapid extension of LLMs to agentic and multi-turn settings, and the growing interest in the GUI setting, this work is very significant. This is especially evident in Android World, which, at the time of writing, is SOTA for a native model (non-agentic system). Open-sourcing the RL codebase is also a significant contribution to open research practices, although the current link is dead; this should be rectified in the rebuttal.

### Weaknesses
Originality: It could be made more evident how the specific components have been explored in other domains using LLMs, RL, and multi-turn interaction. For example, it is now commonplace to avoid vanishing learning signal through sampling strategies (e.g. DAPO). Similarly, curriculum learning and reward shaping exist in other non-LLM RL settings. It would be helpful to clarify whether the specific contributions, such as the SPA reward, are present in different settings.

Quality: Some experiments and conclusions may be slightly misleading, possibly due to a lack of clarity at points, as I explain below.

Clarity: Some aspects seek clarification, which I expand on in the questions section below.

Significance: While the Android World results are significant, I encourage the authors to clarify that it is SOTA for native agents, but not all agentic systems (see the current leaderboard.

### Questions
1.	The problem formulation (line 150) implies that the agent is provided with no form of memory, only the most recent screenshot. The results seem surprising for this setting. Can the authors please clarify that this is correct?
2.	In line 225, could you please clarify whether the SPA reward is still broadcast to every timestep?
3.	In Figure 3b, are the trajectory-level rewards before or after the SPA penalty has been applied?
4.	In Figure 3b, could the authors provide some intuition on why the MobileRL w/o SPA & AdaPR collapses? Specifically, why does only the combination of these two components prevent collapse? Intuitively, introducing oversampling of positive trajectories would avert collapse. But it is less intuitive that the SPA reward (in isolation from oversampling) would prevent collapse, since this reward changes the learning problem but not necessarily the dynamics.
5.	Can the authors please clarify whether difficult tasks are only removed for training, but still included for testing? Naturally, there would be instant gains in reward from removing tasks.
6.	Could the authors please clarify whether MobileRL w/o FCF includes AdaPR or SPA?
7.	It is unsurprising that the reward ceiling for MobileRL w/o FCF is lower, since low-reward tasks have been removed. However, why does the curve in Figure 3b start so much lower? It is my understanding that tasks are removed throughout training after at least two epochs (line 260), which suggests that all lines in the figure should start with the same reward.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an online agentic reinforcement learning framework named MOBILERL, designed to enhance mobile GUI agents. The framework aims to solve challenges in developing GUI agents with RL, such as the heavy-tailed distribution of task difficulty and inefficient large-scale environment sampling. The core of MOBILERL is its Difficulty-ADAptive GRPO (ADAGRPO) algorithm. This algorithm stabilizes training and improves sample efficiency through three strategies including Difficulty-Adaptive Positive Replay (AdaPR), Failure Curriculum Filtering (FCF), and Shortest-Path Reward Adjustment (SPA).

### Strengths
ADAGRPO improves training stability and efficiency through three key strategies : 1) Difficulty-Adaptive Positive Replay (AdaPR) to amplify the learning signal from rare, informative successful trajectories; 2) Failure Curriculum Filtering (FCF) to optimize computational budget by dynamically down-weighting or removing persistently unsolvable tasks ; and 3) Shortest-Path Reward Adjustment (SPA), which uses reward shaping to encourage more efficient (shorter) solutions. The framework's effectiveness is thoroughly validated on two major benchmarks, AndroidWorld and AndroidLab, where the MOBILERL-9B model achieves state-of-the-art success rates of 80.2% and 53.6%, respectively, significantly outperforming previous results.

### Weaknesses
1. I question the generality of the MobileRL-9B model's SOTA results. Appendix F and D.1 state the best 9B model on AndroidWorld was trained "purely on the AndroidWorld environment". This raises doubts about whether the 53.6% score on AndroidLab in Table 1  came from the same model. Attributing SOTA scores from different training strategies to a single model name (MobileRL-9B) may be inconsistent.

2. there is no testing on "out-of-domain" public-use apps.

### Questions
1. Given that MobileRL uses both images and XML as input, did the baseline models also utilize XML?

2. Appendices F and D.1 seem to indicate that the 80.2% score on AndroidWorld was achieved by a model trained only on that dataset. Does this mean the 53.6% score on AndroidLab came from a separate model using a different training strategy?

3. The FCF strategy permanently removes tasks with consecutive failures from the training pool. Could this distort the training objective, leading the model to achieve a high success rate simply by "giving up" on difficult tasks and optimizing for the remaining, simpler task distribution?

### Soundness
3

### Presentation
3

### Contribution
3
