# Multi-agent cooperation through learning-aware policy gradients

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8, 5, 6

## Abstract
Self-interested individuals often fail to cooperate, posing a fundamental challenge for multi-agent learning. How can we achieve cooperation among self-interested, independent learning agents? Promising recent work has shown that in certain tasks cooperation can be established between ``learning-aware" agents who model the learning dynamics of each other. Here, we present the first unbiased, higher-derivative-free policy gradient algorithm for learning-aware reinforcement learning, which takes into account that other agents are themselves learning through trial and error based on multiple noisy trials. We then leverage efficient sequence models to condition behavior on long observation histories that contain traces of the learning dynamics of other agents. Training long-context policies with our algorithm leads to cooperative behavior and high returns on standard social dilemmas, including a challenging environment where temporally-extended action coordination is required. Finally, we derive from the iterated prisoner's dilemma a novel explanation for how and when cooperation arises among self-interested learning-aware agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a new paradigm for MARL that enables agents to be aware of inter-episode dynamics and presents compelling experimental results in both the iterated prisoner’s dilemma and a grid-world sequential social dilemma.

### Strengths
- This work addresses a central challenge in MARL.
- The proposed framework (Figures 1 and 2) is important.
- The method parameters are thoroughly detailed in the appendix.
- Finding 1 is especially intriguing.

### Weaknesses
- Line 114: The metric used in this work is ambiguous. Could you clarify which metric is applied in subsequent sections? What expectations should we have to claim that the proposed method outperforms other algorithms?
	- In Line 485, it appears the objective is solely the meta-agent's reward rather than equilibrium. Can we then say that this work is aimed at developing an algorithm to exploit naive agents?
	- Or is this an agent-based simulation method?

- Line 191: The definition states that the "state" in this game includes all agents' policy parameters. This creates an exponentially large space, especially when agents are assumed to use neural networks. In this case, the state encompasses the combined network parameters of all agents.
	- To implement a meta-agent in this setting, does the input of its policy network include the policy network parameters of other naive agents? If so, how is this optimized and implemented?
	- According to equation (4) and (5), I suppose that the meta agent does not input the others' parameters to its own policy network, but treats it by sampling the gradient. Then why is this estimation unbiased? Using sampling method does not seem to solve the non-stationarity issue. And this contradicts the motivation of this method.
	- If there are two meta-agents in the training phase, how do they treat each other? Are their gradient chains intertwined, or do they simply treat one another as naive agents?
- Line 998: The proof relies on Sutton’s policy gradient theorem, which assumes a fully observable environment. But Line 72 states that this environment is partially observable. Does the derivation still hold in this case? (I doubt the gradient computation is exact in this context, but I understand it may be an acceptable estimate for experimental purposes.)
	- Equation (4) shouldn’t be an equality but should have a scaling factor.
- Line 370: This experiment begins from a specific initialization point. The maximum expected reward is slightly lower than the cooperation reward of 1. Although this is a challenging case, what would results look like for different initialization points? Can a learning-aware agent achieve better outcomes (with an expected reward greater than 1)?
- Figure 3B and 3C: Does the proposed method require interaction with naive agents to initiate training?
- Why is there no comparison between COALA and M-FOS in Figure 5?

Minors:

- Why are all references in blue font? This formatting seems differ from other submissions.
- In Lines 14, 33, 49, 106, 122, 161, 185, 478, and 479, the single quotation marks should be replaced with double quotation marks.

### Questions
Questions are mentioned in the weaknesses part.

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
The paper brings sound improvements over the current opponent shaping methods for non-cooperative general-sum games with self-interested learners. The core contributions over the existing methods are introducing a policy-gradient method (COALA-PG) capable of shaping other learners without making any assumptions on learners' optimization functions due to the sequential and batched nature of the expected return to be maximized. The method is evaluated on Iterated Prisoner's Dilemma and CleanUp environments. COALA-PG is compared to M-FOS and variants of itself as a baseline and investigation of how the pool of learners, whether they are meta or naive learners, affect reaching cooperation.

### Strengths
* Introduction of a novel unbiased policy gradient formulation to shape naive learners that does not need to know the learners' optimization functions but can infer them by sequence modeling.
* Comparisons showing the effect of how learners are pooled (meta vs naive or meta vs meta etc.) give insight into how shaping is achieved.
* COALA-PG improves performance over M-FOS.
* The figures are informative.

### Weaknesses
* There is insufficient explanation on why COALA-PG is so much better than batch-unaware COALA-PG which perform very similar to M-FOS. The credit assignment figure is a nice visualization to give intuition, but it does not provide concrete explanation and we have to speculate. Since batch-awareness is the main factor that makes COALA-PG's performance superior, it is difficult to attribute the success to sequence modeling and inferring the learning dynamics. There should be an investigation into how batch-awareness affect and change gradients and why batch-unaware COALA-PG is performs similar to MFOS.
* Baselines are insufficient. The only shaping method compared is the MFOS. Although mentioned in the paper, no comparison to Shaper is done in experiments. There is no comparison to LOLA either; even though MFOS is already expected to outperform LOLA, still a verification should have been done.
* There is no versus for different shaping methods, i.e. there is no experiment on how COALA-PG shapes M-FOS and vice versa.

### Questions
* Why is there no experiment for LOLA?
* Why does batch-unaware COALA-PG perform so similar to M-FOS, while the batch-aware COALA-PG surpasses it? Did you investigate how gradients change and why the batch-aware gradients work so well?

### Soundness
2

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
The paper introduces a novel policy gradient algorithm for learning-aware reinforcement learning, which allows agents to model the learning dynamics of other agents. This approach enables cooperative behavior among self-interested agents, even in complex scenarios like temporally extended action coordination. The algorithm avoids higher-order derivatives, and can handle mini-batched learning. Additionally, the paper provides a theoretical explanation for the emergence of cooperation in the iterated prisoner's dilemma based on learning awareness and agent heterogeneity.

### Strengths
1. The paper tackles a core issue in multi-agent learning - fostering cooperation among self-interested agents.
2. The introduction of the first unbiased, higher-derivative-free policy gradient algorithm for learning-aware reinforcement learning is a significant contribution.
3. The proposed algorithm appears to outperform existing methods in achieving cooperation.
4.The paper proposes a novel explanation for cooperation emergence based on learning-awareness and agent heterogeneity in the iterated prisoner's dilemma.

### Weaknesses
In my view, the “elegant” formulation and the batched co-player shaping POMDP appear similar to a parallel environment POMDP setup, while COALA-PG seems more like a delayed optimization applied after several episodes. This raises concerns about the true novelty and contribution of the work. I have a few specific questions regarding this point.

It’s entirely possible that I may not yet have fully understood the paper. However, improving the paper’s clarity and structure would make it more accessible and easier to follow.

I look forward to discussing these points with the authors, and I am open to raising my score if my concerns are addressed satisfactorily.

### Questions
1.	What are the key differences between the proposed batched co-player shaping POMDP and a standard POMDP with parallel environments? Equation 3 appears to simply sum rewards across parallel environments, making it unclear what distinct advantages this formulation provides.
2.	The font size in the figures is too small, making it difficult to read. Enhancing font size and clarity would significantly improve readability.
3.	 There are well-known baseline methods in the mixed-motive game literature, particularly within the Cleanup environment. Why does the paper not include comparisons with these established methods?
4.       More information is needed on the simplified Cleanup environment. How does it differ from the standard Cleanup environment, and what are the implications of these differences for the study’s findings?
5.	 While the paper aims to address a mixed-motive game, the problem is formulated as a general-sum game. In my view, general-sum games are not fully equivalent to mixed-motive games. Could you clarify the rationale behind this formulation choice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
To solve two challenges of general-sum games: 1. Non-stationarity of the environment, hard to choose a good equilibrium. The author propose an unbiased, higher derivative-free policy gradient algorithm for learning-aware RL, which considers how the action will influence the other agent's policy changing: They do policy gradient in meta-episode. They have extensive theoratic comparison of their meta policy gradient with prior works. And have interesting findings in the iterated prisoner's dilemma and a grid-world-like social dilemma "CleanUp".

### Strengths
1. Clear writing, clear concept definition.
2. extensive theoratic comparison with prior works and experiments on two non-trivial settings.
3. enough details of implementation in the appendix.

### Weaknesses
This may be difficult, but it would be great if you could show the efficiency of your method on more difficult environments of deep MARL beyond matrix game or grid world (like Agar.io[1])

[1]: Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization, ICLR 2021

### Questions
None

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This study presents a framework for multi-agent cooperation in partially observable general-sum games to enable agents to adaptively cooperate in non-stationary environments by conditioning their policies on past interactions. The COALA-PG method introduces improved policy calculation through accumulated gradients and variance reduction that enhances stability and computational efficiency. The framework is tested on benchmark tasks such as the Iterated Prisoner´s Dilemma and the Cleanup game, demonstrating cooperative behavior among decentralized agents.

### Strengths
1. **History-Dependent Adaptation for Multi-Agent Cooperation:** The paper introduces a promising approach that enables agents to adaptively cooperate by conditioning policy updates on observation histories. This allows agents to respond to non-stationarity in general-sum games, specifically handling the evolving distributions of co-agent strategies as each agent independently learns and adapts over time. By incorporating these historical observations, the framework aims to maintain effective cooperation even as the behaviors of other agents shift dynamically, without explicitly modeling the non-stationarity or learning dynamics of other agents.
2. **Theoretical and Practical Advancements in Policy Gradient Estimation:** Theorems 3.1 and 3.2 present a robust policy gradient estimation method specifically designed for cooperative multi-agent learning. COALA-PG´s use of accumulated gradients and minibatch summation approach improves stability and adaptability without requiring higher-order gradients. This makes COALA-PG more scalable in comparison to methods like LOLA-DICE. Additionally, the use of value function baselines in gradient estimation reduces variance which increases COALA-PG´s practical applicability.
3. **Supporting Results Against Naive Learner Baselines:** Empirical comparisons demonstrate that COALA-PG outperforms naive, non-history-conditioned learners, demonstrating the benefit of history dependency in non-stationary environments.

### Weaknesses
1. **Full History Dependency:** While history dependency enables adaptability, it may approximate full observability, especially in discrete environments. By accumulating state-action information over time, agents in discrete settings could essentially reconstruct the environment as if fully observable, reducing the framework´s applicability in scenarios where true partial observability is intended.
2. **Simplistic Experimental Scenarios:** The chosen experiments, such as the Iterated Prisoner’s Dilemma (IPD) and Cleanup, are relatively simple and do not fully demonstrate the framework´s ability to manage partial observability or complex, dynamic cooperation. IPD lacks multi-step state evolution, and while Cleanup provides more interaction, it still operates in a low-dimensional grid setting, which is easier for agents to handle than high-dimensional or real-world environments.
3. **Lack of Convergence Analysis:** COALA-PG lacks formal guarantees of stability or convergence in non-stationary environments. This leaves open questions about COALA-PG´s ability to adapt as agent strategies evolve. The study would benefit from additional theoretical or empirical validation to support its effectiveness in highly dynamic settings.

### Questions
1. Given that full history dependency may reduce partially observable settings to fully observable ones in discrete environments, have the authors conducted a sample complexity analysis for handling true partial observability, where agents have limited access to or storage for historical data?
2. Could the authors elaborate on their choice of the Iterated Prisoner´s Dilemma and Cleanup as experimental benchmarks? 
3. Could the authors discuss whether any convergence properties have been considered for COALA-PG? Given the absence of a formal convergence analysis, how do the authors anticipate the framework will behave in non-stationary environments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new learning-aware reinforcement learning rule derived as a policy gradient estimator. Instead of using long histories of samples, the policy gradients consider the stochastic minibatch updates used by other agents. The results show improvements in some mixed environments. Overall, the paper is well-written and well-organized. The notations are standard, and the theorems are sound.

### Strengths
1. The paper is well-written and well-organized. 
2. Theoretical proofs are complete and sound.

### Weaknesses
1. The motivation for proposing COALA-PG is unclear. It’s not obvious whether the issue is related to variance or other issues when using mini-batches. The authors suggest that larger mini-batches could pose a problem, but this may lead to higher variance in reward summation. However, these points are not extensively discussed in the manuscript. Additionally, compared to M-FOS, it appears that COALA-PG uses the 1/B term to scale rewards, but it seems that this scaling is still related to controlling the policy gradient’s variance.

2. Why not use a Q-function instead of summing rewards?

3. In Theorem 3.1, what is the advantage of using batches of inner episode histories instead of a long history? Are there any theoretical insights supporting this approach?

### Questions
please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
