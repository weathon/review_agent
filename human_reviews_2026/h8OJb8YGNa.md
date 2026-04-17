# Towards Unpredictable Worlds: Continual In-Context Reinforcement Learning in Non-Stationary Environments

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Traditional In-Context Reinforcement Learning (ICRL) demonstrates impressive rapid adaptation, but its reliance on static environments limits its applicability.
In contrast, real-world scenarios are inherently non-stationary, with continuous and unpredictable changes that challenge an agent's ability to adapt. 
To bridge this gap, we formally define and systematically investigate Continual In-Context Reinforcement Learning in Non-Stationary Environments. 
Our central question is: what model architectures and training strategies enable an agent not only to rapidly master new dynamics in a continuously evolving environment, but also to efficiently discard or isolate outdated information, thereby achieving robust online adaptation?
To ground our investigation, we construct a new benchmark suite featuring two complementary non-stationary domains---a symbolic reasoning task and a physics-based control task---each modified to exhibit unpredictable, intra-lifetime dynamic changes.
On these benchmarks, we conduct  extensive evaluations at both the model and training-strategy levels.  
At the model level, we compare state-of-the-art sequence model architectures.
At the training strategy level, we systematically analyze the influence of stationary versus non-stationary training, dynamic change frequency, context length, and interaction scale.
Our findings demonstrate the necessity of non-stationary training and reveal critical factors shaping continual adaptation. 
These results provide actionable insights and design principles for building agents capable of learning and adapting in truly open and dynamic worlds.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Continual In-Context Reinforcement Learning (CICRL), which extends traditional in-context reinforcement learning to handle non-stationary environments where dynamics change unpredictably within a single episode. The authors formalize this new setting and create two benchmark suites: a symbolic reasoning task and a physics-based control task with dynamic shifts occurring during an agent’s lifetime. They evaluate several sequence model architectures, including Transformers, Mamba2, and GatedDeltaNet, under different training conditions. The results show that non-stationary training is essential for effective continual adaptation and that models with stronger sequence modeling capabilities adapt more robustly. Overall, the paper provides a clear framework, new benchmarks, and key insights for developing agents that can learn and adapt in continuously changing environments.

### Strengths
- The paper presents a clear and well-motivated formulation of Continual In-Context Reinforcement Learning (CICRL), addressing an important gap in adapting to non-stationary environments.
- The introduction of two complementary benchmark suites (symbolic and physics-based) provides a good empirical foundation for future research.
- The evaluation protocol and metrics (e.g., $\Delta$Switch, $\Delta$Adapt) are well-designed to isolate aspects of continual adaptation and forgetting.
- The ablation studies provide valuable insights into how factors like context length, change frequency, and scale influence continual adaptation.

### Weaknesses
- The authors do not provide access to the code or benchmarks, even though they state an intention to release them. This makes it difficult to assess reproducibility and verify experimental results, especially since the proposed benchmarks are a core contribution of the paper.
- The study relies exclusively on PPO as the underlying RL algorithm, which limits the generality of the findings and makes it unclear whether the observed adaptation behaviors would hold under different RL algorithms.
- Some results are difficult to interpret quantitatively or theoretically, with overlapping performance among models in certain settings.

### Questions
- How sensitive are the results to the choice of PPO as the underlying RL algorithm? Would other RL algorithms yield similar adaptation behavior?
- How do the models handle gradual versus abrupt changes in environment dynamics?
- Have you analyzed catastrophic interference within context representations, and can the agent actively forget outdated information?

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
3

### Summary
The paper introduces Continual In-Context Reinforcement Learning (CICRL), an extension of ICRL designed to address intra-lifetime non-stationarity. It defines a framework where environments evolve dynamically without episodic resets, proposes two non-stationary benchmarks, i.e., modified XLand-Minigrid and Kinetix, and evaluates several sequence models (Transformer, Mamba2, GatedDeltaNet) under static and non-stationary training regimes. The main finding is that non-stationary training improves adaptation to dynamically changing environments.

### Strengths
1. The proposed benchmarks are well-designed and may aid future reproducible research.

2. The evaluation metrics ($\Delta_{\rm Switch}, \Delta_{\rm Adapt}, \Delta_{\rm In-Context}$) provide a structured view of adaptation and forgetting.

3. The experiments are extensive and include several ablation factors (context length, change frequency, scale).

### Weaknesses
1. The paper may have limited conceptual novelty. CICRL is mostly a rebranding of meta-RL with drifting task distributions. No new theoretical formulation or learning mechanism distinguishes it from existing continual or meta-RL setups.

2. The paper only benchmarks off-the-shelf sequence models using standard PPO. There’s no method, modification, or insight addressing how to handle non-stationarity, it only shows that training on non-stationary data helps.

3. XLand-Minigrid and Kinetix involve rule or physics changes that are arbitrary and discontinuous. These synthetic shifts may not convincingly approximate real-world or open-ended non-stationarity.

4. The results find that non-stationary training helps. However, there’s no analysis of why certain models perform better.  

5. Since model parameters are frozen, the framework lacks actual continual learning. There are no weight adaptation, consolidation, or transfer across shifts. The agent merely performs repeated context-based inference.

6. No comparisons to true continual or meta-RL methods (e.g., EWC, online PPO, fine-tuning). Without them, it’s unclear whether CICRL offers any advantage beyond being static PPO trained on varied data.

### Questions
1. How is CICRL mathematically or behaviorally distinct from meta-RL with changing task distributions?

2. How does “forgetting” occur in a model without parameter updates?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Continual In-Context Reinforcement Learning (CICRL): adapting to non-stationary environments without gradient updates. Authors propose a new benchmark for CICRL and perform experiments regarding model architectures and training strategies.

### Strengths
- Proposes an important problem, extending ICRL to more real-world-like scenarios.
  
- Builds a new benchmark consisting of both discrete grid-world environments and continuous physics-based environments.
  
- The proposed new evaluation metrics and analysis around these metrics are interesting.

### Weaknesses
- The evaluation protocol and certain evaluation metrics seem to only take discrete, episode-wise dynamics change into account, while nonstationarity may well be continuous (e.g. a robot continuously draining its battery).
  
- The overall formulation could focus more on the characterization of the continual aspect of CICRL. The current one (a sequence of POMDPs) can arguably be subsumed into a single POMDP with an unobserved, time-varying environmental variable.
  
- Experiment results regarding model architectures are confusing. For example, non-static Mamba2 is the best in minigrid in terms of avg-return, but the worst in kinetix.

### Questions
- How are the 'Static' models trained? Why do they mostly perform much worse than non-static models even under a zero-shot setting without any contexts?
  
- What is the difference between CICRL and regular POMDPs?
  
- In Fig.4 freq-1 results, why is Random(1-10) the overall best strategy? Wouldn't Random(1-5) be closer to the freq-1 evaluation setup?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a continual in-context RL framework where the enrivonments continously change. The authors propose a new benchmark suite featuring two complementary non-stationary domains and benchmark sequence models.

### Strengths
1. clear formalization of CICRL and why standard ICRL (stationary rules) is insufficient
2. two complementary, well-motivated environments with continous changes. They are general and reproducible
3. the analysis and insights are interesting.

### Weaknesses
1. This paper proposed a benchmark and conducted baseline evaluations with an analysis of the results. There is no novel algorithm development, which limits the novelty and contribution of this work. 
2. The topic itself seems to be in alignment with continual learning and continual RL, or lifelong RL. But I don't see what the difference is between this work and other continuously changing environments proposed in continual RL environments. It seems to me that, in context, learning is simply learning history-dependent policies rather than Markovian ones. 
3. The experiments are largely simplistic. I am not sure how significant the results are, given the simplicity of these problems. 
4. In RL, I think domain randomization is the primary technique for training robust policies. I think the authors miss a large portion of the literature, especially on the aspects of robot learning, which is one of the largest fields of RL applications.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
