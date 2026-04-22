# From Objects to Skills: Interpretable Meta-Policies for Neural Control

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Despite its success in learning high-performing policies for diverse control and decision-making tasks, deep reinforcement learning remains difficult to interpret and align due to the black-box nature of its neural network representations. Neuro-symbolic approaches improve transparency by incorporating symbolic reasoning, but when applied to low-level actions, they result in overly complex policies. We introduce NEXUS, a hierarchical Reinforcement Learning framework that integrates neural skills with neuro-symbolic meta-policies to balance efficiency and interpretability. In its core, it allows transparent reasoning on disentangled high-level actions (i.e. interpretable skills), greatly reducing complexity of symbolic policies. Object-centric representations enable extracting rewards and meta-policies from language models, while the hierarchical structure allow reasoning over skills rather than atomic actions. We experimentally demonstrate that NEXUS agents are interpretable, less prone to reward hacking, and more robust to environment simplifications. We further evaluate how differing levels of meta-policy interpretability (i.e. purely neural or symbolic) influences performance. Overall, NEXUS enables interpretable and robust control via neuro-symbolic reasoning over high-level skills.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LENS, a hierarchical RL framework that achieves strong compositional generalization and interpretability. It decouples high-level reasoning from low-level control by using an object-centric perception module, a pre-trained library of neural skills, and a meta-policy that selects which skill to apply to which objects (as tool/target). The key contribution is enabling zero-shot generalization to structurally novel tasks by composing known skills in new ways.

### Strengths
*   By design, the agent's high-level decisions are explicit and human-readable, representing a significant advantage over opaque end-to-end policies and facilitating debugging.
*  The experiments convincingly demonstrate the ability to solve complex, multi-step manipulation tasks that were not seen during training, highlighting the framework's practical effectiveness.

### Weaknesses
*   The system's success is critically dependent on a high-quality, pre-trained skill library and a flawless perception module. The paper does not fully address the scalability of creating this skill library or how the system handles perception failures.
*   The meta-policy is trained via supervised learning, which requires expert demonstrations for high-level actions. This raises concerns about scalability and autonomy compared to a pure RL approach for discovering high-level strategies.

### Questions
How does the system handle failures in underlying modules? For instance, what is the recovery mechanism if a low-level skill fails to execute or the perception module makes an error?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents NEXUS, a hierarchical structure of neuro-symbolic RL, which uses LLMs to identify skills for a given task and compose reward function for learning neural policies regarding each skill. Specifically, before policy learning NEXUS identifies the pre-conditions for each skill using a LLM, what are used to activate skills for each state. Crucially, both the reward function and the pre-conditions are defined for concepts/symbols at each state, provided by the simulator. The authors evaluate NEXUS on a few atari games and compares it with its neural counterpart, PQN and PPO.

### Strengths
1. The paper presents effort in learning disentangled skills, such as "Rescue Divers" and "Shoot Enemy" for the Atari game Seaquest. Such a design sheds light on more effect usage of neural policies for neuro-symbolic RL.

2. The paper showcases that LLM-generated reward functions are quite competitive when visual symbols can be reliably obtained.

### Weaknesses
1. While using LLMs to identify pre-conditions and reward functions for learning hierarchical policies, the paper does not include a discussion for prior attempt on LLM-generated rules or rewards for NSRL. 

2. No performance comparison with other neuro-symbolic approach is presented, even though Atari is a common benchmark for this line of work.

3. The paper lacks a discussion for the failure patterns of LLM-generated rules or rewards. 

4. The entire framework is based on the assumption that visual concepts can be reliably identified at real-time, which is quite impractical, and there is no discussion for how to address such assumption.

### Questions
1. Since both the LLM-generated rules and rewards are not updated during policy learning, how do you ensure their alignments with the task of interest and eliminate the influence of hallucination?

2. Is is possible to apply this method on benchmarks other than OCAtari, especially those that do not provide ground-truth visual concepts?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the challenge of making RL agents more interpretable and robust by **reasoning at the level of skills rather than low-level actions**. The authors propose NEXUS, a hierarchical framework that combines neural skill controllers with symbolic or neuro-symbolic meta-policies to select which skill to execute based on object-centric state representations. By integrating symbolic reasoning with learned control, NEXUS enables transparent decision-making while maintaining flexibility and performance. Experiments on several Atari-style and Crafter environments show that NEXUS learns distinct, meaningful skills, achieves competitive rewards, avoids reward hacking, and generalizes better to distribution shifts compared to neural baselines.

### Strengths
- Experiments are very thoughtfully designed to support the key claims about interpretability, robustness, and performance. 
- The perspective of shifting interpretability  from the action level to the skill level is novel and creative.
- The paper is well-written and conceptually clear, with motivating examples, intuitive figures to follow along.

### Weaknesses
- Although some skill rewards are LLM-generated, the framework still relies on meaningful skill definitions provided a priori.
- Along the same point, while symbolic meta-policies are interpretable, it’s not evident how their rule sets scale with larger skill repertoires or more complex environments. The evaluation too mainly focuses on 2D domains. 
- The paper could engage more directly with established hierarchical RL frameworks to clarify what is genuinely new in the meta-policy formulation beyond symbolic labeling.

### Questions
- To what extent can the framework autonomously discover useful skills, rather than relying on manually defined or LLM-suggested ones? Could the meta-policy guide skill discovery dynamically during training? How does the system behave when symbolic rules conflict or are incorrect?
- Since object-centric state representations are assumed, how sensitive is performance to noise or inaccuracies in object detection?
- Have the authors considered any quantitative or user-centered evaluation of interpretability (e.g., human predictability or trust metrics)?
- how might NEXUS extend to continuous control or robotics tasks where symbolic conditions and skills are less clear?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes NEXUS, a hierarchical reinforcement learning (RL) framework that combines neural skills (low-level policies) with interpretable meta-policies (high-level decision-making) to achieve transparency in control tasks. Drawing from dual-process theory, it structures agents with fast neural execution (System 1) and deliberative symbolic reasoning (System 2). Key components include object-centric representations for symbolic states, LLM-generated reward functions for disentangled skills, and three meta-policy variants: fully neural, fully symbolic, and neuro-symbolic (NeSy). The framework extends Parallelised Q-Networks (PQN) to hierarchical settings, training skills off-policy with skill-specific rewards. Experiments on Atari games (Seaquest, Kangaroo) and Crafter evaluate disentanglement, interpretability, performance, and robustness to simplifications, claiming reduced reward hacking and better generalization.

### Strengths
The paper addresses a timely issue in RL: interpretability in hierarchical agents, which is crucial for alignment and debugging. The integration of LLMs for generating rewards and rules is practical, reducing manual effort and enabling adaptation. The neuro-symbolic variant offers a balanced trade-off between flexibility and transparency, with clear visualizations aiding understanding. Experiments show promising results, like balanced skill learning and robustness to environment simplifications, highlighting potential advantages over baselines like PPO and PQN.

### Weaknesses
Methodologically, the reliance on pre-extracted object-centric states assumes perfect perception, ignoring real-world challenges like noisy or incomplete object detection, which limits applicability. LLM generation of rewards/rules is underexplored—prompts are vague (Section E), and no analysis of LLM errors or sensitivity to model choice (e.g., GPT-4 vs. others). The hierarchical PQN extension is incremental, building on existing works without novel theoretical contributions. Baselines are weak—HPQN is a strawman without skill rewards—and no comparison to state-of-the-art like neuro-symbolic RL. Robustness claims are overstated; simplifications (e.g., removing enemies) favor symbolic policies by design, but no ablation on rule quality or generalization to harder shifts (e.g., added obstacles).

### Questions
While interpretability in RL is important, NEXUS largely recombines established ideas—hierarchical RL, object-centric states, and LLM-aided RL—without breakthroughs. The core innovation (NeSy meta-policy) is a simple mask on Q-values, lacking depth compared to prior neuro-symbolic works. Do I miss anything here?

The motivation is not well motivated. There have been approaches that use symbolic reasoning to help low-level policy learning. What advantages does the proposed approach have compared to previous work, intuitively speaking?

Is the sentence "(1) a high-level meta-policy deciding between" incomplete ?

### Soundness
3

### Presentation
3

### Contribution
2
