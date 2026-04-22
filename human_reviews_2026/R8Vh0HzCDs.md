# Action-Adaptive Continual Learning: Enabling Policy Generalization under Dynamic Action Space

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Continual Learning (CL) is a romising paradigm that enables agents to learn a sequence of tasks, accumulating knowledge learned in the past and using it for problem-solving or future task learning. However, existing CL methods often assume that the agent’s capabilities remain static within dynamic environments, which doesn’t reflect real-world scenarios where capabilities dynamically change. This paper introduces a new and realistic problem: *Continual Learning with Dynamic Capabilities* (CL-DC), posing a significant challenge for CL agents: How can policy generalization across different action spaces be achieved? Inspired by the cortical functions, we propose an **A**ction-**A**daptive **C**ontinual **L**earning framework (AACL) to address this challenge. Our framework decouples the agent’s policy from the specific action space by building an action representation space. For a new action space, the encoder-decoder of action representations is adaptively fine-tuned to maintain a balance between stability and plasticity. Furthermore, we release a benchmark based on three environments to validate the effectiveness of methods for CL-DC. Experimental results demonstrate that our framework outperforms popular methods by generalizing the policy across action spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces AACL, which decouples the policy from task-specific action spaces by learning an action representation space via SSL (encoder–decoder) and adaptively fine-tuning the decoder when the action space changes, balancing stability and plasticity.

### Strengths
The proposed AACL framework effectively decouples policy and action representations, achieving a better stability–plasticity trade-off and superior forward transfer on benchmarks such as MiniGrid.

Experimental results are comprehensive and reproducible, showing that AACL consistently outperforms replay-, regularization-, and architecture-based baselines in both expansion and contraction scenarios.

### Weaknesses
The approach currently supports only discrete and limited action spaces, assuming shared state spaces and similar task semantics, which restricts its applicability to continuous-control or heterogeneous domains.

In action-space contraction settings, AACL still experiences performance degradation and forgetting, indicating incomplete mitigation of catastrophic forgetting.

The method requires an exploration phase at the start of each task to collect transition data for self-supervised training, which limits its efficiency and practicality in strictly offline settings.

### Questions
1. In line 239, typo: "the auxiliary task of the action representation"
2. In line 287, why does the encoder need fine-tuning but does not require adding constraints? If the encoder only needs to be fine-tuned on new tasks, will the encoder quickly forget the state to encoder vector mapping relationships learned on historical tasks?
3. In Algorithm 1, the pseudocode does not show the sequential relationship of task training. The current pseudocode appears to randomly select and train across all tasks.
4. If the action space is contracted, does it require prior knowledge of the environment to perform masking? Can the method proposed in this paper only adapt to scenarios where environmental dynamics do not change significantly, because in this paper, the state space seems to be assumed unchanged by default, whereas in real-world scenarios the state space may change?
5. When the environment's action space is reduced, the action mask requires prior knowledge of the environment, so I believe the statement in line 344 is inappropriate.
6. In Figures 4 and 5, all methods need to be tested on all tasks during training, so do the curves in the figures represent the average test performance across all tasks? If each subfigure represents the model's performance on each corresponding subtask, then for Figure 6, we can clearly see that when AACL is trained to the third task, the first two tasks show significant performance degradation, which means the model experiences severe forgetting. However, in Table 2's Bigfish experiment, AACL's forgetting is very small. Why is this?
7. In Figures 16 and 17, why does the ALL algorithm exceed the return range shown in the figures? Does this mean that if the performance of all other algorithms were converted to normalized scores, they would show relatively poor performance? I suggest the authors use normalized performance as a metric, as this would better enable comparison of performance differences between different models.
8. The authors need to clarify which module parameters are shared and which module parameters are task-specific in the current method.
9. The current experiments contain task sequences of only three tasks, but in classic CRL tasks, such as CW20 and CW10, the task sequence lengths are 20 and 10 respectively. Therefore, I suggest the authors consider longer task sequences to fully demonstrate the effectiveness of the proposed method.
10. The method proposed in this paper is not applicable to situations where the state space changes, and state space changes frequently occur in real-world scenarios, such as changes in the number of robot sensors or robotic arm models. Furthermore, compared to the environments used in most current CRL tasks (Continual World), the tasks adopted in this paper are simpler, and the environmental dynamics changes are smaller. This can be seen from Figures 16 and 17, where we can observe that when training the first task, the model's performance on the second and third tasks also improves simultaneously, which means these three continual learning tasks are essentially almost identical.
11. Since the model parameters used by the current authors are not large, and the continual learning task sequence is very short, why not directly train a separate model for each task? In my view, this would not significantly increase storage and computational overhead, and parallel training could even further reduce physical time consumption.
12. This method has the potential of being applied to continuous high-dimensional CRL environments, so I suggest the authors compare with recent methods on classic tasks such as CW10 and CW20, for example L2M [1], t-DGR [2], Doya-DaYu[3], and UPGD [4].

[1] Learning to modulate pre-trained models in rl

[2] dgr: A trajectory-based deep generative replay method for continual learning in decision making

[3] Lifelong reinforcement learning via neuromodulation

[4] Addressing loss of plasticity and catastrophic forgetting in continual learning

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a tokenized, transformer-based world model for continual reinforcement learning (CRL). It introduces a compact VQ-VAE tokenizer to discretize frames, a causal Transformer world model (WM) trained for next-step token/reward/done prediction, and an actor–critic policy trained on imagined rollouts. Adaptation is localized using LoRA adapters inserted only into the world model, while a FiLM-based task conditioning module fuses explicit task IDs with trajectory-inferred context.

### Strengths
# Strengths
* The use of tokenized (VQ-VAE) state representations in a world model tailored for CRL is innovative and bridges discrete representation learning with continual adaptation.
* Adapting only the world model while freezing the policy is a well-motivated design choice to stabilize learning and isolate plasticity.
* The study includes detailed metrics (Isolated Forgetting, Forward Transfer) on six Atari tasks with clear ablations and competitive baselines.
* The paper is technically detailed, with well-structured methodology, hyperparameter listings, and pseudo-code. It’s easy to reproduce and interpret.

### Weaknesses
# Weaknesses
* Evaluation is confined to Atari CORA; no experiments on non-visual or real-world-like domains (e.g., DM Control or robotic settings).
* Dependency on Task Cues
* It remains somewhat unclear whether improvements arise mainly from LoRA-based modularity or from the discrete tokenization (the ablation focuses only on LoRA).
* Some tables are a bit nonsensical (mainly Table 1. Why do you need a table that has a single row?)

### Questions
# Questions / Suggestions
* Including a runtime or compute comparison to baselines would strengthen the practical relevance.
* How does the tokenizer quality (e.g., codebook size or reconstruction error) affect CRL performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the problem of continual learning (CL) under dynamic capability, where the agent must handle changing action spaces across tasks. To address this, the authors propose the Action-Adaptive Continual Learning (AACL) framework, which decouples the policy network from task-specific action spaces by introducing a shared action representation space. When encountering a new action space, AACL fine-tunes the encoder–decoder components to adapt the representation accordingly. Furthermore, the authors release a benchmark designed to evaluate this setting and demonstrate that AACL achieves superior performance compared to existing methods.

### Strengths
1.	The paper tackles an underexplored and practically relevant problem—continual learning with dynamically changing action spaces—thereby extending the conventional CL setting beyond fixed-action environments.
2.	Experimental results across the proposed benchmark indicate that AACL achieves consistently strong performance, validating the effectiveness of the action representation mechanism.

### Weaknesses
1.	The paper should more explicitly articulate the fundamental challenges of continual learning with dynamic capability (CL-DC) compared to standard CL. Is the key difficulty solely the expansion or contraction of the action dimension, or does it introduce deeper representational conflicts? Moreover, how does this setting relate to class-incremental CL, where the output head expands over time?
2.	It remains unclear how the model determines whether the action space has expanded or contracted. Does the framework assume access to a task identifier during training or inference? If the encoder or decoder dynamically changes its structure, how is inference performed without explicit knowledge of the current task or action configuration?
3.	Although regularization is used to mitigate catastrophic forgetting in the decoder, continuous fine-tuning may still alter the mapping from latent representations to action distributions. When the action space expands or contracts, previously learned action distributions may shift undesirably. The paper should explain how AACL preserves decoding consistency for earlier tasks under such structural transitions.
4.	The selection of baselines appears somewhat dated, with the most recent comparison being Mask (2023). Considering that the paper targets ICLR 2026, a fair evaluation would require including more recent continual RL methods. Furthermore, as AACL combines structure-based (encoder) and regularization-based (decoder) strategies, both categories (also combination) should be represented among the baselines to ensure methodological fairness.
5.	The ablation analysis focuses solely on regularization components, while neglecting structural variations such as applying structured expansion or mask to the decoder. Additionally, the current analysis provides limited insight into the causal factors underlying the observed results; deeper interpretation of why certain components contribute more substantially would enhance the paper’s scientific rigor.

### Questions
See the weakness.

### Soundness
2

### Presentation
1

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
This paper introduces Continual Learning with Dynamic Capabilities (CL-DC), a problem where agents must learn sequentially while their action spaces dynamically change through expansion, contraction, partial change, or complete change. The authors propose Action-Adaptive Continual Learning (AACL), which decouples policies from specific action spaces by learning an action representation space via self-supervised learning (SSL). An encoder maps actions to representations, while a decoder maps representations to action probabilities. When action spaces change, the decoder structure is adaptively updated and both encoder-decoder are fine-tuned with EWC-based regularization. The authors release a benchmark based on MiniGrid, Procgen (Bigfish), and ALE (Atlantis) environments. Experiments show AACL outperforms baselines including FT, EWC, Online-EWC, CLEAR, and Mask on expansion and contraction scenarios.

### Strengths
### **1. Problem formulation**: 
CL-DC is a practical and underexplored problem in continual RL with clear real-world motivation like hardware failures and software updates.

### **2. Consistent improvements**: 
AACL outperforms a comprehensive set of baselines across most metrics and environments.

### Weaknesses
### **1. Limited Technical Novelty**
The approach essentially combines existing techniques (SSL for representation learning + EWC regularization + adaptive architecture). The core contribution is incremental rather than introducing fundamentally new methods. The SSL objective (predicting actions from state transitions) is adapted from prior work (Chandak et al., 2019; Fang & Stachenfeld, 2024), and EWC is a standard continual learning technique.

### **2. Doubtful Scalability**
The authors only tested their approach where actions are discrete and with very low dimensionality (maximum 9 actions). It also requires the agents to explore whenever action spaces change in order to fine-tune the action encoder-decoder. I believe this can work in simple environments as tested by the authors, but in an environment with complex dynamics or in real-world settings, the sample efficiency and applicability remain unknown. Specific concerns include:

- Fisher information matrix storage scales as O(|parameters|²), raising scalability doubts
- Random exploration policy acknowledged as potentially insufficient for complex spaces
- No evaluation on continuous action spaces, critical for real robotic applications

It would be better if the authors provide further discussion and empirical evidence addressing these scalability concerns.

### **3. Catastrophic Forgetting's Significance Not Justified**
The authors propose avoiding catastrophic forgetting as one of their main novelties, but its importance needs further justification. Two examples are given to reason the need to adapt to a dynamic action space: hardware/software updates in robotic systems, and structural change during the life of a biological system. In the latter example, a swift and stable adaptation is needed, but it never requires the agents to maintain good performance in the exact previous forms.
For robotics, I also don't see the need to maintain the performance of previous versions in the same policy network. On the other hand, if generalizing across different action spaces for same tasks is the goal, there is a whole line of research on cross-embodiment learning that fits this case better, including:
- **"RoboCat: A Self-Improving Robotic Agent" (Bousmalis et al., 2023)** - learns action representations across multiple robot embodiments and action spaces
- **"Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation"** (Doshi et al., 2024) - learns abstract action representations that transfer across embodiments

### 4. **Overly Simple Experiment Design for a Benchmark**
The experimental design has only discrete actions and there is only one task for each environment. More critically, the authors define 4 types of action space changes (Section 3.2): Expansion, Contraction, Partial Change, and Complete Change, but only expansion and contraction are tested in the main experiments. Appendix D.3 shows sequential expansion-then-contraction, which is **not** the same as true partial change. There is no further justification or discussion other than to punt it to future work in Appendix G.1.

### 5. **Clarity in Writing**
- The first sentence of the abstract says "Continual Learning (CL) is a powerful tool..." but continual learning is a research topic or problem setting, not a "tool." Tools would be specific methods or algorithms designed for continual learning.
- The authors repeatedly use the overloaded word "capability" to refer to action space, which can be confusing. Simply using "action space" would be clearer.
- Page 5, line 239: "Therefore, the auxiliary task **off** the action representation" should be "of".

### Questions
Can you please address the concerns over novelty, scalability, limitations in experiment design, and why mitigating catastrophic forgetting is essential in this particular continual learning setting?

### Soundness
2

### Presentation
2

### Contribution
2
