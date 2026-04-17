# Let it Calm: Exploratory Annealed Decoding for Verifiable Reinforcement Learning

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) is a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs), yet its success hinges on effective exploration. An ideal exploration strategy must navigate two fundamental challenges: it must preserve sample quality while also ensuring training stability. While standard fixed-temperature sampling is simple, it struggles to balance these competing demands, as high temperatures degrade sample quality and low temperatures limit discovery. In this work, we propose a simpler and more effective strategy, Exploratory Annealed Decoding (EAD), grounded in the insight that exploration is most impactful on early tokens which define a sequence's semantic direction. EAD implements an intuitive *explore at the beginning, exploit at the end* strategy by annealing the sampling temperature from high to low during generation. This dynamic schedule encourages meaningful, high-level diversity at the start, then gradually lowers the temperature to preserve sample quality and keep the sampling distribution close to the target policy, which is essential for stable training. We demonstrate that EAD is a lightweight, plug-and-play method that significantly improves sample efficiency, consistently outperforming fixed-temperature sampling across various RLVR algorithms and model sizes. Our work suggests that aligning exploration with the natural dynamics of sequential generation offers a robust path to improving LLM reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an intuitively motivated strategy for controlling the exploration-exploitation trade-off during LLM generation for RLVR. Later on, they found this training strategy leads to many improvements on model performances in:
1. Mitigates entropy collapse
2. Improved Pass@k performance 
3. Improves Majority@k performance during inference time

Overall I find this to be a solid, well executed, technical paper, though the idea is highly incremental. I think the paper meets the standard but is not necessarily the most exciting.

### Strengths
1. Clearly written paper
2. Significant Empirical Gains and Robustness.
3. Well measured benefits and good ablation studies 

Exploration coefficient annealing has a long history in RL -- it's often known as epsilon-annealing in the epsilon-greedy exploration strategy. Modern LLMs exploration is controlled by temperature, therefore, annealing temperature intuitively will work.

### Weaknesses
1. Training Stability and Off-Policy Issues: The dynamic nature of EAD creates a discrepancy between the behavior policy and the target policy, introducing an off-policy issue that risks training instability.
2. Hyperparameter Sensitivity: The optimal configuration of EAD depends on model capability.
3. Diminished Relative Gains at High Exploration: While EAD remains superior, the relative performance advantage diminishes when the number of rollouts is scaled up extensively.

### Questions
The benefits of temperature annealing on pass@k, majority@k (inference time) is more like an **after-the-fact realization**. If inference-time scaling is all I care about, then I could try other methods, such as in [1]. They directly proposed policy optimization algorithm using pass@k as reward. How does temperature annealing, a more general method, compare to those that specifically made to target pass@k? 

I know it might be difficult to add a new experiment during rebuttal -- so I don't want to force too much work on the authors. But if they resubmit the paper to another venue, they should consider adding some comparison studies to show that a general method such as EAD can even outperform methods specifically designed for inference-time scaling. That would make this paper's result extremely surprising and exciting since EAD is in an order of magnitude simpler to implement than method proposed in [1]. However, without the comparison, I have no idea which one to use. 

[1] Tang, Yunhao, et al. "Optimizing language models for inference time objectives using reinforcement learning." arXiv preprint arXiv:2503.19595 (2025).

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Exploratory Annealed Decoding (EAD), a novel sampling strategy for Reinforcement Learning with Verifiable Rewards (RLVR). The method is motivated by the observation that the conditional entropy of a language model's output distribution naturally decreases as a sequence is generated. Grounded in this insight, EAD proposes an "explore at the beginning, exploit at the end" strategy by dynamically annealing the sampling temperature from a high value to a low value during the generation of a single sequence. This approach aims to encourage high-level semantic diversity in the early tokens, which are argued to be most critical for a sequence's overall direction, while ensuring coherence and stability by reducing randomness in later tokens.

### Strengths
The paper is well-written and easy to follow. The motivation is clearly articulated, grounding the proposed method in an intuitive principle ("explore early, exploit late") and supporting it with theoretical reasoning and empirical evidence.

### Weaknesses
The paper's contributions are undermined by its conceptual simplicity and an over-generalized central assumption.

1. Limited Novelty: The core idea of decreasing temperature over time to balance exploration and exploitation is the foundational principle of Simulated Annealing, a classic optimization algorithm. While the authors correctly note that they are the first to apply this intra-sequence, this adaptation feels more like an incremental application of a well-known heuristic rather than a fundamentally new technique. The conceptual leap from annealing over optimization steps to annealing over generation steps in a sequence is not substantial enough to be considered a major contribution on its own.

2. Overly Simplistic and Brittle Heuristic: The central premise that exploration is most valuable for early tokens is a strong assumption that does not hold universally, particularly in the complex reasoning tasks that RLVR targets. Many reasoning processes require precision and the discovery of critical, low-probability tokens late in the sequence. For example:

i) In a multi-step mathematical proof, the final numerical answer or a key simplification step might occur near the end. A very low temperature at this point would suppress the model's ability to find the correct, precise token if it's not the most probable one.

ii) In code generation, a bug might be fixed by changing a single character or keyword in the final line of a function. Early exploration might set the overall structure, but late-stage exploration is needed to find that specific, crucial fix. 

In general, the "explore early, exploit late" strategy seems brittle and tailored to problems where the initial plan is all that matters, which is a limiting assumption for general-purpose reasoning. Other works have proposed more nuanced exploration strategies, such as focusing on high-entropy "branching" tokens regardless of their position, which seems like a more robust and principled approach (e.g. https://arxiv.org/abs/2504.15466)

### Questions
1. The paper's core heuristic seems to break down for reasoning tasks that require high precision or discovery in later generation steps. Could you discuss the failure modes of EAD? Have you analyzed its performance on tasks where a single, critical token late in the sequence determines correctness (e.g., solving an equation for a final numerical value)?

2. The primary baseline is fixed-temperature sampling. While a standard choice, it doesn't fully challenge the dynamic nature of EAD. Have you considered comparing EAD to other dynamic schedules? For example, a "reverse" annealing schedule (exploit then explore) or a schedule that adapts the temperature based on the model's current uncertainty (e.g., increasing temperature when the next-token entropy is high), irrespective of position?

3. The global-step-aware decay rate appears to be a hand-tuned heuristic (line 240). Could you provide more justification for this specific functional form? An ablation study on the sensitivity of the linear growth factor (5) and the cap (40000) would strengthen the claim that EAD is a robust, "plug-and-play" method.

4. You mention that for the 7B model, a higher \tau_min was needed to prevent it from generating "plausible but incorrect solutions." This seems to contradict the core motivation of exploiting (i.e., generating high-probability, high-quality tokens) at the end. Could you elaborate on this? Does this suggest that for more capable models, a low temperature late in the sequence is actually harmful, thus undermining the "exploit late" principle?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new method of llm decoding that uses a high temperature for the token at the beginning of the sequence and gradually decay the temperature towards the end. The authors show that this method empirically improves inference time scaling, and can be incorporated into multiple RL algorithms to see further gains.

### Strengths
1. The method proposed is simple and can be easily integrated with current machine learning algorithms.

2. The empirical observation that exploring at the beginning phase is better than exploring at the later phase is insightful.

3. The paper provides comprehensive experiments to show the effectiveness of the method. They also provide empirical data related to the randomness to justify the statistical significance 

4. The authors clearly noted that in some cases, their algorithms' advantage may diminish / become smaller. This significantly improve the clearity of the result.

### Weaknesses
1. While it is plausible that the inequality behind 172-174 approximately holds, it is not a consequence of data processing inequality unless there are strong assumptions.

2. [Minor] The arragement of Figure 9 and 10 brings some confusions as Figure 9 is positioned at the section discussing incorporation with RL algorithms.

### Questions
Please refer to the weaknesses. Some additional questions include

1. Why EAD seems to have a collapsing worst-of-all in Figure 4 for llama model and at the same time has the highest pass@16? What is the intuition behind this?

2. How is the global-step-aware decay rate decided and is it strictly required to sample the trained model from the RL process using the same EAD temperature schedule as it is trained?

### Soundness
3

### Presentation
3

### Contribution
2
