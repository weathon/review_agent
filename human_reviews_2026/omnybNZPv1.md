# Inference-based Rewards for Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
A central challenge in reinforcement learning (RL) is defining reward signals that reliably capture human values and intentions. Recent advances in vision–language models (VLMs) suggest they can serve as a powerful source of semantic rewards, offering a flexible alternative to environment-defined objectives. Unlike hand-crafted signals, VLM-based feedback can reflect high-level human goals such as safety, efficiency, and comfort. We first analyze the conditions under which VLM-based rewards enable effective learning. In particular, we highlight the importance of monotonicity with respect to true task performance and the satisfaction of the Markov property. When these conditions hold, VLMs provide a viable basis for reward inference. On the algorithmic side, we identify what learning strategies are best suited for such rewards. Trajectory-based methods such as policy gradient (e.g., REINFORCE, PPO) are naturally aligned with inferred returns, whereas Q-learning style algorithms are more fragile because they operate on step-wise Bellman updates (e.g., DQN) and implicitly assume the Markov property of rewards. This perspective reframes RL around reward inference rather than reward specification, highlighting both the promise of VLM-based alignment and the theoretical and practical boundaries of when such methods are effective.
Experiments across control domains provide supporting evidence for these insights. In particular, monotonicity appears to align with learning outcomes, REINFORCE and PPO show greater robustness than DQN when trained with inferred rewards, and natural language prompts can guide the emergence of instruction-driven behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies inferred rewards framework to analyze the use of Vision-Language Models (VLMs) as reward providers. The paper proposes two key properties: monotonicity (two trajectories order based on true reward should remain the same based on the any provided reward mechanism) and the Markov property. The paper claims that policy gradient-based methods are more robust to the non-Markovian nature of VLM rewards because they depend on trajectories, rather than step-wise Bellman updates which are fully dependent on the Markovian nature. Experimental results support this claim.

### Strengths
1. The paper attempts to systematize the discussion around VLM-based rewards. By explicitly identifying and naming monotonicity and the non-Markovian nature of these rewards as key properties, it provides concepts to analyze this problem. While the necessity the non-Markovian in VLM have been noted in prior work [1], [2], the paper aims to frame this. Moreover, the monotonicity they defined seems to be obvious property we desired, which states that optimizing proposed reward function should imply 'true' reward function.

2. The paper aims to bridge its conceptual framework with empirical validation. It doesn't just discuss monotonicity in the abstract; it proposes concrete metrics (e.g., pairwise agreement, Kendall's $\tau$) for measuring it.

### Weaknesses
1. This paper tries to frame the necessary properties for VLM-based rewards, but there is no rigorous theoretical arguments on it. Also, some of claim can be more clear. For example, line 255, the equation shows: $R\left(\tau_1\right)>R\left(\tau_2\right) \Rightarrow  \hat{R}\left(\tau_1\right)>\hat{R}\left(\tau_2\right)$. I think they want to claim 'order preserving' regarding some rewards. Intuitively, the same logic, but different equation looks much clear in my opinion; $\hat{R}\left(\tau_1\right)\leq \hat{R}\left(\tau_2\right) \Rightarrow  R\left(\tau_1\right) \leq R\left(\tau_2\right)$. Since even if we change the original equation to $R\left(\tau_1\right) \leq R\left(\tau_2\right) \Rightarrow  \hat{R}\left(\tau_1\right) \leq \hat{R}\left(\tau_2\right)$. Their claim should still hold.

2. In section 4.3, they mention policy gradient methods and Q-learning based methods. However, during experiments, they work on PPO and DQN, where PPO depends on Actor-Critic method relying on value difference lemma, which is not pure policy gradients method as they rely on Bellman-update as well.

3. To measure monotonicity, they still need "True return of the given trajectory", which leads to impracticality when we don't have any true reward function.

### Questions
1. Your measurement method still need 'true' reward function, how much this will be useful when we don't have any access to the true reward function to solve the problems. 

2. On line 255, conceptually, the following equation looks much easier to interpret even though they have the same meaning; $\hat{R}\left(\tau_1\right) \leq \hat{R}\left(\tau_2\right) \Rightarrow  R\left(\tau_1\right) \leq R\left(\tau_2\right)$, which simply says order preserving regarding reward. I think even if you change the original equation in Line 255 to $R\left(\tau_1\right) \leq R\left(\tau_2\right) \Rightarrow  \hat{R}\left(\tau_1\right) \leq \hat{R}\left(\tau_2\right)$, your definition of 'monotonicity' should remain the same. Is this correct? 

[1] J. Beck, "Offline RLAIF: Piloting VLM Feedback for RL via SFO," arXiv preprint arXiv:2503.01062, 2025.

[2] H. Kang, E. Sachdeva, P. Gupta, S. Bae, and K. Lee, "GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks," arXiv preprint arXiv:2503.06514, 2025.

### Soundness
2

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
3

### Summary
This paper addresses the common misalignment between the reward function and the true intended target in reinforcement learning (RL). The authors propose an algorithm that replaces the original reward with a VLM-based reward, which evaluates the alignment between a trajectory window and a text-specified goal. The paper also discusses the conditions that ensure the effectiveness of such rewards and analyzes which RL algorithms are suitable for this inferred reward through theoretical reasoning and experiments, with an emphasis on the monotonicity of the inferred reward and the Markovian requirement.

### Strengths
1. The paper provides a practical and simple approach for leveraging the prior knowledge of VLMs to generate high-quality reward signals.

2. Theoretical analysis of the proposed method is provided, explaining why and when it achieves better performance. In addition, the paper offers practical methods for quantifying the monotonicity of the inferred reward.

3. The experimental design is detailed and effectively supports the analysis and assumptions.

### Weaknesses
1. Although the experiments provide supportive and intuitive results for the analysis (monotonicity and Markovian condition), there are few experiments that directly compare performance under the proposed reward with that under traditional rewards. More experimental results are expected to further confirm the effectiveness of the framework.

2. The experimental results show reduced correlation in more complex environments. Does this imply that the proposed method is currently limited to simpler tasks?

3. The quality of the inferred reward largely depends on the pretrained CLIP model, which is not fine-tuned during the InfeRL process. If the VLM model becomes the bottleneck, how can performance be further improved?

### Questions
See the weaknesses above.

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
The paper proposes Inference-Based Reinforcement Learning (InfeRL): instead of using environment rewards, an agent infers rewards from a pretrained vision–language model (VLM) conditioned on a natural-language goal. The core theoretical stance is that two structural properties largely determine whether such rewards are useful for learning: (i) trajectory-level monotonicity—the inferred return preserves the ordering of trajectories by true performance—and (ii) (quasi-)Markovianness—per-step reward can be made Markovian with a bounded temporal window. The paper argues that policy-gradient methods (e.g., PPO) depend mainly on trajectory-level signals and thus tolerate non-Markov rewards better than value-based methods (e.g., DQN), whose Bellman backups assume Markovian per-step rewards. Empirically, the authors compute rank-correlation metrics between true and inferred returns and compare PPO vs. DQN on CartPole variants and MuJoCo control.

### Strengths
- Recasting reward design as reward inference aligns with a fast-growing line of work that uses pretrained models or language to specify goals. The paper’s explicit articulation of trajectory-level monotonicity as a practical diagnostic is helpful for practitioners who currently rely on ad‑hoc prompt tinkering.
- The PPO vs. DQN contrast is plausible: policy gradients can optimize any scalar trajectory return (REINFORCE/GPOMDP), whereas Bellman targets fail when per-step rewards are history-dependent.
- Using Kendall’s τ / Spearman’s ρ to validate whether a prompt yields a usable reward signal is straightforward and reproducible.

### Weaknesses
- Much of the paper’s core message—“optimize trajectory-level signals from learned or inferred rewards; PG tolerates non-Markov structure better than Q-learning”—has been made in previous works.
  - Preference-based RL / RLHF already builds on trajectory orderings (pairwise comparisons) and uses policy-gradients over those signals [1].
  - VLM‑as‑reward and VLM‑feedback papers (zero‑shot rewards, RoboCLIP, RL‑VLM‑F) already examine robustness/instabilities and how to elicit better signals; comparisons here are too light [2-4].
- The paper motivates monotonicity conceptually but does not establish conditions under which monotonicity of returns implies true policy improvement under stochastic sampling (e.g., how much anti‑monotone noise can PPO tolerate?). A tighter connection to classical reward transformation invariance would help distinguish what is genuinely new (monotone trajectory ranking) vs. known (potential‑based shaping, affine transforms).
- The Markov critique of DQN hinges on the reward’s history dependence, but state augmentation (frame stacking / recurrent Q‑nets) and reward delay often restore Bellman validity. The experiments don’t report whether DQN received comparable temporal context or whether reward delays were tuned, so the conclusion may partly reflect architecture mismatch, not an inherent limitation.

# References

[1] Deep reinforcement learning from human preferences. 

[2] RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback.

[3] RoboCLIP: One Demonstration is Enough to Learn Robot Policies.

[4] Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a framework, InfeRL (Inference-based Reinforcement Learning), that characterizes the conditions under which VLM-based rewards, particularly similarity-based rewards, are effective for policy learning. The authors identify two key factors for stabilizing policy learning: the monotonicity of trajectory-level VLM-based returns and the Markov property of per-step VLM-based rewards. Based on these conditions, the paper examines which classes of RL algorithms, including policy gradient and Q-learning, are most suitable for training with such rewards. To quantify reward monotonicity, the authors introduce metrics such as pairwise agreement and rank correlation (Kendall's $\tau$ and Spearman's $\rho$). The framework is evaluated on two classical control tasks (CartPole and InvertedPendulum) and two locomotion tasks (Walker2D and Ant). Results indicate that policy gradient methods are generally more robust and effective for learning with VLM-based rewards.

### Strengths
1. The paper is clearly written and easy to follow, and the problem is well-motivated.
2. It addresses an important question: under what conditions are VLM-based rewards useful for reinforcement learning. From an algorithmic perspective, the paper provides insightful analyses of why common RL algorithms succeed or fail when trained with VLM-based rewards.
3. The introduced metrics for measuring monotonicity are valuable, as they can help design better language prompts or select appropriate VLMs.
4. The work has the potential to make a meaningful contribution, though it would benefit from further clarification of the framework and experimental setup.

### Weaknesses
1. The proposed framework uses the CLIP model as a VLM, but since the CLIP model takes a single image as input, it is unclear how it is used to derive rewards for a trajectory fragment. Is the reward obtained by averaging the individual similarity scores across frames?
2. The paper mentions goal-conditioned policies within the InfeRL framework and their ability to generalize at test time, but it appears that this type of policy was not actually used in the experiments.
3. In the experiment investigating monotonicity (Section 5.2), the evaluated trajectories are generated by a random policy. Is this sufficient to include both successful and failed trajectories? For instance, in Walker2D, random actions are likely to cause the robot to collapse. Therefore, the statement that "the inferred rewards distinguish between successful walking and collapsed robots" seems inappropriate. Could the authors elaborate on how this benchmark dataset was constructed?
4. Although Table 1 reports monotonicity scores across different environments, there is no analysis of how the degree of monotonicity correlates with the agent's actual performance. Including such insights would greatly strengthen the paper. For example, how does varying the degree of monotonicity (e.g., by adjusting language prompts or selecting different VLM models) affect the overall performance of the agent?
5. There is limited investigation into the Markov property. If the MDP is augmented to make it more Markovian (e.g., by using the same history or future context for both the reward and Q-function, as mentioned in Section 4.2), would the stability of DQN improve?
6. The experiment in Walker2D (Section 5.4) is not particularly compelling in supporting the paper's contribution, primarily due to the lack of ground-truth rewards. Without these, it is unclear how the monotonicity metrics are being utilized in this context. Additionally, the learning curve referenced in Line 456 seems to be missing from Figure 2.
7. Missing implementation details: What is the window size value? How does the VLM provide rewards for a trajectory, at the end (sparse) or at each timestep (dense)? Without these details, it may be difficult to understand the Markov property of VLM-based rewards.

Minor comments:
- In line 306, should $s_{t+k}$ be $s_{t+k-1}$?
- In Table 2, is the Cartpole-Base result missing? 

Suggestion:
- The experimental design could be strengthened: The paper focuses on two axes of VLM-based rewards: monotonicity and Markov properties. If we consider each property as being either "strong" or "weak", the current experiments seem to focus primarily on strong monotonicity and weak Markov property. To provide deeper insights, it would be helpful if the experiments could also explore cases where both properties vary, reflecting a broader range of scenarios.

### Questions
Please see the Weakness.

### Soundness
2

### Presentation
3

### Contribution
3
