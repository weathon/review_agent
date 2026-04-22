# The Alignment Waltz: Jointly Training Agents to Collaborate for Safety

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Harnessing the power of LLMs requires a delicate dance between being helpful and harmless, leading to two critical challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the metaphorical music entirely—it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. 
At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe responses or overrefusals from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical challenge of balancing helpfulness and harmlessness in LLMs, aiming to mitigate both adversarial vulnerability and overrefusal on benign prompts. The authors propose WALTZRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. This system jointly trains a conversation agent and a feedback agent, where the latter is incentivized by a Dynamic Improvement Reward to provide useful suggestions that improve the conversation agent's response, rather than simply blocking it. At inference, this feedback mechanism is applied adaptively, preserving low latency on safe queries. The paper's primary contribution is this collaborative framework, which, as demonstrated across five datasets, significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench), effectively advancing the Pareto front between safety and utility without degrading general capabilities.

### Strengths
1. Novelty of Method: The paper proposes a novel multi-agent RL method (WALTZRL) that frames the alignment problem as a collaborative, positive-sum game. This is a significant departure from standard safeguard models, which act as blunt classifiers, or other multi-agent approaches that often rely on zero-sum (adversarial) setups. By jointly training a conversation agent and a feedback agent to cooperate, this framework creates a new mechanism for adaptive, fine-grained safety corrections that can improve a response rather than just rejecting it.

2. Insightful Analysis: The authors provide insightful analysis and thorough ablation studies that successfully validate their core design choices. The investigation into the DIR variants in Section 3.3 is particularly strong, clearly demonstrating why conditioning the improvement reward on label correctness is crucial for balancing feedback usefulness and label accuracy. Furthermore, the justification for the two-stage training process effectively shows how it overcomes the sample diversity challenge as the conversation agent improves.

3. Clarity and Presentation: The paper is exceptionally well-written, and the overall presentation is clear, logical, and easy to follow. The "Alignment Waltz" metaphor is effective, and figures like Figure 1 provide an excellent, intuitive overview of both the inference-time protocol and the RL training step. This clarity makes the paper's novel contributions highly accessible despite the complexity of the underlying framework.

### Weaknesses
1. Limited Model Diversity: The experiments are confined to a single model architecture, Llama-3.1-8B, which is insufficient to claim broad generalizability. It is unclear how WALTZRL would perform with other prominent models, such as Qwen, or models known for different reasoning behaviors. Exploring the framework's effectiveness on models with diverse architectures and reasoning capabilities would significantly strengthen the paper's claims.

2. Simplified Datasets: The study relies on single-turn adversarial (WildJailbreak) and overrefusal (OR-Bench) datasets. These, while useful, do not reflect the complexity of multi-step, real-world interactions, such as those found in coding or computer-use agent datasets. The paper lacks analysis of how WALTZRL would handle more complex, emergent safety failures that can occur over extended, stateful interactions in agentic tasks.

### Questions
1. Can you provide additional results of Qwen?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. A conversation agent generates responses to users’ prompts and a feedback agent provide revision feedback for the conversation agent to improve the responses. Both agents are optimized simultaneously using reinforcement learning with rewards obtained from LLM judgers. A dynamic improvement reward (DIR) is integrated to encourage the feedback agent to focus on the improvements made by the conversation agent.

### Strengths
The paper is overall easy to follow. 
The idea is novel, compared to previous multi-agent framework where a zero-sum game is played between the agents, the proposed framework encourage the collaboration between agents, by designing the dynamic improvement reward that uses the improvements of conversation agent to guide the feedback agent.
The effectiveness of inference-time collaboration and RL-based joint training is well demonstrated.
Overall, the main contributions include: introduction of a multi-agent RL framework that optimizes two agents with a positive-sum game, introduction of the dynamic improvement reward that encourage the collaboration between the agents, and demonstration of the effectiveness of the proposed framework on multiple metrics.

### Weaknesses
The motivation is not clearly described, why the collaborative training can enhance helpfulness and reduce harmlessness?

The core drawback is the experimental setting where the collaborative round number is set to 1, I wonder whether such a setting is sufficient to fully harness/demonstrate the power of the proposed DIR. Also, it remains unknown whether the feedbacks are inherently consistent across collaborative rounds. There is a lack of ablation study to compare the performance of systems with and without DIR, currently, in Sec. 3.3, all (A), (B), and (C) settings include DIR. The stoping criteria is also not verified in the experiments.

The experiments should be also benchmarked with other multi-agent RL frameworks, e.g., (Zheng et al. 2024)

Zheng et al., Toward Optimal LLM Alignments Using Two-Player Games, 2024.

### Questions
The motivation may be improved, by describing the role that the feedback agent plays during training and inference, i.e., to provide reflection on over refusal for revision.

The ablation study on DIR, mentioned above, may be provided to better demonstrate its importance. The visualization of DIR can be also added to show the convergence of DIR.

There are grammatical errors that need to be fixed.

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
5

### Summary
This paper introduces WALTZRL, a novel multi-agent reinforcement learning framework designed to address the fundamental tension between helpfulness and harmlessness in LLMs. The central problem is the dual challenge of models being susceptible to generating unsafe content while also exhibiting over-refusal on benign but sensitive prompts. WALTZRL reframes safety alignment as a collaborative, positive-sum game between a conversation agent, which generates responses, and a feedback agent, which provides constructive suggestions for improvement. Through extensive experiments across five diverse datasets, the authors demonstrate that WALTZRL significantly reduces both ASR and Over-Refuse Rat compared to strong baselines.

### Strengths
1. A key strength is the work's direct focus on reducing over-refusal. Unlike many safety methods that sacrifice helpfulness, this approach successfully improves the model's ability to respond to sensitive yet benign prompts.
2. The method is supported by comprehensive experiments across diverse benchmarks for safety, over-refusal, and general capabilities, showing clear improvements over strong baselines.

### Weaknesses
1. The experiments are limited to a single model (Llama-3.1-8B). This raises questions about how the approach generalizes to other model families or sizes. Including results on another model would significantly strengthen the paper's claims.
2. The paper only evaluates a single round of feedback (T_max=1), even though the framework supports more. The potential effects of multi-round interactions, including possible failure modes, are not explored.

### Questions
Did you try an ablation where you only train the feedback agent and keep the conversation agent frozen throughout? This would help clarify the benefits of co-evolving the agents versus simply pairing a static model with a trained feedback module.

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
3

### Summary
This work proposes a MARL framework to make the safety alignment as a positive-sum game by training a conversation agent and a feedback agent. The dynamic improvement reward encourages the conversation agent to take the feedback effectively. Experiments show results that unsafe and over-refusal responses are reduced for a better trade-off.

### Strengths
The paper is well-written and easy to follow. The novelty is sound, and the idea of game theory is interesting for the storytelling. The presentation of figures and algorithms are clear to me.

### Weaknesses
- Overall, I feel like the implementation is full of tricks and the details are not clear enough, which makes the story of game theory less convincing in practice.

- When gathering RL states and actions, it is expected to justify why only one random round is chosen as final feedback agent trajectory. Why not use the all T trajectories?

- Since the conversation agent trajectory is randomly chosen from A and B, more experiments need to show its benefits by comparing the cases where only A or B is taken.

- I feel like it is kind of what we want when initial responses from a conversation agent are already safe and not overrefusing, so the adaptive feedback needs clearer justification.

- In table 1, it seems the results of single-model RL and oracle label-converted feedback are better and might be more generalizable. More experiments of safety and over-refusal on different datasets are needed to show the generalizability.  Also the Pareto front is supposed to be a curve not a single point in the 2D plane for each method.

- Also, inference-time collaboration based on prompts seems to be more generalizable to me since it is training-free. It needs more justification regarding why RL post-training is needed.

- Figure 2 lacks confidence intervals or standard deviations, making it hard to distinguish between setups B and C.

### Questions
See Weakness

### Soundness
2

### Presentation
3

### Contribution
2
