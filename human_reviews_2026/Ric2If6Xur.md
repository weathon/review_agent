# Difficulty-Aware Reasoning for Mobile GUI Automation via Reinforcement Fine-Tuning

- Avg Score: 3.60
- Decision: Reject
- Scores: 4, 4, 2, 4, 4

## Abstract
Automating GUI tasks remains challenging due to layout complexity, element density, and intent ambiguity, which requires effective and efficient reasoning to facilitate each operation. Existing agents typically employ a uniform chain-of-thought (CoT) reasoning process for all actions, a one-size-fits-all approach that incurs unnecessary computational overhead and even performance degradation on trivial steps.
To address this, we introduce \textbf{AdaGUI-R1}, a GUI agent that pioneers a difficulty-aware reasoning paradigm by dynamically modulating its reasoning depth based on action complexity. Our methodology consists of reasoning inducing and reasoning enhancing.
During reasoning inducing, we introduce a self-supervised mechanism to generate high-quality, difficulty-aware reasoning trajectories. Fine-tuning on this curated data endows the agent with the fundamental capability to adjust its reasoning depth according to action complexity. Subsequently, Group Adaptive Policy Optimization (GAPO) algorithm is implemented to enhance reasoning performance. It leverages an adaptive thought reward to encourage thinking on challenging steps, and a novel exploration reward with a difficulty-aware Gaussian bandwidth to improve action accuracy.Extensive experiments demonstrate that AdaGUI-R1 sets a new state-of-the-art. It concurrently reduces unnecessary reasoning tokens by 40% while improving action accuracy by 5%, underscoring the power of adaptive reasoning in GUI automation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper *“Difficulty-Aware Reasoning for Mobile GUI Automation via Reinforcement Fine-Tuning (AdaGUI-R1)”* introduces a new framework for mobile GUI agents that can dynamically adjust how much reasoning they perform based on the difficulty of each step. Traditional GUI automation models apply uniform reasoning chains to all tasks, which leads to inefficiency—simple steps are overanalyzed while complex ones lack adequate reasoning. AdaGUI-R1 addresses this by integrating **difficulty-aware reasoning** into both supervised and reinforcement fine-tuning stages.

In the first stage, the model learns when to think by generating “Think–Action” pairs only for difficult steps, while easy ones receive a placeholder “None” thought. A self-supervised consistency mechanism ensures that generated reasoning aligns with the correct actions. In the second stage, the authors propose Group Adaptive Policy Optimization (GAPO), which introduces two key rewards: an adaptive thought reward that encourages longer reasoning for hard steps and shorter for easy ones, and a Gaussian exploration reward that provides smoother, distance-based feedback for click actions with difficulty-sensitive variance.

Experimental results on multiple mobile GUI benchmarks show that AdaGUI-R1 outperforms prior state-of-the-art methods, achieving around **5% higher action accuracy** while reducing reasoning token usage by **about 40%**. The study demonstrates that allocating reasoning effort adaptively—thinking deeply only when needed—can improve both efficiency and robustness in GUI automation agents.

### Strengths
The paper contributes a new difficulty-aware reasoning paradigm for GUI automation. Instead of applying a uniform Chain-of-Thought across all tasks, it introduces a principled way to adjust reasoning depth based on estimated step difficulty. This rethinking of how reasoning effort should be distributed marks a clear conceptual advancement over prior “one-size-fits-all” reasoning frameworks. The work provides several concrete and novel algorithmic components:  
   - A self-supervised CoT generation mechanism that ensures consistency between thought and action.  
   - The Group Adaptive Policy Optimization (GAPO) algorithm, integrating adaptive thought rewards and Gaussian exploration rewards.  

   Together, these elements enhance stability, exploration efficiency, and reasoning adaptability, forming a cohesive and technically sound framework.

AdaGUI-R1 achieves substantial performance improvements on multiple GUI automation benchmarks, increasing success rates while reducing reasoning token usage by about 40%. These empirical gains demonstrate that adaptive reasoning not only improves efficiency but also sets a foundation for broader applications in multimodal and interactive AI systems.

### Weaknesses
1. The paper primarily evaluates the model on multiple offline benchmarks, where the metrics focus on step-level accuracy. However, these benchmarks differ from the more commonly used online interactive benchmarks (e.g., AndroidWorld, AndroidLab) that measure full-task success rates (SR). It is recommended to include a discussion on how the proposed method relates to these online benchmarks, and to report additional results of AdaGUI-R1-7B and its ablation models on such interactive benchmarks. Demonstrating effectiveness on SR metrics would significantly strengthen the paper’s empirical validity.

2. This paper emphasizes step-level difficulty awareness, with extensive design innovations in the “think” component compared to prior work. It would be beneficial to provide several case analyses, including examples of how steps are categorized by difficulty, and how the thought content changes before and after training with the Thought Reward mechanism. Such qualitative insights would clarify the behavioral impact of the proposed difficulty-aware design.

3. The Action Exploration Reward section improves upon the conventional binary (0/1) feedback by introducing a Gaussian Exploration Reward, which smooths the reward function for individual actions. However, computing Gaussian functions introduces additional computational cost compared to previous approaches. It is recommended to provide comparative experiments—such as evaluating against baselines that use bounding-box inclusion or distance-based penalty smoothing—to demonstrate that this extra computational overhead yields meaningful performance benefits.

### Questions
The questions are already included within the weaknesses section.

### Soundness
3

### Presentation
3

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
This paper presents AdaGUI-R1, a mobile GUI agent that introduces difficulty-aware reasoning, dynamically adjusting its reasoning depth based on task complexity. The method integrates a self-supervised CoT generation process to produce consistent reasoning-action pairs and a Group Adaptive Policy Optimization (GAPO) algorithm with adaptive thought and exploration rewards. Experiments on multiple GUI automation benchmarks demonstrate significant improvements in both accuracy and efficiency.

### Strengths
- Innovative reward design: The paper introduces a difficulty-aware reward mechanism that assigns different reward functions to actions of varying difficulty, effectively aligning reasoning depth with task complexity.

- Strong empirical results: The proposed AdaGUI-R1 achieves strong performance on three benchmarks, surpassing prior models in both accuracy and efficiency.

### Weaknesses
- More qualitative examples (e.g., reasoning traces comparing easy vs. hard steps) are required.
- Lack of novelty in "Self-Supervised CoT Generation": The proposed "self-supervised CoT generation" closely mirrors the STaR [1] approach and does not introduce a fundamentally new mechanism. The pipeline of generating initial CoT, validating actions, and revising reasoning is nearly identical to prior methods.
- Soft reward on coordinates is not new: Using spatially smoothed rewards for click or grounding actions has been explored in previous RL-based GUI grounding works.

[1] STaR: Bootstrapping Reasoning With Reasoning.

### Questions
- What is the performance of model after SFT.
- According to the experimental results, the average reasoning token length of AdaGUI-R1 is less than 20 tokens, which suggests that the model performs almost no explicit reasoning. So does it mean that it is enough to train the model to predict actions? And have you observed that the decrease in thought length is mainly due to the omission of content?
- What content or reasoning elements are being omitted in the reduction of cot length?

### Soundness
3

### Presentation
3

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
Automating GUI tasks is challenging because of the task complexity. State-of-the-art employs chain-of-thought reasoning in order to deal with such complexity; however, they equally apply the reasoning protocol without considering the difficulty or complexity of the sub-tasks. This results in computational inefficiency because of applying unnecessary reasoning steps for trivial sub-tasks. Also, such a one-fit-for-all approach results in performance degradation, especially in complex subtasks, since a fixed number of reasoning steps may not be enough for some complex sub-tasks, while being more than enough for trivial ones. This paper introduces a difficulty-aware reasoning, which adapts the depth of reasoning to the action complexity. The core idea is determining the difficulty of subtasks via a pre-trained VLM for GUI tasks. Then, a reward component is employed to reward the model for thinking longer for harder tasks and shorter for easier ones. The proposed method shows performance improvements over baselines for GUI tasks.

### Strengths
- **Difficulty-aware Reasoning**: Determining the difficulty and encouraging the agent to think longer for harder sub-tasks for GUIs looks promising for GUI agents. 
- **Improvements over Baselines**: Results are promising, showing significant improvements over the base model they compared.
- **Reward Components**: Reward components are ablated nicely, and it is shown that each component helps the agent to reach a better performance.
- **Difficulty-threshold Analysis**: The effects of the difficulty threshold, which takes a key role in $R_{thought}$, on the model performance is analyzed nicely.

### Weaknesses
**Major**:
- **Novelty**: The paper's claims and coverage are scoped entirely to GUI automation, and the key contribution described as the difficulty-aware CoT. However, there are already works in literature where CoT is adapted based on the task difficulty (see below). These works must be discussed in detail, and the proposed approach should be compared with them, since the novelty of the proposed method is questionable beyond the experimental setting.  

[a]: Waheed, Abdul, et al. "Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation." arXiv preprint arXiv:2509.05226 (2025).

[b]: Yu, Zishun, et al. "Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization." Forty-second International Conference on Machine Learning.

[c]: Wang, Xinglin, et al. "Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning." Findings of the Association for Computational Linguistics: NAACL 2025. 2025.

[d]: Han, Tingxu, et al. "Token-budget-aware llm reasoning." arXiv preprint arXiv:2412.18547 (2024).

[e]: Aggarwal, Pranjal, Aman Madaan, and Yiming Yang. "Let’s Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs." Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023.

[f]: Damani, Mehul, et al. "Learning How Hard to Think: Input-Adaptive Allocation of LM Computation." The Thirteenth International Conference on Learning Representations.

- **Computational Overhead**: The proposed approach requires difficulty estimation and CoT pre-training. However, the computational overhead over the baselines are not discussed. This must be clearly elaborated, and equal-compute comparisons and a quality–cost curve (tokens/FLOPs/wall-clock) should be presented.
- **Discrete Difficulty Levels**: It is unclear why only five discrete difficulty levels are selected. This design choice is not explained.
- **CoT Pre-training**: It is mentioned that they first teach how to generate CoT to their agent, using a curated, annotated dataset. The details of such fine-tuning and the data size are unclear. 
- **Reward Design**: It is unclear how R_{thought} is designed.  It would also be great to show the impact of this reward design on the model's performance by comparing it with simpler designs.
- **Experimental Results**: In the experimental results in Table 1, for AITZ, UI-TARS-7B works better than the proposed approach. However, these results are not written in bold; instead, the results of the proposed approach are written in bold, which is misleading. This must be corrected. Also, please elaborate on why UI-TARS-7B performs better than the proposed approach.
- **Confidence Intervals**: No confidence intervals are presented for the results in Table 1. Please report the confidence intervals either in the main table or in the appendix. This is important since it looks like UI-TARS-7B performs closely to the proposed approach.

**Minor**: 
- **Presentation**: Figure 1-right is very unclear. Please either explain what this figure shows in the caption in detail or remove it.

### Questions
- In figure 1-right, why do we have multiple bars for different levels? What are these levels, the difficulty levels according to Eq. 4?
- Why are there five discrete difficulty levels? Have you ever considered continuous difficulty levels, or are there any drawbacks to using them?
- Please explain how you teach "how" and "when" to generate CoT to your agent. Can you also elaborate on the data curation and the size of the data used for such pre-training?
- How is R_{thought} designed? It is mentioned that the function is smooth and strictly monotonic, but is this the only design consideration? Have you ever considered different functions?
- Why does UI-TARS-7B perform better than the proposed approach in AITZ? This must be clearly elaborated.
- Can you please report confidence intervals for the results in Table 1?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces reinforcement learning fine-tuning algorithm that induces difficulty-aware reasoning. The intuition is that the model should "reason" only in difficult states.

The proposed method works in two stages:
1. Inducing reasoning using supervised fine-tuning.
2. Refining reasoning to be difficulty-aware through a novel reinforcement learning algorithm: GAPO that defines rewards based on the difficulty of the decision in that state. Precisely, during fine-tuning an agent is penalised when providing an answer to a difficult step without reasoning, or for too much "thinking" in low-difficulty states.

In the first stage the difficulty is measured by another VLLM, while in the second stage the difficulty is computed on the fly based on the group of generated samples.

The suggested method outperforms other recent grounding models with or without reasoning.

### Strengths
- Comprehensive set of ablations for all components of the algorithm.
- Cost analysis demonstrating efficient use of tokens (a performance / tokens 2D plot would be useful for visualisation here)
- Strong empirical results.

### Weaknesses
1. No direct comparison with other adaptive reasoning methods.
2. Reduced novelty: adaptive thinking has been introduced before.

### Questions
1. Why isn't the current solution compared to AdaptThink, AdaCoT, ThinkSwitcher... works that were correctly mentioned in Section 2?
2. When computing the difficulty level $l$ during "enhancement", rather than sampling multiple times and counting how many generations exactly match the target, wouldn't be easier to measure the log-probs of the correct answer against a threshold?
3. Do all the models in Table 1 follow the same protocol of using half of the data for training and half for testing. Is the split the same?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to create a difficulty aware reasoning paradigm that aims to reduce token budget and adaptively reason longer on hard steps and avoid reasoning longer on easy ones. The paper overall manages to reduce the token budget and not reduce performance at the same time. It is a timely and interesting work  that aims to increase the metacognitive abilities of the agent by helping in its resource allocation of thinking.

### Strengths
The paper tackles the problem of thinking budget of the agent thereby increasing the metacognitive aspects of decision making in GUI environments. The paper also increases the efficiency of exploration of agent in a high dimensional setting as GUIs by including an action exploration reward for the hard steps where reward is sparse and making it more dense with a Gaussian exploration reward
Ablations and experiments seem robust.

### Weaknesses
Even though the agent is advertised as self supervised, it does require ground truth labels from existing datasets that are human annotated and in that the approach isn’t general or scalable. 

There's a possibility that there is a lot of redundancy in these datasets, so I am not sure the degree of generalization that this approach promises. In general, on exploration task fine-tuning on similar trajectories improve performance. Hence results could also be explained by data leak, as the model get finetuned  on half the trajectories randomly selected. So some sort of stratified split up of the tasks in these datasets showing some analysis would be helpful to understand as in where exactly the performance is coming from. 

Some figures are not referenced in the text.

### Questions
In line 399, the paper claims and I quote “not only improves the accuracy of hard steps, but also avoids the hallucination triggered by introducing over analysis in easy steps”, this is a serious claim but I would like to see some evidence from some analysis which you could do to show it.

There are eight actions: key, click, swipe, long press, type, system button, terminate, and wait. Is the gaussian exploration function helpful for all these actions or is it limited to some?

Any reason why this algorithm was tested only on mobile GUI environments only (not a demerit of the paper)?

Any reason you went for an easy/hard dissociation instead of a graded difficulty thinking budget. What I mean is it could have easily been a continuous function ( like agreement of high accuracy outputs) of difficulty adaptation instead of just these two distinct categories.

How do you ensure that the agent doesn't reward hack the length bonus and increases the token budget unnecessarily for hard problems which such a long length might not be needed. It could be that you are managing to reduce the token budget for easy steps but increasing it for harder ones unnecessarily with your design choices.

I would be willing to revise my scores if you could answer these questions satisfactorily and address weakness.Thank you.

### Soundness
2

### Presentation
2

### Contribution
2
