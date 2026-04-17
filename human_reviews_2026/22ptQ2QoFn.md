# Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Recent thinking models are capable of solving complex reasoning tasks by scaling test-time compute across various domains, but this scaling must be allocated in line with task difficulty. On one hand, short reasoning (underthinking) leads to errors on harder problems that require extended reasoning steps; but, excessively long reasoning (overthinking) can be token-inefficient by generating unnecessary steps even after reaching a correct intermediate solution. We refer to this as under-adaptivity, where the model fails to modulate its response length appropriately given problems of varying difficulty. To address under-adaptivity and strike a balance between under- and overthinking, we propose TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model’s self-attention over a long reasoning trajectory to identify important steps and prune redundant ones. TRAAC also estimates difficulty and incorporates into training rewards, thereby learning to allocate reasoning budget commensurate with example difficulty. Our approach improves accuracy, reduces reasoning steps, and enables adaptive thinking compared to base models and other RL baselines. Across a variety of tasks (AIME, AMC, GPQA-D, BBEH), TRAAC (Qwen3-4B) achieves an average absolute accuracy gain of 8.4% with a relative reduction in reasoning length of 36.8% compared to the base model, and a 7.9% accuracy gain paired with a 29.4% length drop compared to the best RL baseline. TRAAC also shows strong generalization: although the models are trained on math datasets, they show accuracy and efficiency gains on out-of-distribution non-math datasets like GPQA-D, BBEH, and OptimalThinkingBench. Our analysis further verifies that TRAAC provides fine-grained adjustments to thinking budget based on difficulty and that a combination of task-difficulty calibration and attention-based compression yields gains across diverse tasks

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
It proposes a novel online post-training reinforcement learning (RL) method designed to address under-adaptivity in reasoning language models—i.e., their tendency to either underthink (terminate too early on hard problems) or overthink (generate excessive, redundant steps on easy problems).
Despite being trained only on math data (DAPO-Math-17k), TRAAC generalizes to out-of-distribution non-math tasks (e.g., GPQA-D, BBEH, OptimalThinkingBench), showing consistent gains in both accuracy and efficiency.

TRAAC enables language models to "think right": neither too little nor too much—optimizing both reasoning quality and token efficiency through adaptive, attention-guided compression.

### Strengths
1. Difficulty-aware compression: Using rollout pass rates during GRPO to estimate problem difficulty and modulate compression intensity。
2. Self-attention as a salience signal: Leveraging the model’s own attention from the </think> token to identify and prune low-importance reasoning steps—without external annotators or auxiliary models.
3. Uniformity-aware pruning: Introducing a mechanism to reduce compression when attention is diffuse (i.e., when no step clearly dominates), preventing harmful over-pruning.
4. Reducing reasoning length by ~37% while boosting accuracy by 8.4% has direct implications for cost, latency, and scalability in real-world deployments.

### Weaknesses
1. The results are mainly on small-scale model size such as qwen3-4b/deepseek-r1-7b, it lacks of large-scale model-size exp.
2. Compared to [1], the length-reduction is not as good as [1].
3. The performance of deepseek-r1-7b on aime24 seems lower compared to [2][1] which is 55.4, while the results on aime24 on this work is only 33.71. It is confusing.



[1] DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforcement Learning
[2] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

### Questions
1. The performance of deepseek-r1-7b on aime24 seems lower compared to [2][1] which is 55.4, while the results on aime24 on this work is only 33.71. It is confusing.
2. The training reward is the summarization of Correctness Reward,  Format Reward and Length Reward. Did you try to do the ratio search to balance the different aspect and which one is important for length-reduction?






[1] DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforcement Learning
[2] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

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
4

### Summary
The paper presents TRAAC, a reinforcement learning framework designed to mitigate under-adaptivity in the reasoning process of large language models.

TRAAC introduces an adaptive attentive compression module that leverages the attention pattern surrounding the `</think>` token and integrates difficulty-level calibration. The method is further enhanced by a reward system that jointly considers correctness, output format, and reasoning length.

Experimental results demonstrate that TRAAC improves both performance and efficiency while maintaining strong generalization capability across different domains.

### Strengths
1. The paper proposes the TRAAC method, offering a novel and well-motivated approach for enhancing the fine-tuning process of large reasoning models.
2. The experimental evaluation includes a comprehensive set of recent and competitive baselines, effectively validating the robustness and effectiveness of the proposed approach.

### Weaknesses
1. The approach relies heavily on the model’s inherent reasoning ability, as the training data are generated from the model itself. This dependence makes it difficult to eliminate biases or artifacts inherited from previous training phases, potentially limiting the robustness of the learned behavior.
2. The experiments are conducted only on small models with similar architectures (DeepSeek-Qwen-7B and Qwen3-4B), both of which are Qwen-based and up to 7B scale. This narrow model selection leaves the reliability and scalability of the TRAAC method on larger or architecturally diverse models unverified.

### Questions
1. What is the motivation for adopting a reinforcement learning (RL) framework in this context? Using truncated reasoning traces for supervised fine-tuning (SFT) appears to be a more intuitive approach—could the authors clarify the specific advantages of RL in this setting?
2. For models without inherent reasoning capabilities, is it possible to directly train reasoning ability by leveraging reasoning traces generated by other, more capable models?
3. In the attention-based compression stage, the selection is guided solely by the attention pattern around the `</think>` token. Is this approach sufficient to capture all relevant dependencies? Why not consider the full-context attention following the `</think>` token for a potentially more comprehensive compression strategy?

### Soundness
3

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
This paper introduces TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method
that leverages the model’s self-attention over a long reasoning trajectory to identify important steps and prune redundant ones.
Trained with the TRAAC algorithm, reasoning models could learn to think adaptively, thereby enabling more efficient reasoning process.

### Strengths
1. This paper is well written and easy to follow.
2. This paper works on an important problem (efficient test time scaling).

### Weaknesses
1. **Ignored extra computations** The main contribution of this paper is the proposed TRAAC method, which incoporates an attention-based score to the computation of the reward in online RL. Trained with TRAAC, the reasoning model could think more efficiently on token level. However, the added Attention-Based Compression introduces extra computation complexity. There is no discussion about this problems. Additionally, the impact from different types of attention like Multi-head self attention, Grouped-Query Attention seems to be ignored.

2. **Insufficient evaluations** The experients only includes few reasoning benchmarks (AIME, GPQA). I'm concerned about the generalization of this method to other areas. It's not clear that whether the TRAAC method would do harm to some other abilities like conversations and agentic tasks. Adding these discussion will make this paper more reproducible.

### Questions
see Weakness.

### Soundness
3

### Presentation
2

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
This paper introduces TRAAC, a post-training reinforcement learning method designed to address the problem of under-adaptivity in large language models (LLMs)—the failure to dynamically allocate reasoning effort (thinking length) based on problem difficulty. Under-adaptivity leads to overthinking on simple problems (wasting compute) and underthinking on hard problems (sacrificing accuracy).

### Strengths
Explicitly incorporating estimated task difficulty to modulate the compression rate is a clear and direct way to tackle under-adaptivity, which is often only implicitly addressed in prior work.
Results show that TRAAC simultaneously improves accuracy and reduces reasoning length compared to strong baselines, demonstrating effective adaptive reasoning that generalizes to out-of-distribution tasks.

### Weaknesses
If the responses are trimmed, they are usually not coherent and readable, which significantly impacts the model's usability and user-friendliness.
The paper relies on the assumption that low attention from </think> implies a reasoning step is redundant. While the results are promising, there is no analysis to validate that the pruned steps are indeed unhelpful or that the attention mechanism is a faithful indicator of reasoning importance. 
While TRAAC improves test-time efficiency, the computational cost of the online GRPO training—which involves generating multiple rollouts, computing attention scores for the entire trajectory, and performing compression for each training step—is likely substantial. The paper does not discuss the training efficiency or compare it to the baselines. A discussion of this trade-off (offline training cost vs. online inference savings) is important for a complete picture.

### Questions
Could you provide evidence that the attention-based compression is faithful? For a sample of correctly answered problems, can you show that the compressed reasoning trajectory remains logically sound and complete? Conversely, for some failures, could over-compression have removed a critical step? For instance, on a subset of problems, you could use a verifier model or human annotators to score the logical coherence of the full vs. compressed reasoning chains. 
What is the comparative computational cost of training TRAAC versus the other RL baselines (L1-Max, AdaptThink)? An estimate of the additional overhead introduced by the attention computation and compression module would help users weigh the benefits against the training costs.

### Soundness
2

### Presentation
2

### Contribution
2
