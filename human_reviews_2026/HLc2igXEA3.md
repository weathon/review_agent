# PEAR: Phase Entropy Aware Reward for Efficient Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large Reasoning Models (LRMs) have achieved impressive performance on complex reasoning tasks by generating detailed chain-of-thought (CoT) explanations. However, these responses are often excessively long, containing redundant reasoning steps that inflate inference cost and reduce usability. Controlling the length of generated reasoning without sacrificing accuracy remains an open challenge. 
Through a systematic empirical analysis, we reveal a consistent positive correlation between model entropy and response length at different reasoning stages across diverse LRMs: the thinking phase exhibits higher entropy, reflecting exploratory behavior of longer responses, while the final answer phase shows lower entropy, indicating a more deterministic solution. This observation suggests that entropy at different reasoning stages can serve as a control knob for balancing conciseness and performance. Based on this insight, this paper introduces Phase Entropy Aware Reward (PEAR), a reward mechanism that incorporating phase-dependent entropy into the reward design. Instead of treating all tokens uniformly, PEAR penalize excessive entropy during the thinking phase and allowing moderate exploration at the final answer phase, which encourages models to generate concise reasoning traces that retain sufficient flexibility to solve the task correctly. This enables adaptive control of response length without relying on explicit length targets or rigid truncation rules. 
Extensive experiments across six benchmarks demonstrate that PEAR consistently reduces response length while sustaining competitive accuracy across model scales. In addition, PEAR demonstrates strong out-of-distribution (OOD) robustness beyond the training distribution.Our code is available at: https://github.com/iNLP-Lab/PEAR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reveals a correlation between model entropy and response length at different reasoning stages in reasoning models, based on which the author introduce Phase Entropy Aware Reward (PEAR) in the new reward design. PEAR penalize excessive entropy during the thinking phase and allowing moderate exploration at the final answer phase, which encourages models to generate concise reasoning traces that retain sufficient flexibility to solve the task correctly. This enables adaptive control of response length without relying on explicit
length targets or rigid truncation rules. Extensive experiments across four benchmarks demonstrate that PEAR consistently reduces response length while sustaining competitive accuracy across model scales. The experiments in the paper demonstrate the validity of this claim and also compare it with some recent works.

### Strengths
A new reward design and some experiments show the positive results.  PEAR achieves the most substantial reduction in response length across all benchmarks and evaluated models, while maintaining accuracy at a level comparable to original models.  The author further analyze how PEAR influences model reasoning across different phases, focusing on changes in entropy, number of reasoning steps, and average tokens per step after training with PEAR

### Weaknesses
1. The risk of over-penalizing necessary complexity.  The central assumption is that lower entropy during the thinking phase is desirable because it correlates with conciseness. However, this may not hold true for all problems.
2. More discussions are need for the hyperparameter sensitivity of "Excessive Entropy"
3. The experiments are conducted on models up to 8B parameters. While this demonstrates a trend, the behavior of language models can change dramatically at larger scales.
4. If the benchmarks are all in similar domains (e.g., mathematical or logical reasoning), the method's effectiveness might be confined to tasks with structured, verifiable reasoning paths. It is unclear how PEAR would perform on tasks requiring commonsense reasoning, planning, or creative generation, where the notion of a "correct trace" is more fluid.

### Questions
Considering the weakness, more experiments and discussions are needed.

### Soundness
2

### Presentation
3

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
This paper aims to address the redundancy in thinking steps in Large Reasoning models. The authors observe a consistant positive correlation between model entropy and response length. Inspired by the observation, this paper introduces Phase Entropy Aware Reward (PEAR) to punish entropy in thinking phase and encourage entropy in final answers. The model trained with GRPO shows much reduced response length with comparable performance.

### Strengths
1. The paper studies an interesting and important problem of efficient reasoning. 
2. The trained model shows consistent reduction in reasoning lengths across datasets, while largely preserving the performance.

### Weaknesses
1. The claimed OOD robustness is only supported by model's performance on other math datasets. Whether current method generalizes to non-math tasks (coding, general reasoning...) is unclear.

### Questions
1. The ablation study only shows how the coefficient in $H_{answer}$ changes the performance. I'm curious about how the coefficient $H_{think}$ changes model's behavior? My understanding is the reduction in reasoning length comes from reduced exploration in model's thinking, which sacrifices reasoning performance. I wonder given a fixed $alpha$, what is a "sweet point" for the exploration-efficiency trade-off. 
2. For table 1, can you provide the standard deviation?
3. Does the proposed method work for other series of models (e.g. llama series)?

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
This paper investigates the problem of large-scale inference models generating excessively long inference chains. Through systematic empirical analysis, the authors find a positive correlation between the model's token-level entropy and response length, exhibiting different entropy patterns at different stages of inference: higher entropy in the thinking stage and lower entropy in the final answer stage. Based on this observation, the paper proposes the PEAR method, a GRPO-based reward mechanism that encourages the model to generate more concise inference trajectories by penalizing excessive entropy in the thinking stage while maintaining moderate flexibility in the final answer stage. Experiments on four mathematical inference benchmarks including GSM8K, MATH500, AIME24, and AMC23 show that PEAR achieves a response length reduction of 37.8% to 59.4% across different model sizes, with an accuracy decrease of less than 1%, and demonstrates good out-of-distribution generalization ability.

### Strengths
S1. The authors established the relationship between entropy and response length through systematic empirical analysis. They not only demonstrated the overall correlation but also decomposed the entropy patterns at different inference stages, providing a solid empirical foundation for subsequent methods. Furthermore, the entropy filtering experiment cleverly proved that high-entropy tokens primarily contribute to redundant exploration rather than correct inference, directly supporting the design philosophy of PEAR.

S2. The experimental results are very convincing. On all tested models and benchmarks, PEAR achieved significant length reductions (37.8%-59.4%), while the accuracy decline was minimal (less than 1% on average).

S3. PEAR's design is simple and elegant, requiring only the addition of a phase entropy-based penalty term to the GRPO reward function. This simplicity makes the method easy to understand and implement, and also facilitates integration into existing training workflows.

S4. The experimental section of the paper is very comprehensive.

### Weaknesses
W1. While the experimental results are promising, it remains unclear why PEAR maintains accuracy while significantly reducing length. Are there any theoretical guarantees? What is the theoretical basis for the design of the reward function, especially the max operation in Equation (7)? The paper does not provide a convergence analysis, nor does it discuss the relationship between the optimal solution of PEAR optimization and the standard GRPO.

W2. The paper only evaluated PEAR on mathematical reasoning tasks, but it is unclear whether PEAR is applicable to other types of reasoning tasks (such as common sense reasoning, code generation, scientific reasoning, etc.).

W3. The paper mentions that Step Entropy uses two-stage training (SFT+RL) with inserted `[SKIP]` tags, while PEAR uses end-to-end RL training, but it doesn't delve into the impact of this difference. A similar comparison with LCPO is made; specifically, LCPO uses explicit length constraints, while PEAR uses entropy as an implicit constraint, but under what circumstances do these two approaches have their advantages? Furthermore, the paper doesn't discuss whether PEAR can be combined with other methods to achieve better results.

W4. Although the paper analyzes the impact of the hyperparameter `α` in Section 4.4, only a limited number of values ​​were tested. The optimal `α` may vary greatly for different types of tasks or different models, but the paper does not provide guidelines on how to choose `α`.

### Questions
Q1. In Equation 5, entropy is calculated using the old policy `πθ_old`, but `πθ_old` is continuously updated during training. Does this mean that the same response will yield different entropy values ​​and rewards in different training iterations? If so, will this non-stationarity of the reward affect training stability?

Q2. In the edge case, if the `think` tag is missing, the paper sets k=T and H_answer=0, but is this reasonable? Would this lead to such responses always receiving higher rewards?

Q3. Could you provide some qualitative examples demonstrating the complete reasoning process of the original model and the PEAR-trained model on the same problem? This will help us determine whether conciseness comes at the expense of clarity. Also, for mathematical reasoning, the correctness of each step is important. Does PEAR increase the error rate in the reasoning steps, even if the final answer is correct?

Q4. From Figure 2(b), the entropy is already low in the final answer stage. Why then subtract `α * H_answer` from the reward? What is the intuitive reasoning behind this? Is it to encourage the model to maintain a certain degree of uncertainty and flexibility when providing the answer? However, in mathematical reasoning, the final answer should be deterministic. Why encourage entropy?

### Soundness
3

### Presentation
3

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
This paper proposes a phase entropy aware reward mechanism which penalizes excessive entropy during the thinking phase and encourages moderate exploration at the final answer phase. This reward design encourages models to generate concise reasoning traces while retaining the original accuracy.

However, the vanilla GRPO training simultaneously improves accuracy and reduces the generation tokens. Compared with the simple GRPO baseline, the proposed PEAR seems to reduce more tokens with the cost of losing the RL training accuracy gain. More experiments using new and larger base models and comparison with more baselines should be conducted to validate the effectiveness.

### Strengths
This paper proposes a simple but effective reward for RL training to improve the reasoning efficiency. By penalizing the entropy in the thinking process, the method effectively reduce the generation tokens  37.8%~59.4% compared with original model while achieving similar reasoning accuracy on 4 math datasets and 3 LRMs. The authors also further analyze the changes in entropy, number of reasoning steps, and average tokens per step after training to validate the effectiveness of PEAR.

### Weaknesses
1. There are several previous related works about RL based training and reward shaping for efficient reasoning [1, 2]. In addition, there are also several SFT and training-free based baselines [3] should be compared to thoroughly validate the effectiveness of the proposed reward design. 

2. Only 3 LRMs are compared. With the release of Deepseek R1 0528, the thinking process of LRMs already changed. Could you please use the recent Deepseek R1 distilled LRMs such as deepseek-ai/DeepSeek-R1-0528-Qwen3-8B to validate the effectiveness on the improved and effective reasoning models? 

[1] Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., Song, L., Liu, Q., Zhou, M., Zhang, Z. and Wang, R., 2024. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187.

[2] Liu, W., Zhou, R., Deng, Y., Huang, Y., Liu, J., Deng, Y., Zhang, Y. and He, J., 2025. Learn to reason efficiently with adaptive length-based reward shaping. arXiv preprint arXiv:2505.15612.

[3] Lin, W., Li, X., Yang, Z., Fu, X., Zhen, H.L., Wang, Y., Yu, X., Liu, W., Li, X. and Yuan, M., 2025. TrimR: Verifier-based Training-Free Thinking Compression for Efficient Test-Time Scaling. arXiv preprint arXiv:2505.17155.

### Questions
1. Does the proposed reward scale with the model size? 

2. How is the performance on general tasks after training with PEAR and other baseline methods?

### Soundness
3

### Presentation
3

### Contribution
2
