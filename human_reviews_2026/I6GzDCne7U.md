# DeepScaleR: Effective RL Scaling of Reasoning Models via Iterative Context Lengthening

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Recent advances in large reasoning models (LRMs) such as OpenAI's o1 and Deepseek-R1 have demonstrated that reinforcement learning (RL) with outcome-based supervision can significantly enhance the reasoning abilities of language models. However, these improvements have so far relied on massive model scales and compute budgets, leaving open the question of whether RL-based scaling can be made both effective and efficient at smaller scales. In this work, we introduce DeepScaleR-1.5B, a 1.5B parameter model trained using reinforcement learning with a novel iterative context lengthening strategy. Our method begins with shorter context windows and progressively extends them throughout training, enabling the model to first learn to reason efficiently before learning to reason longer. This approach yields substantial performance gains with dramatically reduced computational cost. DeepScaleR-1.5B achieves 43.3% Pass@1 on the AIME2024 math benchmark—a 14.3 percentage point improvement over its base model and on par with OpenAI's o1-preview—while requiring a fraction of the compute. We provide a full training recipe, including dataset, code, hyperparameters, and training methodology, demonstrating that small models can be effectively scaled into strong math reasoners via RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This approach yields substantial performance gains with dramatically reduced computational cost. DeepScaleR-1.5B achieves 43.3% Pass@1 on the AIME2024 math benchmark—a 14.3 percentage point improvement over its base model.

### Strengths
1.DeepScaleR-1.5B achieves 43.3% Pass@1 on the AIME2024 math benchmark.
2.The GRPO algorithm is advanced.

### Weaknesses
1.The performance for 1.5B-LLM is not in the sota range, for example nvidia-1,5b
2.These is not very much novelty in the proposed algorithm.
3.These is not evluation on the code dataset in the experiments as code is also a good task for the reasoning ability of LLMs.
4.There is big gap netween the performacne of the proposed LLM and the performance of the sota LLMs on 1.5B paramater scale.

### Questions
1.Why does the 1.5B-parameter LLM fail to achieve state-of-the-art performance, particularly when compared to models like NVIDIA's 1.5B?
2.What novel contributions does the proposed algorithm offer beyond existing methods?
3.Why is there no evaluation of the proposed method on code-related datasets, despite code being a strong indicator of reasoning ability in LLMs?
4.Why is there a significant performance gap between the proposed 1.5B-parameter LLM and current state-of-the-art LLMs at the same scale?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents DeepScaleR, an efficient and effective training recipe for reasoning models. Specifically, it proposes high-quality data curation and iterative context lengthening, which gradually extends the context window during RL training (from 8K to 16K to 24K) to help the model learn efficient short reasoning before longer reasoning. Evaluation shows the effectiveness of context scheduling, as the trained 1.5B model achieves 43.3% Pass@1 on AIME2024, a 14.3% improvement over the base model and comparable to OpenAI’s o1-preview. This paper aims at reporting a RL-based training recipe that enables models to achieve good reasoning performance efficiently.

### Strengths
The paper is easy to follow.

The idea of iterative context lengthening scheduler is straightforward.

The method leads to a 1.5B model that has shows good performance in reasoning benchmarks.

### Weaknesses
Limitation of the model and experimental results. Although the claim is that the technique can enable small model with efficient training to have reasoning ability, the current technique is applied only to train a 1.5B model. It is currently not clear whether the length scaling can be universally effective for other model configurations, i.e. different sizes or different architectures. Furthermore, is this technique applicable for models with larger size, i.e. 7B model.

Question regarding length cutting: during training, by cutting at i.e. ctx length = 8k, do you explicitly let the model generate the final answer, i.e. by appending the final <think> token after it reaches 8k output length? How is this step done?

Question regarding the 24k ctx length ablation. Why does the plot show almost no improvement (figure 4) with static context length during training?

### Questions
See weakness.

### Soundness
3

### Presentation
3

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
The paper addresses the challenge that training large reasoning models with Reinforcement Learning is computationally expensive and believed to be ineffective for smaller models. The authors propose iterative context lengthening, a training strategy that acts as an implicit curriculum. Instead of training at a large, fixed context (e.g., 24K), iterative context lengthening starts with a short context (8K) to force the
model to learn efficient reasoning, then progressively increases the context length (to 16K, then 24K) as performance plateaus. Using this method, their 1.5B parameter DeepScaleR model achieves 43.3% Pass@1 on AIME2024, a 14.3% gain over its base model, matching o1-preview. This was achieved with a 2.6x reduction in compute cost compared to a direct 24K training baseline.

### Strengths
- Simple and Effective Method: Iterative context lengthening is an intuitive, simple, and highly effective training strategy that provides a more stable and efficient curriculum than direct
long-context training.
- Strong Performance and Efficiency: The 1.5B model achieves a significant +14.3% absolute
gain on AIME2024, demonstrating that small models can be scaled with RL. This is achieved
with 2.6x less training compute and results in a model that is also more efficient at inference time.
- Good Ablations: The ablation study (Figure 4) clearly proves iterative context lengthening's
superiority over a direct 24K baseline.

### Weaknesses
- Questionable Base Model Choice: The base model selection of a Qwen-2.5B-Math model is a concern, as this model series is known for potential test-set contamination on math benchmarks. This makes it difficult to definitively attribute the +14.3% AIME gain solely to the iterative context lengthening technique rather than the base model's pre-existing (and potentially "tainted") capabilities. The claims would be far more convincing if the authors either:
  - Replicated the experiment with a different base model (e.g., the Qwen3-0.6B used for the
COUNTDOWN task or Gemma-3-1B).
  - Evaluated the base and final trained models on a contamination-resistant benchmark,
such as LiveMathBench, to confirm the gains are from reasoning and not memorization.

- Unclear Mechanism for "Shorter Reasoning": The paper's claim that a constrained window "Encourages shorter reasoning" is not well substantiated. A constrained window merely truncates long responses, filtering them from the gradient update. This doesn't necessarily teach the model to be concise. It's plausible that the RL algorithm (GRPO, which is known to increase response length) still favors longer reasoning paths, which are then simply cut off. This would result in fewer valid, complete responses within the constrained window, not more efficient ones.
  - To truly support this claim, the authors should show that the 8K-trained model produces a
higher percentage of valid, complete responses (i.e., those reaching a final answer) within
a fixed 8K/16K/24K context than the base model. The data in Table 1, which only shows
average token length, is insufficient proof of this efficiency gain.

- Ambiguous Test-Time Scaling Evaluation: The test-time scaling analysis in Figure 5 is missing a critical detail: the maximum context window used for generating the 64 samples. It is unclear if this was capped at the 24K training limit or was uncapped.
  - A more insightful comparison, given the paper's theme, would be to evaluate how both the base model and DeepScaleR perform at test-time when the context window is expanded beyond the training limit with scaling at test time (e.g., progressively to 32K or
64K).

I am happy to increase my score if all the concerns are resolved.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
