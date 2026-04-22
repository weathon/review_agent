# Explore Data Left Behind in Reinforcement Learning for Reasoning Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach for improving the reasoning abilities of large language models (LLMs). The Group Relative Policy Optimization (GRPO) family has demonstrated strong performance in training LLMs with RLVR. However, as models train longer and scale larger, more training prompts become residual prompts—those with zero-variance rewards that provide no training signal. Consequently, fewer prompts contribute to training, reducing diversity and hindering effectiveness. To fully exploit these residual prompts, we propose the Explore Residual Prompts in Policy Optimization (ERPO) framework, which encourages exploration on residual prompts and reactivates their training signals. ERPO maintains a history tracker for each prompt and adaptively increases the sampling temperature for residual prompts that previously produced all-correct responses. This encourages the model to generate more diverse reasoning traces, introducing incorrect responses that revive training signals. Empirical results on the Qwen2.5 series demonstrate that ERPO consistently surpasses strong baselines across multiple mathematical reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper study the issue in GRPO algorithm family, the residual prompts, which are the prompts that all rolled out responses are correct and therefore does not provide any training signal. To address this issue, the paper propose ERPO. Two ways are introduced to leverage the residual prompts. The first is reactivating training signal, which adds a small negative reward to the zero advantage. The second is to gradually increase the sampling temperature to encourage more diverse responses. Experiments on Qwen models demonstrates that proposed method outperforms DAPO baseline.

### Strengths
The strengths of the paper are listed as follows

1. This paper propose two interesting way to leverage the residual prompt, adding a pseudo-negative reward when computing the advantage and increasing the sampling temperature. These two way, while seemingly simple, make sense to the reviewer.

2. Most of the claims are properly consolidated by empirical evidences. The results in Table 1 and Table 3 are good justification of the motivation of the method.

### Weaknesses
The weakness of this paper are listed as follows

1. The experiments are limited to DAPO baseline. Could the author also provide the results based on vanilla GRPO (that is, GRPO vs GRPO+ERPO)? Also, since PPO does not suffers from the issue of residual prompt, could the author also compare ERPO with PPO?

2. While the experiment results in Table 2 looks good over DAPO, some results in Table 4 drops by a innegligible margin. Could the author provide more justification on how the temperature-related hyperparamter are chosen and how to choose them for some new tasks?

3. Could the author provide some cases comparing the responses generated with different temperature? As far as the reviewer's knowledge, high temperature sometimes results in poor responses, which might not provide enough supervision. A few more case might be useful to provide a better knowledge of how temperature affect response qualities and might also justify your choice of temperature-related hyperparameter.

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an algorithmic improvement named ERPO. The algorithm increases the sampling temperature whenever they encounter a prompt with fully saturated (all correct rollouts) during RLVR. Increasing the temperature helps in generating more diverse responses which could possibly lead to a better training signal because of non-zero advantages during GRPO/DAPO. They evaluate their technique on the Qwen2.5 3B and 7B models.

### Strengths
The idea of increasing temperature during training is novel, simple, and easy to implement. The paper is also well written and easy to follow.

### Weaknesses
I have 2 major concerns with the numbers reported in the paper, specifically for the MATH500 dataset. 

1. **Underperforming baselines:**

The paper reports mean@4 for the 3B model to reach 50.4% and for the 7B model to reach 60.3% using DAPO. However, these numbers seem to be severely underperforming the Qwen2.5 3B and 7B Instruction tuned models at 65.9% and 75.5%. These numbers are from the official report of the Qwen Team from Table 8 and Table 9 [1]. 

Why is there such a large discrepancy in performance? Is this because the model hasn't been trained for enough steps? If so, would be possible to show that the algorithm has actually been trained to convergence using the training curves?

2. **Dealing with off-policy rollouts:**

How is the DAPO algorithm modified when the temperature changes? Since increasing the temperature makes the rollouts off-policy, there needs to be either a correction using some importance ratio or a correction in how log-probs are computed. The authors do not talk about this issue. How did they deal with it?


[1] Qwen2.5 Technical Report (https://arxiv.org/abs/2412.15115)

### Questions
Please look at the weaknesses mentioned above.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies a limitation in GRPO family algorithms: as training progresses and models scale up, more prompts become "residual prompts" that yield all-correct responses, resulting in zero-variance rewards and providing no training signal. To address this, the authors propose ERPO, which maintains a history tracker for each prompt and adaptively increases sampling temperature for residual prompts to encourage exploration and reactivate training signals. Experiments on Qwen2.5-3B and 7B models across AIME2025, AIME24, AMC2023, and MATH500 demonstrate improvements over the DAPO baseline, particularly on AIME2025.

### Strengths
1. The paper clearly identifies a practical limitation in current GRPO-family algorithms where residual prompts accumulate and reduce training diversity. Table 1 provides compelling evidence that this problem worsens with larger models and longer training.

2. The proposed ERPO framework is straightforward to implement and can be easily integrated into existing RLVR algorithms.

### Weaknesses
1. Recent work has shown that in RLVR, even when models achieve nearly 100% accuracy on the training set, continued training can still improve performance on validation/test sets—a phenomenon known as grokking, which is also observable in reward curves and val accuracy curves. From this perspective, the problem this paper proposed may not be as significant as claimed.

2. The paper primarily conduct experiments based on DAPO. Missing some RLVR algs including vanilla GRPO, GPG, RLOO, REINFORCE++, GSPO, etc.

3. While RA shows strong performance on 3B model, it underperforms on 7B. The paper attributes this to "overfitting" but doesn't provide concrete evidence. A more detailed analysis comparing RA and ERPO on different model scales would strengthen the paper.

4. Table 2 shows that ERPO's improvements are very small, making it difficult to be convinced by the paper's claims.

### Questions
N/A

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
3

### Summary
This papers proposes a method named ERPO to reduce the problems that residual prompts would increase as RL training progress. It would add pesudo negative reward for all positive response batch, and add a problem-specific temperature adaptive sampling based on the historical performance. It improves the performance on Qwen2.5-3/7B compared with DAPO.

### Strengths
1. The paper is easy to follow, the method is clean and easy. Temperature increase can prevent RA overfits too fast
2. Performs well on Qwen2.5-3/7B on math tasks compared with DAPO

### Weaknesses
1. Only train on Qwen2.5 models, haven't try other models like Llama/OpenThinker/Octothinker etc. From spurious reward[1], maybe better evaluate on non-Qwen2.5 models.
2. And I'm kind of concerned about whether the method can be well generalized when scaling model size and compute. In the paper the training step is only 175, and the curves look hasn't converge, and when the temperature touches T_max, keeping RA may make model overfit to some correct outputs. It's also kind of tricky to tune the temperature, which may be model-dependent. Increase temperature is kind of similar to add large entropy loss, which is not exactly equivalent to semantic level diversity, and may increase the risk of instable training. I still think DAPO would be better choice for long-term RL training, though wasting some rollout.

If there are results for ERPO on more steps (for example, > 500 or better > 1k) or on 32B models (better on non-Qwen2.5 models), and show that ERPO consistently beat DAPO, then I would increase my score




[1] Shao, Rulin, et al. "Spurious rewards: Rethinking training signals in rlvr." arXiv preprint arXiv:2506.10947 (2025).

### Questions
1. Where is the Qwen2.5-32B +DAPO results comes? Do you use the DAPO-trained checkpoint or train by yourself?
2. What's the pass@k performance of baseline and ERPO? Is it possible to report the entropy curve of baselines and ERPO during training?
3. what's the RL resource used for different models in Tab. 1

### Soundness
2

### Presentation
3

### Contribution
2
