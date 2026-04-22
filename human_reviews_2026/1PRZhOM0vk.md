# Reflective Reinforcement Tool Learning

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Tool learning enables large language models (LLMs) to interact with real-world environments. While prior work mainly relies on supervised fine-tuning (SFT), recent reinforcement learning (RL) methods have shown promise in improving the tool-use capabilities of LLMs by leveraging richer reward signals. However, during RL rollouts, failures often stem from environmental perturbations such as network issues or tool instability rather than policy errors. These failed trajectories are typically discarded, resulting in low data efficiency and high costs, especially when using paid tools. To solve the issue, we find that many failures can be recovered through simple retries, reasoning, or reflection. Yet these augmented new policies for self-correction introduce distribution shifts that hinder the reuse of recovered data for origin policy learning. In this paper, we propose Tool-Reflective Reinforcement Learning (Tool-ReRL), an off-policy RL framework that equips LLMs with a reflection mechanism to temporarily adjust the rollout policy, thus analyzing failures, attempting self-correction, and exploring diverse solution paths. To bridge the distribution gap between modified and original policy, we introduce an importance sampling estimator, enabling rewards from reflection-enhanced trajectories to effectively guide the optimization of the original policy. Our extensive experiments on four tool-learning benchmarks demonstrate that, given the same training data, Tool-ReRL significantly improves data efficiency and achieves average performance gains of up to 7.60% and 6.11% over standard RL algorithms based on Qwen2.5-7B and LLaMA3.1-8B, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates reinforcement learning for large language models in tool-use scenarios. The authors propose a new off-policy framework Tool-Reflective Reinforcement Learning with two key designs: (i) a reflection mechanism to adjust the failure trajectory during the rollout; (ii) an importance sampling estimator to bridge the gap between the current policy and the one that guides the reflection.

### Strengths
- This paper is well-originalized and clearly written.
- The motivation is well-established. Specifically, this paper seeks to improve the data efficiency during reinforcement learning in tool-LLM.

### Weaknesses
- The authors propose using samples with reflective responses, which may introduce distributional shifts. I recommend that the authors provide experimental validation, such as a distributional comparison, to make the argument more convincing.
- The description of **Assumption 1** is vague. Does it imply that $\pi_{\theta_{rfl}}(\tau^\*|q^\*) = \pi_{\theta}(\tau^\*|q^\*)$? If so, the use of importance sampling would be unnecessary.
- Based on the experimental results (e.g., **CoT** vs. **CoT + IS**), importance sampling does not appear to significantly improve performance. This point requires further explanation, even though the proposed method achieves notable improvements.
- The paper lacks a concrete training example to illustrate how the reflection mechanism is designed.
- The overall contribution of this work is limited. For example, importance sampling is a standard component of PPO, and the paper does not offer substantial changes.

### Questions
See them in Weaknesses.

### Soundness
2

### Presentation
3

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
The paper demonstrates that fixing negative samples during the RL training process with an importance sampling to offset the distribution mismatch between refined trajectory and existing trajectory can improve the model's tool learning performance.

* It shows many negative samples generated are almost positive samples with minor correction.
* It proposes a way to fix the negative sample issue by invoking an additional model call to point out the error with suggested edit and ask the model being trained to regenerate a new trajectory based on existing input, original output and the reflection.
* It shows using the reflection based behavior policy in the importance sampling function can resolve the degradation from the new generated trajectory and further improve the performance.

### Strengths
* The paper performs extensive experiments on various tool use dataset including RotBench, TaskBench, BFCL and Seals with different training approaches.
* The reflection-recovery strategy is clearly explained and the overall approach is easy to understand.

### Weaknesses
* The detailed setup for the reflection module used in the experiment is missing.
* The ablation should include training with positive samples only as well to show the increment is not from removing negative samples and training with same number of positive samples after refinement to show the  increment is not from more training signals.
* Experiment on PPO + IS should also be included.
* Details of the reflection module are missing, such as the actual module used, what’s the input/output.
* The results from 2 model on the distribution shift does not provide strong evidence that additional refinement will cause a performance drop given one of the results does not show improvement.
* The table 1’s benchmark variant is not defined.
* Using the same policy to generate a refined trajectory should still consider on-policy. The distribution shift is not due to mixing off-policy data with on-policy data but rather mixing different task’s data
* The analysis on data with near-success or environment-related issues in a single dataset is not extensive to support the claim that many failures can be recovered through simple retries, reasoning, or reflection.

### Questions
* What’s the efficiency cost with additional refinement module call.
* Do all the CoT and Ref variants in all experiments only apply to negative samples?

### Soundness
2

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
4

### Summary
This paper introduces Tool-Reflective Reinforcement Learning (Tool-ReRL), a framework that addresses data inefficiency in reinforcement learning for tool use by large language models. The authors identify that 44.7% of failed trajectories in standard RL training are near-success cases discarded due to environmental perturbations rather than policy errors. Tool-ReRL incorporates a reflection mechanism that analyzes failures and generates corrective feedback, combined with importance sampling to handle the distributional shift from reflection-augmented trajectories. Experiments on four benchmarks using Qwen-2.5-7B and LLaMA-3.1-8B models demonstrate performance gains of up to 7.60% and 6.11% respectively over standard RL algorithms, with the framework consistently outperforming supervised fine-tuning, DPO, and PPO baselines. The results show that both reflection and importance weighting components are necessary - reflection alone decreased performance by 3.4% without proper correction, while the complete framework transforms previously wasted near-success trajectories into valuable training signals, improving data efficiency in tool learning scenarios where environmental instabilities are common.

### Strengths
1. Good problem statement, data efficiency in RL training is an important problem
2. Fixing issues with tools or minor changes to trajectory as correction is a good idea and makes sense
3. Comparison against a good set of baselines and a broad set of tool use benchmarks
4. Importance weighting is clearly motivated, explained, and evaluated with ablations
5. Presentation is clear and easy to follow

### Weaknesses
- One of the baseline methods should be on-policy distillation [1]. The "reflection" model is much more powerful than the model being trained. Such a "teacher" model is not available in SFT or PPO. And use of distillation will likely be more efficient than PPO. 
- The paper claims that the method is more data efficient. But there is no measurement of how efficient the proposed method is. How much less number of epochs does the proposed method need to reach the same score as the baseline models?
- The paper lacks a measure of computational efficiency. A much larger model (Deepseek R1) is used as a reflection model. It is possible that the compute and time used to create such reflections exceeds the time it takes to recover from discarding erroneous data. 




[1] Agarwal, Rishabh, et al. "On-policy distillation of language models: Learning from self-generated mistakes." The twelfth international conference on learning representations. 2024.

### Questions
1. "we identified 834 failures in total, of which 373 (44.7%) were attributable to near-success or environment-related issues." How do you identify "near-success"? 
2. How do you verify that the reflection model is correct? What happens when it is incorrect? 
3. Why is this method only limited to "near-success" trajectories? Why not apply it to all failed trajectories?
4. "if the policy lacks sufficient capability to sample positive examples with reasonable success rates, its contribution
to RL effectiveness becomes negligible." What is the evidence for this claim?
5. "tool invocation does not require long-form reasoning" -- what is the evidence for this claim?
6. I did not understand this: "+CoT+IS allows the model to generate its own internal thought sequence but does not perform reflection, while applying importance weights to reduce distributional discrepancy."
7. Paper claims PPO halves the compute required compared to DPO. PPO requires use of a critic model, which is very expensive. Can you substantiate this claim?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Tool-ReRL, an off-policy RL framework for tool learning that repairs failed rollouts online via a reflection module and re-uses the repaired trajectories with importance-weighted correction inside a PPO objective. This design converts many environment-induced or near-success failures—typically discarded by prior work—into useful training signals while controlling distributional drift. On four tool-use benchmarks, Tool-ReRL consistently improves average performance over strong baselines with the same data budget (up to +7.60% / +6.11% on Qwen2.5-7B / LLaMA3.1-8B).

### Strengths
- The motivation is clear and reasonable. Environment perturbations create many “false negatives”; turning them into learnable signals can improve the sampling efficiency.

- The analysations and ablations are strong and comprehensive.

### Weaknesses
- The method hinges on the inverse-prompt equivalence to estimate behavior policy for IS correction. However, quantitative (or theoretical) bias analysis of the assumption is omitted.

- Computational cost & scalability are not carefully analysed. Reflection attempts and extra sampling add non-trivial wall-clock costs. The paper should report env-steps / wall-clock time / money against strong RL baselines under matched budgets.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
