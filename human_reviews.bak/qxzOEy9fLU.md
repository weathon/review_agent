# Adaptive Uncertainty-Aware Reinforcement Learning from Human Feedback

- Decision: Reject
- Scores: 5, 3, 5

## Abstract
Reinforcement learning from human feedback (RLHF) is a popular technique to align large language models (LLMs) to human preferences. It requires learning a reward model that predicts scalar values given a generated text sequence, acting as a proxy for human preference scores. A central problem of RLHF is \textit{reward hacking}, i.e., overoptimization. LLMs can easily exploit the reward model by generating text that can receive high scores but no longer align with human preferences. We address this problem by proposing a new objective which adapts the tradeoff between reward model score and regularisation based on reward uncertainty. We hypothesize that when the reward model uncertainty is low, RLHF should make a larger step size by lowering the regularization coefficient. On the other hand, when the uncertainty is high, optimization should slow down by staying closer to the original model. We present a novel re-formulation of the RLHF objective and derive our approach from its generalization to account for reward model variance. We demonstrate that our uncertainty-aware RLHF objective mitigates overoptimization and outperforms vanilla RLHF by 50% on a standard summarization task.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel approach to mitigate reward hacking in reinforcement learning from human feedback (RLHF) for large language models (LLMs). Traditional RLHF often encounters issues with overoptimization, where the model learns to exploit reward models, resulting in outputs that score well on proxies but misalign with actual human preferences. The authors propose an adaptive method that adjusts the regularization term based on the reward model's uncertainty, effectively modulating training based on confidence. They demonstrate that this method not only reduces overoptimization but also improves performance on a summarization task by 50% compared to standard RLHF.

### Strengths
* Originality: The uncertainty-aware mechanism is a novel solution to a well-known RLHF problem, showing how dynamically adjusting regularization based on confidence can prevent overoptimization.

* Quality: The paper’s experiments provide robust support for the method’s effectiveness, with comprehensive tests and relevant comparisons to standard baselines.

* Significance: Mitigating reward hacking while improving alignment is crucial for RLHF, especially for LLM applications where maintaining human alignment over time is essential.

* Clarity: The approach’s mathematical foundation is well-explained, and the provided algorithmic steps (e.g., Algorithm 1) are easy to follow.

### Weaknesses
* Computational cost: The adaptive approach involves running ensembles, which adds computational overhead. While the paper notes this is only a slight increase, exploring less expensive uncertainty estimation techniques could improve practicality. In online training, wondering how authors can load so many reward models into memories?

* Bias in the reward models (RMs): the variance estimation of reward model can be a good proxy for variance estimation. However, the error of reward model should be quantified as $\mathbb{E}_{\hat{r} \text{ trained on } D}(\hat{r}(x,y)-r(x,y))^2 = bias^2 + variance$. The paper only address the variance term and does not address the bias. Assume for example, the RM prefer longer responses with 90%, we should still debias the length instead of only tackling the variance.

* Data/prompt shift issues unaddressed: if the RM training and inference time face distribution shift in prompt, response, how does this work address that?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the challenge of reward hacking of LLMs in RLHF. Reward hacking occurs when models over-optimize based on inaccurate signals from a proxy reward model, leading to a misalignment with true human preferences. The authors propose an adaptive uncertainty-aware RLHF approach, which incorporates reward model uncertainty into the optimization objective.

### Strengths
1. The approach is reasonable in its use of uncertainty-based adaptivity in RLHF. By integrating uncertainty estimation through an ensemble reward model, the paper contributes a new method to dynamically adjust optimization.
2.The paper is well-structured and clearly presents both the theoretical basis and empirical methods.

### Weaknesses
1. The approach assumes that reward model uncertainty, measured as ensemble variance, provides a reliable indicator of alignment confidence. However, the generalizability of this assumption across different tasks or domains remains unclear. I think a more important problem should be considered is the evolution of the reward model. The upper bound of the reward model capability is the core issue that restricts the continuous optimization of RLHF.
2. The experiments are limited to a single summarization task using the Reddit TL;DR dataset. While the results are promising, testing on a more diverse set of tasks would better demonstrate the robustness of the proposed approach.
3.

### Questions
Could you test the proposed adaptive uncertainty-aware RLHF approach on a more general-purpose large model, such as LLaMA 3 or other recent LLMs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses the common issue of "reward hacking" in reinforcement learning from human feedback (RLHF). Reward hacking occurs when language models exploit inaccuracies in reward models, producing outputs that receive high reward scores but deviate from real human preferences (i.e., overoptimization). The authors propose a straightforward adaptive RLHF objective that adjusts the regularization based on reward uncertainty. The idea is to dynamically adjust the regularization weight and learning rate based on the confidence level of the reward model. Experimental results on RL;DR dataset demonstrate that this method outperforms standard RLHF approaches, maintaining alignment with human preferences and reducing the risk of over-optimization.

### Strengths
- Good motivation
- Straightforward approach, very reasonable for the over-optimization issue
- Good results
- Clean presentation

### Weaknesses
The biggest problem of the work is limited evaluation. The evaluation is restricted to a single summarization task with limited model sizes. In my opinion, the authors need to evaluate their approach on more diverse tasks and larger models, as the proposed method is a general technique for RLHF. Moreover, the main results in Figures 5 and 7 appear to be based on a single run, largely limiting the robustness of the conclusions. Repeating the experiments with different random seeds would improve the reliability of the findings. Analysis ana ablation studies are also missing.

### Questions
1. What is the y-axis of Figure 1? How is the "Quality of LLM" evaluated? Is it same as gold-reward defined in Section 4?
2. In Line 398, could you explain what do you mean by "well-calibrated uncertainty estimates"?
3. What is the best value of M? To what extent does the number of reward models affect the results?

### Soundness
2

### Presentation
3

### Contribution
3
