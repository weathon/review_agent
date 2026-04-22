# VL Norm: Rethink Loss Aggregation in RLVR

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
We propose VL Norm (Variance-reduced Length-dependent Normalization), a simple yet effective loss aggregation method tailored to the characteristic of dynamic generation lengths in Reinforcement Learning with Verifiable Rewards (RLVR). Recently, RLVR has demonstrated strong potential in improving the reasoning capabilities of large language models (LLMs), but a major challenge lies in the large variability of response lengths during training, which leads to high gradient variance and unstable optimization. Although previous methods such as GRPO, DAPO, and Dr. GRPO introduce different loss normalization terms to address this issue, they either produce biased estimates or still suffer from high gradient variance. By analyzing the effect of varying lengths on policy loss both theoretically and empirically, we reformulate the problem as finding a minimum-variance unbiased estimator. Our proposed VL Norm not only provides an unbiased estimate of the true policy loss but also minimizes gradient variance in theory. Extensive experiments show that it consistently achieves superior results across different model sizes, maximum lengths, and tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes VL Norm, a new loss aggregation method designed to reduce the gradient variance caused by variable response lengths in reinforcement learning with verifiable rewards (RLVR). The method reformulates the gradient aggregation problem as a minimum variance unbiased estimation (MVUE) problem and introduces a tunable parameter α to control the trade-off between variance and utilization of long responses. Empirical results on tasks such as CountDown and Math using Qwen2.5 models are presented, showing that VL Norm outperforms prior approaches like GRPO, DAPO, and Dr. GRPO in stability and final accuracy.

### Strengths
1. The formulation of loss aggregation as an MVUE problem is mathematically elegant and conceptually interesting.
2. The method generalizes several prior approaches (e.g., GRPO, Dr. GRPO) as special cases of VL Norm via the α parameter, which adds flexibility.
3. The paper provides comprehensive experiments across multiple tasks, model sizes, and maximum response lengths.

### Weaknesses
1. Lack of standard deviation or variance reporting, despite the paper’s core focus on variance reduction. In Table 2, Table 3, and Table 4, performance metrics such as Avg@8 and Pass@8 are reported without any standard deviation or confidence interval:
“Table 2: Detailed results on the CountDown task…”
“Table 3: Performance comparison across different hyperparameters…”
This makes it impossible to assess whether the performance improvements are statistically significant, especially when the observed differences are small (e.g., Math task in Table 4).

2. GRPO Norm appears to achieve similar entropy stability, which undermines VL Norm’s claimed benefit in stabilizing training. The paper states: “VL Norm consistently maintains an entropy between 0.1 and 0.2, which is beneficial for stable training.” (p. 7, §4.2)
However, in Figure 6, GRPO Norm shows comparable entropy behavior, suggesting that low entropy is not unique to VL Norm.

3. VL Norm may exacerbate length bias, contrary to its motivation. In Figure 7, VL Norm exhibits a sharp increase in response length (especially in Math 7B), unlike the baselines. Yet the paper interprets this as a positive signal:
“The response length exhibits sharp increases…coinciding with noticeable boosts in test accuracy.” (p. 7)
This suggests that longer responses are being disproportionately rewarded, which could reinforce verbosity bias—a known issue in reward modeling. This contradicts the motivation to control length-induced gradient variance (p. 2–3).

### Questions
1. In Figure 7, VL Norm results in longer responses, which seems to contradict the paper’s motivation of mitigating length effects. How does the method avoid reinforcing verbosity bias, especially given prior work (e.g., BSR) that explicitly addresses this issue?
2. In Figure 6, GRPO Norm maintains similarly low entropy, yet the paper presents this as a unique property of VL Norm. Could the authors clarify whether VL Norm has distinct stability benefits beyond what GRPO already offers?
3. The paper states:
“Extensive experiments show that it consistently achieves superior results…” (Abstract)
Yet in CountDown 7B (Table 2), GRPO Norm actually outperforms VL Norm. Why is this not discussed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VL Norm (Variance-reduced Length-dependent Normalization), a new loss aggregation method designed to address challenges in Reinforcement Learning with Verifiable Rewards (RLVR), especially the variability in response lengths during training. RLVR has shown strong potential in improving the reasoning capabilities of large language models (LLMs), but it faces the challenge of gradient instability due to fluctuating response lengths.  The proposed VL Norm not only provides an unbiased estimate of the true policy loss but also minimizes gradient variance, achieving superior training stability and performance.

### Strengths
VL Norm introduces a fresh approach from statistical perspective to loss aggregation by addressing the length-dependent variability in RLVR tasks. The method's ability to provide both unbiased estimations and minimized variance is a valuable contribution to this field.

### Weaknesses
The paper heavily relies on the assumption that the variance structure of the token-level gradient grows proportionally with the response length. However, I found Figure 3 unclear. Specifically, how the Q, K, V matrices reflect the variability of the token-level gradient. Could you provide more details about the computational process involved in this and how these matrices accurately capture this variability?

### Questions
1. Could you further give an explanation about the connection and distinction between VL-norm and GSPO [1]. GSPO modifies the importance sampling weight
2. DAPO employs four key techniques that can significantly impact the variance of the gradient. For instance, the reward shaping strategy  avoid excessively long responses, which can significantly reduce the variance of the gradient. Does it seem fair to directly compare the variance across DAPO and other methods?
3. What is the impact of the hyperparameter $\alpha$ on the performance of VL Norm? Have the authors conducted experiments  to evaluate the sensitivity of VL Norm’s performance to variations in this hyperparameter?
4. The experiments mainly focus on CountDown and Math tasks. Could the authors conduct additional experiments on tasks with more complex reward structures, or tasks that require long-term decision-making?



[1] Group Sequence Policy Optimization. Qwen Team, Alibaba.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies that existing RLVR methods suffer from biased estimation or high variance when performing response length normalization. The authors innovatively propose the VL Norm method, framing length normalization as a minimum variance unbiased estimation problem, which stabilizes RLVR training and facilitates convergence.

### Strengths
1. Although the length normalization issue addressed in the paper is a minor component of RLVR algorithms, it significantly impacts training stability and convergence speed. The authors provide a novel perspective by analyzing gradient variance.
2. The theoretical analysis of minimum variance unbiased estimation is highly convincing. Table 1 clearly demonstrates the superiority of the proposed method through comparisons of gradient mean and gradient variance with baselines.
3. The experimental results in Section 4 are analyzed in great detail.

### Weaknesses
1. All experiments are conducted on Qwen models. As noted in [1][2], the Qwen series may suffer from data contamination issues in evaluation datasets such as MATH500. Therefore, I strongly recommend that the authors perform additional experiments on more open-source models (e.g., the Llama3 series) to enhance the robustness of their conclusions. I would be willing to raise my score if more experimental results are provided.
2. The entire theoretical framework is based on the assumption that gradient variance is positively correlated with response length. However, the authors only use statistics from one or two samples to support this claim, which is insufficient. Theoretical support for this argument would be preferable. Alternatively, the authors should conduct experiments with more samples, model series, and layer types.
3. The proposed method introduces a hyperparameter alpha, but the final experimental results (Table 4) do not show a convincing improvement.

**Reference**

[1] Wu et al., Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination, arxiv 2025

[2] Shaoe et al., Spurious Rewards: Rethinking Training Signals in RLVR, arxiv 2025

### Questions
1. Why do the main experiments in Fig. 2 and Fig. 5 not show the complete results of VL Norm? Is this due to the instability of RL training causing performance degradation in later stages?
2. [1] also bases on the assumption that gradient variance is positively correlated with response length and starts from the goal of minimizing gradient variance. Could the author briefly explain the similarities and differences between this paper and your paper? If possible, could this method be included as a baseline in your paper?

Reference

[1] Hao et al., On-Policy RL with Optimal Reward Baseline, arxiv 2025

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the bias and variance of gradients induced by response length, along with related normalization methods. The authors analyze how varying response lengths affect policy loss and propose a normalization technique that achieves an unbiased gradient estimate with minimal variance. Extensive experiments are conducted to demonstrate that the proposed method improves both performance and training stability (in terms of entropy dynamics).

### Strengths
- The paper is well written and easy to follow.

- The authors present a novel and well-founded perspective on length normalization, which is both important and closely related to the training stability of RL algorithms. They provide a theoretical analysis of the effects of different length normalization methods and propose a new normalization approach based on this theoretical insight.

- The experimental results are sound, evaluating both the final performance and the training stability of the proposed method.

### Weaknesses
- The main assumption (even in the derivation of VL Norm) is the independence of token gradients (i.e., zero covariance; Line 174). However, intuitively, sequential tokens are unlikely to be independent. Although Figure 4 shows that gradient variance is correlated with response length, I am still wondering how this independence assumption might affect the results. Could the authors provide more discussion or justification regarding this assumption?
- The experimental datasets seem relatively “easy” compared to the AIME dataset commonly used in RLVR papers. The length issue would likely be more significant on such a challenging dataset, as it requires the model to generate longer responses to solve the problems. In other words, evaluating on a more difficult dataset like AIME would better highlight the importance of the length normalization issue.

### Questions
Please refer to Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
