# Segmental Advantage Estimation: Enhancing PPO for Long-Context LLM Training

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Training Large Language Models (LLMs) for reasoning tasks is increasingly driven by Reinforcement Learning with Verifiable Rewards (RLVR), where Proximal Policy Optimization (PPO) provides a principled framework for stable policy updates. However, the practical application of PPO is hindered by unreliable advantage estimation in the sparse-reward RLVR regime. This issue arises because the sparse rewards in RLVR lead to inaccurate intermediate value predictions, which in turn introduce significant bias when aggregated at every token by Generalized Advantage Estimation (GAE). To address this, we introduce Segmental Advantage Estimation (SAE), which mitigates the bias that GAE can incur in RLVR. Our key insight is that aggregating $n$-step advantages at every token(as in GAE) is unnecessary and often introduces excessive bias, since individual tokens carry minimal information. Instead, SAE first partitions the generated sequence into coherent sub-segments using low-probability tokens as heuristic boundaries. It then selectively computes variance-reduced advantage estimates only from these information-rich segment transitions, effectively filtering out noise from intermediate tokens.  Our experiments demonstrate that SAE achieves superior performance, with marked improvements in final scores, training stability, and sample efficiency.  These gains are shown to be consistent across multiple model sizes, and a correlation analysis confirms that our proposed advantage estimator achieves a higher correlation with an approximate ground-truth advantage, justifying its superior performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on the problem of instability of GAE estimation of token-level PPO method, when used in a typical RLVR setting with terminal reward. Most of the contemporary solutions employ $\lambda=1$ further amplifying the instability in training an accurate value estimate.

To mitigate this issue, authors propose a clever solution of segmenting the responses into chunks of high probablity tokens and reducing the overall effective "actions" in a response trajectory from MDP perspective. They call this method Segmental Advantage Estimation (SAE). SAE effectively reduced the number of steps in a response trajectory from number of tokens to number of segmented chunks and in-effect reduces the inefficiency in estimating the advantage estimate.

Experiments on multiple model scales across standard math datasets and evals, show that SAE consistently outperforms PPO ($\lambda=1$ and adaptive $\lambda$) and GRPO.

### Strengths
- Well motivated problem of instablity of GAE estimation in RLVR where $\lambda$ is set to 1
- Insightful solution to focus on segments of response for GAE estimation instead of per token
- Theoretical analysis to justify that SAE reduces the bias in estimation.
- Emprirical analysis showcasing SAE has highest correlation with true Advantage compared to other baselines in a controlled setting.

### Weaknesses
- The SAE method uses a fixed threshold of 0.2 on the probability to decide the segments. I would have preferred an abilation study for the choice of this parameter.
- I would prefer to have SAE compared with the simple baseline of fixed length segments from the theoretical analysis of section 4.2. For example, what is the effect when I naive let chunks to be of size $M=100$ or $200$ tokens irrespective of the probablity. Does the choice of segmentation method matter towards the downstream performance of SAE?

### Questions
1. I would have preferred to see some analysis of average segment length for different model sizes in a practical setting. The paper will greatly benefit with more analysis and the effects of varying the probablity threshold $p$ on the segment size.
2. Did authors try other segmentation methods such as entropy instead of raw token probability?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes the Segmental Advantage Estimation (SAE) method to improve value estimation in PPO algorithms. SAE first partitions the generated sequence into coherent sub-segments, using low-probability tokens as heuristic boundaries, and then treats each segment as an action for GAE computation. Experiments are conducted to demonstrate the effectiveness of the proposed method compared to standard PPO.

### Strengths
- The paper is well written and easy to follow.
- Accurate value estimation is crucial for PPO algorithms. The idea of segmenting responses based on low-probability tokens is intuitive and makes sense.
- Experiments are conducted to demonstrate the effectiveness of the proposed method compared to standard PPO.

### Weaknesses
My main concern is the lack of comparison with related baseline:
-  There is no comparison with the mentioned related works, such as VC-PPO and VAPO
-  Previous studies have proposed computing GAE at the step level (e.g., by splitting sequences using special tokens such as ‘\n’) [1]. This paper is closely related to those approaches, and a comparison with them would help better demonstrate the effectiveness of the proposed method.

[1]Chen, Guoxin, et al. "Alphamath almost zero: process supervision without process." Advances in Neural Information Processing Systems 37 (2024): 27689-27724

### Questions
Please refer to the weakness part.

### Soundness
2

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
3

### Summary
This paper introduces Segmental Advantage Estimation (SAE) to improve Proximal Policy Optimization (PPO) for training Large Language Models (LLMs) on long-horizon reasoning tasks with verifiable rewards (RLVR).  It aims to address the unreliable advantage estimation in sparse-reward settings, where traditional Generalized Advantage Estimation (GAE) amplifies bias by performing token-level bootstrapping using noisy value predictions. SAE mitigates this by first partitioning the generated sequence into semantically coherent segments, using low-probability tokens as heuristic boundaries, and then selectively computing advantages only at these segment transitions. This reduces bootstrapping bias by filtering out noise from intermediate, low-information tokens.

### Strengths
(1) The proposed method in this paper is practically elegant, as its recursive formulation allows for seamless integration into existing PPO frameworks with minimal computational overhead. 
(2) The empirical evaluation is thorough, benchmarking against strong baselines like GRPO and adaptive PPO variants across multiple out-of-distribution test sets (AIME, AMC). The consistent performance gains across 4B, 8B, and 14B model sizes strongly support the method's robustness and scalability.

### Weaknesses
(1) While the use of low-probability tokens is intuitive, it is an unsupervised method that may not always align perfectly with true semantic boundaries, potentially introducing its own form of noise.
(2) The evaluation is confined to mathematical reasoning. While this is a canonical domain for RLVR, the paper does not demonstrate SAE's efficacy in other long-context scenarios like code generation or complex dialogue, limiting the claimed generality of the approach.

### Questions
(1) How might more sophisticated, learned segmentation strategies (e.g., leveraging an auxiliary model or syntactic features) further improve the performance and robustness of SAE compared to the current probability-based heuristic?
(2) The paper sets the segmentation threshold p=0.2 universally. How sensitive is the performance of SAE to this hyperparameter, and could an adaptive or dynamically learned threshold offer benefits?

### Soundness
2

### Presentation
2

### Contribution
2
