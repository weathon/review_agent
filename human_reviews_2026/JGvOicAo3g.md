# GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training for LLM Reasoning

- Decision: Reject
- Scores: 4, 4, 6, 4, 4

## Abstract
Reinforcement learning (RL)  has recently emerged as a central paradigm for enhancing large language models' (LLMs) reasoning abilities. State-of-the-art RL with Verifiable Rewards (RLVR) methods have demonstrated remarkable effectiveness in mathematical reasoning tasks. Recent studies suggest that high-entropy tokens play an exceptionally important role in model training, since training with only the highest 20\% entropy tokens yields significant performance gains. In this work, we find that while high-entropy tokens within one answer tend to correlate with large gradient magnitude, entropy alone fails to consistently reflect token importance across different answers, considering the variations in the answer-level reward signals. Based on this observation, we introduce the **G**radient **M**agnitude-based **T**oken **S**election (GMTS) method to quantify tokens. We find that training with the top 20\%  tokens ranked by GMTS achieves substantially better performance than entropy-based selection on well-known math benchmarks (**+1.55** on Qwen2.5-math-1.5B, **+1.33** on Qwen2.5-math-7B, **+1.85** on Qwen3-8B models). These findings indicate that GMTS provides a more refined quantification than entropy, thereby improving the performance of RLVR training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper builds up on prior work showing that during finetuning with RL, training only on the tokens with highest entropy improves performance. It is found that this relates to the gradient of the tokens, and actually selecting tokens with largest gradient improves performance even more. Experiments with GRPO and DAPO finetuning on math benchmarks demonstrate that the proposed method, GMTS, improves over entropy token selection (ETS) by 1-2%.

### Strengths
1. The paper presents a convincing analysis relating entropy to gradients and derives a new method with performance gains.
2. There are experiments on several math benchmarks, various training methods and models.

### Weaknesses
1. The differences between using the proposed method, GMTS, and the base model, are very small, usually <2% accuracy. The ETS paper showed that the differences become more pronounced when using larger models, e.g. 16B or 32B, but I assume this is computationally expensive. If possible, it would be good to show a comparison to ETS on a larger model.
2. Some results are not really supporting the value of sub-set selection, e.g. in Figure 5 the curve is pretty much flat (selecting 10% or 90% best tokens has similar performance), and in Figure 3a, it looks like ETS might be better when just using more training steps. 
3. In the end, the GMTS seems to be a simple weighting of the entropy by \omega. How different are those two usually? It would be good to see a scatterplot of  E(o_i) vs E(o_i) * \omega_i.

### Questions
1. Was the model trained until convergence (figure 3a)? 
2. Are there other advantages of GMTS, e.g. more conscise answers, apart from the 1% performance gain?
3. How is the relation in Equation 3 derived? A first order Taylor expansion around x_0=1 would just yield (1-p_k) for me and not p_k (1-p_k).

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
This paper introduces Gradient Magnitude-based Token Selection (GMTS), a novel method for improving the training of Large Language Models (LLMs) on reasoning tasks using Reinforcement Learning with Verifiable Rewards (RLVR). The authors first establish a relationship between a token's entropy and its log-probability gradient magnitude, providing an explanation for the success of prior work on Entropy-based Token Selection (ETS). However, they argue that entropy alone is an insufficient proxy for token importance across different generated answers, as it does not account for variations in answer-level rewards. To address this, GMTS proposes a new metric for token importance that approximates the gradient magnitude by combining token entropy with the answer-level advantage signal. The paper demonstrates through extensive experiments on mathematical reasoning benchmarks that training on the top 20% of tokens ranked by GMTS consistently and significantly outperforms the ETS baseline across various model sizes (1.5B, 7B, 8B) and RLVR algorithms (DAPO, GRPO).

### Strengths
1. The paper provides a very clear and intuitive motivation. It begins by explaining why entropy-based selection (ETS) works, the correlation between entropy and gradient magnitude within a single answer, and then clearly demonstrates its limitations. The analysis showing that gradient distributions shift for answers with different advantages (Figure 2) provides a compelling argument that entropy alone is not a complete picture, motivating the need for a more robust metric like GMTS.
2. The proposed GMTS method is simple to understand and implement. It leverages quantities (advantage, entropy) that are already available during standard RLVR training, thus introducing minimal computational overhead. Despite its simplicity, the method is shown to be highly effective, yielding consistent and often significant performance improvements over a strong baseline (ETS) on a variety of challenging math reasoning benchmarks.

### Weaknesses
1. The performance gains of GMTS over ETS and DAPO, while consistent, appear to be in the range of 1-2 percentage points. This marginal improvement may not be sufficient to demonstrate the method's broad effectiveness and generality. Additionally, since the method is a modification of ETS, the overall contribution might seem relatively marginal.

2.	Figure 3(b) shows that entropy still trends downwards, suggesting a collapse phenomenon. This indicates the method may not fully mitigate issues in long training runs (e.g., thousands of steps). When overall entropy drops to a low point, the entropies of the top 20% of tokens will also be very low and difficult to distinguish. Can GMTS continue to provide gains in such a scenario?

### Questions
1.	Could you provide a comparison of GMTS against ETS and DAPO on models known for long chain-of-thought reasoning, such as DeepSeek-Qwen2.5-1.5b-Distill and DeepSeek-Qwen2.5-7b-Distill?
2.	Was the RL training conducted exclusively on the Math-12k dataset? Given that this dataset might be relatively simple, have you considered evaluating your method using more challenging or diverse training data, such as the datasets from Deep-Scaler, DAPO, or Skywork-OR1?
3.	Could you provide a visualization that illustrates which specific tokens are selected by ETS versus GMTS during training on a given example?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GMTS (Gradient Magnitude-based Token Selection), a technique designed to enhance Reinforcement Learning with Verifiable Rewards (RLVR) for reasoning tasks in large language models.
Building on prior work (ETS), which improved reasoning by training only on high-entropy tokens, the authors identify a limitation: entropy correlates with gradient magnitude within a single answer but not across answers with different reward values.
GMTS addresses this by weighting token importance using the product of entropy and the gradient coefficient derived from RLVR advantage terms.
Experiments on multiple Qwen math models (1.5B, 7B, and 8B) across six reasoning benchmarks demonstrate consistent performance improvements with minimal computational overhead.

### Strengths
[S1] The authors provide a clear and well-motivated problem formulation, and the overall argument for improving token selection in RLVR training is convincing.

[S2] The presentation is strong: the paper is clearly written, well-organized, and effectively communicates both the motivation and methodology.

### Weaknesses
[W1] Comparison with ETS. The paper’s central motivation that entropy-based token selection (ETS) fails because entropy varies across samples with different rewards is only qualitatively supported. While the authors provide correlation analyses between entropy and gradient magnitude, no concrete examples are shown where ETS explicitly selects misleading or low-reward tokens. As a result, the argument that “entropy cannot reliably measure token importance across answers” remains intuitive rather than empirically verified. A few explicit case studies or quantitative analyses (e.g., proportion of high-entropy tokens in low-reward trajectories) would significantly strengthen the paper’s motivation and clarity.

[W2] Variation of gradients. The paper claims that GMTS can be applied to both GRPO and DAPO training, but it does not discuss how the absence of a KL regularization term in DAPO affects the underlying gradient structure. The theoretical motivation of GMTS, that token importance correlates with gradient magnitude, is derived assuming a KL-regularized objective (as in GRPO), where the gradient is partially stabilized by the KL term. In contrast, DAPO omits this regularization, leading to higher variance and potentially different gradient scaling across tokens. The paper does not analyze whether GMTS’s gradient-based weighting remains theoretically valid or empirically stable under this setting.

[W3] Generalization to other domains. The experiments focus exclusively on mathematical reasoning benchmarks, such as AIME, AMC, and Minerva, which limits the generality of the conclusions. Since GMTS is proposed as a general token selection method for RLVR training, its effectiveness on other domains (e.g., commonsense reasoning, code generation, or science) remains unverified. It is unclear whether the observed improvements are specific to math reasoning, where token uncertainty and reward structures are relatively well-behaved, or if they extend to more diverse or noisy tasks.

### Questions
- Could the authors provide specific examples or quantitative evidence showing cases where ETS selects high-entropy tokens from low-reward answers, illustrating how it fails across samples? (W1)

- As DAPO lacks the KL regularization term, why is GMTS expected to remain generally applicable across GRPO variants with different objective forms? In other words, what makes the method robust to whether or not the training objective includes KL regularization? (W2)

- Does GMTS demonstrate generality beyond mathematical reasoning tasks, and is it expected to perform similarly in other domains? (W3)

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
This paper proposes Gradient Magnitude-based Token Selection (GMTS), a method to improve reinforcement learning with verifiable rewards (RLVR) for training large language models (LLMs) in reasoning tasks. While prior work emphasized high-entropy tokens for training, the authors find that entropy alone fails to consistently reflect token importance due to variations in answer-level reward signals. GMTS addresses this by ranking tokens based on their gradient magnitude, which better captures their contribution to learning. Experiments on mathematical reasoning benchmarks show GMTS outperforms entropy-based selection, achieving notable performance gains (+1.55 to +1.85 across Qwen models).

### Strengths
The paper focuses on improving the performance of reinforcement learning (RL) for large language models (LLMs), which is a critical and impactful area of research. RL plays a key role in enhancing LLM reasoning capabilities, and exploring methods to optimize RL training is relevant to advancing the field.

### Weaknesses
1. The paper does not provide code or implementation details, making it difficult for researchers to replicate the results or apply GMTS to other models. Providing well-documented code would enhance the paper's impact and accessibility.

2. While the experiments use multiple backbone models (Qwen2.5-math-1.5B, Qwen2.5-math-7B, and Qwen3-8B), these are relatively small-scale models. The effectiveness of GMTS on larger models (e.g., 13B, 32B) remains untested. Expanding experiments to larger-scale LLMs would strengthen the paper's claims and demonstrate broader applicability.

3. The reported baseline performance on benchmarks like Math500 and AIME24 is significantly lower than what is typically reported in official documentation or other math-related papers. This discrepancy raises questions about the experimental setup or evaluation methodology. The authors should clarify how these results were obtained and ensure alignment with standard practices to validate their findings.

### Questions
refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a Gradient Magnitude-based Token Selection (GMTS) method aimed at improving the training efficiency of large language models in mathematical reasoning tasks under reinforcement learning with verifiable rewards. The core idea of this method is to more accurately identify subsets of tokens that are critical for training by analyzing their gradient magnitudes rather than relying on traditional entropy-based measures. The approach incorporates an importance scoring mechanism that integrates entropy and advantage signals, and during training, parameter updates are performed only on high-scoring tokens. Experiments demonstrate that GMTS enhances model performance compared to other methods across mathematical reasoning benchmarks.

### Strengths
- The paper proposes using gradient magnitude as a more stable and robust importance metric. The analysis of the relationship between entropy and gradient magnitude provides a theoretical foundation for the method.  

- GMTS is designed as a plug-and-play module that can be easily incorporated into existing RLVR frameworks, which enhances its practical value and impact.

### Weaknesses
- The idea of using gradients to identify important words is rather common and has been explored in various previous studies.

- The evaluation is restricted to models under 8B parameters, making it unclear whether the proposed GMTS can generalize to larger and more capable LLMs.

- The experimental gains reported in Tables 1 and 2 are modest, raising concerns about the practical impact of the proposed method.

- The experiments primarily focus on reasoning tasks. Although mathematical reasoning is a representative complex reasoning task, the generalizability of the method to other disciplines (such as physics or chemistry) or other types of reasoning tasks has not been verified.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3
