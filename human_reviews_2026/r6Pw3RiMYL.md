# Beyond Magnitude: Leveraging Direction of RLVR Updates for LLM Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning capabilities of large language models. 
While existing analyses identify that RLVR-induced changes are sparse, they primarily focus on the **magnitude** of these updates, largely overlooking their **direction**. 
In this work, we argue that the direction of updates is a more critical lens for understanding RLVR's effects, which can be captured by the signed, token-level log probability difference $\Delta\log p$ between the base and final RLVR models.
Through statistical analysis and token-replacement interventions, we demonstrate that $\Delta\log p$ more effectively identifies sparse, yet reasoning-critical updates than magnitude-based metrics (e.g., divergence or entropy).
Building on this insight, we propose two practical applications:
(1) a *test-time extrapolation* method that amplifies the policy along the learned $\Delta\log p$ direction to improve reasoning accuracy without further training;
(2) a *training-time reweighting* method that focuses learning on low-probability (corresponding to higher $\Delta\log p$) tokens, which improves reasoning performance across models and benchmarks.
Our work establishes the direction of change as a key principle for analyzing and improving RLVR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper identifies the direction of policy updates as a critical yet underexplored factor underlying RLVR’s effects. The authors show that reformulating the sampling policy using $\Delta \log p$ between the RL policy and the base policy yields improved performance on math-focused datasets. They also provide a concise theoretical explanation for this phenomenon and introduce a simple, training-based method that further strengthens the model.

Overall, the paper is well organized and easy to follow. I enjoyed reading it and believe it offers useful insights to the community. However, the experimental evaluation is limited to math-related tasks. Expanding the experiments to other domains—such as coding, agentic tasks, or logical reasoning—would substantially strengthen the work.

### Strengths
1. The paper is clearly written and easy to follow.
2. The observed phenomenon is important and, to my knowledge, has not been systematically explored; it offers meaningful insight.
3. The theoretical explanation is concise yet sufficiently direct to convey the key ideas. The references are appropriate, and I did not notice overclaims.
4. The proposed methodology is simple to implement and appears effective. If the results generalize, it could be broadly useful to practitioners.

### Weaknesses
1. The experimental setup is restricted to math-related tasks. It is unclear whether the gains extend to other tasks such as coding, agent-based evaluation, or logical reasoning.
2. Minor issues (easy to address):
   - The “averaged KL divergence” defined in lines 133–134 corresponds to the Jensen–Shannon divergence, which has a standard name.
   - Figure 1(c) uses “token replacement,” but the corresponding setup is only introduced in Section 3.2. I recommend adding a hyperlink or forward reference around lines 073–077 to improve readability.

### Questions
1. Could the authors clarify the y-axis in Figure 1(b)? It appears to be neither raw counts nor frequency (since it exceeds 1 at some points). In addition, how were the tokens collected—how many tokens, how many RL training steps, which tasks, and which algorithm?
2. In Figure 2, replacement stops when performance reaches the RLVR value. What happens if the RLVR replacement ratio is increased further for the three curves?
3. What is the effect of applying the test-time method on the model trained with your training-based method? Does the combination yield additional gains?
4. Could the authors provide results on at least one non-math task, as noted above? Adding another domain would significantly strengthen the paper.

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
This paper emphasizes the direction of RLVR updates via a token-level signed metric Δlog p and shows it better localizes sparse, high-impact tokens than entropy or KL. Building on this, the authors propose (i) selective test-time extrapolation that nudges the policy along Δlog p at a small set of salient tokens, and (ii) a training-time advantage reweighting that emphasizes low-probability, reasoning-critical tokens. On math-reasoning benchmarks, both strategies improve over baselines.

### Strengths
1.Clear idea. The paper puts the direction of change (signed Δlog p) at the center, not just the size.
2. Practical methods. The two tools—selective test-time extrapolation and training-time reweighting—are straightforward and easy to try.
3. Low integration cost. The approach needs only log-probs from a base and an RL model; it’s a small code change with no extra labels.
4. Token-level evidence. The replacement test shows that a small set of tokens drives most of the gains, making the story concrete.

### Weaknesses
1. The paper asserts prior work focuses on magnitude and neglects direction, which is only partly true. While "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning"（arXiv:2506.01939) is largely magnitude-centric (entropy-based selection), Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs (arXiv:2505.12929) explicitly measures for positive-advantage tokens and proposes methods to reduce direction errors(Fig3). Please revise the positioning to acknowledge signed-direction analysis and clarify that your contribution is to directionality (e.g., Δlog p) more directly.
2. The comparison in the token‐replacement experiment may misinterpret prior works’ use of entropy or KL divergence, which aimed to guide optimization directions during training rather than to describe post‐training entropy distributions — please clarify this distinction.
3. Hyperparameter robustness. The method relies on a gate τ (token selection) and an extrapolation strength γ (fixed at 0.1). A lack of sensitivity analysis leaves open whether gains are robust across tasks or temperatures.

### Questions
1. Direction correctness：Could you show the rate at which positive-advantage tokens are increased and negative-advantage tokens are decreased (signed Δlog p), tracked over training, and compare this to arXiv:2505.12929
2. Sensitivity to τ and γ (robustness): Could you share the performance across a grid of τ and γ to identify a stable operating region rather than a single tuned point.

### Soundness
3

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
This paper argues that the direction of token-level updates, measured by the signed log-probability difference between base and RLVR models, is more informative than magnitude-based metrics for understanding RLVR's effect on LLM reasoning. The authors propose and validate two methods, test-time extrapolation and training-time advantage reweighting, which exploit the directional insight to improve reasoning performance.

### Strengths
The paper introduces a novel and intuitive directional metric that effectively captures sparse, reasoning-critical updates. The metric is supported by rigorous token-replacement experiments and gradient analysis.

The proposed methods are simple yet effective, with consistent gains across multiple models and benchmarks without the need for additional training data.

### Weaknesses
The test-time extrapolation method requires access to both the base and RLVR models, which may limit its practicality in settings where only the fine-tuned model is available.

The paper focuses primarily on mathematical reasoning benchmarks (e.g., AIME, AMC); it remains unclear whether the findings generalize to other reasoning domains or more diverse tasks.

Theoretical justification in Theorem 4.1 relies on a simplified tabular softmax bandit setting, which may not fully reflect the complexity of modern LLM training dynamics.

### Questions
How does the proposed direction-based extrapolation perform in non-mathematical reasoning tasks?

The token-replacement experiment convincingly shows that \Delta\log p identifies critical tokens. However, does this intervention sometimes hurt the performance? 

In which instances does replacing a base model's token with the RLVR model's choice result in an incorrect answer, and what characterizes these tokens?

### Soundness
3

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
The paper analyzes reinforcement learning with verifiable rewards through how probabilities shift between the base and RLVR-trained models. The paper finds that upweight advantages of low-probability tokens during RL training improves reasoning accuracy.

### Strengths
The proposed advantage reweighting methods is straightforward to implement. Despite the simplicity, it consistently improves reasoning performance across multiple models (Qwen2.5, Qwen3) and benchmarks (AIME, AMC).

### Weaknesses
1. How sensitive are performance gains to γ and τ remain unclear.

### Questions
1. Can you provide examples of tokens with large ∆log p? What qualitative insights do they reveal?
2. Will amplifying the penalty of negative tokens also bring benefits as indicated in this work [1]?

[1] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning. NeurIPS 2025

### Soundness
2

### Presentation
2

### Contribution
2
