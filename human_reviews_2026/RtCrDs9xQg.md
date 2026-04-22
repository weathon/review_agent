# Multi-Token Policy Gradient Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Policy-gradient optimization methods like PPO typically operate at the token level, estimating action probabilities for each next-token prediction. 
While effective, this formulation overlooks the structured nature of reasoning, where meaningful decisions often span multiple tokens—such as defining variables or composing equations. 
To bridge this gap, we propose Multi-token Policy Gradient Optimization (MPO), a framework that treats contiguous blocks of K tokens as unified semantic actions. 
This block-level perspective better captures the compositional structure of reasoning trajectories and supports optimization over coherent, higher-level objectives. 
Experiments on mathematical reasoning and coding benchmarks show that MPO consistently outperforms standard token-level policy gradient baselines, demonstrating the effectiveness of modeling multi-token actions for structured reasoning in LLM post-training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the challenge of granularity mismatch between sequence-level rewards and token-level actions in auto-regressive language models. The authors propose Multi-token Policy Gradient Optimization (MPO), a method that addresses this by aggregating contiguous blocks of future tokens into unified action units, rather than treating each token as an isolated action. The method is evaluated on the GSM8K and MATH benchmarks.

### Strengths
- This paper is generally well-written and easy to follow.
- The idea of treating semantically continuous tokens as a group for optimization is novel and intuitively sound: disadvantageous token spans should be suppressed collectively, while advantageous ones should be encouraged as a whole.
- This paper tries to address the important challenge of the granularity mismatch between token-level optimization and sequence-level reward signals in current RLVR scenario.

### Weaknesses
- **The motivation behind this paper is somewhat confusing** 

While MPO is proposed to address the granularity mismatch between token-level generation and sequence-level rewards by aggregating tokens into blocks, a mismatch still remains between these block-level units and the sequence-level reward. The authors should clarify how this approach truly resolves the core issue.

- **Lack of Justification for hyperparameter K**

The paper's motivation for MPO is based on grouping semantically coherent segments, as illustrated in Figure 1(a) in the paper, where segments like equations vary in length. For MPO to be most effective, one would expect it to identify the natural boundaries of these semantic units. However, as described in Section 4, the method does not appear to identify such boundaries and instead relies on a fixed hyperparameter K to evenly divide the token sequence into contiguous blocks.

A related question concerns the optimal value of K. Table 2(a) indicates that K=5 yields the best performance among the values {2,3,5}. This seems to contradict the intuitive example in Figure 1(a), where the authors state that the "model's decision-making process needs to switch constantly between long token segments with more than 10 tokens." Could the authors explain this apparent mismatch? Specifically, why does a relatively small K=5 achieve optimal performance when the semantic segments it is meant to approximate are typically much longer?

- **Limited experimental evaluation**

The experimental evaluation is limited to two mathematical reasoning benchmarks, GSM8K and MATH. This raises concerns about the generalizability of MPO's effectiveness. To robustly support the claim that MPO is effective for "large language model post-training," additional evaluations on diverse tasks (e.g., code generation and instruction following) are necessary.

### Questions
In section 6.2, the authors state, "(clip fraction) reduction becomes more pronounced as the proportion of future information increases. This suggests that future-aware training stabilizes policy updates" (lines 405-407). However, on lines 417-419, they state, "This highlights that excessive incorporation of future information may lead to training instability." Both experiments were conducted using the same settings (MPO-10%/20%/30%/40%) but appear to have contradictory conclusions. I hope the authors can clarify this for me.

### Soundness
2

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
3

### Summary
This paper introduces Multi-Token Policy Gradient Optimization (MPO), a framework that extends token-level policy gradient methods by treating blocks of $K$ future tokens as atomic actions. The key motivations are: (1) to stabilize advantage estimation by aggregating future knowledge through multi-token prediction, and (2) to encourage higher-level planning over coherent text segments rather than individual tokens.
To realize this, MPO replaces the standard per-token importance sampling ratio with a block-level ratio computed over $K$ consecutive tokens. Because directly multiplying per-token ratios can cause high variance, the authors approximate this product using a weighted log-sum of log-ratios, inspired by the Log-COP-TD method.
Empirically, MPO consistently outperforms PPO and GRPO baselines on mathematical reasoning benchmarks (GSM8K and MATH), demonstrating improved stability, lower clip fraction, and higher accuracy in sequence-level reasoning tasks.

### Strengths
MPO outperforms GRPO and PPO on mathematical reasoning benchmarks (GSM8K and MATH)

### Weaknesses
Although this paper is practical, it lacks sufficient experiments.
- The experimental results are based on relatively small language models (1B or 1.5B).
- The GSM8K and MATH benchmarks are not sufficient to demonstrate the performance of LLMs. Additional tasks should be included (at least those used in the GRPO paper).

### Questions
I have following questions about this paper:

1. Additional Experiments (related to Weaknesses)

    The experimental results are based on relatively small language models (1B and 1.5B), which may not be sufficient to evaluate the alignment effect. Are there any additional results on larger models? In addition, the GSM8K and MATH benchmarks may not adequately demonstrate the performance of LLMs. Do the authors have results on other tasks, such as those used in the GRPO paper?

2. About $\beta\_k$

    The authors state that $\beta\_k$ are normalized to $\sum\_{n=1}^K \beta\_n=1$. I assume this means $\beta\_k=\frac{\tilde\beta\_k}{\sum\_{n=1}^K \tilde\beta\_n}$ for arbitrary $\{\tilde\beta\_k\}$. However, the paper seems to adopt a different kind of normalization. Could the authors clarify what “normalization” specifically means in this context?

3. Approximation of $\widetilde{R}$

    Equations (11) and (12) are equivalent when $\beta\_n=1$ for all $n$. However, enforcing $\sum\_{n=1}^K \beta\_n=1$ may introduce additional bias, even if it helps reduce variance, as authors mention in line 289 - 290. How do the authors justify this design choice? Moreover, given the potential bias, is the comparison in Figure 4 fair?

4. Trade-off between variance and bias

    In this paper, both the choice of $K$ and the set of $\{\beta\_n\}$ involve a trade-off between variance and bias. Could the authors clarify how these trade-offs are characterized or managed in the proposed method?

### Soundness
3

### Presentation
2

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
This paper proposes **Multi-Token Policy Gradient Optimization (MPO)**, a novel framework for post-training large language models (LLMs) using reinforcement learning. The central claim is that standard policy-gradient methods like PPO operate at a token-level granularity, which is a poor match for sequence-level rewards in complex reasoning tasks like mathematics. To address this, MPO redefines the policy gradient "action" as a **block of $K$ future tokens**. It computes the importance sampling ratio over this entire multi-token block rather than a single token.

To implement this, the method incorporates auxiliary **Multi-Token Prediction (MTP) modules**, which are first "warmed up" to predict future token probabilities. To manage the high variance that arises from multiplying multiple probability ratios, MPO introduces a **log-space approximation** (a weighted log-sum) to stabilize training. Experiments conducted on the GSM8K and MATH benchmarks show that MPO consistently outperforms token-level baselines, PPO and GRPO, in mathematical reasoning accuracy.

### Strengths
-  **Important Problem:** The paper addresses a significant and timely problem: the granularity mismatch between the token-level optimization of standard policy gradient methods and the sequence-level, holistic nature of rewards in complex reasoning tasks. Improving RL alignment for multi-step reasoning is a critical research direction.

- **Sufficient Literature Review:** The paper provides a comprehensive review of related work. It clearly positions MPO relative to both existing policy gradient methods (PPO, GRPO, DAPO) and the separate field of multi-token prediction (MTP) techniques.

- **Novel Method:** The core contribution—modifying the policy gradient objective to compute importance sampling ratios over multi-token blocks—is a novel and interesting approach for LLM alignment.

- **Clarity and Presentation:** The paper is generally well-written and easy to follow. The method is explained clearly, and the inclusion of diagrams like Figure 1 and Figure 2 is helpful for understanding the MTP module's role and the overall MPO training process.

### Weaknesses
- **Insufficient Motivation:** The paper is not adequately motivated. While it claims MPO "better captures the structure of reasoning", it doesn't provide a strong intuitive or theoretical explanation for *why* computing the importance ratio over $K$ future tokens leads to a more effective or stable policy update compared to the standard single-token ratio.

  

- **Methodological Soundness and Cost:** The proposed method's reliance on *new* MTP modules raises significant concerns about soundness and efficiency:

  - **Redundancy:** The authors do not explain why the original, backbone LLM cannot be used to auto-regressively compute the probabilities for the $K$ future tokens. The introduction of separate MTP modules seems redundant and adds significant complexity.

  - **Parameter Overhead:** These new modules add a substantial number of parameters, increasing the model size significantly (e.g., 1.52x to 2.05x for $K=4$ on a Llama3.2-1B model).

  - **Training Complexity:** The MTP modules require a separate "warm-up" phase before MPO training can begin, complicating the overall training pipeline.

  - **Computational Cost:** When computing the MPO loss, the objective (Eq. 12) appears to require $K$ forward passes (or at least $K$ probability lookups) *for each token* $t$ in the trajectory to calculate the multi-token ratio. This represents a potential $K$-fold increase in computational cost for the loss computation, which is a major drawback.

- **Limited Empirical Evaluation:** The empirical results could be improved. The method is only tested on two math benchmarks (GSM8K, MATH) and compared against only two baselines (PPO, GRPO). A key omission is a comparison against **DAPO**, which is cited as an advanced method for reasoning tasks and would serve as a much stronger baseline.

- **Insignificant Performance vs. Cost:** The reported performance gains are not very significant, especially when weighed against the massive increase in computational and parameter costs. For example, on GSM8K (Deepseek-Qwen2.5-1.5B), MPO ($K=5$) improves over PPO by only 1.6 percentage points (0.882 vs. 0.866), while on MATH the gain is also 1.6 points (0.779 vs. 0.763). This marginal improvement does not seem to justify the 26-48% slowdown in training iteration speed and the 1.2x-2.05x increase in model size.

### Questions
1. Could the authors clarify the precise reason for introducing separate MTP modules? Why not use the backbone LLM's own decoder to compute the future token probabilities $\pi_{\theta}(o_{i,t+n}|o_{i,1:t+n-1})$ for the importance ratio? What advantages do the specialized MTP module architectures (as shown in Figure 1b)  provide over the standard backbone?

2. The paper motivates MPO as "encouraging higher-level planning" Could the authors elaborate on this mechanism? How does aggregating $K$ ratios via a weighted log-sum (Eq. 12) translate to a better policy gradient for long-term reasoning, beyond the variance reduction shown in Figure 4a?

3. The best results are reported for $K=5$. However, the cost analysis in Table 3 only provides data for $K=2$ and $K=4$. What are the specific model size and training speed costs for the $K=5$ setting, which yielded the best performance?

4. Given that DAPO is also a policy gradient method designed to improve upon PPO/GRPO for reasoning tasks, why was it omitted from the experimental comparison? How would the authors hypothesize MPO compares to DAPO in terms of both performance and computational efficiency?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Multi-Token Policy Gradient Optimization (MPO) that extends token-level policy optimization to multi-token policy optimization for matching the sequence-level nature of rewards in reasoning tasks. More specifically, MPO calculates importance sampling ratios (of PPO) over K future tokens. Then, MPO proposes the MPO objective in Eq. 13. This paper evaluates MPO on two representative mathematical reasoning benchmarks including GSM8K and MATH by using two language models including DeepSeek-Distilled-Qwen2.5-1.5B and Llama3.2-1B.

### Strengths
- S1. [Presentation] First of all, this paper is very well written and organized.

- S2. [Motivation] It is clearly motivated why multi-token policy optimization of LLMs is required for mathematical reasoning. 

- S3. [Analysis] Besides the main results, this paper provides a comprehensive empirical analysis on the effectiveness of MPO. It includes (1) variance of importance sampling ratios, (2) effect of future information injection, (3) effect of MTP hyper-parameters, and (4) reliability of incorporated future information. Also, this paper provides an analysis of time and memory cost, comparing MPO with PPO.

### Weaknesses
- W1. [Performance] One of main weaknesses of this paper may be the performance gain, compared to the training cost. According to Figure 3, in case of DeepSeek-Qwen2.5-1.5B, MPO achieves the accuracy of 0.882, while PPO provides 0.866. According to Table 3, MPO takes 30% more training time and 40% more memory.

### Questions
- Q1. This paper uses MPO to train relatively small language models such as Deepseek-Qwen2.5-1.5B and Llama3.2-1B. Is the performance improvements increased, if larger language models are used?

- Q2. Current implementation of MPO is based on PPO. Is it possible to apply MTP to GRPO?

### Soundness
3

### Presentation
3

### Contribution
3
