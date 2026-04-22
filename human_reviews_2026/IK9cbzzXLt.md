# Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Speculative decoding accelerates large language model (LLM) inference by using a small draft model to generate candidate tokens for a larger target model to verify. The efficacy of this technique hinges on the trade-off between the time spent on drafting candidates and verifying them. However, current state-of-the-art methods rely on a static time allocation, while recent dynamic approaches optimize for proxy metrics like acceptance length, often neglecting the true time cost and treating the drafting and verification phases in isolation. To address these limitations, we introduce Learning to Draft (LTD), a novel method that directly optimizes for throughput of each draft-and-verify cycle. We formulate the problem as a reinforcement learning environment and train two co-adaptive policies to dynamically coordinate the draft and verification phases.  This encourages the policies to adapt to each other and explicitly maximize decoding efficiency.
We conducted extensive evaluations on five diverse LLMs and four distinct tasks. Our results show that LTD achieves speedup ratios ranging from 2.24x to 4.32x, outperforming the state-of-the-art method Eagle3 up to
36.4\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes Learning to Draft (LTD), a novel speculative decoding method that directly optimizes for throughput using reinforcement learning. LTD trains two distinct policies -- a depth policy for controlling the tree depth in drafting and a size policy for deciding the number of candidate tokens to verify -- that are co-optimized to jointly improve drafting and verification. Experimental results using the Eagle3 framework shows that LTD can yield substantial throughput improvements for greedy decoding across the Eagle3 baseline with various model families and sizes.

### Strengths
1. The paper's main premise, which is that optimizing for acceptance length is suboptimal, is well explain and is empirically supported. 

2. The proposed RL method, which jointly trains two policies to directly optimize throughput, is intuitive and demonstrates promising empirical results for greedy decoding. The initial policy training and iterative optimization offer practical implementation benefits.

3. Experiments with multiple variants of the Eagle3 baseline with five different LLMs.

### Weaknesses
1. Missing discussion on training cost. While using sufficiently well trained policies have shown to often yield substantial throughput improvements, it is unclear if the cost of RL training is justified in all cases. Can one grid search be more efficient at the end than having to do RL training (which involves initial policy training + iterative optimization), especially for those cases where LTD doesn't lead to improvements (e.g., Dpsk 8B, Qwen3 14B, Qwe3 32B on several of the benchmarks)

2. Missing analysis on inference overhead. The lightweight MLP policies may add only "negligible overhead", more discussion on inference cost of running the policies in isolation would be useful. Also, do MLP policies with a greater capacity give better predictions? If so, what are the trade-offs between using larger MLP policies for better predictions vs. inference overhead. How do the answers change for different target models?

3. Training data generalization. It is somewhat surprising that the policies trained on HumanEval data give substantial improvements for non-coding benchmarks. There needs to be more discussion on the potential limits of generalization. Does training on more in-domain data lead to better results? How do policies trained on non-coding data transfer to other types of benchmarks?

### Questions
Q. How do the methods compare in case of sampling (as opposed greedy)?

Q. The amount of compute needed to train the policies seems to heavily depend on the target model. Have the authors done any analysis on potential policy generalization, i.e., how the two policies trained for a smaller target possibly generalize to a larger target?

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
This paper proposes Learning to Draft (LTD), a reinforcement learning-based speculative decoding method that directly optimizes throughput, defined as the number of accepted tokens per unit time in this paper, by jointly training two co-adaptive policies to control draft depth and verification size. Unlike prior static or heuristic approaches (e.g., Eagle3), LTD dynamically balances drafting and verification costs, achieving up to 36.4% speedup across five LLMs and four benchmarks while maintaining robust performance under different decoding settings.

### Strengths
- The paper introduces a principled reinforcement learning framework that directly optimizes throughput rather than proxy metrics like acceptance length, addressing a key limitation in prior speculative decoding methods.
- Extensive experiments across multiple LLMs and tasks demonstrate consistent and significant speedups (up to 36.4%) over strong baselines such as Eagle3. The analyses are also comprehensive and well-designed.
- The method generalizes well to high-temperature decoding scenarios where other speculative decoding methods typically degrade.

### Weaknesses
- The RL-based policy training (on HumanEval for 100K and 1M PPO steps) is expensive and tuned on a specific dataset. I am concerned about the training cost and the transferability to unseen domains or longer-context tasks. Could you provide the training details, including the actual GPU hours and why training for such extremely long steps?
- I am also concerned about the iterative optimization part. From Table 7 and 8, we can see that some models/datasets benefit from iterative optimization, while others demonstrate marginalized effect. The orders for optimization (size first or depth first) also lacks investigation. It would be great if the authors could provide some guidance on the design choices for iterative optimization, so that LTD will be more useful in practice.

### Questions
- In Figure 3, any intuitions as to Llama saturates while DeepSeek continues to gain advantage from iterative training?
- Some typos: (i) Make sure to add spacing before left brackets, e.g., lines 153 and 190. (ii) Figure 3 right, "token" should be "size".

### Soundness
3

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
3

### Summary
This paper presents Learning to Draft (LTD), a reinforcement learning (RL) framework for optimizing speculative decoding efficiency in large language models.
Unlike prior methods, which use fixed or heuristic strategies to adjust the draft depth and verification size, LTD directly maximizes throughput (accepted tokens per total time) rather than surrogate metrics such as acceptance length.
The authors model the drafting–verification process as an RL environment with two co-adaptive policies: a Depth Policy (controls how deep the draft tree expands), and a Size Policy (controls how many candidate tokens are verified).
Both are trained via PPO using throughput as a direct reward.
LTD achieves 2.24×–4.32× speedup across five LLMs and four benchmarks, improving upon EAGLE-3 by up to 36.4% without quality degradation, and maintains robustness even under high-temperature sampling.

### Strengths
- The paper formalizes speculative decoding as a reinforcement learning (RL) environment that directly optimizes throughput rather than proxy metrics such as acceptance length. It introduces two interacting policies (Depth and Size), and the co-adaptive training framework is conceptually clear and empirically validated.
- The algorithmic structure is clearly explained, and the figures effectively illustrate the RL formulation and the draft–verify cycle.
- The proposed method demonstrates robustness across different sampling temperatures and models, showing consistent performance improvements.
- The approach can enhance real-world LLM inference speed without retraining or modifying the base model, which makes it practically valuable.

### Weaknesses
- The RL formulation lacks theoretical guarantees or convergence analysis under the throughput-based reward.
- The paper does not report quantitative metrics related to training efficiency (e.g., sample efficiency, or convergence speed).
- The paper does not discuss the rationale behind selecting PPO as the optimization algorithm for policy learning, nor whether alternative RL methods were considered.
- While the ablation studies demonstrate co-adaptation effects, the analysis remains qualitative. A more quantitative study (e.g., correlation between policy actions or mutual information) could strengthen the argument for policy synergy.
- The policies are trained on the HumanEval code dataset but evaluated on text-based benchmarks (e.g., GSM8K, MT-bench). The rationale or analysis explaining this generalization behavior is missing.
- The experimental section lacks comparisons with other adaptive or self-speculative methods such as DEL, or LayerSkip which would provide a stronger baseline context.

### Questions
- Can you report sample efficiency or convergence statistics (e.g., number of steps to reach stable reward or variance across runs)?
- What was the rationale for selecting PPO as the optimization algorithm? Were alternative RL methods considered?
- Have you analyzed the interaction strength between the Depth and Size policies (e.g., correlation of their actions during co-training)?
- What motivated the choice of HumanEval as the primary training environment?
- How much additional inference-time latency is introduced by running the two policy networks?

### Soundness
3

### Presentation
4

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
The paper introduces Learning to Draft (LTD), a novel method for accelerating large language model (LLM) inference using adaptive speculative decoding. Unlike previous methods that rely on static configurations or optimize for proxy metrics like acceptance length, LTD directly optimizes for the throughput of each draft-and-verify cycle, accounting for the crucial time cost of both the drafting and verification phases. The authors solve this problem using a Reinforcement Learning (RL) environment and train two distinct, co-adaptive policies: a depth policy to dynamically control the draft tree's depth (draft cost) and a size policy to manage the verification size (verification cost). These policies are jointly optimized to maximize the throughput, defined as the number of accepted tokens divided by the total time ($L_A / T_{total}$). Through extensive evaluations, LTD demonstrated significant throughput improvements of up to 36.4% over the state-of-the-art Eagle3 baseline across various LLMs and tasks, proving a more robust and efficient dynamic strategy for speculative decoding.

### Strengths
1. The proposed method LTD directly optimizes the practical system throughput ($L_A / T_{total}$), i.e., the number of accepted tokens divided by the total time. This ensures the strategy is genuinely maximizing inference speed while prior works mostly focused on indirect metrics like acceptance length. 

2. The method uses a Reinforcement Learning (RL) environment to train two dynamic, co-adaptive policies: one for the draft tree's depth and one for the verification size. This dynamic approach allows the model to continuously adjust both the drafting cost and the verification cost for optimal performance in real-time.

3. LTD demonstrates significant and robust performance gains, achieving speedup ratios up to 4.32x and improving throughput by up to 36.4% over state-of-the-art baselines like Eagle3. This indicates a highly effective and generalizable strategy for accelerating LLM inference across various models and tasks.

### Weaknesses
1. Training the adaptive policies requires setting up and running a complex RL environment. This adds significant computational overhead and complexity during the training phase, which is a major barrier to adoption compared to simpler, non-adaptive speculative decoding methods.

2. The core of LTD relies on accurately modeling the time cost of both the drafting and verification phases to compute the throughput objective ($T_{total}$). If the real-world environment introduces variances or non-linearities in time cost that the model does not capture, the policies trained in the simplified environment may become sub-optimal in production.

3. Like all speculative decoding methods, LTD's performance still fundamentally relies on the quality of the small draft model to accurately predict future tokens. While the adaptive policies optimize the use of the draft model, they cannot compensate for a draft model that frequently generates incorrect tokens, which would lead to low acceptance rates and minimal speedup.

4. The paper only compares against Eagle3 as the base framework, but does not evaluate on other tree-based methods like SpecInfer or Medusa. The contribution would be strong if more comparisons with recent adaptive methods are provided.

### Questions
1. How does the computational cost of training and deploying the two co-adaptive policies using the RL environment compare to the total inference time savings achieved? Is the initial overhead of the RL training process outweighed by the long-term throughput gains in practical, high-volume production settings?

2. If the small draft model's quality degrades (e.g., due to a mismatch between its training data and the target task) how effectively can the adaptive depth and size policies mitigate the resulting low token acceptance rate to maintain a high throughput? What is the minimum acceptable acceptance rate before LTD's advantages over static methods disappear?

### Soundness
3

### Presentation
4

### Contribution
3
