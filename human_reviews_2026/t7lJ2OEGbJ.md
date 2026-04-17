# Learning to Evict from Key-Value Cache

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
The growing size of Large Language Models (LLMs) makes efficient inference challenging, primarily due to the memory demands of the autoregressive Key-Value (KV) cache. Existing eviction or compression methods reduce cost but rely on heuristics, such as recency or past attention scores, which serve only as indirect proxies for a token’s future utility and introduce computational overhead. We reframe KV cache eviction as a reinforcement learning (RL) problem: learning to rank tokens by their predicted usefulness for future decoding. To this end, we introduce KV Policy (KVP), a framework of lightweight per-head RL agents trained on pre-computed generation traces using only key and value vectors. Each agent learns a specialized eviction policy guided by a holistic reward, derived from future utility, that evaluates the quality of the ranking across all cache budgets, requiring no modifications to the underlying LLM or additional inference. Evaluated on the long-context benchmark RULER and the multi-turn dialogue benchmark OASST2-4k, KVP significantly outperforms baselines. Furthermore, zero-shot tests on standard downstream tasks indicate that KVP generalizes well beyond its training distribution. These results demonstrate that learning to predict future token utility is a powerful and scalable paradigm for adaptive KV cache management.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces KV Policy, a reinforcement learning framework for KV cache eviction in LLMs. Instead of using heuristic policies (recency, attention score, etc.), KVP learns to rank tokens by predicted future utility. Each attention head in the LLM is paired with a lightweight RL agent trained offline on pre-computed KV traces using only keys, values, and positional embeddings. The reward function evaluates the ranking quality across all cache budgets without additional inference. Experiments on long-context benchmarks (RULER, OASST2) and zero-shot downstream tasks (BoolQ, ARC-Challenge, MMLU, HellaSwag) show that KVP achieves consistently better accuracy and lower perplexity than heuristic and attention-based baselines such as SnapKV, TOVA, and StreamingLLM.

### Strengths
1. The use of per-head lightweight RL agents is well-motivated and efficiently parallelizable.

2. The authors include careful ablations isolating the contribution of the RL objective and the reward design.

### Weaknesses
1. While the paper claims that KVP introduces minimal overhead, there are no quantitative measurements of inference latency or throughput. Reporting actual wall-clock runtimes or speedups relative to heuristic and attention-based baselines would strengthen the empirical evaluation.

2. The paper claims offline efficiency, but training 112 separate agents still requires substantial GPU resources. A clearer cost-benefit analysis would help.

3. Because KVP requires a dedicated offline training phase, comparing it only with training-free heuristic baselines is not entirely fair. It would strengthen the evaluation to include comparisons with other training-based or learned compression methods, such as Gisting Token (https://arxiv.org/abs/2509.15763) or Activation Beacon (https://arxiv.org/abs/2401.03462).

4. Experiments are limited to one base model (Qwen2.5-7B-Chat). It would strengthen the paper to demonstrate that the learned policies generalize to other architectures (e.g., Llama, Mistral).

5. Since the core motivation is to reduce memory and latency, experiments on truly long-context settings (e.g., 10k–100k tokens or more) are necessary. The current benchmarks (RULER and OASST2-4k) do not fully test KVP’s scalability under extreme context lengths, limiting the conclusions about its real-world applicability.

6. The generalizability of KVP remains uncertain. It would be important to assess how sensitive the learned policies are to the choice of offline training data. For example, would a policy trained on conversational datasets transfer effectively to domains such as code generation or complex reasoning tasks?

7. KVP operates offline and applies a fixed learned policy per head. This limits adaptivity during runtime when generation dynamics may differ from training data distributions.

### Questions
See the limitations.

### Soundness
2

### Presentation
2

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
This submission investigates the problem of KV cache management in LLMs and proposes a method called KVP which adopts Reinforcement Learning to learn a strategy minimizing the future value of tokens to be evicted from the cache. While this is an interesting, novel approach, the submission does not make a compelling case that KVP improves the performance of SOTA methods in a significant way.

### Strengths
The formulation of KV cache eviction as a learning problem is original. The authors prove that, under two reasonable assumptions, the subset selection problem can be reduced to a ranking problem, which can then be formulated as an RL problem.

It is interesting that the policy requires only the keys, values, and their positions as input and no attention information. It is a strength of KVP that it can be pretained and does not incur any overhead at inference time.

I appreciate the ablation study demonstrating that supervised learning does not work and RL is necessary.

The paper is written clearly.

### Weaknesses
The experiments show that KVP achieves the best accuracy or perplexity on RULER and OASST2 for most cache sizes (fig 2) and competitive accuracy on the downstream tasks BOOLQ and ARC CHALLENGE (fig 3). However, there is typically a tradeoff between accuracy and latency and storage space. Therefore, the authors should also report the latency and storage space of the various tested methods.

The authors have only performed experiments with a version of Owen, and I would like to see whether their results generalize to another LLM, such as a Llama model.

While the learning approach of KVP is interesting and well-described, the main benefit of KVP remains unclear: is it better accuracy, reduced latency, reduced storage space, etc.?

### Questions
What is the latency and storage space of the various tested methods?

Are the perplexity gains reported in figure 2 statistically significant?

Why do you use different performance metrics for RULER (accuracy) and for OASST2 (perplexity)?

How does KVP work for another LLM, such as a Llama model?

What is the main benefit of KVP: better accuracy, reduced latency, reduced storage space, etc.?

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
The paper recasts KV-cache eviction as a reinforcement-learning (RL) ranking problem and proposes KV Policy (KVP). This is a lightweight policy that has per-head agents that learn to rank tokens by predicted future usefulness using only cache features, improving both efficiency and downstream accuracy. Under some low uniqueness and nestedness assumptions, eviction reduces to learning a single budget-agnostic ranking. KVP instantiates a Plackett–Luce stochastic ranking policy and leverages Gumbel-Sort for parallel, one-shot permutation sampling.  Each head’s agent is a small MLP scoring (key, value, position) and is trained offline on precomputed traces with a global reward equal to the negative cumulative future attention of evicted tokens summed over all budgets; the reward is normalized and optimized with REINFORCE using an RLOO baseline.  At inference time, the learned per-head rankings evict the lowest-ranked entries for any budget using only K/V/position (no queries or attention), requiring no extra LLM calls. The paper shows that KVP outperforms strong heuristics and attention-based baselines on RULER accuracy and OASST2-4k perplexity across most cache sizes. Additionaly, it generalizes competitively in zero-shot tests on BoolQ and ARC.

### Strengths
The paper provides a principled, budget-agnostic formulation of KV eviction and a practical, lightweight per-head policy that uses only cache-local features. Due to this, the technique is fast, query-free, and easy to deploy. Empirically, it outperforms strong heuristics and attention-aware baselines across cache sizes and tasks, showing robust generalization.

### Weaknesses
Training optimizes a proxy based on future attention computed from offline Q/K/V traces. This can misalign with downstream utility and requires precomputing and storing full-sequence Q/K/V (attention matrices omitted only due to size).

### Questions
1. Your reduction to a single budget-agnostic ranking hinges on uniqueness and nestedness. Do you have empirical evidence that generation traces satisfy these, and how sensitive is performance when nesting is violated (for instance, due to head/layer complementarity)?

2. Since KVP is attention-free and trained on offline "future-attention" signals, can you quantify alignment with downstream metrics vs. attention-aware policies that exploit query-specific information at prefil?

3. Could you add direct comparisons (same prefill-then-compress protocol and absolute-budget axes) to KeyFormer and MorphKV (Dialogue without Limits), and discuss where their key-centric selection/compression differs from your per-head RL ranking? Also, would their ideas change your conclusion about using a uniform per-head/layer budget?

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
In this paper, the authors propose KV Policy (KVP), a reinforcement learning (RL) framework that reframes cache eviction as a learning-to-rank problem. Each attention head in the transformer is assigned a lightweight RL agent trained on precomputed generation traces to predict each token's future utility using only key, value, and positional embeddings. Empirical results on RULER and OASST2-4k show that KVP outperforms other attention-based (e.g., SnapKV, TOVA) and attention-free (e.g., StreamingLLM, KeyDiff) baselines in accuracy and perplexity under the same budgets.

### Strengths
S1. This paper tackles an important problem of KV cache compression by eviction.

S2. This paper proposes to learn to evict the KV states, which is less explore in the area.

S3. The paper is well written and structured.

### Weaknesses
W1. Comparisons with recent sparse kv cache retrieval approaches, e.g., IceCache, ArkVale, MagicPig, InfiniGen, should also be included in addition to the kv cache eviction approaches.

W2. More backbone LLMs should be included. Qwen2.5 should be upgraded to Qwen3-8B. At least Llama3.1-8B should be included for the different variety of the models. One of a medium size LLMs, i.e., ~32B, should also be included to demonstrate the scalability of the proposed approach.

W3. More long-context benchmarks, e.g., longbench, should be included. Moreover, long-generation benchmarks, e.g., longgenbench, should also be included in experiments.

### Questions
Q1. What is the training cost, i.e., training data size, training time, etc., of the RL-based approach?

Q2. Would the RL-based approach be able to be combined with the heuristics-based approaches?

Q3. Can you show some failure case studies, such that we can better understand the pros and cons of the RL-based approach?

### Soundness
3

### Presentation
3

### Contribution
3
