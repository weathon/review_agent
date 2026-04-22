# Adaptive Test-Time Compute Allocation via Training-Free Difficulty Proxies

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Large language models (LLMs) excel at complex tasks but incur prohibitive computational costs, particularly when using techniques like self-consistency that require multiple generation attempts. This paper addresses the challenge of adaptive test-time compute allocation. We propose a framework that leverages **training-free difficulty proxies** derived directly from the LLM generation process to distribute a fixed compute budget across the test queries, without requiring specialized training for the allocation mechanism. Our objective is to maximize the number of solved instances by dynamically allocating more compute to difficult instances and less to simpler ones, while adhering to a total budget constraint. We first introduce several training-free proxies and empirically demonstrate their effectiveness in estimating instance difficulty. We then design an adaptive allocation strategy guided by these proxies, which is theoretically grounded in a novel bandit formulation. Experiments across math (MATH, GSM8K), coding (LiveCodeBench), and Q\&A (e.g., GPQA-Diamond) benchmarks demonstrate that our method significantly outperforms both uniform budget allocation and training-based allocation baselines, solving substantially more problems under identical budget constraints. This work presents a practical and readily deployable approach to enhance the resource efficiency of LLM inference for demanding reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem of adaptive test-time compute allocation for large language models (LLMs). Traditional inference strategies, such as self-consistency and Best-of-N sampling, apply uniform compute budgets to all inputs, regardless of their difficulty. The authors propose DIPA, a training-free framework that dynamically allocates compute based on difficulty proxies derived directly from the LLM’s generation process (e.g., entropy, variance of gradient norms, generation length, and consistency). The method formulates adaptive allocation as a multi-armed bandit (MAB) problem, introducing probabilistic sampling to balance exploration and exploitation. Theoretically, they provide a regret bound showing that performance depends on the correlation between the proxy and true difficulty. Experiments on reasoning and coding benchmarks demonstrate that DIPA significantly outperforms uniform and training-based baselines under fixed compute budgets.

### Strengths
- The proposed solution is interesting. The reformulation of compute allocation as a multi-armed bandit with arm elimination is well-justified.
- The paper systematically investigates multiple training-free proxies (entropy, gradient norms, generation length, etc.) and analyzes their correlation with oracle difficult
- The paper is easy to follow.

### Weaknesses
- The individual proxies (e.g., generation length, entropy) are adapted from prior uncertainty estimation works. The main contribution lies in combining and evaluating them.
- The method's performance heavily depends on the proxy's correlation with true difficulty.
- Some works on test-time scaling are missing, such as [1,2,3].

[1] Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving.

[2] Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers.

[3] Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling.

### Questions
- Can the authors provide some guidelines for how to use these proxies in practical scenarios? For example, how to achieve good performance when encountering a brand new dataset?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper focuses on adaptive test-time compute allocation for LLM reasoning under a fixed budget, arguing that uniform “N-samples per instance” and training-based difficulty predictors are either inefficient or costly to train/deploy. The authors instead propose training-free difficulty proxies and formulates allocation as a stochastic MAB with arm-elimination, introducing DIPA, which samples instances with probability inversely proportional to their current difficulty, initializes from cheap input-based priors, and updates online using generation-based proxies.

### Strengths
This paper addresses the issue that training-based difficulty predictors are inefficient or costly, provide training-free difficulty proxies insights, and present corresponding experimental results.

### Weaknesses
- I believe that the Easy2Hard strategy aligns with DIPA’s fundamental principle of “probabilistically prioritizing arms (instances) estimated to be easier.” The reason Easy2Hard underperforms in the experiments is likely due to inaccurate initial difficulty estimation. Did the authors compare the performance of Easy2Hard when using existing training-based difficulty predictors to estimate difficulty before allocation?
- The proposed method requires sequential rollouts, which sacrifices parallelism. Although the overall compute budget is fixed, this design increases wall-clock time. While the authors mention that DIPA could, in theory, be extended to a batched version, I think that sampling multiple instances simultaneously under the same probability distribution would likely result in over-selecting easy instances, leading to budget waste on already simple problems.

### Questions
See weakness

### Soundness
2

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
4

### Summary
This paper addresses the inefficiency of uniform test-time compute allocation for LLMs and proposes DIPA, a training-free framework for adaptive compute distribution. It leverages training-free difficulty proxies derived from LLM inputs or generation processes to estimate instance difficulty. It reformulates the allocation problem as a multi-armed bandit task with arm elimination upon success, using a probabilistic policy to balance exploration and exploitation. Experiments on varies benchmarks show DIPA outperforms uniform allocation, deterministic strategies, and training-based baselines, solving more problems under fixed budgets.

### Strengths
1) The proposed method achieves effective inference-budget allocation without any additional training, outperforming baselines such as BoN and SC under the same compute budget.  
2) The formalization as a multi-armed bandit and the accompanying regret-bound analysis link proxy quality to performance, providing a theoretical grounding for the approach.

### Weaknesses
1) The experimental baselines are limited to relatively easy problems, the authors should include AIME24, a benchmark commonly used in recent reasoning-model studies, which demands greater reasoning capability and larger budgets and would better highlight the proposed method’s incremental value.  
2) The selected training-free proxies are rather trivial.  
3) Although casting budget allocation as a multi-armed bandit is interesting, it is not “a novel bandit formulation” but a very classical algorithmic template; the authors should acknowledge this.  
4) The computational overhead of obtaining the difficulty proxies must also be accounted for—comparing only inference budgets is insufficient, explicit end-to-end run-time comparisons with each baseline are necessary.

### Questions
Please refer to the "Weakness" part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the problem of adaptive resource allocation during test-time scaling, proposing a training-free difficulty proxy metric and providing analysis and proofs regarding the modeling and associated regret bounds via multi-armed bandits. The effectiveness of the approach is demonstrated across mathematical, programming, and document-related problems.

### Strengths
- The problem studied in this paper is highly important and practical.
- The proposed method has a certain mathematical foundation, though its relevance to practice remains limited.

### Weaknesses
- The problem is oversimplified, particularly in terms of measuring computational budget solely based on the number of steps. More specific measurements, such as FLOPs and time, are needed.
- While the paper emphasizes the theoretical contributions of the method, further demonstration of its practical application value and significance is required.
- Evaluations of computational overhead and FLOPs for inference deployment should be provided with specific calculations.

### Questions
- The paper also mentions the application of reward models and suggests incorporating more procedural information for difficulty analysis. It is recommended to consider the application analysis of generative reward models, such as GenGRM:

[1] GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning

### Soundness
3

### Presentation
3

### Contribution
2
