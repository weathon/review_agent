# Strategic Scaling of Test-Time Compute: A Bandit Learning Approach

- Decision: Accept (Poster)
- Scores: 6, 2, 4

## Abstract
Scaling test-time compute has emerged as an effective strategy for improving the performance of large language models. However, existing methods typically allocate compute uniformly across all queries, overlooking variation in query difficulty. To address this inefficiency, we formulate test-time compute allocation as a novel bandit learning problem and propose adaptive algorithms that estimate query difficulty on the fly and allocate compute accordingly. Compared to uniform allocation, our algorithms allocate more compute to challenging queries while maintaining accuracy on easier ones. Among challenging queries, our algorithms further learn to prioritize solvable instances, effectively reducing excessive computing on unsolvable queries. We theoretically prove that our algorithms achieve better compute efficiency than uniform allocation and empirically validate their effectiveness on math and code benchmarks. Specifically, our algorithms achieve up to an 11.10\% performance improvement (15.04\% relative) on the MATH-500 dataset, up to 10.82\% (14.44\% relative) on the AIME25 dataset, and up to an 11.23\% performance improvement (15.29\% relative) on the LiveCodeBench dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the inefficiency of uniform test-time compute allocation for LLMs and proposes a bandit learning framework for strategic compute distribution. It formulates test-time compute allocation as a bandit problem, treating each query as an action and allocating compute sequentially to maximize the number of correctly answered queries. Theoretical analysis and experiments validates the algorithm’s superior compute efficiency over uniform allocation.

### Strengths
1) The paper formulates test-time compute allocation as a bandit problem and demonstrates its superiority over uniform allocation through both theoretical and empirical analyses.  
2) The framework supports multiple exploration strategies, aggregation methods, and settings, ensuring broad applicability.

### Weaknesses
1) The computational overhead of calculating Entropy and UCB should also be taken into account; comparing only the inference budget is insufficient. Explicit end-to-end run-time comparisons with each baseline are necessary.  
2) The impact of the Entropy and UCB hyper-parameter $\lambda$ has not been thoroughly analyzed; ablation experiments showing how $\lambda$ affects performance across different tasks should be provided.

### Questions
Please refer to the "Weakness" part.

### Soundness
3

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
4

### Summary
This paper tackles the problem of test-time compute allocation for large language models (LLMs). Traditional approaches such as Best-of-N sampling uniformly allocate the same amount of compute to each query, regardless of difficulty. The authors argue that this is inefficient, as some queries are trivially solvable while others are much harder. They reformulate the problem as a bandit learning task, where each query is an “arm” and compute allocation is treated as exploration. Their proposed algorithms estimate query difficulty adaptively and allocate compute dynamically across queries. The core method progressively eliminates solved queries, redistributing compute to unsolved or more uncertain ones. Extensions include exploration variants, support for streaming queries, and token-level compute control. The authors provide theoretical results showing superior compute efficiency over uniform allocation and validate the method empirically on math and code benchmarks, achieving improvements under the same compute budget.

### Strengths
- The idea of using bandit on allocating compute budget seems relatively novel.
- A theory is provided.

### Weaknesses
- The major concern is that the authors should use stronger PRMs as reward "oracles", e.g. Reasonflux-PRM [1]. On the one hand, Qwen2.5-Math-PRM-7B is relatively old compared with the latest SOTA PRMs. On the other hand, Qwen3-1.7B and DeepSeek-R1-Distill-Llama-8B are reasoning models, thus using Qwen2.5-Math-PRM-7B, which is not trained for reasoning models, as the reward oracle may lead to suboptimal results. Reasonflux-PRM is a more recent SOTA PRM that can be used as the reward oracle to further validate the effectiveness of the proposed method on both non-reasoning and reasoning models.
- Another concern on experiments is that self-consistency [2] is also a widely used algorithm to aggregate multiple response without PRMs, it will also be helpful to see how the proposed method will perform with that, such as [3].
- The current paper fails to provide a systematic related work. While it is understandable that test-time scaling algorithms are a relatively new concept and most work can be classified as concurrent and does not need to be compared, it is not acceptable not to formally mention them.


[1] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs.

[2] Self-Consistency Improves Chain of Thought Reasoning in Language Models.

[3] Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning.

### Questions
Please refer to the weaknesses above.

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
3

### Summary
This paper studies online algorithms for optimizing test-time compute allocation across a set of queries. Instead of uniformly allocating compute, the authors formulate the problem as a bandit learning task and develop an adaptive framework that learns to prioritize solvable instances, thereby avoiding excessive computation on unsolvable queries. The paper provides theoretical analysis showing improved compute efficiency over uniform allocation and demonstrates empirical gains on math and coding datasets.

### Strengths
1. The paper is clearly written and easy to follow.
2. The proposed algorithmic framework is simple yet effective. It is modular and flexible—different components such as exploration and exploitation strategies can be adjusted to fit various scenarios.
3. The work presents both theoretical and empirical results, and the experiments show consistent improvements over uniform allocation baselines.

### Weaknesses
1. While simplicity is a strength, the algorithmic contribution feels somewhat incremental. The framework extends a single-query setting to multiple queries in a relatively straightforward manner, and the theoretical analysis relies on standard probabilistic arguments. The results mainly show that the proposed method outperforms uniform allocation—which is expected—but do not clarify whether it outperforms other adaptive baselines such as ETC from a theoretical standpoint, or whether the approach is optimal or near-optimal.
2. The assumption of having access to an accurate reward model and threshold parameter $\gamma$ seems restrictive. It would be valuable to discuss how the method behaves or could be adapted under model inaccuracies.
3. The current analysis focuses on a high-budget regime where every query can, in principle, be solved correctly. In realistic scenarios with a limited budget $B$, it would be more informative to characterize how many queries are expected to be solved correctly given small $B$, or how the algorithm trades off between query coverage and accuracy.

### Questions
Please refer to the above weakness part.

### Soundness
3

### Presentation
4

### Contribution
2
