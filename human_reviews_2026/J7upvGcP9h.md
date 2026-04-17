# Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Test-time scaling methods improve the capabilities of large language models (LLMs) by increasing the amount of compute used during inference to make a prediction. Inference-time compute can be scaled *in parallel* by choosing among multiple independent solutions or *sequentially* through self-refinement. We propose Recursive Self-Aggregation (RSA), a test-time scaling method inspired by evolutionary methods that combines the benefits of both parallel and sequential scaling. Each step of RSA refines a population of candidate reasoning chains through aggregation of subsets to yield a population of improved solutions, which are then used as the candidate pool for the next iteration. RSA exploits the rich information embedded in the reasoning chains -- not just the final answers -- and enables bootstrapping from partially correct intermediate steps within different chains of thought. Empirically, RSA delivers substantial performance gains with increasing compute budgets across diverse tasks, model families and sizes. Notably, RSA enables Qwen3-4B-Instruct-2507 to achieve competitive performance with larger reasoning models, including DeepSeek-R1 and o3-mini (high), while outperforming purely parallel and sequential scaling strategies across AIME-25, HMMT-25, Reasoning Gym, LiveCodeBench-v6, and SuperGPQA. We further demonstrate that training the model to combine solutions via a novel aggregation-aware reinforcement learning approach yields significant performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduce RSA, a novel test-time scaling algorithm which maintain a set of response and keep improving through self-aggregation on different subsets. The proposed method are evaluated on different models and different datasets to show its effectiveness, and further show how such approach can be intergrated to aggregation aware-RL to improve make the RL more effective.

### Strengths
- The paper is well-written and easy to follow.
- The experiments are comprehensive, including many SOTA models on different datasets. A detailed ablation study is provided to support the parameter choice of the method and provide a recommendation on more general cases.
- The experiments on integrating RSA to RL are very insightful.

### Weaknesses
- While the experiments have included many models, one concern is that using self-aggregation on non-thinking models provides an unfair benefit to make the model think longer. Similar concerns applies on reasoning models as the response length is capped. While there are still a decent number of models left for the main results, ablation study and further discussions like aggregation-aware RL lack such support.
- The sampling parameter used is relatively weird, and making the reported numbers far away from typical performance of the models. For example, the reported AIME and HMMT numbers are around 4% lower than the official report, while Livecodebench v6 is significantly higher (14% more). Similar difference on numbers also applies to results in Fig. 4 even if the exact numbers are not provided. While I personally think the idea could still provide a certain amount of benefit, given the fact that most researchers and users will adopt the recommended parameters of the models, I would recommend the authors to rerun the experiments to align to the recommended parameters, i.e., _Temperature=0.7, TopP=0.8, TopK=20, and MinP=0_ for Qwen and _temperature to 0.6, top_p to 0.95_ for nematron.
- How is the budget-match actually conducted? Is it per-prompt? Or per-dataset?

### Questions
No specific questions. Please refer to the weakness. Specifically, I would expect some rerun on the experiments to make the numbers match the normal results of the models on the datasets.

### Soundness
2

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
This paper proposes a sequential-parallel test-time scaling method for LLMs. In each step, the proposed method first samples multiple response in parallel and then combine random subsets of these responses as LLM inputs for the next step.

### Strengths
* This paper connects sequential-parallel test-time scaling for LLMs with evolutionary algorithms.

### Weaknesses
* The primary weakness is the omission of highly relevant previous work on sequential-parallel scaling [1-3]. The paper must provide a detailed comparison outlining the technical differences and performance margins relative to these existing methods.
* Specifically, the basic underlying idea presented here has already been explored in prior work [1], making the technical contribution and insights appear highly incremental. 
* In addition, the claimed improvements are achieved through the use of an expensive RL process, which renders the comparison with non-RL baseline methods inherently unfair and diminishes the practical viability of the proposed approach.


[1] Wang, Fei, et al. "DynScaling: Efficient Verifier-free Inference Scaling via Dynamic and Integrated Sampling." arXiv preprint arXiv:2506.16043 (2025).

[2] Snell, Charlie Victor, et al. "Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning." The Thirteenth International Conference on Learning Representations. 2025.

[3] Wang, Junlin, et al. "Think deep, think fast: Investigating efficiency of verifier-free inference-time-scaling methods." arXiv preprint arXiv:2504.14047 (2025).

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes Recursive Self-Aggregation (RSA), a hybrid test-time scaling method that combines the benefits of both parallel and sequential scaling. Specifically, it recursively combines multiple reasoning chains and refines a population of candidates through repeated aggregation. This allows the models to reuse correct intermediate steps and self-correct the mistakes. RSA outperforms existing across diverse reasoning tasks such as math and code domains. The authors also propose integration with reinforcement learning by incorporating additional candidates in the optimization objective.

### Strengths
* The proposed RSA method is a novel and effective test-time scaling method that combines the benefits of both parallel and sequential scaling.
* Experiments are comprehensive: results on multiple benchmarks, such as HMMT, LiveCodeBench-v6, and SuperGPQA, demonstrate effective performance improvements over existing methods. The fact that applying RSA to a small model (Qwen3-Instruct-4B) can surpass a large model (DeepSeek-R1) is a significant finding.
* The writing is clear and easy to understand and follow.

### Weaknesses
* While RSA matches generation budgets with baselines, the paper does not provide detailed measurements of actual inference-time latency, which are crucial in real-world deployments. Note that methods such as majority voting should not increase the latency if conducted in parallel, while the proposed method might increase the latency, even if compute is not an issue.
* RSA's performance relies on carefully designed aggregation prompts. The paper lacks a principled or automated way to set aggregation prompts for new tasks or models.
* Lack of discussions and/or comparisons with other test-time scaling methods. For example, adaptive parallel scaling for reasoning methods, such as APR [1] and Multiverse [2], are also adaptive and could reason sequentially and in parallel. These methods demonstrate improved pareto frontier. Discussions and comparisons with these methods are important to highlight the advantages of RSA.

[1] Learning Adaptive Parallel Reasoning with Language Models. Pan, et al. https://arxiv.org/abs/2504.15466

[2] Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation. Yang, et al. https://arxiv.org/abs/2506.09991

### Questions
* Could the authors provide a detailed analysis of the latency of the proposed method, compared to the baseline methods? This would clarify RSA’s practicality under real deployment constraints and help quantify compute–performance trade-offs.
* Given the same base model, does the proposed method achieve an improved pareto frontier compared to adaptive parallel scaling methods, especially Multiverse, which solves Math reasoning problems as well?

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
This paper introduces a inference time paradigm called Recursive Self-Aggregation. In each step of RSA, N candidate solutions is kept and N subset of size K will be sampled and aggregated to get N new solutions for next step. Fixing the sampling budget, RSA achieves better performance than majority voting and self-refinement by a sizable margin. Furthermore, while standard long cot training hurts the performance after RSA, the authors propose an aggregation aware RL pipeline that can improve the end-to-end performance of RSA after training.

### Strengths
1. The paper clearly illustrated the proposed method.

2. The method is tested across multiple benchmarks and settings and the improvement over baseline is significant.

3. The proposed aggregation-aware RL show that this method can be used in training time as well.

### Weaknesses
1. As the authors noted, the method here bear certain similarity with the mixture-of-agents (MoA) paper. Either MoA or self-MoA should be included as a stronger baseline.

2. It is unclear why standard RL training will hurt the RSA performance. This may limit the usability of the methods in other settings.

### Questions
Please refer to the weakness for the major questions. Some other questions include

1. The largest K in the paper is only 4. Is the bounding factor here context length? 

2. Given a small K, the information mixing between different responses may be slow. How should T and N be scaled jointly?

### Soundness
3

### Presentation
3

### Contribution
3
